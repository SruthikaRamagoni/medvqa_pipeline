"""
agents/training_agent.py

TrainingAgent — loads encoded features from FeatureEngineeringAgent
and fine-tunes the model selected by ModelSelectionAgent using PEFT/LoRA.

Changes from previous version (fixes the
"not enough values to unpack (expected 3, got 1)" crash on Qwen2-VL /
Qwen2.5-VL training):

  ROOT CAUSE #1 (fixed previously) — wrong batch collation strategy
  ------------------------------------------------------------------
  Qwen2-VL / Qwen2.5-VL processors emit `pixel_values` already patchified,
  per example, as a 2D tensor of shape (num_patches, patch_dim) — e.g.
  (256, 1176) — NOT the classic (C, H, W) layout that BLIP-2 / InstructBLIP
  use. They pair this with `image_grid_thw`, shape (3,) per image, telling
  the vision tower how to reshape the flat patch sequence back into a
  spatial (t, h, w) grid.

  `default_data_collator` blindly `torch.stack()`s every column, which is
  wrong for both these fields: `pixel_values` has a different patch count
  per example (different image resolutions) so it must be CONCATENATED,
  not stacked; `image_grid_thw` must be stacked/concatenated into shape
  (num_images, 3). This was fixed with a dedicated `_qwen_vl_collator()`.

  ROOT CAUSE #2 (fixed in this version) — stray leading batch dim
  ------------------------------------------------------------------
  After fixing pixel_values/image_grid_thw collation, a second, distinct
  bug surfaced: "The shape of the mask [1, 128] at index 0 does not match
  the shape of the indexed tensor [4, 1] at index 0".

  Qwen2.5-VL's `get_rope_index()` requires `input_ids` and
  `attention_mask` to be exactly 2D: (batch_size, sequence_length). If
  FeatureEngineeringAgent calls the HF processor per-example with
  `return_tensors="pt"` (its natural mode) and saves the result to the HF
  Dataset WITHOUT first calling `.squeeze(0)`, each example's `input_ids`
  / `attention_mask` / `labels` is stored with a stray leading dim of 1:
  shape (1, seq_len) instead of (seq_len,). `torch.stack()`-ing 4 such
  examples then produces a 3D tensor (4, 1, seq_len) instead of the
  required 2D (4, seq_len) — and `get_rope_index`'s internal per-batch-item
  masking logic, written for strictly 2D input, throws a mask/tensor shape
  mismatch when it receives this 3D tensor.

  This was confirmed by direct reproduction: stacking four (1, 128)
  per-example tensors produces (4, 1, 128); squeezing the stray leading
  dim of 1 before stacking (only when present — see below) produces the
  correct (4, 128) that matches Qwen2.5-VL's documented contract.

  FIX
  ---
  1. `_qwen_vl_collator()` now squeezes a stray leading dim of 1 off
     `input_ids` / `attention_mask` / `labels` (only if present — a
     dataset that already stores (seq_len,) per example is left
     untouched, so this is non-destructive either way) BEFORE stacking,
     guaranteeing the batch is 2D (batch_size, seq_len) as required.
  2. `image_grid_thw` is reshaped to (-1, 3) per example before
     concatenating, so it's robust whether stored as (3,) or (1, 3) per
     example.
  3. `pixel_values` is concatenated (not stacked) across the batch, since
     patch counts vary by image resolution.
  4. `_prepare_columns()` / `_normalize_pixel_values()` take `hf_id` and
     EXPLICITLY skip the (C, H, W) rank-3 pixel_values normalization for
     Qwen-VL models — that normalization was written for classic vision
     encoders (BLIP-2 / InstructBLIP) and does not apply here; the 2D
     (num_patches, patch_dim) shape is correct as-is.
  5. `_build_trainer()` receives `hf_id` so it can select the correct
     collator and forces `remove_unused_columns=False` (required for any
     vision model — Trainer's automatic column pruning doesn't understand
     pixel_values / image_grid_thw).
  6. Model-family detection centralized in `_is_qwen_vl()` so it's
     consistent across all call sites instead of repeated string checks.

  All fixes verified by direct reproduction against realistic per-example
  tensor shapes (see accompanying test script) before being included here.

  All other behaviour (PhiData + Groq pattern, CLASS_MAP, HF_ID_OVERRIDES,
  strict-no-fallback loading, early stopping, cosine LR schedule, gradient
  clipping, failure_reason propagation for ModelSelectionAgent retries) is
  unchanged. This file is designed to work with whatever model_plan
  ModelSelectionAgent.select_model() returns — no changes needed on that
  side.
"""

from phi.agent import Agent
from phi.model.groq import Groq

from typing import Any, Dict, List, Optional, Tuple
import json, logging, re
from pathlib import Path

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path("./artifacts/checkpoints")

# Tensor columns the Trainer accepts — all others are dropped
TENSOR_COLUMNS = {
    "input_ids", "attention_mask", "pixel_values",
    "labels", "decoder_input_ids", "token_type_ids",
    "image_patches", "image_sizes", "image_grid_thw",
    "qformer_input_ids", "qformer_attention_mask",
}

# Optional per-model hf_id overrides (populated by ModelSelectionAgent hints)
HF_ID_OVERRIDES: Dict[str, str] = {}


class TrainingAgent:
    """
    Fine-tunes the model selected by ModelSelectionAgent on features
    produced by FeatureEngineeringAgent. Loads encoded HF Dataset from
    disk, applies LoRA, and trains.

    STRICT MODE: if the requested model fails to load, the agent halts and
    returns a failed status — it does NOT silently fall back to a lighter
    model. (Retries happen by calling ModelSelectionAgent.select_model()
    again with failure_context — see CoordinatorAgent / main.py.)

    Handles two distinct vision-language batching layouts automatically:
      • Qwen2-VL / Qwen2.5-VL family — pre-patchified pixel_values +
        image_grid_thw, concatenation-based collation.
      • Classic vision models (BLIP-2, InstructBLIP, etc.) — (C, H, W)
        pixel_values, stack-based collation via default_data_collator.

    Return dict always includes `failure_reason` (empty string on success)
    so the Coordinator can pass it directly to ModelSelectionAgent for a
    retry.
    """

    def __init__(self, model_id: str = "mistral"):
        self.agent = Agent(
            name="TrainingAgent",
            model=Groq(id="llama-3.1-8b-instant"),
            instructions=[
                "You are a machine learning training expert.",
                "You receive training result metrics and assess whether they are acceptable.",
                "Always reply with ONLY a JSON object like this:",
                '{"status": "ok", "train_loss": "<value>", "message": "<one sentence>"}',
                "Do not write code. Do not add any text outside the JSON.",
            ],
            show_tool_calls=False,
            markdown=False,
        )

    # ── Public method ─────────────────────────────────────────────────────────

    def train(
        self,
        feature_path: str,
        model_plan:   Dict[str, Any],
        device:       str = "",
    ) -> Dict[str, Any]:
        """
        Fine-tune the model using pre-encoded feature dataset.

        Args:
            feature_path : Path returned by FeatureEngineeringAgent (base dir).
                           Contains train/ and val/ subdirectories.
            model_plan   : Dict from ModelSelectionAgent.select_model().
            device       : 'cuda' | 'cpu' | '' (auto-detect).

        Returns:
            Dict with checkpoint_path, train_loss, status, model_used,
            and failure_reason (empty string on success).
        """
        if not device:
            device = self._detect_device()
        logger.info(f"[Training] Device: {device}")

        train_ds, val_ds = self._load_feature_datasets(feature_path)

        if train_ds is None or len(train_ds) == 0:
            return self._fail(
                f"No encoded training data found at: {feature_path}",
                model_used="",
            )

        logger.info(
            f"[Training] Loaded → Train={len(train_ds)}  Val={len(val_ds)}"
        )
        logger.info(f"[Training] Columns: {train_ds.column_names}")

        try:
            model, tokenizer, active_plan = self._load_model(model_plan, device)
        except RuntimeError as e:
            return self._fail(str(e), model_used=model_plan.get("hf_id", ""))

        hf_id        = active_plan["hf_id"]
        name         = active_plan["name"]
        architecture = active_plan.get("architecture",          "seq2seq")
        lora_r       = active_plan.get("lora_r",                8)
        lora_alpha   = active_plan.get("lora_alpha",            16)
        lora_drop    = active_plan.get("lora_dropout",          0.05)
        target_mods  = active_plan.get("target_modules",        ["q", "v"])
        batch_size   = active_plan.get("batch_size",            2)
        epochs       = active_plan.get("epochs",                3)
        lr           = active_plan.get("learning_rate",         2e-4)
        precision    = active_plan.get("precision",             "fp16")
        # ── Configurable training knobs ──────────────────────────────────────
        max_grad_norm    = active_plan.get("max_grad_norm",               1.0)
        es_patience      = active_plan.get("early_stopping_patience",     2)
        lr_scheduler     = active_plan.get("lr_scheduler_type",           "cosine")

        is_qwen_vl = self._is_qwen_vl(hf_id)
        if is_qwen_vl:
            logger.info(
                f"[Training] Detected Qwen-VL family model ({hf_id}) — "
                f"will use patch-concatenation collator instead of "
                f"default_data_collator."
            )

        safe_name = name.replace(" ", "_").replace("/", "_")
        out_dir   = CHECKPOINT_DIR / safe_name
        out_dir.mkdir(parents=True, exist_ok=True)

        model = self._apply_lora(
            model, lora_r, lora_alpha, lora_drop, target_mods, architecture
        )

        train_ds, val_ds = self._prepare_columns(train_ds, val_ds, hf_id)

        if len(train_ds) == 0:
            return self._fail(
                "No valid tensor columns after column preparation.",
                model_used=hf_id,
                checkpoint_path=str(out_dir),
            )

        try:
            trainer = self._build_trainer(
                model, tokenizer, train_ds, val_ds,
                str(out_dir), batch_size, epochs, lr,
                precision, device,
                max_grad_norm=max_grad_norm,
                es_patience=es_patience,
                lr_scheduler=lr_scheduler,
                hf_id=hf_id,
            )
            metrics    = self._run_training(trainer)
            train_loss = round(float(metrics.get("train_loss", 0.0)), 4)
        except Exception as e:
            logger.error(f"[Training] Training failed: {e}")
            return self._fail(
                f"Training loop error: {e}",
                model_used=hf_id,
                checkpoint_path=str(out_dir),
            )

        self._save(model, tokenizer, str(out_dir), active_plan)
        logger.info(f"[Training] Complete. Loss={train_loss}  Ckpt={out_dir}")

        assessment = self._get_llm_assessment(
            hf_id, lora_r, lora_alpha, epochs,
            batch_size, lr, train_loss, out_dir,
        )
        assessment["checkpoint_path"] = str(out_dir)
        assessment["model_used"]      = hf_id
        assessment["train_loss"]      = str(train_loss)
        assessment["failure_reason"]  = ""          # success — no failure
        return assessment

    # ── Compatibility method ──────────────────────────────────────────────────

    def train_from_processed(
        self,
        processed_data_path: str,
        model_plan:          Dict[str, Any],
        device:              str = "",
    ) -> Dict[str, Any]:
        """
        Fallback: if FeatureEngineeringAgent was not run, encode inline.
        Keeps backward compatibility with old pipeline.
        """
        logger.info(
            "[Training] No feature_path provided. "
            "Running inline encoding via FeatureEngineeringAgent …"
        )
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from agents.feature_engineering_agent import FeatureEngineeringAgent

        fe_agent  = FeatureEngineeringAgent()
        fe_result = fe_agent.engineer_features(
            processed_data_path, model_plan, device or self._detect_device()
        )
        if fe_result.get("status") == "failed":
            return self._fail(
                f"Inline feature engineering failed: {fe_result.get('message')}",
                model_used="",
            )
        return self.train(fe_result["feature_path"], model_plan, device)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _fail(
        self,
        message:         str,
        model_used:      str = "",
        checkpoint_path: str = "",
    ) -> Dict[str, Any]:
        """
        Unified failure return.  Always includes `failure_reason` so the
        Coordinator can pass it to ModelSelectionAgent without extra checks.
        """
        logger.error(f"[Training] FAILED — {message}")
        return {
            "status":          "failed",
            "message":         message,
            "checkpoint_path": checkpoint_path,
            "model_used":      model_used,
            "failure_reason":  message,   # ← key used by retry logic
        }

    def _detect_device(self) -> str:
        try:
            import torch
            if torch.cuda.is_available():         return "cuda"
            if torch.backends.mps.is_available(): return "mps"
        except ImportError:
            pass
        return "cpu"

    def _is_qwen_vl(self, hf_id: str) -> bool:
        """
        Centralized model-family detection. Matches both Qwen2-VL and
        Qwen2.5-VL hf_id naming conventions (e.g.
        'Qwen/Qwen2.5-VL-3B-Instruct', 'Qwen/Qwen2-VL-7B-Instruct').

        Kept as a single source of truth so column preparation and
        collator selection never disagree about which model family is
        active.
        """
        h = hf_id.lower()
        return "qwen" in h and "-vl" in h

    # ── Load encoded datasets ─────────────────────────────────────────────────

    def _load_feature_datasets(self, feature_path: str):
        if not feature_path or not Path(feature_path).exists():
            return None, None
        try:
            from datasets import load_from_disk

            train_path = Path(feature_path) / "train"
            val_path   = Path(feature_path) / "val"

            if not train_path.exists():
                logger.error(f"[Training] train/ not found in {feature_path}")
                return None, None

            train_ds = load_from_disk(str(train_path))
            val_ds   = load_from_disk(str(val_path)) if val_path.exists() else None

            if val_ds is None or len(val_ds) == 0:
                split    = train_ds.train_test_split(test_size=0.1, seed=42)
                train_ds = split["train"]
                val_ds   = split["test"]
                logger.info("[Training] Created val split from train (10%).")

            return train_ds, val_ds
        except Exception as e:
            logger.error(f"[Training] Feature dataset load failed: {e}")
            return None, None

    # ── Model loading — strict, no fallback ───────────────────────────────────

    def _load_model(
        self, model_plan: Dict, device: str
    ) -> Tuple[Any, Any, Dict]:
        hf_id = model_plan.get("hf_id", "")
        if not hf_id:
            raise RuntimeError("No hf_id specified in model_plan.")

        # Apply any runtime overrides
        hf_id = HF_ID_OVERRIDES.get(hf_id, hf_id)
        effective_plan = {**model_plan, "hf_id": hf_id}

        logger.info(f"[Training] Loading requested model: {hf_id}")
        try:
            model, tok = self._load_single(
                hf_id=hf_id,
                loader_hint=effective_plan.get("loader", "auto"),
                use_4bit=effective_plan.get("use_4bit", False),
                device=device,
                precision=effective_plan.get("precision", "fp16"),
            )
            logger.info(f"[Training] Successfully loaded: {hf_id}")
            return model, tok, effective_plan
        except Exception as e:
            raise RuntimeError(
                f"Selected model '{hf_id}' failed to load. "
                f"Halting — no fallback will be attempted.\nError: {e}"
            )

    def _load_single(
        self,
        hf_id:       str,
        loader_hint: str,
        use_4bit:    bool,
        device:      str,
        precision:   str,
    ) -> Tuple[Any, Any]:
        import torch
        import transformers as tf

        # ── Sanity check: PyTorch backend actually usable ──────────────────────
        # transformers' lazy-loading masks a missing/broken torch backend as a
        # generic "Unrecognized configuration class ... for AutoModelForX"
        # error from whichever fallback class happened to be a real class.
        # Catch it here with an explicit, actionable message instead.
        try:
            _ = torch.tensor([0.0])
        except Exception as e:
            raise RuntimeError(
                "PyTorch backend is not usable in this environment "
                f"(torch.tensor() failed: {e}). transformers' lazy-loader "
                "will silently substitute Placeholder objects for "
                "vision-language model classes when this happens, producing "
                "misleading 'Unrecognized configuration class' errors. "
                "Fix the PyTorch install before retrying."
            )

        dtype = (
            torch.float16
            if precision == "fp16" and device == "cuda"
            else torch.float32
        )

        load_kw: Dict = dict(
            pretrained_model_name_or_path=hf_id,
            trust_remote_code=True,
        )

        if use_4bit and device == "cuda":
            from transformers import BitsAndBytesConfig
            load_kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            load_kw["device_map"] = "auto"
        elif device == "cuda":
            load_kw["dtype"]      = dtype
            load_kw["device_map"] = "auto"
        else:
            load_kw["dtype"] = dtype

        CLASS_MAP: Dict[str, List[str]] = {
            "AutoModelForSeq2SeqLM": [
                "AutoModelForSeq2SeqLM",
            ],
            "Blip2ForConditionalGeneration": [
                "Blip2ForConditionalGeneration",
                "AutoModelForImageTextToText",
            ],
            "InstructBlipForConditionalGeneration": [
                "InstructBlipForConditionalGeneration",
                "AutoModelForImageTextToText",
            ],
            "Qwen2_5_VLForConditionalGeneration": [
                "Qwen2_5_VLForConditionalGeneration",
                "AutoModelForImageTextToText",
            ],
            "Qwen2VLForConditionalGeneration": [
                "Qwen2VLForConditionalGeneration",
                "AutoModelForImageTextToText",
            ],
            # FIX: AutoModelForVision2Seq was removed in modern transformers
            # (5.x) and replaced by AutoModelForImageTextToText, which is the
            # universal Auto class covering Qwen2-VL, Qwen2.5-VL, InstructBLIP,
            # BLIP-2, LLaVA, and most current vision-language models. Keeping
            # the old name as a key (mapped to the new class) preserves
            # backward compatibility with any model_plan that still sets
            # loader="AutoModelForVision2Seq".
            "AutoModelForVision2Seq": [
                "AutoModelForImageTextToText",
            ],
            "AutoModelForImageTextToText": [
                "AutoModelForImageTextToText",
            ],
            "auto": [
                "InstructBlipForConditionalGeneration",
                "Blip2ForConditionalGeneration",
                "Qwen2_5_VLForConditionalGeneration",
                "Qwen2VLForConditionalGeneration",
                "AutoModelForSeq2SeqLM",
                "AutoModelForImageTextToText",   # FIX: was AutoModelForVision2Seq
                "AutoModelForCausalLM",
            ],
        }
        class_order = CLASS_MAP.get(loader_hint, CLASS_MAP["auto"])

        # Explicit submodule fallback paths for classes that are sometimes NOT
        # exported at the top-level `transformers` namespace depending on
        # version (this was the root cause of the InstructBLIP failure: the
        # top-level getattr silently returned None and the loop fell through
        # to AutoModelForCausalLM, which can't parse InstructBlipConfig at all).
        SUBMODULE_FALLBACKS: Dict[str, str] = {
            "InstructBlipForConditionalGeneration":
                "transformers.models.instructblip",
            "Blip2ForConditionalGeneration":
                "transformers.models.blip_2",
            "Qwen2_5_VLForConditionalGeneration":
                "transformers.models.qwen2_5_vl",
            "Qwen2VLForConditionalGeneration":
                "transformers.models.qwen2_vl",
        }

        def _is_real_class(cls) -> bool:
            """
            transformers' lazy-loading system (esp. v5.x) can return a
            'Placeholder' stand-in object for classes whose backend
            (e.g. PyTorch) failed to import or isn't installed in this
            environment — getattr() does NOT return None in that case,
            it returns the Placeholder, which only raises when actually
            called. This check catches that before we waste a load attempt.
            """
            if cls is None:
                return False
            return "Placeholder" not in type(cls).__name__ and \
                   "Placeholder" not in getattr(cls, "__name__", "")

        def _resolve_class(cls_name: str):
            """
            Resolve a model class by name.
            1. Try top-level transformers namespace (normal case).
            2. If missing or a lazy-load Placeholder, try the documented
               submodule path.
            Returns (cls_or_None, resolution_note) — note explains *why*
            it was unavailable when both attempts fail, instead of a bare
            'skipping' with no diagnostic value.
            """
            cls = getattr(tf, cls_name, None)
            if _is_real_class(cls):
                return cls, "top-level"
            if cls is not None:
                logger.debug(
                    f"[Training] {cls_name} resolved to a lazy-load "
                    f"Placeholder (backend e.g. torch likely missing/broken) "
                    f"— trying submodule fallback."
                )

            submodule_path = SUBMODULE_FALLBACKS.get(cls_name)
            if submodule_path:
                try:
                    import importlib
                    mod = importlib.import_module(submodule_path)
                    sub_cls = getattr(mod, cls_name, None)
                    if _is_real_class(sub_cls):
                        return sub_cls, f"submodule:{submodule_path}"
                    if sub_cls is not None:
                        return None, (
                            f"resolved via submodule but still a Placeholder "
                            f"— required backend (torch) is likely not "
                            f"importable in this environment"
                        )
                except Exception as e:
                    return None, f"submodule import failed: {e}"

            return None, "not found at top-level and no submodule fallback registered"

        model         = None
        last_error    = None
        skip_reasons:  List[str] = []
        attempt_errors: List[Tuple[str, str]] = []   # (cls_name, error_str) for EVERY real attempt

        for cls_name in class_order:
            cls, note = _resolve_class(cls_name)
            if cls is None:
                skip_reasons.append(f"{cls_name} ({note})")
                logger.warning(
                    f"[Training] Class {cls_name} unavailable — {note}. Skipping."
                )
                continue
            try:
                model = cls.from_pretrained(**load_kw)
                logger.info(
                    f"[Training] {hf_id} loaded with {cls_name} ({note})"
                )
                break
            except Exception as e:
                last_error = e
                attempt_errors.append((cls_name, str(e)))
                logger.warning(f"[Training] {cls_name} load failed for {hf_id}: {e}")

        if model is None:
            reason_block = (
                "; ".join(skip_reasons) if skip_reasons else "no classes skipped"
            )

            # The class the caller actually asked for (loader_hint) is the
            # one whose error matters most — earlier versions of this agent
            # only kept the LAST error in the chain, which is almost always
            # the least relevant fallback (e.g. AutoModelForCausalLM), and
            # silently buried the real reason the intended class failed.
            # Surface that class's own error explicitly here.
            #
            # NOTE: loader_hint may be an alias (e.g. the deprecated
            # "AutoModelForVision2Seq") that CLASS_MAP maps to a different
            # actual class (e.g. "AutoModelForImageTextToText"). Match
            # against the FIRST entry actually attempted in class_order
            # (the primary/intended class for this hint) rather than the
            # raw hint string, so the PRIMARY block is correct even when
            # an alias was resolved.
            primary_class_name = class_order[0] if class_order else loader_hint
            primary_error = next(
                (err for name, err in attempt_errors if name == primary_class_name),
                None,
            )

            per_class_block = "\n".join(
                f"  - {name}: {err[:300]}" for name, err in attempt_errors
            ) or "  (no class was actually attempted — all were unavailable)"

            alias_note = (
                f" (loader_hint '{loader_hint}' resolved to '{primary_class_name}')"
                if primary_class_name != loader_hint else ""
            )

            primary_block = (
                f"\nPRIMARY (requested loader '{loader_hint}'{alias_note}) error:\n"
                f"  {primary_error[:500]}\n"
                if primary_error else
                f"\nNote: intended class '{primary_class_name}' for requested "
                f"loader '{loader_hint}' was never actually attempted (see "
                f"skipped classes below) — this is itself likely the real "
                f"problem.\n"
            )

            raise RuntimeError(
                f"Cannot load {hf_id} with any class in {class_order}.\n"
                f"Skipped (unavailable) classes: {reason_block}\n"
                f"{primary_block}"
                f"All attempted classes and their errors:\n{per_class_block}\n"
                f"Last error (informational only — see PRIMARY above for the "
                f"actual relevant failure): {last_error}"
            )

        for method in ("gradient_checkpointing_enable", "enable_input_require_grads"):
            fn = getattr(model, method, None)
            if fn:
                try:
                    fn()
                except Exception:
                    pass

        tokenizer = None
        try:
            from transformers import AutoProcessor
            tokenizer = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
        except Exception:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)

        inner = getattr(tokenizer, "tokenizer", tokenizer)
        if hasattr(inner, "pad_token") and inner.pad_token is None:
            inner.pad_token = inner.eos_token

        return model, tokenizer

    # ── LoRA ──────────────────────────────────────────────────────────────────

    def _apply_lora(self, model, r, alpha, dropout, target_modules, architecture):
        try:
            from peft import (
                LoraConfig, TaskType,
                get_peft_model, prepare_model_for_kbit_training,
            )
            try:
                model = prepare_model_for_kbit_training(model)
            except Exception:
                pass

            task_type = (
                TaskType.SEQ_2_SEQ_LM
                if architecture == "seq2seq"
                else TaskType.CAUSAL_LM
            )
            cfg = LoraConfig(
                r=r, lora_alpha=alpha, lora_dropout=dropout,
                target_modules=target_modules,
                bias="none", task_type=task_type,
            )
            model = get_peft_model(model, cfg)
            model.print_trainable_parameters()
            logger.info(
                f"[Training] LoRA applied: r={r}  alpha={alpha}  task={task_type}"
            )
        except ImportError:
            logger.warning("[Training] PEFT not installed — training without LoRA.")
        except Exception as e:
            logger.warning(f"[Training] LoRA skipped: {e}")
        return model

    # ── Column preparation ────────────────────────────────────────────────────

    def _prepare_columns(self, train_ds, val_ds, hf_id: str = ""):
        drop_tr = [c for c in train_ds.column_names if c not in TENSOR_COLUMNS]
        drop_vl = [c for c in val_ds.column_names   if c not in TENSOR_COLUMNS]
        if drop_tr:
            logger.info(f"[Training] Dropping non-tensor columns: {drop_tr}")
            train_ds = train_ds.remove_columns(drop_tr)
        if drop_vl:
            val_ds = val_ds.remove_columns(drop_vl)

        if ("labels"    not in train_ds.column_names and
                "input_ids" in  train_ds.column_names):
            train_ds = train_ds.map(
                lambda x: {"labels": x["input_ids"]}, batched=True
            )
            val_ds = val_ds.map(
                lambda x: {"labels": x["input_ids"]}, batched=True
            )

        is_qwen_vl = self._is_qwen_vl(hf_id)

        if "pixel_values" in train_ds.column_names:
            if is_qwen_vl:
                # Qwen2-VL / Qwen2.5-VL emit pre-patchified pixel_values of
                # shape (num_patches, patch_dim) — e.g. (256, 1176) — per
                # example. This is the CORRECT shape, paired with
                # image_grid_thw. Do NOT run the (C, H, W) rank-3
                # normalization here; it was written for classic vision
                # encoders (BLIP-2 / InstructBLIP) and does not apply.
                logger.info(
                    "[Training] Skipping pixel_values rank normalization — "
                    f"{hf_id} is a Qwen-VL family model and uses "
                    "pre-patchified (num_patches, patch_dim) pixel_values "
                    "paired with image_grid_thw. This shape is expected, "
                    "not a bug."
                )
            else:
                try:
                    train_ds, val_ds = self._normalize_pixel_values(
                        train_ds, val_ds, column="pixel_values"
                    )
                except Exception as e:
                    logger.warning(f"[Training] pixel_values normalization skipped: {e}")

        logger.info(f"[Training] Final columns: {train_ds.column_names}")
        return train_ds, val_ds

    def _normalize_pixel_values(self, train_ds, val_ds, column: str = "pixel_values"):
        """
        Ensure `column` is exactly 3D per example: (channels, height, width).

        Only ever called for classic (non Qwen-VL) vision-language models —
        see _prepare_columns(). Qwen-VL's pre-patchified 2D pixel_values
        must never reach this function, since collapsing it toward rank 3
        would corrupt the patch layout.

        ROOT CAUSE OF "too many values to unpack (expected 4)" (the classic-
        model failure mode this function fixes):
        A single fixed unwrap (`pv[0]`) is correct ONLY if the column is
        exactly one level too deep. If the processor wrapped the image in
        an extra list (e.g. producing shape (1, 1, 3, H, W) or
        (1, 1, 1, 3, H, W) per row instead of (3, H, W)), a single unwrap
        pass leaves a residual singleton dimension, and the Trainer
        collates into a 5D+ tensor instead of the 4D (batch, channels,
        height, width) the model expects — raising "too many values to
        unpack (expected 4)" inside `B, C, H, W = pixel_values.shape`.

        Fix: repeatedly strip leading singleton dimensions (via torch
        tensor rank, not list-nesting guesses) until each example is
        exactly rank-3 (C, H, W), regardless of how many extra wrapping
        levels were introduced upstream.
        """
        import torch

        sample = train_ds[0][column]
        t = torch.as_tensor(sample)
        original_shape = tuple(t.shape)
        logger.info(f"[Training] {column} per-example shape before normalize: {original_shape}")

        strips_needed = max(0, t.dim() - 3)

        if strips_needed == 0 and t.dim() == 3:
            logger.info(f"[Training] {column} already correct rank (3) — no normalization needed.")
            return train_ds, val_ds

        if t.dim() < 3:
            logger.warning(
                f"[Training] {column} has rank {t.dim()} (< 3) — cannot "
                f"normalize automatically. Leaving as-is; expect a shape "
                f"error downstream if this is wrong. (If this is a "
                f"Qwen-VL model, this function should not have been "
                f"called at all — check _is_qwen_vl() detection.)"
            )
            return train_ds, val_ds

        def _strip_leading_singletons(arr, n: int):
            """Strip n leading dims of size 1 from a nested-list array."""
            for _ in range(n):
                if isinstance(arr, list) and len(arr) == 1:
                    arr = arr[0]
                else:
                    break
            return arr

        def normalize_batch(batch):
            return {
                column: [
                    _strip_leading_singletons(pv, strips_needed)
                    for pv in batch[column]
                ]
            }

        train_ds = train_ds.map(normalize_batch, batched=True)
        val_ds   = val_ds.map(normalize_batch,   batched=True)

        new_sample = train_ds[0][column]
        new_shape  = tuple(torch.as_tensor(new_sample).shape)
        logger.info(
            f"[Training] {column} normalized: {original_shape} → {new_shape} "
            f"(stripped {strips_needed} leading singleton dim(s))"
        )
        if len(new_shape) != 3:
            logger.warning(
                f"[Training] {column} still not rank-3 after normalization "
                f"(got {new_shape}). Training may still fail with a shape "
                f"error — inspect FeatureEngineeringAgent's image encoding "
                f"step for this model family."
            )

        return train_ds, val_ds

    # ── Collators ─────────────────────────────────────────────────────────────

    def _qwen_vl_collator(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Batch collator for Qwen2-VL / Qwen2.5-VL.

        Handles two distinct shape problems that default_data_collator's
        blanket torch.stack() gets wrong for this model family:

        1. pixel_values / image_grid_thw — ragged per-example shapes
           ---------------------------------------------------------
          • pixel_values   — shape (num_patches_i, patch_dim) per example,
                              where num_patches_i varies with each image's
                              resolution. Must be torch.cat()'d along dim 0
                              into a single (total_patches, patch_dim)
                              tensor — exactly what the Qwen vision tower's
                              forward() expects, alongside image_grid_thw,
                              for the whole batch.
          • image_grid_thw — logically (3,) per image ([t, h, w]), but may
                              be stored as (3,) or (1, 3) depending on how
                              FeatureEngineeringAgent saved it. Reshaped to
                              (-1, 3) per example, then concatenated into
                              (num_images, 3) so the model receives one
                              full (t, h, w) triple per image.

        2. input_ids / attention_mask / labels — stray leading batch dim
           ---------------------------------------------------------------
          HF processors return (1, seq_len) per call (they always assume
          a batch). If FeatureEngineeringAgent saved that shape directly
          to the HF Dataset without squeezing the leading dim, each
          example is stored as (1, seq_len) instead of (seq_len,).
          torch.stack()-ing N such examples then produces a 3D tensor
          (N, 1, seq_len) instead of the 2D (N, seq_len) that
          get_rope_index() requires — causing a mask/tensor shape
          mismatch deep inside the model's forward pass (e.g. "The shape
          of the mask [1, 128] at index 0 does not match the shape of the
          indexed tensor [4, 1] at index 0").

          Fix: squeeze a leading dim of size 1 off each example BEFORE
          stacking, but ONLY when it's actually present — a dataset that
          already stores plain (seq_len,) per example passes through
          unchanged, so this is safe regardless of how
          FeatureEngineeringAgent encoded the data.
        """
        import torch

        def _squeeze_stray_leading_dim(t: "torch.Tensor") -> "torch.Tensor":
            # Only squeeze a genuine (1, seq_len) -> (seq_len,) case.
            # Do NOT touch anything else, so already-correct 1D examples
            # and any other shape are left exactly as they are.
            if t.dim() == 2 and t.shape[0] == 1:
                return t.squeeze(0)
            return t

        batch: Dict[str, Any] = {}

        for key in ("input_ids", "attention_mask", "labels"):
            if key in features[0]:
                tensors = [
                    _squeeze_stray_leading_dim(torch.as_tensor(f[key]))
                    for f in features
                ]
                batch[key] = torch.stack(tensors)

        if "pixel_values" in features[0]:
            batch["pixel_values"] = torch.cat(
                [torch.as_tensor(f["pixel_values"]) for f in features], dim=0
            )

        if "image_grid_thw" in features[0]:
            grids = []
            for f in features:
                g = torch.as_tensor(f["image_grid_thw"])
                # Normalize to (num_images_in_this_example, 3) before
                # concatenating, so a single-image example stored as (3,)
                # OR (1, 3), and a multi-image example stored as (n, 3),
                # all end up correctly shaped in the final batch tensor.
                g = g.reshape(-1, 3)
                grids.append(g)
            batch["image_grid_thw"] = torch.cat(grids, dim=0)

        # Carry through any other tensor columns generically (stack), in
        # case FeatureEngineeringAgent adds model-specific extras later.
        handled = {"input_ids", "attention_mask", "labels",
                   "pixel_values", "image_grid_thw"}
        for key in features[0]:
            if key in handled:
                continue
            try:
                batch[key] = torch.stack(
                    [torch.as_tensor(f[key]) for f in features]
                )
            except Exception as e:
                logger.debug(
                    f"[Training] Qwen-VL collator: could not stack extra "
                    f"column '{key}' ({e}) — dropping from batch."
                )

        return batch

    # ── Trainer ───────────────────────────────────────────────────────────────

    def _build_trainer(
        self,
        model,     tok,
        train_ds,  val_ds,
        out_dir,   batch_size,
        epochs,    lr,
        precision, device,
        # ── new knobs ─────────────────────────────────────────────────────────
        max_grad_norm: float = 1.0,
        es_patience:   int   = 2,
        lr_scheduler:  str   = "cosine",
        hf_id:         str   = "",
    ):
        """
        Build a HuggingFace Trainer with:
          • cosine LR scheduler + linear warmup
          • gradient clipping (max_grad_norm)
          • early stopping (EarlyStoppingCallback, patience=es_patience)
          • the CORRECT data collator for the model family — this is the
            fix for the Qwen-VL "not enough values to unpack" crash.
        """
        import torch
        import transformers
        from transformers import (
            TrainingArguments, Trainer,
            EarlyStoppingCallback,
            default_data_collator,
        )

        use_fp16 = precision == "fp16" and torch.cuda.is_available()
        use_bf16 = False
        if torch.cuda.is_available():
            if torch.cuda.get_device_properties(0).major >= 8:
                use_bf16, use_fp16 = True, False

        steps_per_epoch = max(1, len(train_ds) // batch_size)
        warmup_steps    = min(100, max(1, steps_per_epoch // 10))
        logging_steps   = max(1, steps_per_epoch // 5)

        logger.info(
            f"[Training] Scheduler={lr_scheduler}  "
            f"Warmup={warmup_steps}  MaxGradNorm={max_grad_norm}  "
            f"EarlyStopPatience={es_patience}"
        )

        args = TrainingArguments(
            output_dir=out_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=max(1, batch_size // 2),
            learning_rate=lr,
            fp16=use_fp16,
            bf16=use_bf16,
            gradient_accumulation_steps=4 if batch_size <= 2 else 2,
            warmup_steps=warmup_steps,
            weight_decay=0.01,
            logging_steps=logging_steps,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            # Must stay False for any vision model: Trainer's automatic
            # column pruning doesn't understand pixel_values /
            # image_grid_thw and will strip them before they reach the
            # collator.
            remove_unused_columns=False,
            report_to="none",
            dataloader_num_workers=0,
            max_grad_norm=max_grad_norm,
            lr_scheduler_type=lr_scheduler,
        )

        # transformers >= 4.46 renamed tokenizer → processing_class
        ver       = tuple(int(x) for x in transformers.__version__.split(".")[:2])
        tok_kwarg = "processing_class" if ver >= (4, 46) else "tokenizer"

        has_pixels = "pixel_values" in train_ds.column_names
        is_qwen_vl = self._is_qwen_vl(hf_id)

        # ── Collator selection ──────────────────────────────────────────────
        # This is the core fix: Qwen-VL's ragged pixel_values / per-image
        # image_grid_thw cannot go through default_data_collator's blanket
        # torch.stack() without corruption (see _qwen_vl_collator docstring).
        if has_pixels and is_qwen_vl:
            collator = self._qwen_vl_collator
            logger.info("[Training] Using Qwen-VL patch-concatenation collator.")
        elif has_pixels:
            collator = default_data_collator
        else:
            collator = None

        callbacks = []
        if es_patience > 0:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=es_patience))
            logger.info(
                f"[Training] Early stopping enabled (patience={es_patience})."
            )

        if has_pixels:
            return Trainer(
                model=model,
                args=args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                data_collator=collator,
                callbacks=callbacks if callbacks else None,
            )
        else:
            return Trainer(
                model=model,
                args=args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                callbacks=callbacks if callbacks else None,
                **{tok_kwarg: tok},
            )

    def _run_training(self, trainer) -> Dict:
        self._log_first_batch_shapes(trainer)
        logger.info("[Training] Starting training loop …")
        result  = trainer.train()
        metrics = getattr(result, "metrics", {})
        logger.info(f"[Training] Metrics: {metrics}")
        return metrics

    def _log_first_batch_shapes(self, trainer) -> None:
        """
        Pull exactly one collated batch through the Trainer's own
        dataloader and log every tensor's shape before training starts.

        This turns any future collation/shape bug into a one-line log
        diagnosis instead of a cryptic mid-forward-pass crash that
        requires another full training attempt to even see the input
        shapes that caused it.
        """
        try:
            import torch
            loader = trainer.get_train_dataloader()
            batch = next(iter(loader))
            shapes = {
                k: (tuple(v.shape) if isinstance(v, torch.Tensor) else type(v).__name__)
                for k, v in batch.items()
            }
            logger.info(f"[Training] First collated batch shapes: {shapes}")
        except Exception as e:
            logger.warning(
                f"[Training] Could not pre-inspect first batch shapes "
                f"(non-fatal, continuing): {e}"
            )

    # ── Save ──────────────────────────────────────────────────────────────────

    def _save(self, model, tokenizer, out_dir: str, plan: Dict) -> None:
        p = Path(out_dir)
        try:
            model.save_pretrained(str(p))
        except Exception as e:
            logger.warning(f"[Training] model save failed: {e}")
        try:
            tokenizer.save_pretrained(str(p))
        except Exception as e:
            logger.warning(f"[Training] tokenizer save failed: {e}")
        with open(p / "model_plan.json", "w") as f:
            json.dump(plan, f, indent=2, default=str)
        logger.info(f"[Training] Checkpoint saved → {out_dir}")

    # ── LLM assessment ────────────────────────────────────────────────────────

    def _get_llm_assessment(
        self, hf_id, lora_r, lora_alpha,
        epochs, batch_size, lr, train_loss, out_dir,
    ) -> Dict:
        prompt = (
            f"A Medical VQA model was fine-tuned.\n"
            f"Model: {hf_id}\n"
            f"LoRA r={lora_r}, alpha={lora_alpha}\n"
            f"Epochs={epochs}  Batch={batch_size}  LR={lr}\n"
            f"Training loss: {train_loss}\n"
            f"Checkpoint: {out_dir}\n\n"
            f"Is this result acceptable for Medical VQA evaluation?\n"
            f'Reply with ONLY: {{"status": "ok", '
            f'"train_loss": "{train_loss}", "message": "<one sentence>"}}'
        )
        try:
            response = self.agent.run(prompt)
            return self._parse_response(response)
        except Exception as e:
            logger.warning(f"[Training] LLM assessment failed: {e}")
            return {"status": "ok", "message": "Training complete."}

    def _parse_response(self, response) -> Dict:
        try:
            text  = response.content if hasattr(response, "content") else str(response)
            match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            pass
        return {"status": "ok", "message": "Training complete."}


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, torch
    sys.path.insert(0, str(Path(__file__).parent.parent))

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice : {device}")
    if device == "cuda":
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {vram:.1f} GB")

    # This block now mirrors the actual pipeline: ModelSelectionAgent
    # picks the plan, TrainingAgent consumes it as-is — no manual plan
    # editing required to make Qwen-VL models work.
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from agents.model_selection_agent import ModelSelectionAgent

    selector = ModelSelectionAgent()
    plan = selector.select_model(dataset_size=3515, modality="X-Ray")
    print("\nModel plan from ModelSelectionAgent:")
    print(json.dumps(plan, indent=2, default=str))

    agent = TrainingAgent()

    feature_path   = f"./data/features/{plan['hf_id'].replace('/','_')}"
    processed_path = "./data/processed/processed_dataset.jsonl"

    if Path(feature_path).exists():
        print(f"\nUsing encoded features: {feature_path}")
        result = agent.train(feature_path, plan, device=device)
    elif Path(processed_path).exists():
        print(f"\nEncoding from processed data: {processed_path}")
        result = agent.train_from_processed(processed_path, plan, device=device)
    else:
        print("\nNo data found. Run full pipeline first:")
        print(
            "  python main.py --image image.jpg "
            "--question 'Is there pneumonia?' --dry-run"
        )
        result = {}

    if result:
        print("\n" + "=" * 50)
        print("Training Result:")
        print(json.dumps(result, indent=2, default=str))
        print("=" * 50)

        # Demo: how to pass failure_reason back to ModelSelectionAgent for retry
        if result.get("status") == "failed":
            print("\n[Demo] Passing failure context to ModelSelectionAgent …")
            retry_plan = selector.select_model(
                dataset_size=3515,
                modality="X-Ray",
                failure_context={
                    "failed_hf_id": result.get("model_used", ""),
                    "reason":       result.get("failure_reason", ""),
                },
            )
            print("\nRetry Model Plan:")
            print(json.dumps(retry_plan, indent=2, default=str))
