"""
agents/training_agent.py

TrainingAgent — loads encoded features from FeatureEngineeringAgent
and fine-tunes the model selected by ModelSelectionAgent using PEFT/LoRA.

SELF-HEALING UPDATE
--------------------
train() is now the orchestration entry point for the full recall loop:

    TrainingAgent._train_once()
        ↓ (on failure)
    classify failure → canonical failure_reason code
        ↓
    ModelSelectionAgent.select_model(failed_models=[...], failure_reason=...)
        ↓
    FeatureEngineeringAgent.engineer_features(new_plan, force_regenerate=True)
        ↓
    TrainingAgent._train_once() retried
        ↓ repeat until success or MAX_RETRIES exhausted

This loop lives entirely inside train(), so main.py's call site
(`training_agent.train(...)`) does not change — the pipeline structure is
untouched, but a single call now transparently retries across models.

To support regeneration on retry, train() accepts an optional
`processed_data_path` kwarg (default None, fully backward compatible).
If it is not supplied, retries cannot regenerate features and the agent
falls back to the old single-attempt failure return.

FAILURE CONTRACT — every failed return now always includes:
    {
      "status": "failed",
      "model_used": "...",
      "failure_reason": "<canonical code>",
      "checkpoint_path": "...",
      "retry_recommended": bool,
    }

A canonical failure_reason ("oom" | "load_error" | "tensor_dim_mismatch" |
"processor_not_supported" | "unmasked_labels" | "training_error") is
always produced via _classify_failure(), in addition to the original
free-text message kept under "message", so ModelSelectionAgent's
family-level exclusion logic gets a reliable signal every time.

LABEL-MASKING FAST-FAIL (this revision)
-----------------------------------------
ROOT CAUSE previously found: FeatureEngineeringAgent's vision-chat encoders
were copying input_ids straight into labels with zero masking, so loss was
computed over the prompt/image tokens too. The model never learned to
answer; loss flatlined; every eval metric (BLEU/ROUGE/exact_match/
medical_accuracy) came back exactly 0.0 after a full ~50-minute training
run, because nothing caught the problem until evaluation.

That root cause is now fixed at the source in FeatureEngineeringAgent
(prompt-length-aware masking). This revision adds a defense-in-depth
fast-fail here too: _validate_schema() now checks, BEFORE loading the
model or starting training, that labels are not an exact unmasked copy of
input_ids and are not 100% masked either. If either condition is hit,
training halts in seconds with a new canonical failure_reason
("unmasked_labels") instead of silently burning a full training run to
produce a useless checkpoint with all-zero eval metrics. This also makes
the self-healing retry loop able to react to this failure mode
automatically (excluding the model / triggering feature regeneration)
instead of it being invisible to the retry contract.

The old `_prepare_columns()` fallback that did
`labels = {"labels": x["input_ids"]}` (the same unmasked-copy bug, for any
dataset that reached TrainingAgent without a labels column at all) has
also been removed in favor of failing fast with a clear, actionable error
— silently fabricating unmasked labels here would just reintroduce the
exact same bug one layer up.
"""

from phi.agent import Agent
from phi.model.groq import Groq

from typing import Any, Dict, List, Optional, Tuple
import json, logging, re
from pathlib import Path

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path("./artifacts/checkpoints")
FAILED_MODELS_LOG = Path("./artifacts/failed_models.json")
MAX_RETRIES = 3

TENSOR_COLUMNS = {
    "input_ids", "attention_mask", "pixel_values",
    "labels", "decoder_input_ids", "token_type_ids",
    "image_patches", "image_sizes", "image_grid_thw",
    "qformer_input_ids", "qformer_attention_mask",
}

HF_ID_OVERRIDES: Dict[str, str] = {}


class TrainingAgent:
    """
    Fine-tunes the model selected by ModelSelectionAgent on features
    produced by FeatureEngineeringAgent.

    train() orchestrates the full self-healing retry loop (model
    selection → feature engineering → training) up to MAX_RETRIES, and
    always returns the structured failure contract on exhaustion, or a
    success dict on a successful attempt.
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

    # ── Public orchestration entry point ────────────────────────────────────

    def train(
        self,
        feature_path: str,
        model_plan:   Dict[str, Any],
        device:       str = "",
        processed_data_path: Optional[str] = None,
        dataset_size: Optional[int] = None,
        modality:     str = "",
    ) -> Dict[str, Any]:
        """
        Fine-tune the model using pre-encoded feature dataset, with
        automatic recall/retry on failure.

        Args:
            feature_path        : Path returned by FeatureEngineeringAgent.
            model_plan           : Dict from ModelSelectionAgent.select_model().
            device                : 'cuda' | 'cpu' | '' (auto-detect).
            processed_data_path  : Optional. If supplied, enables retries to
                                    regenerate features for a newly selected
                                    model via FeatureEngineeringAgent.
            dataset_size, modality: Optional. Needed by ModelSelectionAgent
                                    on retry; if omitted, sane defaults are
                                    inferred from the dataset / model_plan.

        Returns:
            Success dict (status="ok", ...) or the structured failure
            contract (status="failed", failure_reason=<canonical code>,
            retry_recommended=bool).
        """
        if not device:
            device = self._detect_device()

        failed_models: List[str] = []
        attempts_log: List[Dict[str, Any]] = []
        current_feature_path = feature_path
        current_plan         = dict(model_plan)

        if dataset_size is None:
            dataset_size = self._infer_dataset_size(current_feature_path)

        for attempt in range(1, MAX_RETRIES + 2):  # initial attempt + MAX_RETRIES
            logger.info(
                f"[Training] === Attempt {attempt}/{MAX_RETRIES + 1} — "
                f"model={current_plan.get('hf_id')} ==="
            )
            result = self._train_once(current_feature_path, current_plan, device)
            attempts_log.append({
                "attempt": attempt,
                "model": current_plan.get("hf_id"),
                "status": result.get("status"),
                "failure_reason": result.get("failure_reason", ""),
            })

            if result.get("status") == "ok":
                result["attempts"] = attempts_log
                return result

            # Failure path
            failed_hf_id = current_plan.get("hf_id", "")
            if failed_hf_id and failed_hf_id not in failed_models:
                failed_models.append(failed_hf_id)

            canonical_reason = self._classify_failure(result.get("message", ""))
            result["failure_reason"] = canonical_reason
            self._log_failed_model(failed_hf_id, canonical_reason, result.get("message", ""))

            retries_left = (MAX_RETRIES + 1) - attempt
            can_retry = retries_left > 0 and processed_data_path is not None

            if not can_retry:
                result["retry_recommended"] = retries_left > 0
                result["attempts"] = attempts_log
                result["failed_models"] = failed_models
                logger.error(
                    f"[Training] Halting — "
                    f"{'max retries exhausted' if retries_left <= 0 else 'no processed_data_path supplied for regeneration'}."
                )
                return result

            # ── Recall ModelSelectionAgent ───────────────────────────────────
            logger.info(
                f"[Training] Recalling ModelSelectionAgent — excluding "
                f"{failed_models}, reason={canonical_reason}"
            )
            try:
                import sys
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from agents.model_selection_agent import ModelSelectionAgent

                selector = ModelSelectionAgent()
                next_plan = selector.select_model(
                    dataset_size=dataset_size or 0,
                    modality=modality,
                    failed_models=failed_models,
                    failure_reason=canonical_reason,
                )
            except Exception as e:
                logger.error(f"[Training] ModelSelectionAgent recall failed: {e}")
                result["retry_recommended"] = False
                result["attempts"] = attempts_log
                result["failed_models"] = failed_models
                return result

            if next_plan.get("hf_id") in failed_models:
                logger.error(
                    "[Training] ModelSelectionAgent returned an already-"
                    "failed model — no further safe candidates. Halting."
                )
                result["retry_recommended"] = False
                result["attempts"] = attempts_log
                result["failed_models"] = failed_models
                return result

            # ── Recall FeatureEngineeringAgent ───────────────────────────────
            logger.info(
                f"[Training] Recalling FeatureEngineeringAgent for "
                f"{next_plan.get('hf_id')}"
            )
            try:
                from agents.feature_engineering_agent import FeatureEngineeringAgent

                fe_agent  = FeatureEngineeringAgent()
                fe_result = fe_agent.engineer_features(
                    processed_data_path, next_plan, device,
                    force_regenerate=True,
                )
            except Exception as e:
                logger.error(f"[Training] FeatureEngineeringAgent recall failed: {e}")
                result["retry_recommended"] = False
                result["attempts"] = attempts_log
                result["failed_models"] = failed_models
                return result

            if fe_result.get("status") == "failed":
                logger.error(
                    f"[Training] Feature regeneration failed for "
                    f"{next_plan.get('hf_id')}: {fe_result.get('message')}"
                )
                # Treat as this model's failure too, loop continues
                current_plan = next_plan
                current_feature_path = ""
                failed_models.append(next_plan.get("hf_id", ""))
                continue

            current_plan         = next_plan
            current_feature_path = fe_result["feature_path"]

        # Should not normally reach here
        return self._fail("Retry loop exhausted unexpectedly.",
                          model_used=current_plan.get("hf_id", ""),
                          excluded_models=failed_models)

    # ── Single training attempt (original core logic) ──────────────────────

    def _train_once(
        self,
        feature_path: str,
        model_plan:   Dict[str, Any],
        device:       str,
    ) -> Dict[str, Any]:
        if not feature_path:
            return self._fail("No feature_path available for this attempt.",
                               model_used=model_plan.get("hf_id", ""))

        train_ds, val_ds = self._load_feature_datasets(feature_path)
        if train_ds is None or len(train_ds) == 0:
            return self._fail(
                f"No encoded training data found at: {feature_path}",
                model_used=model_plan.get("hf_id", ""),
            )

        logger.info(f"[Training] Loaded → Train={len(train_ds)}  Val={len(val_ds)}")
        logger.info(f"[Training] Columns: {train_ds.column_names}")

        # ── Pre-training schema validation ──────────────────────────────────
        schema_error = self._validate_schema(train_ds, model_plan)
        if schema_error:
            return self._fail(schema_error, model_used=model_plan.get("hf_id", ""))

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
        max_grad_norm = active_plan.get("max_grad_norm",         1.0)
        es_patience   = active_plan.get("early_stopping_patience", 2)
        lr_scheduler  = active_plan.get("lr_scheduler_type",     "cosine")

        is_qwen_vl = self._is_qwen_vl(hf_id)
        if is_qwen_vl:
            logger.info(
                f"[Training] Detected Qwen-VL family model ({hf_id}) — "
                f"will use patch-concatenation collator."
            )

        safe_name = name.replace(" ", "_").replace("/", "_")
        out_dir   = CHECKPOINT_DIR / safe_name
        out_dir.mkdir(parents=True, exist_ok=True)

        model = self._apply_lora(model, lora_r, lora_alpha, lora_drop, target_mods, architecture)

        train_ds, val_ds = self._prepare_columns(train_ds, val_ds, hf_id)
        if len(train_ds) == 0:
            return self._fail(
                "No valid tensor columns after column preparation.",
                model_used=hf_id, checkpoint_path=str(out_dir),
            )

        try:
            trainer = self._build_trainer(
                model, tokenizer, train_ds, val_ds,
                str(out_dir), batch_size, epochs, lr,
                precision, device,
                max_grad_norm=max_grad_norm, es_patience=es_patience,
                lr_scheduler=lr_scheduler, hf_id=hf_id,
            )
            metrics    = self._run_training(trainer)
            train_loss = round(float(metrics.get("train_loss", 0.0)), 4)
        except Exception as e:
            logger.error(f"[Training] Training failed: {e}")
            return self._fail(
                f"Training loop error: {e}",
                model_used=hf_id, checkpoint_path=str(out_dir),
            )

        self._save(model, tokenizer, str(out_dir), active_plan)
        logger.info(f"[Training] Complete. Loss={train_loss}  Ckpt={out_dir}")

        assessment = self._get_llm_assessment(
            hf_id, lora_r, lora_alpha, epochs, batch_size, lr, train_loss, out_dir,
        )
        assessment["checkpoint_path"]   = str(out_dir)
        assessment["model_used"]        = hf_id
        assessment["train_loss"]        = str(train_loss)
        assessment["failure_reason"]    = ""
        assessment["retry_recommended"] = False
        return assessment

    # ── Compatibility method ──────────────────────────────────────────────────

    def train_from_processed(
        self,
        processed_data_path: str,
        model_plan:          Dict[str, Any],
        device:              str = "",
    ) -> Dict[str, Any]:
        """Fallback: encode inline, then delegate to the full retry-aware train()."""
        logger.info("[Training] No feature_path provided. Running inline encoding …")
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
        return self.train(
            fe_result["feature_path"], model_plan, device,
            processed_data_path=processed_data_path,
        )

    # ── Failure contract / classification helpers ───────────────────────────

    def _fail(self, message: str, model_used: str = "", checkpoint_path: str = "", excluded_models: list = None) -> Dict[str, Any]:
        logger.error(f"[Training] FAILED — {message}")
        return {
            "status":            "failed",
            "message":           message,
            "checkpoint_path":   checkpoint_path,
            "model_used":        model_used,
            "failure_reason":    self._classify_failure(message),
            "retry_recommended": True,
            "excluded_models":   excluded_models or [],
        }

    def _classify_failure(self, message: str) -> str:
        """Maps a free-text error onto a canonical failure_reason code."""
        m = (message or "").lower()
        if any(k in m for k in ("out of memory", "cuda error", "oom", "cudaoutofmemory")):
            return "oom"
        if any(k in m for k in ("unmasked", "labels identical to input_ids", "labels fully masked")):
            return "unmasked_labels"
        if any(k in m for k in ("number of dimensions", "shape", "unpack", "dimension", "mask")):
            return "tensor_dim_mismatch"
        if any(k in m for k in ("processor", "autoprocessor", "tokenizer load", "unrecognized configuration")):
            return "processor_not_supported"
        if any(k in m for k in ("failed to load", "cannot load", "no module",
                                 "not found in transformers", "weight", "checkpoint")):
            return "load_error"
        return "training_error"

    def _log_failed_model(self, hf_id: str, reason_code: str, message: str) -> None:
        FAILED_MODELS_LOG.parent.mkdir(parents=True, exist_ok=True)
        try:
            existing = json.loads(FAILED_MODELS_LOG.read_text()) if FAILED_MODELS_LOG.exists() else []
        except Exception:
            existing = []
        existing.append({"model": hf_id, "reason": reason_code, "message": message[:500]})
        try:
            FAILED_MODELS_LOG.write_text(json.dumps(existing, indent=2))
        except Exception as e:
            logger.warning(f"[Training] Could not write failed_models.json: {e}")

    def _infer_dataset_size(self, feature_path: str) -> int:
        if not feature_path:
            return 0
        try:
            from datasets import load_from_disk
            train_dir = Path(feature_path) / "train"
            if train_dir.exists():
                return len(load_from_disk(str(train_dir)))
        except Exception:
            pass
        return 0

    # ── Pre-training validation ──────────────────────────────────────────────

    def _validate_schema(self, train_ds, model_plan: Dict[str, Any]) -> str:
        """
        Validate dataset schema against the model's expected feature
        strategy BEFORE attempting to load the model or build a trainer.
        Returns an empty string if valid, or an error message otherwise.

        Includes a fast-fail check (NEW) for the unmasked-labels bug class:
        if labels are an exact copy of input_ids, or are 100% masked, this
        fails immediately instead of letting a full training run complete
        and silently produce a checkpoint with all-zero eval metrics.
        """
        cols = set(train_ds.column_names)
        strategy = model_plan.get("feature_strategy", "")
        is_vision = model_plan.get("vision", False)

        if "input_ids" not in cols:
            return "Schema validation failed: missing 'input_ids' column."
        if "labels" not in cols:
            return "Schema validation failed: missing 'labels' column."
        if is_vision and "pixel_values" not in cols:
            return (
                "Schema validation failed: model_plan marks vision=True but "
                "'pixel_values' column is missing from encoded features — "
                "processor_not_supported."
            )
        if strategy == "vision2seq_patchified" and "image_grid_thw" not in cols:
            return (
                "Schema validation failed: feature_strategy="
                "'vision2seq_patchified' requires 'image_grid_thw' column — "
                "processor_not_supported."
            )

        try:
            import torch
            sample = train_ds[0]
            ids = torch.as_tensor(sample["input_ids"])
            if ids.dim() == 2 and ids.shape[0] == 1:
                # Recoverable: a stray batch-of-one leading dim. TrainingAgent's
                # Qwen-VL collator (and the general column-prep squeeze) strip
                # this before stacking, so it's not fatal — just log it.
                logger.warning(
                    "[Training] input_ids has a leading singleton dim "
                    f"{tuple(ids.shape)} — will be squeezed before training. "
                    "If this persists, fix FeatureEngineeringAgent's encoder "
                    "to save flat per-example sequences."
                )
            elif ids.dim() != 1:
                return (
                    f"Schema validation failed: per-example input_ids rank "
                    f"{ids.dim()} (shape={tuple(ids.shape)}, expected rank 1) "
                    f"— tensor_dim_mismatch."
                )
        except Exception:
            pass

        # ── Unmasked-labels fast-fail (NEW) ─────────────────────────────────
        # Check a small sample of examples, not just one — a single example
        # could coincidentally have its answer span cover (almost) the
        # whole sequence. Checking several makes a false positive unlikely
        # while still failing in well under a second.
        try:
            n_check = min(10, len(train_ds))
            all_identical = True
            all_fully_masked = True
            for i in range(n_check):
                row = train_ds[i]
                ids = list(row["input_ids"])
                lbl = list(row["labels"])
                if ids != lbl:
                    all_identical = False
                if any(t != -100 for t in lbl):
                    all_fully_masked = False
                if not all_identical and not all_fully_masked:
                    break

            if all_identical:
                return (
                    "Schema validation failed: 'labels' are an exact unmasked "
                    "copy of 'input_ids' for every sampled example — loss "
                    "would be computed over prompt/image tokens instead of "
                    "just the answer span, which produces a flat loss curve "
                    "and near-zero eval metrics (BLEU/ROUGE/exact_match all "
                    "0.0). This is the unmasked_labels failure — fix label "
                    "masking in FeatureEngineeringAgent's encoder for this "
                    "model family before retraining."
                )
            if all_fully_masked:
                return (
                    "Schema validation failed: 'labels' are entirely -100 "
                    "(fully masked) for every sampled example — there are no "
                    "real training targets left, so the model has nothing to "
                    "learn from. This is the unmasked_labels failure family — "
                    "check the prompt-length calculation in "
                    "FeatureEngineeringAgent's encoder for this model family."
                )
        except Exception as e:
            logger.warning(f"[Training] Unmasked-labels fast-fail check skipped: {e}")

        return ""

    # ── Helpers (unchanged from previous version) ────────────────────────────

    def _detect_device(self) -> str:
        try:
            import torch
            if torch.cuda.is_available():         return "cuda"
            if torch.backends.mps.is_available(): return "mps"
        except ImportError:
            pass
        return "cpu"

    def _is_qwen_vl(self, hf_id: str) -> bool:
        h = hf_id.lower()
        return "qwen" in h and "-vl" in h

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

    def _load_model(self, model_plan: Dict, device: str) -> Tuple[Any, Any, Dict]:
        hf_id = model_plan.get("hf_id", "")
        if not hf_id:
            raise RuntimeError("No hf_id specified in model_plan.")

        hf_id = HF_ID_OVERRIDES.get(hf_id, hf_id)
        effective_plan = {**model_plan, "hf_id": hf_id}

        logger.info(f"[Training] Loading requested model: {hf_id}")
        try:
            import gc, torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
                gc.collect()
                free_gb = _torch.cuda.mem_get_info()[0] / 1e9
                logger.info(f"[Training] CUDA cache cleared before loading {hf_id} (free={free_gb:.2f}GB)")
        except Exception:
            pass
        try:
            # FIX: After a prior model OOM'd, PyTorch may hold stale CUDA
            # state / fragmented memory. Calling empty_cache() + gc here
            # releases all unreferenced tensors before attempting the next
            # load, preventing the "NCCL Error 1: unhandled cuda error" that
            # otherwise fires on the very first training step of the next
            # attempt even when the model itself loads successfully.
            import gc, torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                logger.info(
                    f"[Training] CUDA cache cleared before loading {hf_id} "
                    f"(free={torch.cuda.mem_get_info()[0] / 1e9:.2f}GB)"
                )
        except Exception:
            pass
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

    def _load_single(self, hf_id: str, loader_hint: str, use_4bit: bool, device: str, precision: str) -> Tuple[Any, Any]:
        import torch
        import transformers as tf

        try:
            _ = torch.tensor([0.0])
        except Exception as e:
            raise RuntimeError(
                "PyTorch backend is not usable in this environment "
                f"(torch.tensor() failed: {e}). Fix the PyTorch install before retrying."
            )

        dtype = torch.float16 if precision == "fp16" and device == "cuda" else torch.float32

        load_kw: Dict = dict(pretrained_model_name_or_path=hf_id, trust_remote_code=True)

        if use_4bit and device == "cuda":
            from transformers import BitsAndBytesConfig
            load_kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
            )
            load_kw["device_map"] = {"": 0}
        elif device == "cuda":
            load_kw["torch_dtype"] = dtype   # FIX: was "dtype" — not a valid kwarg for from_pretrained;
                                              # causes a warning and dtype is silently ignored for some
                                              # model classes (e.g. Qwen2_5_VLForConditionalGeneration).
            load_kw["device_map"] = {"": 0}
        else:
            load_kw["torch_dtype"] = dtype   # FIX: same kwarg fix for CPU path

        CLASS_MAP: Dict[str, List[str]] = {
            "AutoModelForSeq2SeqLM": ["AutoModelForSeq2SeqLM"],
            "Blip2ForConditionalGeneration": [
                "Blip2ForConditionalGeneration", "AutoModelForImageTextToText"],
            "InstructBlipForConditionalGeneration": [
                "InstructBlipForConditionalGeneration", "AutoModelForImageTextToText"],
            "Qwen2_5_VLForConditionalGeneration": [
                "Qwen2_5_VLForConditionalGeneration", "AutoModelForImageTextToText"],
            "Qwen2VLForConditionalGeneration": [
                "Qwen2VLForConditionalGeneration", "AutoModelForImageTextToText"],
            "AutoModelForVision2Seq": ["AutoModelForImageTextToText"],
            "AutoModelForImageTextToText": ["AutoModelForImageTextToText"],
            "auto": [
                "InstructBlipForConditionalGeneration", "Blip2ForConditionalGeneration",
                "Qwen2_5_VLForConditionalGeneration", "Qwen2VLForConditionalGeneration",
                "AutoModelForSeq2SeqLM", "AutoModelForImageTextToText", "AutoModelForCausalLM",
            ],
        }
        class_order = CLASS_MAP.get(loader_hint, CLASS_MAP["auto"])

        EXPLICIT_CLASS_MODEL_TYPES: Dict[str, set] = {
            "InstructBlipForConditionalGeneration": {"instructblip"},
            "Blip2ForConditionalGeneration":          {"blip-2", "blip_2"},
            "Qwen2_5_VLForConditionalGeneration":     {"qwen2_5_vl"},
            "Qwen2VLForConditionalGeneration":        {"qwen2_vl"},
        }

        checkpoint_model_type = ""
        try:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)
            checkpoint_model_type = getattr(cfg, "model_type", "") or ""
            logger.info(f"[Training] {hf_id} checkpoint model_type='{checkpoint_model_type}'")
        except Exception as e:
            logger.warning(
                f"[Training] Could not pre-fetch config.model_type for {hf_id} "
                f"({e}) — architecture compatibility gate will be skipped for "
                f"this load (falls back to try/except behaviour)."
            )

        def _architecture_compatible(cls_name: str) -> bool:
            expected = EXPLICIT_CLASS_MODEL_TYPES.get(cls_name)
            if expected is None:
                return True  # AutoModelFor* — safe by construction
            if not checkpoint_model_type:
                return True  # couldn't determine — don't block, fall through to try/except
            return checkpoint_model_type in expected

        filtered_class_order = [c for c in class_order if _architecture_compatible(c)]
        gated_out = [c for c in class_order if c not in filtered_class_order]
        if gated_out:
            logger.info(
                f"[Training] Architecture gate excluded {gated_out} — "
                f"checkpoint model_type='{checkpoint_model_type}' does not "
                f"match their expected type(s). This prevents loading "
                f"{hf_id} with the wrong model class."
            )
        class_order = filtered_class_order or class_order  # never end up with an empty list

        SUBMODULE_FALLBACKS: Dict[str, str] = {
            "InstructBlipForConditionalGeneration": "transformers.models.instructblip",
            "Blip2ForConditionalGeneration": "transformers.models.blip_2",
            "Qwen2_5_VLForConditionalGeneration": "transformers.models.qwen2_5_vl",
            "Qwen2VLForConditionalGeneration": "transformers.models.qwen2_vl",
        }

        def _is_real_class(cls) -> bool:
            if cls is None:
                return False
            return "Placeholder" not in type(cls).__name__ and "Placeholder" not in getattr(cls, "__name__", "")

        def _resolve_class(cls_name: str):
            cls = getattr(tf, cls_name, None)
            if _is_real_class(cls):
                return cls, "top-level"
            submodule_path = SUBMODULE_FALLBACKS.get(cls_name)
            if submodule_path:
                try:
                    import importlib
                    mod = importlib.import_module(submodule_path)
                    sub_cls = getattr(mod, cls_name, None)
                    if _is_real_class(sub_cls):
                        return sub_cls, f"submodule:{submodule_path}"
                    if sub_cls is not None:
                        return None, "resolved via submodule but still a Placeholder"
                except Exception as e:
                    return None, f"submodule import failed: {e}"
            return None, "not found at top-level and no submodule fallback registered"

        model, last_error = None, None
        skip_reasons: List[str] = []
        attempt_errors: List[Tuple[str, str]] = []

        for cls_name in class_order:
            cls, note = _resolve_class(cls_name)
            if cls is None:
                skip_reasons.append(f"{cls_name} ({note})")
                logger.warning(f"[Training] Class {cls_name} unavailable — {note}. Skipping.")
                continue
            try:
                model = cls.from_pretrained(**load_kw)
                logger.info(f"[Training] {hf_id} loaded with {cls_name} ({note})")
                break
            except Exception as e:
                last_error = e
                attempt_errors.append((cls_name, str(e)))
                logger.warning(f"[Training] {cls_name} load failed for {hf_id}: {e}")

        if model is None:
            reason_block = "; ".join(skip_reasons) if skip_reasons else "no classes skipped"
            primary_class_name = class_order[0] if class_order else loader_hint
            primary_error = next((err for name, err in attempt_errors if name == primary_class_name), None)
            per_class_block = "\n".join(f"  - {name}: {err[:300]}" for name, err in attempt_errors) or \
                "  (no class was actually attempted — all were unavailable)"
            alias_note = (
                f" (loader_hint '{loader_hint}' resolved to '{primary_class_name}')"
                if primary_class_name != loader_hint else ""
            )
            primary_block = (
                f"\nPRIMARY (requested loader '{loader_hint}'{alias_note}) error:\n  {primary_error[:500]}\n"
                if primary_error else
                f"\nNote: intended class '{primary_class_name}' was never actually attempted.\n"
            )
            raise RuntimeError(
                f"Cannot load {hf_id} with any class in {class_order}.\n"
                f"Skipped (unavailable) classes: {reason_block}\n"
                f"{primary_block}"
                f"All attempted classes and their errors:\n{per_class_block}\n"
                f"Last error: {last_error}"
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

    def _apply_lora(self, model, r, alpha, dropout, target_modules, architecture):
        try:
            from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
            try:
                model = prepare_model_for_kbit_training(model)
            except Exception:
                pass
            task_type = TaskType.SEQ_2_SEQ_LM if architecture == "seq2seq" else TaskType.CAUSAL_LM
            cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                              target_modules=target_modules, bias="none", task_type=task_type)
            model = get_peft_model(model, cfg)
            model.print_trainable_parameters()
            logger.info(f"[Training] LoRA applied: r={r}  alpha={alpha}  task={task_type}")
        except ImportError:
            logger.warning("[Training] PEFT not installed — training without LoRA.")
        except Exception as e:
            logger.warning(f"[Training] LoRA skipped: {e}")
        return model

    def _prepare_columns(self, train_ds, val_ds, hf_id: str = ""):
        drop_tr = [c for c in train_ds.column_names if c not in TENSOR_COLUMNS]
        drop_vl = [c for c in val_ds.column_names   if c not in TENSOR_COLUMNS]
        if drop_tr:
            logger.info(f"[Training] Dropping non-tensor columns: {drop_tr}")
            train_ds = train_ds.remove_columns(drop_tr)
        if drop_vl:
            val_ds = val_ds.remove_columns(drop_vl)

        # NOTE: a "labels missing -> copy input_ids" fallback used to live
        # here. It has been removed on purpose: that copy is unmasked by
        # definition (no prompt/answer boundary is known at this layer),
        # so it silently reproduced the exact unmasked-labels bug that
        # caused 0.0 eval metrics. _validate_schema() now requires
        # 'labels' to already exist (and be properly masked) before
        # training ever reaches this point — if it's missing, that's a
        # FeatureEngineeringAgent bug to fix at the source, not something
        # to paper over here.

        # Squeeze a stray leading batch-of-one dim off sequence fields for
        # EVERY model family, not just Qwen-VL. Some HF processors return
        # [[ids...]] (batch-of-one) instead of [ids...] for a single example
        # even with return_tensors=None; if that nested shape ever slips
        # past FeatureEngineeringAgent's own flattening (e.g. an older
        # cached feature set), torch.stack() during collation would produce
        # a 3D tensor here too, not just for Qwen-VL.
        train_ds = self._squeeze_sequence_columns(train_ds)
        val_ds   = self._squeeze_sequence_columns(val_ds)

        is_qwen_vl = self._is_qwen_vl(hf_id)

        if "pixel_values" in train_ds.column_names:
            if is_qwen_vl:
                logger.info(
                    "[Training] Skipping pixel_values rank normalization — "
                    f"{hf_id} is Qwen-VL and uses pre-patchified pixel_values."
                )
            else:
                try:
                    train_ds, val_ds = self._normalize_pixel_values(train_ds, val_ds, column="pixel_values")
                except Exception as e:
                    logger.warning(f"[Training] pixel_values normalization skipped: {e}")

        logger.info(f"[Training] Final columns: {train_ds.column_names}")
        return train_ds, val_ds

    def _squeeze_sequence_columns(self, ds):
        """Flatten a stray [[ids...]] batch-of-one wrapping to [ids...] for
        input_ids / attention_mask / labels, if present."""
        import torch

        def _needs_squeeze(col: str) -> bool:
            if col not in ds.column_names or len(ds) == 0:
                return False
            sample = ds[0][col]
            t = torch.as_tensor(sample)
            return t.dim() == 2 and t.shape[0] == 1

        cols_to_fix = [c for c in ("input_ids", "attention_mask", "labels") if _needs_squeeze(c)]
        if not cols_to_fix:
            return ds

        logger.info(f"[Training] Squeezing leading singleton dim on columns: {cols_to_fix}")

        def _squeeze_batch(batch):
            out = {}
            for c in cols_to_fix:
                out[c] = [row[0] if isinstance(row, list) and len(row) == 1 and isinstance(row[0], list) else row
                          for row in batch[c]]
            return out

        return ds.map(_squeeze_batch, batched=True)

    def _normalize_pixel_values(self, train_ds, val_ds, column: str = "pixel_values"):
        import torch
        sample = train_ds[0][column]
        t = torch.as_tensor(sample)
        original_shape = tuple(t.shape)
        logger.info(f"[Training] {column} per-example shape before normalize: {original_shape}")

        strips_needed = max(0, t.dim() - 3)
        if strips_needed == 0 and t.dim() == 3:
            return train_ds, val_ds
        if t.dim() < 3:
            logger.warning(f"[Training] {column} has rank {t.dim()} (< 3) — cannot normalize automatically.")
            return train_ds, val_ds

        def _strip_leading_singletons(arr, n: int):
            for _ in range(n):
                if isinstance(arr, list) and len(arr) == 1:
                    arr = arr[0]
                else:
                    break
            return arr

        def normalize_batch(batch):
            return {column: [_strip_leading_singletons(pv, strips_needed) for pv in batch[column]]}

        train_ds = train_ds.map(normalize_batch, batched=True)
        val_ds   = val_ds.map(normalize_batch,   batched=True)

        new_shape = tuple(torch.as_tensor(train_ds[0][column]).shape)
        logger.info(f"[Training] {column} normalized: {original_shape} → {new_shape}")
        return train_ds, val_ds

    def _qwen_vl_collator(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        import torch

        def _squeeze_stray_leading_dim(t: "torch.Tensor") -> "torch.Tensor":
            if t.dim() == 2 and t.shape[0] == 1:
                return t.squeeze(0)
            return t

        batch: Dict[str, Any] = {}
        for key in ("input_ids", "attention_mask", "labels"):
            if key in features[0]:
                tensors = [_squeeze_stray_leading_dim(torch.as_tensor(f[key])) for f in features]
                batch[key] = torch.stack(tensors)

        if "pixel_values" in features[0]:
            batch["pixel_values"] = torch.cat(
                [torch.as_tensor(f["pixel_values"]) for f in features], dim=0)

        if "image_grid_thw" in features[0]:
            grids = []
            for f in features:
                g = torch.as_tensor(f["image_grid_thw"]).reshape(-1, 3)
                grids.append(g)
            batch["image_grid_thw"] = torch.cat(grids, dim=0)

        handled = {"input_ids", "attention_mask", "labels", "pixel_values", "image_grid_thw"}
        for key in features[0]:
            if key in handled:
                continue
            try:
                batch[key] = torch.stack([torch.as_tensor(f[key]) for f in features])
            except Exception as e:
                logger.debug(f"[Training] Qwen-VL collator: could not stack '{key}' ({e}) — dropping.")

        return batch

    def _build_trainer(
        self, model, tok, train_ds, val_ds, out_dir, batch_size, epochs, lr,
        precision, device, max_grad_norm: float = 1.0, es_patience: int = 2,
        lr_scheduler: str = "cosine", hf_id: str = "",
    ):
        import os, torch
        import transformers
        from transformers import TrainingArguments, Trainer, EarlyStoppingCallback, default_data_collator

        # FIX: NCCL Error 1 on T4 VMs.
        # T4 GPUs on GCP/Colab do NOT support NCCL peer-to-peer (NVLink/IB
        # is not present). When PyTorch's Trainer initialises the distributed
        # backend (even for single-GPU runs that triggered NCCL via a prior
        # multi-GPU device_map), NCCL probes for P2P and fires
        # "unhandled cuda error" on the very first all-reduce operation.
        # Disabling P2P and IB forces NCCL to fall back to host-memory
        # transfers, which work correctly on T4 single-GPU setups.
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
        os.environ.setdefault("NCCL_IB_DISABLE", "1")
        # Also set PYTORCH_ALLOC_CONF to reduce fragmentation after prior OOMs.
        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        logger.info("[Training] NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 set (T4 P2P workaround)")

        use_fp16 = precision == "fp16" and torch.cuda.is_available()
        use_bf16 = False
        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
            use_bf16, use_fp16 = True, False

        steps_per_epoch = max(1, len(train_ds) // batch_size)
        warmup_steps    = min(100, max(1, steps_per_epoch // 10))
        logging_steps   = max(1, steps_per_epoch // 5)

        args = TrainingArguments(
            output_dir=out_dir, num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=max(1, batch_size // 2),
            learning_rate=lr, fp16=use_fp16, bf16=use_bf16,
            gradient_accumulation_steps=4 if batch_size <= 2 else 2,
            warmup_steps=warmup_steps, weight_decay=0.01,
            logging_steps=logging_steps, eval_strategy="epoch", save_strategy="epoch",
            load_best_model_at_end=True, metric_for_best_model="eval_loss",
            greater_is_better=False, remove_unused_columns=False,
            report_to="none", dataloader_num_workers=0,
            max_grad_norm=max_grad_norm, lr_scheduler_type=lr_scheduler,
        )

        ver       = tuple(int(x) for x in transformers.__version__.split(".")[:2])
        tok_kwarg = "processing_class" if ver >= (4, 46) else "tokenizer"

        has_pixels = "pixel_values" in train_ds.column_names
        is_qwen_vl = self._is_qwen_vl(hf_id)

        if has_pixels and is_qwen_vl:
            collator = self._qwen_vl_collator
        elif has_pixels:
            collator = default_data_collator
        else:
            collator = None

        callbacks = []
        if es_patience > 0:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=es_patience))

        if has_pixels:
            return Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                            data_collator=collator, callbacks=callbacks if callbacks else None)
        else:
            return Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                            callbacks=callbacks if callbacks else None, **{tok_kwarg: tok})

    def _run_training(self, trainer) -> Dict:
        self._log_first_batch_shapes(trainer)
        logger.info("[Training] Starting training loop …")
        result  = trainer.train()
        metrics = getattr(result, "metrics", {})
        logger.info(f"[Training] Metrics: {metrics}")
        return metrics

    def _log_first_batch_shapes(self, trainer) -> None:
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
            logger.warning(f"[Training] Could not pre-inspect first batch shapes: {e}")

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

    def _get_llm_assessment(self, hf_id, lora_r, lora_alpha, epochs, batch_size, lr, train_loss, out_dir) -> Dict:
        prompt = (
            f"A Medical VQA model was fine-tuned.\nModel: {hf_id}\n"
            f"LoRA r={lora_r}, alpha={lora_alpha}\nEpochs={epochs}  Batch={batch_size}  LR={lr}\n"
            f"Training loss: {train_loss}\nCheckpoint: {out_dir}\n\n"
            f"Is this result acceptable for Medical VQA evaluation?\n"
            f'Reply with ONLY: {{"status": "ok", "train_loss": "{train_loss}", "message": "<one sentence>"}}'
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


if __name__ == "__main__":
    import sys, torch
    sys.path.insert(0, str(Path(__file__).parent.parent))
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice : {device}")

    from agents.model_selection_agent import ModelSelectionAgent
    selector = ModelSelectionAgent()
    plan = selector.select_model(dataset_size=3515, modality="X-Ray")
    print("\nModel plan:")
    print(json.dumps(plan, indent=2, default=str))

    agent = TrainingAgent()
    feature_path   = f"./data/features/{plan['hf_id'].replace('/','_')}_{plan.get('model_family','')}"
    processed_path = "./data/processed/processed_dataset.jsonl"

    if Path(feature_path).exists():
        result = agent.train(feature_path, plan, device=device, processed_data_path=processed_path,
                              dataset_size=3515, modality="X-Ray")
    elif Path(processed_path).exists():
        result = agent.train_from_processed(processed_path, plan, device=device)
    else:
        print("\nNo data found. Run the pipeline first.")
        result = {}

    if result:
        print("\n" + "=" * 50)
        print("Training Result:")
        print(json.dumps(result, indent=2, default=str))
        print("=" * 50)
