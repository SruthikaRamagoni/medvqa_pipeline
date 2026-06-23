"""
agents/training_agent.py

TrainingAgent — Contract-aware adaptive fine-tuning for Medical VQA.

CONTRACT VALIDATION
--------------------
Before training starts, validates that the loaded model is compatible
with the encoded features by checking:
  1. feature_strategy in model_plan matches metadata.json in feature_path
  2. model_hf_id in metadata.json matches model_plan["hf_id"]
  3. Required tensor columns exist in the dataset
  4. No dimension mismatches (pixel_values shape checked for vision models)

If validation fails → returns structured failure with retry_recommended=True
so the outer recovery loop can call ModelSelectionAgent + FeatureEngineeringAgent.

RECOVERY LOOP
--------------
train() wraps _train_once() in a retry loop (MAX_RETRIES=3):
  attempt 1: train with model_plan from main.py
  on failure: call ModelSelectionAgent.select_model(failed_models=[...])
              call FeatureEngineeringAgent.engineer_features(new_plan)
  attempt 2: train with new model and new features
  ...up to MAX_RETRIES

TRAINING OPTIMISATIONS
------------------------
  - Early stopping         : patience=2 epochs (EarlyStoppingCallback)
  - LR scheduler           : cosine annealing (smooth decay)
  - Gradient clipping      : max_grad_norm=1.0
  - Collator selection     : DataCollatorForSeq2Seq for seq2seq models
                             default_data_collator for vision models
  - Label clamping         : padding positions → -100

FAILURE HISTORY
----------------
Each failed attempt is recorded in:
  artifacts/checkpoints/failed_models.json

Format:
  [{"model": "...", "reason": "...", "failure_class": "..."}]

Uses PhiData + Groq. Same structure as all other agents in the project.
"""

from phi.agent import Agent
from phi.model.groq import Groq

from typing import Any, Dict, List, Optional, Tuple
import json, logging, re
from pathlib import Path

logger = logging.getLogger(__name__)

CHECKPOINT_DIR    = Path("./artifacts/checkpoints")
FAILED_MODELS_LOG = CHECKPOINT_DIR / "failed_models.json"
MAX_RETRIES       = 3

TENSOR_COLUMNS = {
    "input_ids", "attention_mask", "pixel_values",
    "labels", "decoder_input_ids", "token_type_ids",
    "image_grid_thw", "image_patches", "image_sizes",
}
TEXT_ONLY_COLUMNS = {
    "input_ids", "attention_mask",
    "labels", "decoder_input_ids", "token_type_ids",
}

FALLBACK_MODELS: List[Dict] = [
    {
        "hf_id":            "google/flan-t5-base",
        "name":             "Flan-T5-Base",
        "architecture":     "seq2seq",
        "vision":           False,
        "target_modules":   ["q", "v"],
        "loader":           "AutoModelForSeq2SeqLM",
        "model_family":     "flan_t5",
        "processor_type":   "AutoTokenizer",
        "feature_strategy": "seq2seq",
        "collator_type":    "DataCollatorForSeq2Seq",
    },
    {
        "hf_id":            "google/flan-t5-large",
        "name":             "Flan-T5-Large",
        "architecture":     "seq2seq",
        "vision":           False,
        "target_modules":   ["q", "v"],
        "loader":           "AutoModelForSeq2SeqLM",
        "model_family":     "flan_t5",
        "processor_type":   "AutoTokenizer",
        "feature_strategy": "seq2seq",
        "collator_type":    "DataCollatorForSeq2Seq",
    },
    {
        "hf_id":            "Salesforce/blip2-opt-2.7b",
        "name":             "BLIP2-OPT-2.7B",
        "architecture":     "causal",
        "vision":           True,
        "target_modules":   ["q_proj", "v_proj"],
        "loader":           "Blip2ForConditionalGeneration",
        "model_family":     "blip2",
        "processor_type":   "AutoProcessor",
        "feature_strategy": "vision2seq",
        "collator_type":    "default_data_collator",
    },
    {
        "hf_id":            "Qwen/Qwen2-VL-2B-Instruct",
        "name":             "Qwen2-VL-2B",
        "architecture":     "causal",
        "vision":           True,
        "target_modules":   ["q_proj", "v_proj"],
        "loader":           "AutoModelForVision2Seq",
        "model_family":     "qwen_vl",
        "processor_type":   "AutoProcessor",
        "feature_strategy": "vision2seq_patchified",
        "collator_type":    "qwen_vl_collator",
    },
]


class TrainingAgent:
    """
    Contract-aware adaptive training agent.

    Validates feature-model compatibility before training.
    Recovers automatically on failure by calling ModelSelectionAgent +
    FeatureEngineeringAgent to regenerate a compatible combination.
    """

    def __init__(self, model_id: str = "mistral"):
        self.agent = Agent(
            name="TrainingAgent",
            model=Groq(id="llama-3.1-8b-instant"),
            instructions=[
                "You are a machine learning training expert.",
                "Assess training results for a Medical VQA model.",
                "Reply with ONLY this JSON: "
                '{"status": "ok", "train_loss": "<value>", "message": "<one sentence>"}',
                "Do not write code. No text outside the JSON.",
            ],
            show_tool_calls=False,
            markdown=False,
        )
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Main public interface ─────────────────────────────────────────────────

    def train(
        self,
        feature_path:        str,
        model_plan:          Dict[str, Any],
        device:              str = "",
        processed_data_path: str = "",
        vram_gb:             float = 0.0,
        ram_gb:              float = 0.0,
        dataset_size:        int   = 1000,
        modality:            str   = "",
    ) -> Dict[str, Any]:
        """
        Fine-tune with automatic recovery on failure.

        Args:
            feature_path        : Path to encoded HF Dataset (train/ val/ inside).
            model_plan          : Complete plan from ModelSelectionAgent.
            device              : 'cuda' | 'cpu' | '' (auto).
            processed_data_path : JSONL path for re-encoding on recovery.
            vram_gb / ram_gb    : Hardware info passed to ModelSelectionAgent on retry.
            dataset_size / modality : Passed to ModelSelectionAgent on retry.
        """
        if not device:
            device = self._detect_device()

        excluded_models: List[str] = list(model_plan.get("excluded_models", []))
        failure_reason:  str       = ""
        current_plan               = model_plan
        current_feature_path       = feature_path

        for attempt in range(1, MAX_RETRIES + 1):
            logger.info(
                f"[Training] Attempt {attempt}/{MAX_RETRIES} | "
                f"model={current_plan.get('name', current_plan.get('hf_id','?'))}"
            )

            result = self._train_once(
                feature_path=current_feature_path,
                model_plan=current_plan,
                device=device,
                processed_data_path=processed_data_path,
            )

            if result.get("status") != "failed":
                return result

            # Record failure
            failure_reason  = result.get("failure_reason", result.get("message", "unknown"))
            failed_hf_id    = current_plan.get("hf_id", "")
            excluded_models.append(failed_hf_id)

            self._log_failure(failed_hf_id, failure_reason)
            logger.warning(
                f"[Training] Attempt {attempt} failed: {failure_reason}. "
                f"Excluded so far: {excluded_models}"
            )

            if attempt >= MAX_RETRIES:
                break

            # ── Recovery: re-select model ──────────────────────────────────
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from agents.model_selection_agent import ModelSelectionAgent

            selector     = ModelSelectionAgent()
            current_plan = selector.select_model(
                vram_gb=vram_gb,
                ram_gb=ram_gb,
                dataset_size=dataset_size,
                modality=modality,
                device=device,
                failed_models=excluded_models,
                failure_reason=failure_reason,
            )
            logger.info(
                f"[Training] Recovery plan: {current_plan.get('name')} "
                f"(family={current_plan.get('model_family')})"
            )

            # ── Recovery: re-encode features ───────────────────────────────
            current_feature_path = self._get_features_for_plan(
                current_plan, processed_data_path, device, current_feature_path
            )

        return {
            "status":           "failed",
            "message":          f"All {MAX_RETRIES} training attempts failed.",
            "failure_reason":   failure_reason,
            "retry_recommended":False,
            "checkpoint_path":  "",
            "model_used":       current_plan.get("hf_id", ""),
        }

    # ── Backward compat ───────────────────────────────────────────────────────

    def train_from_processed(
        self,
        processed_data_path: str,
        model_plan:          Dict[str, Any],
        device:              str = "",
    ) -> Dict[str, Any]:
        if not device:
            device = self._detect_device()
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from agents.feature_engineering_agent import FeatureEngineeringAgent
        fe     = FeatureEngineeringAgent()
        result = fe.engineer_features(processed_data_path, model_plan, device)
        if result.get("status") == "failed":
            return {
                "status": "failed",
                "message": f"Inline encoding failed: {result.get('message')}",
                "checkpoint_path": "", "model_used": "",
            }
        return self.train(
            result["feature_path"], model_plan, device,
            processed_data_path=processed_data_path,
        )

    # ── Single attempt ────────────────────────────────────────────────────────

    def _train_once(
        self,
        feature_path:        str,
        model_plan:          Dict[str, Any],
        device:              str,
        processed_data_path: str,
    ) -> Dict[str, Any]:
        """One complete training attempt with pre-flight contract validation."""
        hf_id           = model_plan.get("hf_id", "")
        feature_strategy= model_plan.get("feature_strategy", "seq2seq")
        collator_type   = model_plan.get("collator_type",    "DataCollatorForSeq2Seq")
        is_vision       = model_plan.get("vision",           False)

        # ── Load feature datasets ──────────────────────────────────────────
        train_ds, val_ds = self._load_feature_datasets(feature_path)
        if train_ds is None:
            return self._fail("Feature dataset not found.", hf_id, retryable=True)

        # ── Pre-flight contract validation ────────────────────────────────
        validation_error = self._validate_contract(
            feature_path, model_plan, train_ds
        )
        if validation_error:
            return self._fail(validation_error, hf_id, retryable=True,
                              failure_class="tensor_dim_mismatch")

        # ── Load model ────────────────────────────────────────────────────
        try:
            model, tokenizer, active_plan = self._load_with_fallback(model_plan, device)
        except RuntimeError as e:
            return self._fail(str(e), hf_id, retryable=True,
                              failure_class="load_error")

        actual_hf_id = active_plan["hf_id"]
        if actual_hf_id != hf_id:
            logger.warning(
                f"[Training] Model changed {hf_id} → {actual_hf_id}. "
                "Re-encoding features …"
            )
            train_ds, val_ds = self._reencode(
                active_plan, processed_data_path, device, train_ds, val_ds
            )
            if train_ds is None:
                return self._fail(
                    f"Re-encoding failed for {actual_hf_id}", actual_hf_id, retryable=True
                )
            feature_strategy= active_plan.get("feature_strategy", "seq2seq")
            collator_type   = active_plan.get("collator_type",    "DataCollatorForSeq2Seq")
            is_vision       = active_plan.get("vision",           False)

        logger.info(
            f"[Training] Model={actual_hf_id} | Train={len(train_ds)} | "
            f"Val={len(val_ds)} | Columns={train_ds.column_names}"
        )

        # ── Apply LoRA ────────────────────────────────────────────────────
        model = self._apply_lora(model, active_plan)

        # ── Prepare columns ───────────────────────────────────────────────
        train_ds, val_ds = self._prepare_columns(train_ds, val_ds, is_vision)
        if len(train_ds) == 0:
            return self._fail("Dataset empty after column prep.", actual_hf_id, retryable=True)

        logger.info(f"[Training] Training columns: {train_ds.column_names}")

        # ── Checkpoint dir ────────────────────────────────────────────────
        safe = active_plan["name"].replace(" ", "_").replace("/", "_")
        out_dir = CHECKPOINT_DIR / safe
        out_dir.mkdir(parents=True, exist_ok=True)

        # ── Train ─────────────────────────────────────────────────────────
        try:
            trainer    = self._build_trainer(
                model, tokenizer, train_ds, val_ds,
                str(out_dir), active_plan, device, collator_type,
            )
            metrics    = self._run_training(trainer)
            train_loss = round(float(metrics.get("train_loss", 0.0)), 4)
        except Exception as e:
            import traceback
            logger.error(f"[Training] Loop failed: {e}")
            logger.debug(traceback.format_exc())
            fclass = self._classify_error(str(e))
            return self._fail(str(e), actual_hf_id, retryable=True, failure_class=fclass)

        # ── Save ──────────────────────────────────────────────────────────
        self._save_checkpoint(model, tokenizer, str(out_dir), active_plan)
        logger.info(f"[Training] Complete. Loss={train_loss}  Ckpt={out_dir}")

        result = self._assess(actual_hf_id, active_plan, train_loss, out_dir)
        result["checkpoint_path"]   = str(out_dir)
        result["model_used"]        = actual_hf_id
        result["train_loss"]        = str(train_loss)
        result["retry_recommended"] = False
        return result

    # ── Contract validation ───────────────────────────────────────────────────

    def _validate_contract(
        self,
        feature_path: str,
        model_plan:   Dict,
        train_ds,
    ) -> str:
        """
        Validate that encoded features match the model_plan contract.
        Returns an error string if invalid, empty string if valid.
        """
        hf_id            = model_plan.get("hf_id", "")
        feature_strategy = model_plan.get("feature_strategy", "")
        is_vision        = model_plan.get("vision", False)

        # Check metadata.json if it exists
        meta_file = Path(feature_path) / "metadata.json"
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text())
                cached_model    = meta.get("model_hf_id",      "")
                cached_strategy = meta.get("feature_strategy", "")

                if cached_model and cached_model != hf_id:
                    return (
                        f"Feature mismatch: features encoded for {cached_model} "
                        f"but model_plan requests {hf_id}"
                    )
                if cached_strategy and feature_strategy and cached_strategy != feature_strategy:
                    return (
                        f"Strategy mismatch: features use {cached_strategy} "
                        f"but model needs {feature_strategy}"
                    )
            except Exception:
                pass

        # Check required columns
        cols = set(train_ds.column_names)
        if "input_ids" not in cols:
            return "input_ids missing from encoded dataset"
        if "labels" not in cols and "decoder_input_ids" not in cols:
            return "labels/decoder_input_ids missing from encoded dataset"

        if is_vision and "pixel_values" not in cols:
            logger.warning(
                "[Training] Vision model selected but pixel_values not in features — "
                "will train on text only."
            )

        # Check pixel_values dimension if present
        if "pixel_values" in cols:
            try:
                sample = train_ds[0]["pixel_values"]
                # Detect stray leading batch dim: [[...]] instead of [...]
                if isinstance(sample, list) and len(sample) == 1 and isinstance(sample[0], list):
                    return (
                        "tensor_dim_mismatch: pixel_values has stray leading dimension "
                        "[1, ...] instead of [...]. Re-encode with updated FeatureEngineeringAgent."
                    )
            except Exception:
                pass

        return ""  # valid

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_with_fallback(
        self, model_plan: Dict, device: str
    ) -> Tuple[Any, Any, Dict]:
        candidates = self._build_candidates(model_plan, device)
        for candidate in candidates:
            hf_id = candidate["hf_id"]
            logger.info(f"[Training] Loading: {hf_id}")
            try:
                model, tok = self._load_single(
                    hf_id=hf_id,
                    loader_hint=candidate.get("loader", "auto"),
                    use_4bit=model_plan.get("use_4bit", False) and device == "cuda",
                    device=device,
                    precision=model_plan.get("precision", "fp16"),
                )
                logger.info(f"[Training] Loaded: {hf_id}")
                return model, tok, {**model_plan, **candidate}
            except Exception as e:
                logger.warning(f"[Training] {hf_id} failed: {e}")
        raise RuntimeError("All models failed to load.")

    def _build_candidates(self, model_plan: Dict, device: str) -> List[Dict]:
        requested_id   = model_plan.get("hf_id", "")
        fallbacks      = [f for f in FALLBACK_MODELS if f["hf_id"] != requested_id]
        text_fallbacks = [f for f in fallbacks if not f["vision"]]
        vis_fallbacks  = [f for f in fallbacks if f["vision"]]

        if device == "cpu":
            if not model_plan.get("vision", False):
                return [{**model_plan, "loader": model_plan.get("loader","auto")}] + text_fallbacks
            return text_fallbacks
        return [{**model_plan, "loader": model_plan.get("loader","auto")}] + fallbacks

    def _load_single(
        self, hf_id, loader_hint, use_4bit, device, precision
    ) -> Tuple[Any, Any]:
        import torch, transformers as tf

        dtype = torch.float16 if precision == "fp16" and device == "cuda" else torch.float32
        load_kw: Dict = dict(pretrained_model_name_or_path=hf_id, trust_remote_code=True)

        if use_4bit:
            from transformers import BitsAndBytesConfig
            load_kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
            )
            load_kw["device_map"] = "auto"
        elif device == "cuda":
            load_kw["torch_dtype"] = dtype
            load_kw["device_map"]  = "auto"
        else:
            load_kw["torch_dtype"] = dtype

        CLASS_MAP = {
            "AutoModelForSeq2SeqLM":             ["AutoModelForSeq2SeqLM"],
            "Blip2ForConditionalGeneration":      ["Blip2ForConditionalGeneration", "AutoModelForVision2Seq"],
            "InstructBlipForConditionalGeneration":["InstructBlipForConditionalGeneration", "AutoModelForVision2Seq"],
            "AutoModelForVision2Seq":             ["AutoModelForVision2Seq", "AutoModelForCausalLM"],
            "AutoModelForImageTextToText":        ["AutoModelForImageTextToText", "AutoModelForVision2Seq"],
            "AutoModelForCausalLM":               ["AutoModelForCausalLM"],
            "auto":                               ["AutoModelForSeq2SeqLM", "AutoModelForVision2Seq", "AutoModelForCausalLM"],
        }
        class_order = CLASS_MAP.get(loader_hint, CLASS_MAP["auto"])

        model = None; last_error = None
        for cls_name in class_order:
            cls = getattr(tf, cls_name, None)
            if cls is None: continue
            try:
                model = cls.from_pretrained(**load_kw)
                logger.info(f"[Training] {hf_id} with {cls_name}")
                break
            except Exception as e:
                last_error = e
                logger.debug(f"[Training] {cls_name}: {e}")

        if model is None:
            raise RuntimeError(f"Cannot load {hf_id}. Last: {last_error}")

        for method in ("gradient_checkpointing_enable", "enable_input_require_grads"):
            fn = getattr(model, method, None)
            if fn:
                try: fn()
                except Exception: pass

        try:
            from transformers import AutoProcessor
            tok = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
        except Exception:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)

        inner = getattr(tok, "tokenizer", tok)
        if hasattr(inner, "pad_token") and inner.pad_token is None:
            inner.pad_token = getattr(inner, "eos_token", None) or "<pad>"

        return model, tok

    # ── Re-encoding on model change ───────────────────────────────────────────

    def _reencode(
        self, active_plan, processed_path, device, train_ds, val_ds
    ) -> Tuple[Any, Any]:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from agents.feature_engineering_agent import FeatureEngineeringAgent

        hf_id  = active_plan["hf_id"]
        cached = Path("./data/features") / hf_id.replace("/", "_")
        if cached.exists() and (cached / "train").exists():
            new_train, new_val = self._load_feature_datasets(str(cached))
            if new_train and len(new_train) > 0:
                return new_train, new_val

        if processed_path and Path(processed_path).exists():
            fe     = FeatureEngineeringAgent()
            result = fe.engineer_features(processed_path, active_plan, device)
            if result.get("status") != "failed":
                return self._load_feature_datasets(result["feature_path"])

        logger.warning("[Training] Re-encoding unavailable — stripping vision cols.")
        return self._strip_vision(train_ds, val_ds)

    def _get_features_for_plan(
        self, plan, processed_path, device, current_fp
    ) -> str:
        hf_id  = plan.get("hf_id", "")
        cached = Path("./data/features") / hf_id.replace("/", "_")
        if cached.exists() and (cached / "train").exists():
            return str(cached)
        if processed_path and Path(processed_path).exists():
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from agents.feature_engineering_agent import FeatureEngineeringAgent
            fe     = FeatureEngineeringAgent()
            result = fe.engineer_features(processed_path, plan, device)
            if result.get("status") != "failed":
                return result["feature_path"]
        return current_fp

    def _strip_vision(self, train_ds, val_ds):
        vcols = {"pixel_values", "image_grid_thw", "image_patches", "image_sizes"}
        tr    = [c for c in train_ds.column_names if c in vcols]
        vl    = [c for c in val_ds.column_names   if c in vcols]
        if tr: train_ds = train_ds.remove_columns(tr)
        if vl: val_ds   = val_ds.remove_columns(vl)
        if "input_ids" not in train_ds.column_names:
            return None, None
        return train_ds, val_ds

    # ── Feature loading ───────────────────────────────────────────────────────

    def _load_feature_datasets(self, feature_path: str):
        if not feature_path or not Path(feature_path).exists():
            return None, None
        try:
            from datasets import load_from_disk
            train_path = Path(feature_path) / "train"
            val_path   = Path(feature_path) / "val"
            if not train_path.exists():
                return None, None
            train_ds = load_from_disk(str(train_path))
            if val_path.exists():
                val_ds = load_from_disk(str(val_path))
            else:
                split    = train_ds.train_test_split(test_size=0.1, seed=42)
                train_ds = split["train"]
                val_ds   = split["test"]
            return train_ds, val_ds
        except Exception as e:
            logger.error(f"[Training] Feature load: {e}")
            return None, None

    # ── LoRA ──────────────────────────────────────────────────────────────────

    def _apply_lora(self, model, plan: Dict):
        try:
            from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
            try: model = prepare_model_for_kbit_training(model)
            except Exception: pass
            arch      = plan.get("architecture", "seq2seq")
            task_type = TaskType.SEQ_2_SEQ_LM if arch == "seq2seq" else TaskType.CAUSAL_LM
            cfg = LoraConfig(
                r=plan.get("lora_r", 8),
                lora_alpha=plan.get("lora_alpha", 16),
                lora_dropout=plan.get("lora_dropout", 0.05),
                target_modules=plan.get("target_modules", ["q", "v"]),
                bias="none",
                task_type=task_type,
            )
            model = get_peft_model(model, cfg)
            model.print_trainable_parameters()
            logger.info(f"[Training] LoRA r={cfg.r}  task={task_type.value}")
        except ImportError:
            logger.warning("[Training] PEFT not installed — no LoRA.")
        except Exception as e:
            logger.warning(f"[Training] LoRA skipped: {e}")
        return model

    # ── Column preparation ────────────────────────────────────────────────────

    def _prepare_columns(self, train_ds, val_ds, is_vision: bool):
        keep    = TENSOR_COLUMNS if is_vision else TEXT_ONLY_COLUMNS
        drop_tr = [c for c in train_ds.column_names if c not in keep]
        drop_vl = [c for c in val_ds.column_names   if c not in keep]
        if drop_tr:
            train_ds = train_ds.remove_columns(drop_tr)
            logger.info(f"[Training] Dropped: {drop_tr}")
        if drop_vl:
            val_ds   = val_ds.remove_columns(drop_vl)

        # Add labels if absent
        if ("labels" not in train_ds.column_names and
                "input_ids" in train_ds.column_names):
            train_ds = train_ds.map(lambda x: {"labels": x["input_ids"]}, batched=True)
            val_ds   = val_ds.map(  lambda x: {"labels": x["input_ids"]}, batched=True)

        # Clamp pad positions to -100
        if "labels" in train_ds.column_names:
            def clamp(batch):
                batch["labels"] = [[t if t > 0 else -100 for t in seq]
                                    for seq in batch["labels"]]
                return batch
            train_ds = train_ds.map(clamp, batched=True)
            val_ds   = val_ds.map(clamp,   batched=True)

        logger.info(f"[Training] Final columns: {train_ds.column_names}")
        return train_ds, val_ds

    # ── Trainer ───────────────────────────────────────────────────────────────

    def _build_trainer(
        self, model, tok, train_ds, val_ds,
        out_dir, plan, device, collator_type,
    ):
        import torch, transformers
        from transformers import (
            TrainingArguments, Trainer,
            EarlyStoppingCallback,
            DataCollatorForSeq2Seq,
            default_data_collator,
        )

        precision  = plan.get("precision",     "fp16")
        batch_size = plan.get("batch_size",    2)
        epochs     = plan.get("epochs",        3)
        lr         = plan.get("learning_rate", 2e-4)

        use_fp16 = precision == "fp16" and torch.cuda.is_available()
        use_bf16 = False
        if torch.cuda.is_available():
            if torch.cuda.get_device_properties(0).major >= 8:
                use_bf16, use_fp16 = True, False

        n_steps  = max(1, len(train_ds) // batch_size)
        warmup   = min(100, max(1, n_steps // 10))
        log_step = max(1, n_steps // 5)

        args = TrainingArguments(
            output_dir=out_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=max(1, batch_size // 2),
            learning_rate=lr,
            fp16=use_fp16,
            bf16=use_bf16,
            gradient_accumulation_steps=4 if batch_size <= 2 else 2,
            # ── LR scheduler: cosine annealing ──────────────────────────────
            lr_scheduler_type="cosine",
            warmup_steps=warmup,
            # ── Gradient clipping ───────────────────────────────────────────
            max_grad_norm=1.0,
            # ── Early stopping support ──────────────────────────────────────
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            weight_decay=0.01,
            logging_steps=log_step,
            remove_unused_columns=False,
            report_to="none",
            dataloader_num_workers=0,
        )

        early_stop = EarlyStoppingCallback(
            early_stopping_patience=2,
            early_stopping_threshold=0.001,
        )

        ver       = tuple(int(x) for x in transformers.__version__.split(".")[:2])
        tok_kwarg = "processing_class" if ver >= (4, 46) else "tokenizer"

        # ── Collator selection from contract ──────────────────────────────
        if collator_type == "DataCollatorForSeq2Seq":
            inner_tok = getattr(tok, "tokenizer", tok)
            collator  = DataCollatorForSeq2Seq(
                tokenizer=inner_tok,
                model=model,
                padding=True,
                label_pad_token_id=-100,
            )
            return Trainer(
                model=model, args=args, callbacks=[early_stop],
                train_dataset=train_ds, eval_dataset=val_ds,
                data_collator=collator,
            )
        elif collator_type in ("default_data_collator", "qwen_vl_collator"):
            return Trainer(
                model=model, args=args, callbacks=[early_stop],
                train_dataset=train_ds, eval_dataset=val_ds,
                data_collator=default_data_collator,
            )
        else:
            return Trainer(
                model=model, args=args, callbacks=[early_stop],
                train_dataset=train_ds, eval_dataset=val_ds,
                **{tok_kwarg: tok},
            )

    def _run_training(self, trainer) -> Dict:
        logger.info("[Training] Starting …")
        result  = trainer.train()
        metrics = getattr(result, "metrics", {})
        logger.info(f"[Training] Metrics: {metrics}")
        return metrics

    # ── Save ──────────────────────────────────────────────────────────────────

    def _save_checkpoint(self, model, tokenizer, out_dir, plan):
        p = Path(out_dir)
        try:    model.save_pretrained(str(p))
        except Exception as e: logger.warning(f"[Training] model save: {e}")
        try:    tokenizer.save_pretrained(str(p))
        except Exception as e: logger.warning(f"[Training] tok save: {e}")
        with open(p / "model_plan.json", "w") as f:
            json.dump(plan, f, indent=2, default=str)
        logger.info(f"[Training] Checkpoint → {out_dir}")

    # ── Failure logging ───────────────────────────────────────────────────────

    def _log_failure(self, hf_id: str, reason: str) -> None:
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        history = []
        if FAILED_MODELS_LOG.exists():
            try:
                history = json.loads(FAILED_MODELS_LOG.read_text())
            except Exception:
                pass
        fclass = self._classify_error(reason)
        history.append({
            "model":         hf_id,
            "reason":        reason,
            "failure_class": fclass,
        })
        FAILED_MODELS_LOG.write_text(json.dumps(history, indent=2))
        logger.info(f"[Training] Logged failure: {hf_id} ({fclass})")

    def _classify_error(self, error: str) -> str:
        e = (error or "").lower()
        if any(k in e for k in ("out of memory", "oom", "cuda error")): return "oom"
        if any(k in e for k in ("dimension", "shape", "unpack")):       return "tensor_dim_mismatch"
        if any(k in e for k in ("processor", "tokenizer")):             return "processor_not_supported"
        if any(k in e for k in ("cannot load", "no module", "weight")): return "load_error"
        return "training_error"

    # ── Groq assessment ───────────────────────────────────────────────────────

    def _assess(self, hf_id, plan, train_loss, out_dir) -> Dict:
        prompt = (
            f"Medical VQA model fine-tuned.\n"
            f"Model: {hf_id} | Family: {plan.get('model_family','?')}\n"
            f"Strategy: {plan.get('feature_strategy','?')} | "
            f"Collator: {plan.get('collator_type','?')}\n"
            f"LoRA r={plan.get('lora_r',8)}  Epochs={plan.get('epochs',3)}\n"
            f"LR scheduler: cosine | Gradient clip: 1.0 | Early stop: patience=2\n"
            f"Training loss: {train_loss}\n\n"
            f"Is this acceptable for Medical VQA evaluation?\n"
            f'Reply ONLY: {{"status": "ok", "train_loss": "{train_loss}", '
            f'"message": "<one sentence>"}}'
        )
        try:
            resp = self.agent.run(prompt)
            return self._parse(resp)
        except Exception as e:
            logger.warning(f"[Training] Groq assessment: {e}")
            return {"status": "ok", "message": "Training complete."}

    def _parse(self, response) -> Dict:
        try:
            text  = response.content if hasattr(response, "content") else str(response)
            match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            pass
        return {"status": "ok", "message": "Training complete."}

    def _fail(
        self, msg: str, hf_id: str = "", retryable: bool = False,
        failure_class: str = "training_error"
    ) -> Dict:
        logger.error(f"[Training] {msg}")
        return {
            "status":           "failed",
            "message":          msg,
            "failure_reason":   msg,
            "retry_recommended":retryable,
            "checkpoint_path":  "",
            "model_used":       hf_id,
            "failure_class":    failure_class,
        }

    def _detect_device(self) -> str:
        try:
            import torch
            if torch.cuda.is_available():          return "cuda"
            if torch.backends.mps.is_available():  return "mps"
        except ImportError: pass
        return "cpu"


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch, sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s][%(levelname)s] %(message)s")

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    vram_gb = 0.0
    if device == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
    print(f"Device: {device}  VRAM: {vram_gb:.1f}GB")

    plan = {
        "hf_id":            "google/flan-t5-base",
        "name":             "Flan-T5-Base",
        "architecture":     "seq2seq",
        "vision":           False,
        "use_4bit":         False,
        "model_family":     "flan_t5",
        "processor_type":   "AutoTokenizer",
        "feature_strategy": "seq2seq",
        "collator_type":    "DataCollatorForSeq2Seq",
        "tensor_schema":    "input_ids[b,s], attention_mask[b,s], labels[b,s]",
        "lora_r":           8,  "lora_alpha": 16, "lora_dropout": 0.05,
        "target_modules":   ["q", "v"],
        "batch_size":       2,  "epochs": 1, "learning_rate": 2e-4,
        "precision":        "fp32" if device == "cpu" else "fp16",
        "loader":           "AutoModelForSeq2SeqLM",
    }

    agent      = TrainingAgent()
    processed  = "./data/processed/processed_dataset.jsonl"
    feature    = "./data/features/google_flan-t5-base"

    if Path(feature + "/train").exists():
        print(f"\nUsing features: {feature}")
        result = agent.train(feature, plan, device=device, processed_data_path=processed)
    elif Path(processed).exists():
        print(f"\nEncoding from: {processed}")
        result = agent.train_from_processed(processed, plan, device=device)
    else:
        print("\nNo data. Run pipeline first:")
        print("  python main.py --image image.jpg --question '...' --dry-run")
        result = {}

    if result:
        print("\n" + "="*55)
        print(json.dumps(result, indent=2, default=str))
        print("="*55)
