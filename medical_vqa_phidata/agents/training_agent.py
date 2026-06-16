"""
agents/training_agent.py

TrainingAgent — loads encoded features from FeatureEngineeringAgent
and fine-tunes the selected model using PEFT/LoRA.

Adaptive: no hardcoded model. Works with any model family selected
by ModelSelectionAgent. Uses encoded HF Dataset from FeatureEngineeringAgent.

Uses same PhiData + Groq structure as all other agents in the project.
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
}

# Fallback model chain — lightest first
FALLBACK_MODELS: List[Dict] = [
    {
        "hf_id":          "google/flan-t5-base",
        "name":           "Flan-T5-Base",
        "architecture":   "seq2seq",
        "vision":         False,
        "target_modules": ["q", "v"],
        "loader":         "AutoModelForSeq2SeqLM",
    },
    {
        "hf_id":          "google/flan-t5-large",
        "name":           "Flan-T5-Large",
        "architecture":   "seq2seq",
        "vision":         False,
        "target_modules": ["q", "v"],
        "loader":         "AutoModelForSeq2SeqLM",
    },
    {
        "hf_id":          "Salesforce/blip2-opt-2.7b",
        "name":           "BLIP2-OPT-2.7B",
        "architecture":   "causal",
        "vision":         True,
        "target_modules": ["q_proj", "v_proj"],
        "loader":         "Blip2ForConditionalGeneration",
    },
    {
        "hf_id":          "Qwen/Qwen2-VL-2B-Instruct",
        "name":           "Qwen2-VL-2B",
        "architecture":   "causal",
        "vision":         True,
        "target_modules": ["q_proj", "v_proj"],
        "loader":         "AutoModelForVision2Seq",
    },
]


class TrainingAgent:
    """
    Fine-tunes the selected model on features produced by FeatureEngineeringAgent.
    Adaptive: loads encoded HF Dataset from disk, applies LoRA, and trains.
    Falls back to lighter models automatically if the selected one fails.
    Works both standalone and inside the full pipeline.
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
        model_plan: Dict[str, Any],
        device: str = "",
    ) -> Dict[str, Any]:
        """
        Fine-tune the model using pre-encoded feature dataset.

        Args:
            feature_path : Path returned by FeatureEngineeringAgent (base dir).
                           Contains train/ and val/ subdirectories.
            model_plan   : Dict from ModelSelectionAgent.
            device       : 'cuda' | 'cpu' | '' (auto-detect).

        Returns:
            Dict with checkpoint_path, train_loss, status, model_used.
        """
        if not device:
            device = self._detect_device()
        logger.info(f"[Training] Device: {device}")

        # Load encoded datasets from FeatureEngineeringAgent output
        train_ds, val_ds = self._load_feature_datasets(feature_path)

        if train_ds is None or len(train_ds) == 0:
            return {
                "status":          "failed",
                "message":         f"No encoded training data found at: {feature_path}",
                "checkpoint_path": "",
                "model_used":      "",
            }

        logger.info(
            f"[Training] Loaded → Train={len(train_ds)}  Val={len(val_ds)}"
        )
        logger.info(f"[Training] Columns: {train_ds.column_names}")

        # Load model with automatic fallback
        try:
            model, tokenizer, active_plan = self._load_with_fallback(
                model_plan, device
            )
        except RuntimeError as e:
            return {
                "status":          "failed",
                "message":         str(e),
                "checkpoint_path": "",
                "model_used":      "",
            }

        hf_id        = active_plan["hf_id"]
        name         = active_plan["name"]
        architecture = active_plan.get("architecture", "seq2seq")
        lora_r       = active_plan.get("lora_r",        8)
        lora_alpha   = active_plan.get("lora_alpha",    16)
        lora_drop    = active_plan.get("lora_dropout",  0.05)
        target_mods  = active_plan.get("target_modules",["q", "v"])
        batch_size   = active_plan.get("batch_size",    2)
        epochs       = active_plan.get("epochs",        3)
        lr           = active_plan.get("learning_rate", 2e-4)
        precision    = active_plan.get("precision",     "fp16")

        # Output checkpoint directory
        safe_name = name.replace(" ", "_").replace("/", "_")
        out_dir   = CHECKPOINT_DIR / safe_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Apply LoRA adapters
        model = self._apply_lora(
            model, lora_r, lora_alpha, lora_drop, target_mods, architecture
        )

        # Prepare columns for Trainer
        train_ds, val_ds = self._prepare_columns(train_ds, val_ds)

        if len(train_ds) == 0:
            return {
                "status":          "failed",
                "message":         "No valid tensor columns after column preparation.",
                "checkpoint_path": str(out_dir),
                "model_used":      hf_id,
            }

        # Build Trainer and run training
        try:
            trainer    = self._build_trainer(
                model, tokenizer, train_ds, val_ds,
                str(out_dir), batch_size, epochs, lr, precision, device,
            )
            metrics    = self._run_training(trainer)
            train_loss = round(float(metrics.get("train_loss", 0.0)), 4)
        except Exception as e:
            logger.error(f"[Training] Training failed: {e}")
            return {
                "status":          "failed",
                "message":         f"Training loop error: {e}",
                "checkpoint_path": str(out_dir),
                "model_used":      hf_id,
            }

        # Save checkpoint
        self._save(model, tokenizer, str(out_dir), active_plan)
        logger.info(f"[Training] Complete. Loss={train_loss}  Ckpt={out_dir}")

        # Groq assessment
        assessment = self._get_llm_assessment(
            hf_id, lora_r, lora_alpha, epochs,
            batch_size, lr, train_loss, out_dir,
        )
        assessment["checkpoint_path"] = str(out_dir)
        assessment["model_used"]      = hf_id
        assessment["train_loss"]      = str(train_loss)
        return assessment

    # ── Compatibility method ──────────────────────────────────────────────────

    def train_from_processed(
        self,
        processed_data_path: str,
        model_plan: Dict[str, Any],
        device: str = "",
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

        fe_agent = FeatureEngineeringAgent()
        fe_result = fe_agent.engineer_features(
            processed_data_path, model_plan, device or self._detect_device()
        )
        if fe_result.get("status") == "failed":
            return {
                "status":          "failed",
                "message":         f"Inline feature engineering failed: {fe_result.get('message')}",
                "checkpoint_path": "",
                "model_used":      "",
            }
        return self.train(fe_result["feature_path"], model_plan, device)

    # ── Device ────────────────────────────────────────────────────────────────

    def _detect_device(self) -> str:
        try:
            import torch
            if torch.cuda.is_available():          return "cuda"
            if torch.backends.mps.is_available():  return "mps"
        except ImportError:
            pass
        return "cpu"

    # ── Load encoded datasets ─────────────────────────────────────────────────

    def _load_feature_datasets(self, feature_path: str):
        """
        Load HuggingFace Datasets saved by FeatureEngineeringAgent.
        Returns (train_ds, val_ds) or (None, None) on failure.
        """
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
                # Create val split from train if missing
                split    = train_ds.train_test_split(test_size=0.1, seed=42)
                train_ds = split["train"]
                val_ds   = split["test"]
                logger.info("[Training] Created val split from train (10%).")

            return train_ds, val_ds

        except Exception as e:
            logger.error(f"[Training] Feature dataset load failed: {e}")
            return None, None

    # ── Model loading with fallback ───────────────────────────────────────────

    def _load_with_fallback(
        self, model_plan: Dict, device: str
    ) -> Tuple[Any, Any, Dict]:
        """Try requested model then fallbacks until one loads."""
        requested  = {**model_plan, "loader": model_plan.get("loader", "auto")}
        already    = requested.get("hf_id", "")
        fallbacks  = [f for f in FALLBACK_MODELS if f["hf_id"] != already]
        candidates = [requested] + fallbacks

        for candidate in candidates:
            hf_id = candidate["hf_id"]
            logger.info(f"[Training] Trying: {hf_id}")
            try:
                model, tok = self._load_single(
                    hf_id=hf_id,
                    loader_hint=candidate.get("loader", "auto"),
                    use_4bit=model_plan.get("use_4bit", False),
                    device=device,
                    precision=model_plan.get("precision", "fp16"),
                )
                logger.info(f"[Training] Loaded: {hf_id}")
                active = {**model_plan, **candidate}
                return model, tok, active
            except Exception as e:
                logger.warning(f"[Training] {hf_id} failed: {e}")

        raise RuntimeError(
            "All models failed to load. "
            "Check internet connection and HuggingFace access."
        )

    def _load_single(
        self,
        hf_id: str,
        loader_hint: str,
        use_4bit: bool,
        device: str,
        precision: str,
    ) -> Tuple[Any, Any]:
        import torch
        import transformers as tf

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
            load_kw["torch_dtype"] = dtype
            load_kw["device_map"]  = "auto"
        else:
            load_kw["torch_dtype"] = dtype

        CLASS_MAP = {
            "AutoModelForSeq2SeqLM":        ["AutoModelForSeq2SeqLM"],
            "Blip2ForConditionalGeneration": ["Blip2ForConditionalGeneration",
                                              "AutoModelForVision2Seq"],
            "AutoModelForVision2Seq":        ["AutoModelForVision2Seq",
                                              "AutoModelForCausalLM"],
            "auto":                          ["AutoModelForSeq2SeqLM",
                                              "AutoModelForVision2Seq",
                                              "AutoModelForCausalLM"],
        }
        class_order = CLASS_MAP.get(loader_hint, CLASS_MAP["auto"])

        model      = None
        last_error = None
        for cls_name in class_order:
            cls = getattr(tf, cls_name, None)
            if cls is None:
                continue
            try:
                model = cls.from_pretrained(**load_kw)
                logger.info(f"[Training] {hf_id} loaded with {cls_name}")
                break
            except Exception as e:
                last_error = e
                logger.debug(f"[Training] {cls_name} failed: {e}")

        if model is None:
            raise RuntimeError(
                f"Cannot load {hf_id}. Last error: {last_error}"
            )

        # Enable gradient checkpointing
        for method in ("gradient_checkpointing_enable", "enable_input_require_grads"):
            fn = getattr(model, method, None)
            if fn:
                try: fn()
                except Exception: pass

        # Load tokenizer / processor
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
            logger.info(f"[Training] LoRA applied: r={r}  alpha={alpha}  task={task_type}")
        except ImportError:
            logger.warning("[Training] PEFT not installed — training without LoRA.")
        except Exception as e:
            logger.warning(f"[Training] LoRA skipped: {e}")
        return model

    # ── Column preparation ────────────────────────────────────────────────────

    def _prepare_columns(self, train_ds, val_ds):
        """
        Keep only tensor-compatible columns. Add labels if missing.
        Also flattens any nested list columns that cause unpack errors.
        """
        import numpy as np

        # ── Drop non-tensor columns ───────────────────────────────────────────
        drop_tr = [c for c in train_ds.column_names if c not in TENSOR_COLUMNS]
        drop_vl = [c for c in val_ds.column_names   if c not in TENSOR_COLUMNS]
        if drop_tr: train_ds = train_ds.remove_columns(drop_tr)
        if drop_vl: val_ds   = val_ds.remove_columns(drop_vl)

        # ── Add labels if missing ─────────────────────────────────────────────
        if ("labels"    not in train_ds.column_names and
                "input_ids" in  train_ds.column_names):
            train_ds = train_ds.map(
                lambda x: {"labels": x["input_ids"]}, batched=True
            )
            val_ds = val_ds.map(
                lambda x: {"labels": x["input_ids"]}, batched=True
            )

        # ── Flatten nested pixel_values if present ────────────────────────────
        # InstructBLIP / BLIP-2 encode pixel_values as list-of-list-of-list.
        # HuggingFace Trainer needs it as a flat list or numpy array.
        # We detect and flatten here to prevent "too many values to unpack".
        if "pixel_values" in train_ds.column_names:
            try:
                sample = train_ds[0]["pixel_values"]
                # If it is a nested list (list of lists), leave as-is —
                # DataCollator will handle it. Only flatten if it is
                # a plain Python list wrapped in an extra list.
                if isinstance(sample, list) and len(sample) == 1 and isinstance(sample[0], list):
                    def unwrap_pixel(batch):
                        return {"pixel_values": [pv[0] for pv in batch["pixel_values"]]}
                    train_ds = train_ds.map(unwrap_pixel, batched=True)
                    val_ds   = val_ds.map(unwrap_pixel,   batched=True)
                    logger.info("[Training] Unwrapped nested pixel_values.")
            except Exception as e:
                logger.debug(f"[Training] pixel_values check skipped: {e}")

        logger.info(f"[Training] Final columns: {train_ds.column_names}")
        return train_ds, val_ds

    # ── Trainer ───────────────────────────────────────────────────────────────

    def _build_trainer(
        self, model, tok, train_ds, val_ds,
        out_dir, batch_size, epochs, lr, precision, device,
    ):
        import torch
        import transformers
        from transformers import TrainingArguments, Trainer

        use_fp16 = precision == "fp16" and torch.cuda.is_available()
        use_bf16 = False
        if torch.cuda.is_available():
            if torch.cuda.get_device_properties(0).major >= 8:
                use_bf16, use_fp16 = True, False

        warmup = min(100, max(1, len(train_ds) // batch_size // 10))

        args = TrainingArguments(
            output_dir=out_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=max(1, batch_size // 2),
            learning_rate=lr,
            fp16=use_fp16,
            bf16=use_bf16,
            gradient_accumulation_steps=4 if batch_size <= 2 else 2,
            warmup_steps=warmup,
            weight_decay=0.01,
            logging_steps=max(1, len(train_ds) // batch_size // 5),
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            remove_unused_columns=False,
            report_to="none",
            dataloader_num_workers=0,
            max_grad_norm=1.0,
        )

        # transformers >= 4.46 renamed tokenizer → processing_class
        ver      = tuple(int(x) for x in transformers.__version__.split(".")[:2])
        tok_kwarg = "processing_class" if ver >= (4, 46) else "tokenizer"

        # Use DataCollatorWithPadding for text models
        # Use default_data_collator for vision models (pixel_values present)
        from transformers import default_data_collator
        has_pixels = "pixel_values" in train_ds.column_names

        if has_pixels:
            return Trainer(
                model=model,
                args=args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                data_collator=default_data_collator,
            )
        else:
            return Trainer(
                model=model,
                args=args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                **{tok_kwarg: tok},
            )

    def _run_training(self, trainer) -> Dict:
        logger.info("[Training] Starting training loop …")
        result  = trainer.train()
        metrics = getattr(result, "metrics", {})
        logger.info(f"[Training] Metrics: {metrics}")
        return metrics

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

    plan = {
        "hf_id":          "google/flan-t5-base",
        "name":           "Flan-T5-Base",
        "architecture":   "seq2seq",
        "vision":         False,
        "use_4bit":       False,
        "lora_r":         8,
        "lora_alpha":     16,
        "lora_dropout":   0.05,
        "target_modules": ["q", "v"],
        "batch_size":     2,
        "epochs":         1,
        "learning_rate":  2e-4,
        "precision":      "fp16",
    }

    agent = TrainingAgent()

    # Check for pre-encoded features first
    feature_path = f"./data/features/{plan['hf_id'].replace('/','_')}"
    processed_path = "./data/processed/processed_dataset.jsonl"

    if Path(feature_path).exists():
        print(f"\nUsing encoded features: {feature_path}")
        result = agent.train(feature_path, plan, device=device)
    elif Path(processed_path).exists():
        print(f"\nEncoding from processed data: {processed_path}")
        result = agent.train_from_processed(processed_path, plan, device=device)
    else:
        print(f"\nNo data found. Run full pipeline first:")
        print("  python main.py --image image.jpg --question 'Is there pneumonia?' --dry-run")
        result = {}

    if result:
        print("\n" + "="*50)
        print("Training Result:")
        print(json.dumps(result, indent=2, default=str))
        print("="*50)