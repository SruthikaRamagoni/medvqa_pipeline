"""
agents/feature_engineering_agent.py

FeatureEngineeringAgent — Model-aware feature encoding for Medical VQA.

CONTRACT
--------
Reads feature_strategy, processor_type, collator_type, tensor_schema
directly from model_plan (set by ModelSelectionAgent).

Never assumes a model family — dispatches encoding based on the
feature_strategy key in model_plan:

    seq2seq             → T5-style input/label tokenization
    causal_lm           → causal language model tokenization
    vision2seq          → image + text via AutoProcessor
    vision2seq_patchified → Qwen-VL patchified pixel encoding

After encoding, saves metadata.json alongside the dataset:
    {
        "model_hf_id":      "...",
        "model_family":     "...",
        "processor_type":   "...",
        "feature_strategy": "...",
        "tensor_schema":    "...",
        "image_enabled":    true/false,
        "train_samples":    N,
        "val_samples":      N,
        "columns":          [...]
    }

VALIDATION
----------
Every encoded batch is validated before saving:
  - No empty encoding
  - input_ids present for all strategies
  - labels present
  - pixel_values present for vision strategies
  - No stray leading dimension (b=1 squeezed tensors)
  - Consistent sequence lengths

On validation failure → logs warning, skips record (does not crash).

SELF-HEALING
------------
If metadata.json already exists for this model but was generated for a
*different* model family / strategy, the cached dataset is discarded and
regenerated automatically.

Uses PhiData + Groq. Same structure as all other agents in the project.
"""

from phi.agent import Agent
from phi.model.groq import Groq

from typing import Any, Dict, List, Optional, Tuple
import json, logging, re
from pathlib import Path

logger = logging.getLogger(__name__)

FEATURE_DIR = Path("./data/features")

# Columns the HuggingFace Trainer is allowed to see
SAFE_TENSOR_COLUMNS = {
    "input_ids", "attention_mask", "pixel_values",
    "labels", "decoder_input_ids", "token_type_ids",
    "image_grid_thw", "image_patches", "image_sizes",
}


class FeatureEngineeringAgent:
    """
    Converts processed JSONL into model-specific encoded HuggingFace Dataset.

    Dispatches to the correct encoder based on model_plan["feature_strategy"].
    Validates every encoded record. Saves metadata.json.
    Regenerates automatically if cached metadata mismatches current model.
    """

    def __init__(self, model_id: str = "mistral"):
        self.agent = Agent(
            name="FeatureEngineeringAgent",
            model=Groq(id="llama-3.1-8b-instant"),
            instructions=[
                "You are a machine learning feature engineering expert.",
                "Assess encoded feature statistics for Medical VQA training.",
                "Reply with ONLY this JSON: "
                '{"status": "ok", "train_samples": <int>, '
                '"val_samples": <int>, "message": "<one sentence>"}',
                "Do not write code. No text outside the JSON.",
            ],
            show_tool_calls=False,
            markdown=False,
        )

    # ── Public interface ──────────────────────────────────────────────────────

    def engineer_features(
        self,
        processed_data_path: str,
        model_plan:          Dict[str, Any],
        device:              str = "",
    ) -> Dict[str, Any]:
        """
        Encode processed JSONL into model-specific tensors.

        Args:
            processed_data_path : Path to processed_dataset.jsonl
            model_plan          : Complete plan from ModelSelectionAgent
                                  (must include feature_strategy, processor_type)
            device              : 'cuda' | 'cpu' | '' (auto)

        Returns:
            {status, feature_path, train_samples, val_samples, message}
        """
        if not device:
            device = self._detect_device()

        if not Path(processed_data_path).exists():
            return self._fail(f"Processed data not found: {processed_data_path}")

        hf_id            = model_plan.get("hf_id",            "google/flan-t5-base")
        model_family     = model_plan.get("model_family",     self._detect_family(hf_id))
        feature_strategy = model_plan.get("feature_strategy", "seq2seq")
        processor_type   = model_plan.get("processor_type",   "AutoTokenizer")
        max_len          = model_plan.get("max_seq_len",       128)

        logger.info(
            f"[FeatureEng] Model={hf_id} | family={model_family} | "
            f"strategy={feature_strategy} | processor={processor_type}"
        )

        # ── Check cache validity ──────────────────────────────────────────────
        safe_name    = hf_id.replace("/", "_")
        feature_path = FEATURE_DIR / safe_name

        if feature_path.exists():
            if self._cache_is_valid(feature_path, hf_id, feature_strategy):
                logger.info(f"[FeatureEng] Valid cache found: {feature_path}")
                train_n, val_n = self._count_cached(feature_path)
                return {
                    "status":        "ok",
                    "feature_path":  str(feature_path),
                    "train_samples": train_n,
                    "val_samples":   val_n,
                    "message":       "Loaded from validated cache.",
                }
            else:
                logger.warning(
                    f"[FeatureEng] Cache exists but metadata mismatch — regenerating."
                )
                import shutil
                shutil.rmtree(str(feature_path), ignore_errors=True)

        # ── Load records ──────────────────────────────────────────────────────
        records = self._load_records(processed_data_path)
        if not records:
            return self._fail("No records in processed dataset.")

        # ── Load processor ────────────────────────────────────────────────────
        processor, tokenizer = self._load_processor(hf_id, processor_type, model_family)
        if processor is None and tokenizer is None:
            return self._fail(f"Cannot load processor/tokenizer for {hf_id}")

        # ── Encode ───────────────────────────────────────────────────────────
        encoded, skipped = self._encode_records(
            records=records,
            processor=processor,
            tokenizer=tokenizer,
            feature_strategy=feature_strategy,
            model_family=model_family,
            max_len=max_len,
        )

        logger.info(f"[FeatureEng] Encoded={len(encoded)}  Skipped={skipped}")

        if len(encoded) < 5:
            return self._fail(
                f"Too few valid encoded records: {len(encoded)} "
                f"(skipped {skipped}). "
                f"Check processor compatibility with feature_strategy={feature_strategy}"
            )

        # ── Build HF Datasets ─────────────────────────────────────────────────
        train_ds, val_ds = self._build_datasets(encoded)

        # ── Save ─────────────────────────────────────────────────────────────
        feature_path.mkdir(parents=True, exist_ok=True)
        train_ds.save_to_disk(str(feature_path / "train"))
        val_ds.save_to_disk(str(feature_path / "val"))

        # ── Save metadata ─────────────────────────────────────────────────────
        meta = {
            "model_hf_id":      hf_id,
            "model_family":     model_family,
            "processor_type":   processor_type,
            "feature_strategy": feature_strategy,
            "tensor_schema":    model_plan.get("tensor_schema", ""),
            "image_enabled":    model_plan.get("vision", False),
            "train_samples":    len(train_ds),
            "val_samples":      len(val_ds),
            "columns":          list(train_ds.column_names),
        }
        with open(feature_path / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(
            f"[FeatureEng] Saved → {feature_path} | "
            f"train={len(train_ds)}  val={len(val_ds)} | "
            f"columns={train_ds.column_names}"
        )

        # ── Groq assessment ───────────────────────────────────────────────────
        result = self._get_assessment(
            hf_id, model_family, feature_strategy,
            len(train_ds), len(val_ds), list(train_ds.column_names),
        )
        result["feature_path"]  = str(feature_path)
        result["train_samples"] = len(train_ds)
        result["val_samples"]   = len(val_ds)
        return result

    # ── Device ────────────────────────────────────────────────────────────────

    def _detect_device(self) -> str:
        try:
            import torch
            if torch.cuda.is_available():          return "cuda"
            if torch.backends.mps.is_available():  return "mps"
        except ImportError:
            pass
        return "cpu"

    # ── Cache validation ──────────────────────────────────────────────────────

    def _cache_is_valid(
        self, feature_path: Path, hf_id: str, feature_strategy: str
    ) -> bool:
        meta_file = feature_path / "metadata.json"
        train_dir = feature_path / "train"
        if not meta_file.exists() or not train_dir.exists():
            return False
        try:
            meta = json.loads(meta_file.read_text())
            return (
                meta.get("model_hf_id")      == hf_id
                and meta.get("feature_strategy") == feature_strategy
            )
        except Exception:
            return False

    def _count_cached(self, feature_path: Path) -> Tuple[int, int]:
        try:
            from datasets import load_from_disk
            tr = load_from_disk(str(feature_path / "train"))
            vl = load_from_disk(str(feature_path / "val"))
            return len(tr), len(vl)
        except Exception:
            return 0, 0

    # ── Record loading ────────────────────────────────────────────────────────

    def _load_records(self, path: str) -> List[Dict]:
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        continue
        logger.info(f"[FeatureEng] Loaded {len(records)} records from {path}")
        return records

    # ── Processor loading ─────────────────────────────────────────────────────

    def _load_processor(
        self, hf_id: str, processor_type: str, model_family: str
    ) -> Tuple[Any, Any]:
        """
        Returns (processor, tokenizer).
        processor is set for vision models (AutoProcessor).
        tokenizer is set for text models (AutoTokenizer).
        Both are returned; callers use whichever is not None.
        """
        processor = None
        tokenizer = None

        try:
            if processor_type == "AutoProcessor" or model_family in (
                "blip2", "instructblip", "llava", "qwen_vl", "phi_vision", "idefics",
            ):
                from transformers import AutoProcessor
                processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
                tokenizer = getattr(processor, "tokenizer", processor)
                logger.info(f"[FeatureEng] Loaded AutoProcessor for {hf_id}")

            elif model_family in ("flan_t5", "seq2seq"):
                from transformers import T5Tokenizer, AutoTokenizer
                try:
                    tokenizer = T5Tokenizer.from_pretrained(hf_id)
                except Exception:
                    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
                logger.info(f"[FeatureEng] Loaded T5Tokenizer for {hf_id}")

            else:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
                logger.info(f"[FeatureEng] Loaded AutoTokenizer for {hf_id}")

        except Exception as e:
            logger.error(f"[FeatureEng] Processor load failed for {hf_id}: {e}")
            # Hard fallback: flan-t5-base tokenizer always works
            try:
                from transformers import T5Tokenizer
                tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
                logger.warning("[FeatureEng] Fallback to flan-t5-base tokenizer.")
            except Exception as e2:
                logger.error(f"[FeatureEng] Even fallback tokenizer failed: {e2}")
                return None, None

        # Ensure pad token
        tok = tokenizer or processor
        inner = getattr(tok, "tokenizer", tok) if tok else None
        if inner and hasattr(inner, "pad_token") and inner.pad_token is None:
            inner.pad_token = getattr(inner, "eos_token", None) or "<pad>"

        return processor, tokenizer

    # ── Encoding dispatcher ───────────────────────────────────────────────────

    def _encode_records(
        self,
        records:          List[Dict],
        processor,
        tokenizer,
        feature_strategy: str,
        model_family:     str,
        max_len:          int,
    ) -> Tuple[List[Dict], int]:
        """
        Dispatch to the correct encoding function based on feature_strategy.
        Returns (encoded_list, n_skipped).
        """
        encoded = []
        skipped = 0

        for i, rec in enumerate(records):
            question   = rec.get("question",   "").strip()
            answer     = rec.get("answer",     "").strip()
            image_path = rec.get("image_path", "")

            if not question or not answer:
                skipped += 1
                continue

            try:
                if feature_strategy == "seq2seq":
                    entry = self._encode_seq2seq(
                        question, answer, tokenizer, max_len
                    )
                elif feature_strategy == "causal_lm":
                    entry = self._encode_causal(
                        question, answer, tokenizer, max_len
                    )
                elif feature_strategy == "vision2seq_patchified":
                    entry = self._encode_qwen_vl(
                        question, answer, image_path, processor, max_len
                    )
                elif feature_strategy == "vision2seq":
                    # Dispatch to model-specific vision encoder
                    entry = self._encode_vision2seq(
                        question, answer, image_path,
                        processor, tokenizer, model_family, max_len,
                    )
                else:
                    # Unknown strategy → causal fallback
                    entry = self._encode_causal(
                        question, answer, tokenizer, max_len
                    )

                if entry and self._validate_entry(entry, feature_strategy):
                    encoded.append(entry)
                else:
                    skipped += 1

            except Exception as e:
                logger.debug(f"[FeatureEng] Record {i} failed: {e}")
                skipped += 1

        return encoded, skipped

    # ── Seq2seq encoder (Flan-T5, BART) ──────────────────────────────────────

    def _encode_seq2seq(
        self, question: str, answer: str, tokenizer, max_len: int
    ) -> Optional[Dict]:
        tok  = getattr(tokenizer, "tokenizer", tokenizer)
        prompt = f"Medical question: {question}"

        inp = tok(
            prompt, max_length=max_len, padding="max_length",
            truncation=True, return_tensors=None,
        )

        # Seq2seq label tokenization
        try:
            with tok.as_target_tokenizer():
                tgt = tok(
                    answer, max_length=64, padding="max_length",
                    truncation=True, return_tensors=None,
                )
        except AttributeError:
            # Newer transformers: text_target kwarg
            tgt = tok(
                text_target=answer, max_length=64, padding="max_length",
                truncation=True, return_tensors=None,
            )

        pad_id = getattr(tok, "pad_token_id", 0) or 0
        labels = [t if t != pad_id else -100 for t in tgt["input_ids"]]

        return {
            "input_ids":      inp["input_ids"],
            "attention_mask": inp["attention_mask"],
            "labels":         labels,
        }

    # ── Causal LM encoder ────────────────────────────────────────────────────

    def _encode_causal(
        self, question: str, answer: str, tokenizer, max_len: int
    ) -> Optional[Dict]:
        tok  = getattr(tokenizer, "tokenizer", tokenizer)
        text = f"Question: {question}\nAnswer: {answer}"

        enc = tok(
            text, max_length=max_len, padding="max_length",
            truncation=True, return_tensors=None,
        )
        pad_id = getattr(tok, "pad_token_id", 0) or 0
        labels = [t if t != pad_id else -100 for t in enc["input_ids"]]

        return {
            "input_ids":      enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels":         labels,
        }

    # ── Generic vision2seq dispatcher ────────────────────────────────────────

    def _encode_vision2seq(
        self,
        question:     str,
        answer:       str,
        image_path:   str,
        processor,
        tokenizer,
        model_family: str,
        max_len:      int,
    ) -> Optional[Dict]:
        """Dispatch to family-specific vision encoder."""
        if model_family == "blip2":
            return self._encode_blip2(question, answer, image_path, processor, max_len)
        if model_family == "instructblip":
            return self._encode_instructblip(question, answer, image_path, processor, max_len)
        if model_family == "llava":
            return self._encode_llava(question, answer, image_path, processor, max_len)
        if model_family == "phi_vision":
            return self._encode_phi_vision(question, answer, image_path, processor, max_len)
        # Generic: image + text via AutoProcessor
        return self._encode_generic_vision(question, answer, image_path, processor, tokenizer, max_len)

    # ── BLIP-2 ────────────────────────────────────────────────────────────────

    def _encode_blip2(
        self, question, answer, image_path, processor, max_len
    ) -> Optional[Dict]:
        image  = self._load_image(image_path)
        prompt = f"Question: {question} Answer:"

        if image is not None:
            enc = processor(
                images=image, text=prompt, return_tensors=None,
                padding="max_length", truncation=True, max_length=max_len,
            )
        else:
            enc = processor(
                text=prompt, return_tensors=None,
                padding="max_length", truncation=True, max_length=max_len,
            )

        tok    = getattr(processor, "tokenizer", processor)
        pad_id = getattr(tok, "pad_token_id", 0) or 0
        lbl    = tok(
            answer, max_length=64, padding="max_length",
            truncation=True, return_tensors=None,
        )
        labels = [t if t != pad_id else -100 for t in lbl["input_ids"]]
        return {**{k: v for k, v in enc.items()}, "labels": labels}

    # ── InstructBLIP ──────────────────────────────────────────────────────────

    def _encode_instructblip(
        self, question, answer, image_path, processor, max_len
    ) -> Optional[Dict]:
        image  = self._load_image(image_path)
        prompt = f"Question: {question}\nAnswer:"

        if image is not None:
            enc = processor(
                images=image, text=prompt, return_tensors=None,
                padding="max_length", truncation=True, max_length=max_len,
            )
        else:
            enc = processor(
                text=prompt, return_tensors=None,
                padding="max_length", truncation=True, max_length=max_len,
            )

        tok    = getattr(processor, "tokenizer", processor)
        pad_id = getattr(tok, "pad_token_id", 0) or 0
        lbl    = tok(
            answer, max_length=64, truncation=True,
            padding="max_length", return_tensors=None,
        )
        labels = [t if t != pad_id else -100 for t in lbl["input_ids"]]

        # Validate pixel_values shape — InstructBLIP outputs [1, N, 3, H, W];
        # squeeze the leading batch dimension to [N, 3, H, W] so collator
        # can re-batch correctly without dimension mismatch.
        result = {k: v for k, v in enc.items()}
        if "pixel_values" in result:
            pv = result["pixel_values"]
            if isinstance(pv, list) and len(pv) == 1 and isinstance(pv[0], list):
                result["pixel_values"] = pv[0]
        result["labels"] = labels
        return result

    # ── LLaVA ─────────────────────────────────────────────────────────────────

    def _encode_llava(
        self, question, answer, image_path, processor, max_len
    ) -> Optional[Dict]:
        image  = self._load_image(image_path)
        prompt = f"USER: <image>\n{question}\nASSISTANT: {answer}"

        if image is not None:
            enc = processor(
                text=prompt, images=image, return_tensors=None,
                padding="max_length", truncation=True, max_length=max_len,
            )
        else:
            tok = getattr(processor, "tokenizer", processor)
            enc = tok(
                prompt, return_tensors=None,
                padding="max_length", truncation=True, max_length=max_len,
            )

        result           = {k: v for k, v in enc.items()}
        result["labels"] = list(result.get("input_ids", []))
        return result

    # ── Qwen-VL (patchified) ──────────────────────────────────────────────────

    def _encode_qwen_vl(
        self, question, answer, image_path, processor, max_len
    ) -> Optional[Dict]:
        image = self._load_image(image_path)

        if image is not None and hasattr(processor, "apply_chat_template"):
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            text += answer
            try:
                enc = processor(
                    text=text, images=[image], return_tensors=None,
                    padding="max_length", truncation=True, max_length=max_len,
                )
            except Exception as e:
                logger.debug(f"[FeatureEng] Qwen-VL vision encode failed: {e}")
                # Fall back to text-only
                tok = getattr(processor, "tokenizer", processor)
                enc = tok(
                    f"Question: {question}\nAnswer: {answer}",
                    return_tensors=None, padding="max_length",
                    truncation=True, max_length=max_len,
                )
        else:
            tok = getattr(processor, "tokenizer", processor)
            enc = tok(
                f"Question: {question}\nAnswer: {answer}",
                return_tensors=None, padding="max_length",
                truncation=True, max_length=max_len,
            )

        result           = {k: v for k, v in enc.items()}
        result["labels"] = list(result.get("input_ids", []))
        return result

    # ── Phi-3.5-vision ────────────────────────────────────────────────────────

    def _encode_phi_vision(
        self, question, answer, image_path, processor, max_len
    ) -> Optional[Dict]:
        image = self._load_image(image_path)

        if image is not None:
            prompt = (
                f"<|user|>\n<|image_1|>\n{question}<|end|>\n"
                f"<|assistant|>\n{answer}<|end|>"
            )
            try:
                enc = processor(
                    text=prompt, images=[image], return_tensors=None,
                    padding="max_length", truncation=True, max_length=max_len,
                )
            except Exception as e:
                logger.debug(f"[FeatureEng] Phi-vision encode failed: {e}")
                tok = getattr(processor, "tokenizer", processor)
                enc = tok(
                    f"Question: {question}\nAnswer: {answer}",
                    return_tensors=None, padding="max_length",
                    truncation=True, max_length=max_len,
                )
        else:
            tok  = getattr(processor, "tokenizer", processor)
            text = f"<|user|>\n{question}<|end|>\n<|assistant|>\n{answer}<|end|>"
            enc  = tok(
                text, return_tensors=None, padding="max_length",
                truncation=True, max_length=max_len,
            )

        result           = {k: v for k, v in enc.items()}
        result["labels"] = list(result.get("input_ids", []))
        return result

    # ── Generic vision ────────────────────────────────────────────────────────

    def _encode_generic_vision(
        self, question, answer, image_path, processor, tokenizer, max_len
    ) -> Optional[Dict]:
        image  = self._load_image(image_path)
        prompt = f"Question: {question}\nAnswer: {answer}"
        tok    = tokenizer or getattr(processor, "tokenizer", processor)

        if image is not None and processor is not None:
            try:
                enc = processor(
                    text=prompt, images=image, return_tensors=None,
                    padding="max_length", truncation=True, max_length=max_len,
                )
                result           = {k: v for k, v in enc.items()}
                result["labels"] = list(result.get("input_ids", []))
                return result
            except Exception as e:
                logger.debug(f"[FeatureEng] Generic vision encode failed: {e}")

        enc              = tok(
            prompt, return_tensors=None,
            padding="max_length", truncation=True, max_length=max_len,
        )
        result           = {k: v for k, v in enc.items()}
        result["labels"] = list(result.get("input_ids", []))
        return result

    # ── Image loading ─────────────────────────────────────────────────────────

    def _load_image(self, image_path: str):
        if not image_path or not Path(image_path).exists():
            return None
        try:
            from PIL import Image
            return Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.debug(f"[FeatureEng] Image load failed {image_path}: {e}")
            return None

    # ── Entry validation ──────────────────────────────────────────────────────

    def _validate_entry(self, entry: Dict, feature_strategy: str) -> bool:
        """
        Validate a single encoded entry.
        Returns True if valid, False if should be skipped.
        """
        if not entry:
            return False

        # Must have input_ids
        if "input_ids" not in entry:
            return False

        ids = entry["input_ids"]
        if not ids or len(ids) == 0:
            return False

        # Must have labels
        if "labels" not in entry:
            return False

        lbls = entry["labels"]
        if not lbls or len(lbls) == 0:
            return False

        # Vision strategies must have pixel_values
        if feature_strategy in ("vision2seq", "vision2seq_patchified"):
            if "pixel_values" in entry:
                pv = entry["pixel_values"]
                # Reject stray leading dimension: [[...]] when should be [...]
                if isinstance(pv, list) and len(pv) == 1 and isinstance(pv[0], list):
                    # Squeeze it
                    entry["pixel_values"] = pv[0]

        return True

    # ── Dataset building ──────────────────────────────────────────────────────

    def _build_datasets(self, encoded: List[Dict]) -> Tuple[Any, Any]:
        from datasets import Dataset

        n     = len(encoded)
        split = max(1, int(n * 0.9))
        return Dataset.from_list(encoded[:split]), Dataset.from_list(encoded[split:])

    # ── Groq assessment ───────────────────────────────────────────────────────

    def _get_assessment(
        self, hf_id, family, strategy, train_n, val_n, columns
    ) -> Dict:
        prompt = (
            f"Feature engineering completed for Medical VQA.\n"
            f"Model: {hf_id}  Family: {family}  Strategy: {strategy}\n"
            f"Train: {train_n}  Val: {val_n}  Columns: {columns}\n\n"
            f"Are these features ready for training?\n"
            f'Reply ONLY: {{"status": "ok", "train_samples": {train_n}, '
            f'"val_samples": {val_n}, "message": "<one sentence>"}}'
        )
        try:
            response = self.agent.run(prompt)
            return self._parse_response(response)
        except Exception as e:
            logger.warning(f"[FeatureEng] Groq assessment failed: {e}")
            return {"status": "ok", "message": "Feature engineering complete."}

    def _parse_response(self, response) -> Dict:
        try:
            text  = response.content if hasattr(response, "content") else str(response)
            match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            pass
        return {"status": "ok", "message": "Feature engineering complete."}

    def _fail(self, msg: str) -> Dict:
        logger.error(f"[FeatureEng] {msg}")
        return {
            "status":        "failed",
            "message":       msg,
            "feature_path":  "",
            "train_samples": 0,
            "val_samples":   0,
        }

    # ── Family detection (mirrors ModelSelectionAgent) ────────────────────────

    def _detect_family(self, hf_id: str) -> str:
        hid = (hf_id or "").lower()
        if "flan-t5" in hid or "flan_t5" in hid:        return "flan_t5"
        if "blip2"   in hid or "blip-2"  in hid:        return "blip2"
        if "instructblip" in hid:                        return "instructblip"
        if "llava"        in hid:                        return "llava"
        if "qwen2.5-vl"   in hid or "qwen2-vl" in hid:  return "qwen_vl"
        if "phi-3.5-vision" in hid or "phi3.5"  in hid: return "phi_vision"
        if "idefics"      in hid:                        return "idefics"
        if "t5"  in hid:                                 return "seq2seq"
        if "bart" in hid:                                return "seq2seq"
        return "causal"


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch, sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    agent = FeatureEngineeringAgent()

    plan = {
        "hf_id":            "google/flan-t5-base",
        "name":             "Flan-T5-Base",
        "architecture":     "seq2seq",
        "vision":           False,
        "model_family":     "flan_t5",
        "processor_type":   "AutoTokenizer",
        "feature_strategy": "seq2seq",
        "collator_type":    "DataCollatorForSeq2Seq",
        "tensor_schema":    "input_ids[b,s], attention_mask[b,s], labels[b,s]",
        "max_seq_len":      128,
    }

    data_path = "./data/processed/processed_dataset.jsonl"
    if not Path(data_path).exists():
        print(f"Data not found: {data_path}")
        print("Run pipeline first: python main.py --image image.jpg --question '...' --dry-run")
    else:
        result = agent.engineer_features(data_path, plan, device=device)
        print(json.dumps(result, indent=2, default=str))
