"""
agents/feature_engineering_agent.py

FeatureEngineeringAgent — converts the cleaned processed dataset into
model-specific tensor format ready for training.

CACHE / METADATA UPDATE
------------------------
Every feature directory now carries a metadata.json:
    {
      "model_hf_id": "...", "architecture": "...", "processor_type": "...",
      "feature_strategy": "...", "tensor_schema": "...", "image_enabled": bool
    }
On each call, if a cache for the target hf_id already exists, its
metadata.json is compared against the current model_plan. On any mismatch
(or missing/corrupt metadata) the cache is discarded and regenerated —
this is what lets TrainingAgent's retry loop safely call this agent again
after ModelSelectionAgent picks a different model.

Encoded records are now validated (_validate_entry) before being added to
the dataset: invalid/missing tensor fields or wrong-rank pixel_values are
rejected rather than silently propagated to TrainingAgent.
"""

from phi.agent import Agent
from phi.model.groq import Groq

from typing import Any, Dict, List, Optional, Tuple
import json, logging, re, shutil
from pathlib import Path

logger = logging.getLogger(__name__)

FEATURE_DIR = Path("./data/features")

VISION_FAMILIES = {"blip2", "instructblip", "llava", "qwen_vl", "phi_vision", "idefics"}


class FeatureEngineeringAgent:
    """
    Converts processed JSONL records into model-specific HuggingFace
    Dataset format (input_ids, pixel_values, labels, etc.).

    Adaptive: detects model family from hf_id at runtime, loads the
    correct processor/tokenizer, encodes accordingly, validates the
    result, and caches it with a metadata.json describing the exact
    model/processor/strategy combination used — so stale caches from a
    previously-failed model are never silently reused.
    """

    def __init__(self, model_id: str = "mistral"):
        self.agent = Agent(
            name="FeatureEngineeringAgent",
            model=Groq(id="llama-3.1-8b-instant"),
            instructions=[
                "You are a machine learning feature engineering expert.",
                "You receive dataset statistics and encoding results.",
                "Assess whether the encoded features are ready for model training.",
                "Always reply with ONLY a JSON object like this:",
                '{"status": "ok", "train_samples": <int>, "val_samples": <int>, "message": "<one sentence>"}',
                "Do not write code. Do not add any text outside the JSON.",
            ],
            show_tool_calls=False,
            markdown=False,
        )

    # ── Public method ─────────────────────────────────────────────────────────

    def engineer_features(
        self,
        processed_data_path: str,
        model_plan: Dict[str, Any],
        device: str = "",
        force_regenerate: bool = False,
    ) -> Dict[str, Any]:
        """
        Convert processed JSONL into model-specific encoded HuggingFace Dataset.

        Args:
            processed_data_path : Path to processed JSONL from DataPreprocessingAgent.
            model_plan          : Dict from ModelSelectionAgent with hf_id,
                                   architecture, vision, processor_type,
                                   feature_strategy (if present, used to
                                   build/validate the cache; if absent,
                                   derived locally — backward compatible
                                   with older model_plans).
            device              : 'cuda' | 'cpu' | '' (auto-detect).
            force_regenerate    : Skip cache check entirely (used by
                                   TrainingAgent's retry loop after a
                                   training failure on this exact model
                                   to rule out a corrupt cache).

        Returns:
            Dict with feature_path, train_samples, val_samples, status.
        """
        if not device:
            device = self._detect_device()

        if not Path(processed_data_path).exists():
            return self._fail(f"Processed data not found: {processed_data_path}")

        hf_id        = model_plan.get("hf_id", "google/flan-t5-base")
        architecture = model_plan.get("architecture", "seq2seq")
        is_vision    = model_plan.get("vision", False)
        max_len      = model_plan.get("max_seq_len", 128)
        model_family = model_plan.get("model_family") or self._detect_model_family(hf_id)

        logger.info(f"[FeatureEng] Model: {hf_id}  Family: {model_family}")

        target_meta = {
            "model_hf_id":       hf_id,
            "architecture":      architecture,
            "processor_type":    model_plan.get("processor_type", self._default_processor_type(model_family)),
            "feature_strategy":  model_plan.get("feature_strategy", self._default_feature_strategy(model_family)),
            "tensor_schema":     model_plan.get("tensor_schema", ""),
            "image_enabled":     bool(is_vision),
        }

        feature_path = self._feature_path_for(hf_id, model_family)

        if not force_regenerate and self._cache_is_valid(feature_path, target_meta):
            logger.info(f"[FeatureEng] Reusing valid cache at {feature_path}")
            try:
                from datasets import load_from_disk
                train_ds = load_from_disk(str(Path(feature_path) / "train"))
                val_ds   = load_from_disk(str(Path(feature_path) / "val"))
                assessment = self._get_llm_assessment(
                    hf_id, model_family, is_vision, len(train_ds), len(val_ds),
                    list(train_ds.column_names),
                )
                assessment["feature_path"]  = feature_path
                assessment["train_samples"] = len(train_ds)
                assessment["val_samples"]   = len(val_ds)
                return assessment
            except Exception as e:
                logger.warning(f"[FeatureEng] Cache load failed ({e}) — regenerating.")

        if Path(feature_path).exists():
            logger.info(f"[FeatureEng] Discarding stale cache: {feature_path}")
            shutil.rmtree(feature_path, ignore_errors=True)

        records = self._load_records(processed_data_path)
        if not records:
            return self._fail("No records found in processed data.")

        processor, tokenizer = self._load_processor(hf_id, model_family)
        if processor is None and tokenizer is None:
            return self._fail(f"Could not load processor for {hf_id}")

        encoded = self._encode_records(
            records, processor, tokenizer,
            model_family, is_vision, architecture, max_len,
        )
        if not encoded:
            diag = getattr(self, "_last_encode_diagnostics", {})
            detail = ""
            if diag.get("no_image_count"):
                detail = (
                    f" {diag['no_image_count']}/{diag.get('total_records', 0)} records had "
                    f"no usable image — check DataPreprocessingAgent's output schema "
                    f"('image_path' or 'image' field)."
                )
            elif diag.get("error_samples"):
                detail = f" Sample errors: {diag['error_samples']}"
            return self._fail(f"Encoding produced no valid records.{detail}")

        train_ds, val_ds = self._build_hf_datasets(encoded)

        saved_path = self._save_to_disk(train_ds, val_ds, feature_path, target_meta)
        logger.info(
            f"[FeatureEng] Done. Train={len(train_ds)}  Val={len(val_ds)}  Path={saved_path}"
        )

        assessment = self._get_llm_assessment(
            hf_id, model_family, is_vision,
            len(train_ds), len(val_ds), list(train_ds.column_names),
        )
        assessment["feature_path"]  = saved_path
        assessment["train_samples"] = len(train_ds)
        assessment["val_samples"]   = len(val_ds)
        return assessment

    # ── Failure helper ────────────────────────────────────────────────────────

    def _fail(self, message: str) -> Dict[str, Any]:
        logger.error(f"[FeatureEng] FAILED — {message}")
        return {
            "status": "failed", "message": message, "feature_path": "",
            "train_samples": 0, "val_samples": 0,
        }

    # ── Device detection ──────────────────────────────────────────────────────

    def _detect_device(self) -> str:
        try:
            import torch
            if torch.cuda.is_available():          return "cuda"
            if torch.backends.mps.is_available():  return "mps"
        except ImportError:
            pass
        return "cpu"

    # ── Load records ──────────────────────────────────────────────────────────

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
        logger.info(f"[FeatureEng] Loaded {len(records)} records.")
        return records

    # ── Model family detection (shared vocabulary with ModelSelectionAgent) ───

    def _detect_model_family(self, hf_id: str) -> str:
        hid = (hf_id or "").lower()
        if "flan-t5" in hid or "flan_t5" in hid:         return "flan_t5"
        if "blip2" in hid or "blip-2" in hid:             return "blip2"
        if "instructblip" in hid:                          return "instructblip"
        if "llava" in hid:                                 return "llava"
        if "qwen2.5-vl" in hid or "qwen2-vl" in hid:     return "qwen_vl"
        if "phi-3.5-vision" in hid or "phi3.5" in hid:    return "phi_vision"
        if "idefics" in hid:                               return "idefics"
        if "t5" in hid:                                    return "seq2seq"
        if "bart" in hid:                                  return "seq2seq"
        return "causal"

    def _default_processor_type(self, family: str) -> str:
        return "AutoProcessor" if family in VISION_FAMILIES else "AutoTokenizer"

    def _default_feature_strategy(self, family: str) -> str:
        if family == "qwen_vl":
            return "vision2seq_patchified"
        if family in VISION_FAMILIES:
            return "vision2seq"
        if family in ("flan_t5", "seq2seq"):
            return "seq2seq"
        return "causal_lm"

    # ── Cache validation ─────────────────────────────────────────────────────

    def _feature_path_for(self, hf_id: str, model_family: str) -> str:
        safe_name = (hf_id + "_" + model_family).replace("/", "_").replace(" ", "_")
        return str(FEATURE_DIR / safe_name)

    def _cache_is_valid(self, feature_path: str, target_meta: Dict[str, Any]) -> bool:
        meta_file = Path(feature_path) / "metadata.json"
        train_dir = Path(feature_path) / "train"
        val_dir   = Path(feature_path) / "val"
        if not (meta_file.exists() and train_dir.exists() and val_dir.exists()):
            return False
        try:
            existing = json.loads(meta_file.read_text())
        except Exception as e:
            logger.warning(f"[FeatureEng] metadata.json unreadable ({e}) — invalid cache.")
            return False

        compare_keys = ["model_hf_id", "architecture", "processor_type",
                         "feature_strategy", "image_enabled"]
        for k in compare_keys:
            if existing.get(k) != target_meta.get(k):
                logger.info(
                    f"[FeatureEng] Cache mismatch on '{k}': "
                    f"cached={existing.get(k)!r} vs target={target_meta.get(k)!r}"
                )
                return False
        return True

    # ── Processor loading ─────────────────────────────────────────────────────

    def _load_processor(self, hf_id: str, model_family: str):
        processor = None
        tokenizer = None
        try:
            if model_family in VISION_FAMILIES:
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
            logger.warning(f"[FeatureEng] Processor load failed for {hf_id}: {e}")
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
                logger.info("[FeatureEng] Fallback to flan-t5-base tokenizer.")
            except Exception as e2:
                logger.error(f"[FeatureEng] All tokenizer loads failed: {e2}")
                return None, None

        tok = tokenizer if tokenizer else processor
        inner = getattr(tok, "tokenizer", tok)
        if hasattr(inner, "pad_token") and inner.pad_token is None:
            inner.pad_token = inner.eos_token

        return processor, tokenizer

    # ── Encoding ──────────────────────────────────────────────────────────────

    def _encode_records(
        self, records, processor, tokenizer, model_family, is_vision, architecture, max_len,
    ) -> List[Dict]:
        encoded = []
        skipped = 0
        no_image_count = 0
        error_samples: List[str] = []   # first few distinct exception messages, surfaced to the caller
        seen_errors: set = set()

        for i, rec in enumerate(records):
            question = rec.get("question", "").strip()
            answer   = rec.get("answer",   "").strip()
            image    = self._resolve_record_image(rec)

            if not question or not answer:
                skipped += 1
                continue

            if is_vision and image is None:
                # A vision model with no usable image for this record is not
                # a recoverable case (most VLM processors require `images=`
                # and will raise if called without one) — skip explicitly
                # instead of falling through to a text-only encode path that
                # silently crashes for every record.
                no_image_count += 1
                skipped += 1
                continue

            try:
                entry = self._encode_single(
                    question, answer, image, processor, tokenizer,
                    model_family, architecture, max_len,
                )
                reject_reason = self._validate_entry(entry, model_family) if entry else "encode_single returned None/empty"
                if entry and not reject_reason:
                    encoded.append(entry)
                else:
                    if reject_reason and reject_reason not in seen_errors and len(error_samples) < 5:
                        seen_errors.add(reject_reason)
                        error_samples.append(f"validation_rejected: {reject_reason}")
                    skipped += 1
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                if msg not in seen_errors and len(error_samples) < 5:
                    seen_errors.add(msg)
                    error_samples.append(msg)
                # Surfaced at WARNING (not DEBUG) so it is visible at default
                # INFO log level instead of being silently swallowed.
                logger.warning(f"[FeatureEng] Record {i} encoding failed: {msg}")
                skipped += 1
                continue

        logger.info(f"[FeatureEng] Encoded {len(encoded)} records. Skipped {skipped}.")
        if no_image_count:
            sample_keys = list(records[0].keys()) if records else []
            sample_image_val = records[0].get("image", records[0].get("image_path", "<absent>")) if records else None
            sample_image_type = type(sample_image_val).__name__
            logger.warning(
                f"[FeatureEng] {no_image_count}/{len(records)} records had no "
                f"usable image for a vision model (model_family={model_family}). "
                f"First record keys={sample_keys}  "
                f"'image'/'image_path' field type={sample_image_type}  "
                f"value_preview={str(sample_image_val)[:120]!r}. "
                f"Check DataPreprocessingAgent's JSONL output schema — "
                f"_resolve_record_image() expects 'image_path' (filesystem "
                f"string), or 'image' as a PIL object, raw bytes, "
                f"{{'bytes':..,'path':..}} dict, or numpy array. If the JSONL "
                f"only stores a dataset index / HF dataset row id, the image "
                f"is being dropped during JSON serialization upstream and "
                f"must be re-exported to disk as actual image files."
            )
        if error_samples:
            logger.warning(f"[FeatureEng] Sample encoding/validation errors: {error_samples}")
        self._last_encode_diagnostics = {
            "no_image_count": no_image_count,
            "error_samples": error_samples,
            "total_records": len(records),
        }
        return encoded

    def _validate_entry(self, entry: Dict, model_family: str) -> str:
        """Returns '' if entry is valid, otherwise a short reason string
        explaining why it was rejected (used both to skip bad records and
        to surface a diagnosable cause in engineer_features()'s failure
        message instead of an opaque 'skipped' count)."""
        if "input_ids" not in entry:
            return "missing input_ids"
        if "labels" not in entry:
            return "missing labels"
        if not isinstance(entry["input_ids"], list) or len(entry["input_ids"]) == 0:
            return f"input_ids not a non-empty list (got {type(entry['input_ids']).__name__})"
        if model_family in VISION_FAMILIES:
            if "pixel_values" not in entry:
                return "vision model but 'pixel_values' missing from encoded entry (processor call likely returned no image tensor)"
            pv = entry["pixel_values"]
            try:
                import torch
                rank = torch.as_tensor(pv).dim()
            except Exception:
                rank = len(_shape_of(pv))
            if model_family == "qwen_vl":
                if rank not in (2, 3):
                    return f"qwen_vl pixel_values rank={rank} (expected 2 or 3)"
            else:
                if rank not in (3, 4):
                    return f"{model_family} pixel_values rank={rank} (expected 3 or 4)"
        return ""

    def _encode_single(
        self, question, answer, image, processor, tokenizer,
        model_family, architecture, max_len,
    ) -> Optional[Dict]:
        if model_family in ("flan_t5", "seq2seq"):
            entry = self._encode_seq2seq(question, answer, tokenizer, max_len)
        elif model_family == "blip2":
            entry = self._encode_blip2(question, answer, image, processor, max_len)
        elif model_family == "instructblip":
            entry = self._encode_instructblip(question, answer, image, processor, max_len)
        elif model_family == "llava":
            entry = self._encode_llava(question, answer, image, processor, max_len)
        elif model_family == "qwen_vl":
            entry = self._encode_qwen_vl(question, answer, image, processor, max_len)
        elif model_family == "phi_vision":
            entry = self._encode_phi_vision(question, answer, image, processor, max_len)
        else:
            entry = self._encode_causal(question, answer, tokenizer, max_len)

        return self._flatten_sequence_fields(entry) if entry else entry

    def _flatten_sequence_fields(self, entry: Dict) -> Dict:
        """
        Normalize 'input_ids' / 'attention_mask' / 'labels' to a flat
        per-example list[int] (rank 1), regardless of how the underlying
        HF processor shaped them.

        ROOT CAUSE this fixes: several processors (notably Qwen2.5-VL's
        AutoProcessor, even when called with return_tensors=None) still
        return these fields as a batch-of-one nested list — [[id, id, ...]]
        instead of [id, id, ...] — because the processor always treats its
        input as a batch internally. Saving that nested shape directly into
        the HF Dataset means torch.as_tensor() on a single example yields
        rank 2 instead of rank 1, which is exactly the
        "input_ids rank 2 (expected 1)" schema-validation failure
        TrainingAgent reported. Squeezing it here, once, at the only place
        all encoders funnel through, keeps every downstream consumer
        (TrainingAgent's validator, the Qwen-VL collator, default_data_collator)
        working with a single consistent per-example rank-1 contract.
        """
        for key in ("input_ids", "attention_mask", "labels"):
            val = entry.get(key)
            if (
                isinstance(val, list) and len(val) == 1
                and isinstance(val[0], list)
            ):
                entry[key] = val[0]
        return entry

    # ── Model-specific encoders (unchanged behaviour) ─────────────────────────

    def _encode_seq2seq(self, question, answer, tokenizer, max_len) -> Dict:
        prompt = f"Medical question: {question}"
        inp = tokenizer(prompt, max_length=max_len, padding="max_length",
                         truncation=True, return_tensors=None)
        with tokenizer.as_target_tokenizer():
            tgt = tokenizer(answer, max_length=64, padding="max_length",
                             truncation=True, return_tensors=None)
        labels = [t if t != tokenizer.pad_token_id else -100 for t in tgt["input_ids"]]
        return {"input_ids": inp["input_ids"], "attention_mask": inp["attention_mask"], "labels": labels}

    def _encode_blip2(self, question, answer, image, processor, max_len) -> Dict:
        prompt = f"Question: {question} Answer:"
        if image is not None:
            enc = processor(images=image, text=prompt, return_tensors=None,
                             padding="max_length", truncation=True, max_length=max_len)
        else:
            enc = processor(text=prompt, return_tensors=None,
                             padding="max_length", truncation=True, max_length=max_len)
        tok = getattr(processor, "tokenizer", processor)
        lbl = tok(answer, max_length=64, padding="max_length", truncation=True, return_tensors=None)
        labels = [t if t != tok.pad_token_id else -100 for t in lbl["input_ids"]]
        result = {k: v for k, v in enc.items()}
        result["labels"] = labels
        return result

    def _encode_instructblip(self, question, answer, image, processor, max_len) -> Dict:
        prompt = f"Question: {question}\nAnswer:"
        if image is not None:
            enc = processor(images=image, text=prompt, return_tensors=None,
                             padding="max_length", truncation=True, max_length=max_len)
        else:
            enc = processor(text=prompt, return_tensors=None,
                             padding="max_length", truncation=True, max_length=max_len)
        tok    = getattr(processor, "tokenizer", processor)
        lbl    = tok(answer, max_length=64, truncation=True, padding="max_length", return_tensors=None)
        labels = [t if t != tok.pad_token_id else -100 for t in lbl["input_ids"]]
        result = {k: v for k, v in enc.items()}
        result["labels"] = labels
        return result

    def _encode_llava(self, question, answer, image, processor, max_len) -> Dict:
        prompt = f"USER: <image>\n{question}\nASSISTANT: {answer}"
        if image is not None:
            enc = processor(text=prompt, images=image, return_tensors=None,
                             padding="max_length", truncation=True, max_length=max_len)
        else:
            tok = getattr(processor, "tokenizer", processor)
            enc = tok(prompt, return_tensors=None, padding="max_length",
                      truncation=True, max_length=max_len)
        result = {k: v for k, v in enc.items()}
        result["labels"] = result.get("input_ids", [])[:]
        return result

    def _encode_qwen_vl(self, question, answer, image, processor, max_len) -> Dict:
        if image is not None and hasattr(processor, "apply_chat_template"):
            messages = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": question},
            ]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            text += answer
            enc = processor(text=text, images=[image], return_tensors=None,
                             padding="max_length", truncation=True, max_length=max_len)
        else:
            tok  = getattr(processor, "tokenizer", processor)
            text = f"Question: {question}\nAnswer: {answer}"
            enc  = tok(text, return_tensors=None, padding="max_length",
                       truncation=True, max_length=max_len)
        result = {k: v for k, v in enc.items()}
        result["labels"] = result.get("input_ids", [])[:]
        if "image_grid_thw" in result:
            g = result["image_grid_thw"]
            # Some processor versions wrap this as [[t,h,w]] (batch-of-one)
            # instead of [t,h,w] per image — normalize to the flat form so
            # the qwen_vl_collator's reshape(-1, 3) in TrainingAgent always
            # receives a consistent shape regardless of processor version.
            if isinstance(g, list) and len(g) == 1 and isinstance(g[0], list) and isinstance(g[0][0], list):
                result["image_grid_thw"] = g[0]
        return result

    def _encode_phi_vision(self, question, answer, image, processor, max_len) -> Dict:
        if image is not None:
            prompt = f"<|user|>\n<|image_1|>\n{question}<|end|>\n<|assistant|>\n{answer}<|end|>"
            enc = processor(text=prompt, images=[image], return_tensors=None,
                             padding="max_length", truncation=True, max_length=max_len)
        else:
            tok  = getattr(processor, "tokenizer", processor)
            text = f"<|user|>\n{question}<|end|>\n<|assistant|>\n{answer}<|end|>"
            enc  = tok(text, return_tensors=None, padding="max_length",
                       truncation=True, max_length=max_len)
        result = {k: v for k, v in enc.items()}
        result["labels"] = result.get("input_ids", [])[:]
        return result

    def _encode_causal(self, question, answer, tokenizer, max_len) -> Dict:
        text = f"Question: {question}\nAnswer: {answer}"
        tok  = getattr(tokenizer, "tokenizer", tokenizer)
        enc  = tok(text, max_length=max_len, padding="max_length",
                   truncation=True, return_tensors=None)
        labels = [t if t != (tok.pad_token_id or 0) else -100 for t in enc["input_ids"]]
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels}

    # ── Image loading helper ──────────────────────────────────────────────────

    def _load_image(self, image_path: str):
        if not image_path or not Path(image_path).exists():
            return None
        try:
            from PIL import Image
            return Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.debug(f"[FeatureEng] Image load failed {image_path}: {e}")
            return None

    def _resolve_record_image(self, rec: Dict[str, Any]):
        """
        Resolve a usable PIL.Image from a processed record, regardless of
        how DataPreprocessingAgent represented it. This is the fix for the
        "0 encoded / N skipped" failure: HF image datasets (e.g.
        flaviagiammarino/vqa-rad) commonly carry the image as an embedded
        PIL object, raw bytes, or a {"bytes":..., "path":...} dict rather
        than a filesystem path string in 'image_path' — the old code only
        ever checked 'image_path', so every record silently fell through
        to a no-image encode path that crashes for processors (like
        InstructBLIP's) that require `images=` to be set.

        Tries, in order:
          1. rec['image_path']      — filesystem path (original behaviour)
          2. rec['image']           — PIL.Image, raw bytes, numpy array, or
                                       a HF datasets-style {"bytes","path"} dict
        Returns a PIL.Image in RGB mode, or None if nothing usable is found.
        """
        from PIL import Image
        import io

        image_path = (
            rec.get("image_path") or rec.get("img_path")
            or rec.get("image_file") or rec.get("file_path") or ""
        )
        if image_path:
            img = self._load_image(image_path)
            if img is not None:
                return img

        raw = rec.get("image", rec.get("img", rec.get("pixel_data")))
        if raw is None:
            return None

        try:
            if isinstance(raw, Image.Image):
                return raw.convert("RGB")
            if isinstance(raw, dict):
                if raw.get("bytes"):
                    return Image.open(io.BytesIO(raw["bytes"])).convert("RGB")
                if raw.get("path"):
                    return self._load_image(raw["path"])
                return None
            if isinstance(raw, (bytes, bytearray)):
                return Image.open(io.BytesIO(raw)).convert("RGB")
            if isinstance(raw, str):
                # Possible base64-encoded image string (with or without a
                # data: URI prefix) — some preprocessing pipelines serialize
                # images this way to keep everything in a single JSONL field.
                import base64
                s = raw.split(",", 1)[-1] if raw.startswith("data:") else raw
                try:
                    decoded = base64.b64decode(s, validate=True)
                    return Image.open(io.BytesIO(decoded)).convert("RGB")
                except Exception:
                    return self._load_image(raw)  # maybe it's actually a path
            # numpy array or similar array-like
            import numpy as np
            if isinstance(raw, np.ndarray):
                return Image.fromarray(raw).convert("RGB")
        except Exception as e:
            logger.debug(f"[FeatureEng] Could not resolve embedded image: {e}")
        return None

    # ── HuggingFace Dataset / cache persistence ────────────────────────────────

    def _build_hf_datasets(self, encoded: List[Dict]):
        from datasets import Dataset
        n     = len(encoded)
        split = max(1, int(n * 0.9))
        train_ds = Dataset.from_list(encoded[:split])
        val_ds   = Dataset.from_list(encoded[split:] or encoded[-1:])
        return train_ds, val_ds

    def _save_to_disk(self, train_ds, val_ds, feature_path: str, meta: Dict[str, Any]) -> str:
        base_path = Path(feature_path)
        base_path.mkdir(parents=True, exist_ok=True)
        train_ds.save_to_disk(str(base_path / "train"))
        val_ds.save_to_disk(str(base_path / "val"))
        (base_path / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
        logger.info(f"[FeatureEng] Saved to {base_path} (metadata.json written)")
        return str(base_path)

    # ── LLM assessment ────────────────────────────────────────────────────────

    def _get_llm_assessment(self, hf_id, model_family, is_vision, train_n, val_n, columns) -> Dict:
        prompt = (
            f"Feature engineering completed for Medical VQA.\n"
            f"Model: {hf_id}\nFamily: {model_family}  Vision: {is_vision}\n"
            f"Train samples: {train_n}  Val samples: {val_n}\n"
            f"Encoded columns: {columns}\n\n"
            f"Are these features ready for model training?\n"
            f'Reply with ONLY: {{"status": "ok", "train_samples": {train_n}, '
            f'"val_samples": {val_n}, "message": "<one sentence>"}}'
        )
        try:
            response = self.agent.run(prompt)
            return self._parse_response(response)
        except Exception as e:
            logger.warning(f"[FeatureEng] LLM assessment failed: {e}")
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


def _shape_of(x) -> List[int]:
    """Pure-python nested-list shape inference (fallback when torch unavailable)."""
    shape = []
    cur = x
    while isinstance(cur, list):
        shape.append(len(cur))
        cur = cur[0] if cur else None
    return shape


if __name__ == "__main__":
    import torch
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice : {device}")

    agent = FeatureEngineeringAgent()
    plan = {
        "hf_id": "google/flan-t5-base", "name": "Flan-T5-Base",
        "architecture": "seq2seq", "vision": False, "max_seq_len": 128,
    }
    data_path = "./data/processed/processed_dataset.jsonl"
    if not Path(data_path).exists():
        print(f"\nData file not found: {data_path}")
    else:
        result = agent.engineer_features(data_path, plan, device=device)
        print("\n" + "=" * 50)
        print("Feature Engineering Result:")
        print(json.dumps(result, indent=2, default=str))
        print("=" * 50)
