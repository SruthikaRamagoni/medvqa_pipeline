"""
agents/feature_engineering_agent.py

FeatureEngineeringAgent — converts the cleaned processed dataset into
model-specific tensor format ready for training.

Adaptive: detects the selected model's requirements at runtime and applies
the correct processor/tokenizer/image encoder automatically.
No hardcoding for any specific model.

Uses same PhiData + Groq structure as all other agents in the project.
"""

from phi.agent import Agent
from phi.model.groq import Groq

from typing import Any, Dict, List, Optional, Tuple
import json, logging, re
from pathlib import Path

logger = logging.getLogger(__name__)

FEATURE_DIR = Path("./data/features")


class FeatureEngineeringAgent:
    """
    Converts processed JSONL records into model-specific HuggingFace
    Dataset format (input_ids, pixel_values, labels, etc.).

    Adaptive behaviour:
    - Detects model family from hf_id at runtime
    - Loads the correct processor/tokenizer for that model
    - Encodes text and images according to model requirements
    - Saves encoded dataset to disk so TrainingAgent can load directly
    - Works for seq2seq, causal LM, and vision-language models
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
    ) -> Dict[str, Any]:
        """
        Convert processed JSONL into model-specific encoded HuggingFace Dataset.

        Args:
            processed_data_path : Path to processed JSONL from DataPreprocessingAgent.
            model_plan          : Dict from ModelSelectionAgent with hf_id, architecture, vision.
            device              : 'cuda' | 'cpu' | '' (auto-detect).

        Returns:
            Dict with feature_path, train_samples, val_samples, status.
        """
        if not device:
            device = self._detect_device()

        if not Path(processed_data_path).exists():
            return {
                "status":        "failed",
                "message":       f"Processed data not found: {processed_data_path}",
                "feature_path":  "",
                "train_samples": 0,
                "val_samples":   0,
            }

        hf_id        = model_plan.get("hf_id", "google/flan-t5-base")
        architecture = model_plan.get("architecture", "seq2seq")
        is_vision    = model_plan.get("vision", False)
        max_len      = model_plan.get("max_seq_len", 128)

        logger.info(f"[FeatureEng] Model: {hf_id}")
        logger.info(f"[FeatureEng] Architecture: {architecture}  Vision: {is_vision}")

        # Step 1: Load records
        records = self._load_records(processed_data_path)
        if not records:
            return {
                "status":        "failed",
                "message":       "No records found in processed data.",
                "feature_path":  "",
                "train_samples": 0,
                "val_samples":   0,
            }

        # Step 2: Detect model family and requirements
        model_family = self._detect_model_family(hf_id)
        logger.info(f"[FeatureEng] Detected model family: {model_family}")

        # Step 3: Load processor / tokenizer adaptively
        processor, tokenizer = self._load_processor(hf_id, model_family)
        if processor is None and tokenizer is None:
            return {
                "status":        "failed",
                "message":       f"Could not load processor for {hf_id}",
                "feature_path":  "",
                "train_samples": 0,
                "val_samples":   0,
            }

        # Step 4: Encode records adaptively based on model family
        encoded = self._encode_records(
            records, processor, tokenizer,
            model_family, is_vision, architecture, max_len,
        )

        if not encoded:
            return {
                "status":        "failed",
                "message":       "Encoding produced no valid records.",
                "feature_path":  "",
                "train_samples": 0,
                "val_samples":   0,
            }

        # Step 5: Build HuggingFace Dataset and split
        train_ds, val_ds = self._build_hf_datasets(encoded)

        # Step 6: Save to disk
        
        feature_path = self._save_to_disk(
            train_ds,
            val_ds,
            hf_id + "_" + model_family
        )
        logger.info(
            f"[FeatureEng] Done. Train={len(train_ds)}  Val={len(val_ds)}  "
            f"Path={feature_path}"
        )

        # Step 7: Ask Groq to assess
        assessment = self._get_llm_assessment(
            hf_id, model_family, is_vision,
            len(train_ds), len(val_ds),
            list(train_ds.column_names),
        )
        assessment["feature_path"]  = feature_path
        assessment["train_samples"] = len(train_ds)
        assessment["val_samples"]   = len(val_ds)
        return assessment

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

    # ── Model family detection ────────────────────────────────────────────────

    def _detect_model_family(self, hf_id: str) -> str:
        """
        Detect model family from hf_id string.
        Returns one of: flan_t5 | blip2 | llava | qwen_vl | phi_vision |
                        instructblip | idefics | causal | seq2seq
        """
        hid = hf_id.lower()

        if "flan-t5" in hid or "flan_t5" in hid:         return "flan_t5"
        if "blip2" in hid or "blip-2" in hid:             return "blip2"
        if "instructblip" in hid:                          return "instructblip"
        if "llava" in hid:                                 return "llava"
        if "qwen2.5-vl" in hid or "qwen2-vl" in hid:     return "qwen_vl"
        if "phi-3.5-vision" in hid or "phi3.5" in hid:    return "phi_vision"
        if "idefics" in hid:                               return "idefics"
        if "t5" in hid:                                    return "seq2seq"
        if "bart" in hid:                                  return "seq2seq"

        # Default by architecture hint
        return "causal"

    # ── Processor loading ─────────────────────────────────────────────────────

    def _load_processor(self, hf_id: str, model_family: str):
        """
        Load the correct processor/tokenizer for the model family.
        Returns (processor_or_None, tokenizer).
        processor is set for vision models; tokenizer is always set.
        """
        processor = None
        tokenizer = None

        try:
            if model_family in ("blip2", "instructblip", "llava",
                                "qwen_vl", "phi_vision", "idefics"):
                from transformers import AutoProcessor
                processor = AutoProcessor.from_pretrained(
                    hf_id, trust_remote_code=True
                )
                tokenizer = getattr(processor, "tokenizer", processor)
                logger.info(f"[FeatureEng] Loaded AutoProcessor for {hf_id}")

            elif model_family in ("flan_t5", "seq2seq"):
                from transformers import T5Tokenizer, AutoTokenizer
                try:
                    tokenizer = T5Tokenizer.from_pretrained(hf_id)
                except Exception:
                    tokenizer = AutoTokenizer.from_pretrained(
                        hf_id, trust_remote_code=True
                    )
                logger.info(f"[FeatureEng] Loaded T5Tokenizer for {hf_id}")

            else:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    hf_id, trust_remote_code=True
                )
                logger.info(f"[FeatureEng] Loaded AutoTokenizer for {hf_id}")

        except Exception as e:
            logger.warning(f"[FeatureEng] Processor load failed for {hf_id}: {e}")
            # Last resort fallback
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    "google/flan-t5-base"
                )
                logger.info("[FeatureEng] Fallback to flan-t5-base tokenizer.")
            except Exception as e2:
                logger.error(f"[FeatureEng] All tokenizer loads failed: {e2}")
                return None, None

        # Ensure pad token
        tok = tokenizer if tokenizer else processor
        inner = getattr(tok, "tokenizer", tok)
        if hasattr(inner, "pad_token") and inner.pad_token is None:
            inner.pad_token = inner.eos_token

        return processor, tokenizer

    # ── Encoding ──────────────────────────────────────────────────────────────

    def _encode_records(
        self,
        records: List[Dict],
        processor,
        tokenizer,
        model_family: str,
        is_vision: bool,
        architecture: str,
        max_len: int,
    ) -> List[Dict]:
        """
        Encode each record into tensors appropriate for the model family.
        Adaptive: different encoding path per model family.
        Returns list of dicts with tensor-compatible fields.
        """
        encoded = []
        skipped = 0

        for i, rec in enumerate(records):
            question   = rec.get("question", "").strip()
            answer     = rec.get("answer",   "").strip()
            image_path = rec.get("image_path", "")

            if not question or not answer:
                skipped += 1
                continue

            try:
                entry = self._encode_single(
                    question, answer, image_path,
                    processor, tokenizer,
                    model_family, architecture, max_len,
                )
                if entry:
                    encoded.append(entry)
            except Exception as e:
                logger.debug(f"[FeatureEng] Record {i} encoding failed: {e}")
                skipped += 1
                continue

        logger.info(
            f"[FeatureEng] Encoded {len(encoded)} records. "
            f"Skipped {skipped}."
        )
        return encoded

    def _encode_single(
        self,
        question: str,
        answer: str,
        image_path: str,
        processor,
        tokenizer,
        model_family: str,
        architecture: str,
        max_len: int,
    ) -> Optional[Dict]:
        """
        Encode one record. Returns dict with tensor-compatible fields.
        Dispatches to model-family-specific encoder.
        """

        # ── Seq2Seq models (Flan-T5, BART, etc.) ─────────────────────────────
        if model_family in ("flan_t5", "seq2seq"):
            return self._encode_seq2seq(
                question, answer, tokenizer, max_len
            )

        # ── BLIP-2 ────────────────────────────────────────────────────────────
        if model_family == "blip2":
            return self._encode_blip2(
                question, answer, image_path, processor, max_len
            )

        # ── InstructBLIP ──────────────────────────────────────────────────────
        if model_family == "instructblip":
            return self._encode_instructblip(
                question, answer, image_path, processor, max_len
            )

        # ── LLaVA ─────────────────────────────────────────────────────────────
        if model_family == "llava":
            return self._encode_llava(
                question, answer, image_path, processor, max_len
            )

        # ── Qwen-VL ───────────────────────────────────────────────────────────
        if model_family == "qwen_vl":
            return self._encode_qwen_vl(
                question, answer, image_path, processor, max_len
            )

        # ── Phi-vision ────────────────────────────────────────────────────────
        if model_family == "phi_vision":
            return self._encode_phi_vision(
                question, answer, image_path, processor, max_len
            )

        # ── Generic causal LM (fallback for unknown models) ───────────────────
        return self._encode_causal(
            question, answer, tokenizer, max_len
        )

    # ── Model-specific encoders ───────────────────────────────────────────────

    def _encode_seq2seq(self, question, answer, tokenizer, max_len) -> Dict:
        """Flan-T5, BART, and other seq2seq models."""
        prompt = f"Medical question: {question}"

        inp = tokenizer(
            prompt,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors=None,
        )
        with tokenizer.as_target_tokenizer():
            tgt = tokenizer(
                answer,
                max_length=64,
                padding="max_length",
                truncation=True,
                return_tensors=None,
            )

        labels = [
            t if t != tokenizer.pad_token_id else -100
            for t in tgt["input_ids"]
        ]

        return {
            "input_ids":      inp["input_ids"],
            "attention_mask": inp["attention_mask"],
            "labels":         labels,
        }

    def _encode_blip2(self, question, answer, image_path, processor, max_len) -> Dict:
        """Salesforce BLIP-2 models."""
        from PIL import Image

        prompt = f"Question: {question} Answer:"
        image  = self._load_image(image_path)

        if image is not None:
            enc = processor(
                images=image,
                text=prompt,
                return_tensors=None,
                padding="max_length",
                truncation=True,
                max_length=max_len,
            )
        else:
            enc = processor(
                text=prompt,
                return_tensors=None,
                padding="max_length",
                truncation=True,
                max_length=max_len,
            )

        tok = getattr(processor, "tokenizer", processor)
        lbl = tok(
            answer, max_length=64, padding="max_length",
            truncation=True, return_tensors=None,
        )
        labels = [
            t if t != tok.pad_token_id else -100
            for t in lbl["input_ids"]
        ]

        result = {k: v for k, v in enc.items()}
        result["labels"] = labels
        return result

    def _encode_instructblip(self, question, answer, image_path, processor, max_len) -> Dict:
        """Salesforce InstructBLIP models."""
        image = self._load_image(image_path)
        prompt = f"Question: {question}\nAnswer:"

        if image is not None:
            enc = processor(
                images=image,
                text=prompt,
                return_tensors=None,
                padding="max_length",
                truncation=True,
                max_length=max_len,
            )
        else:
            enc = processor(
                text=prompt,
                return_tensors=None,
                padding="max_length",
                truncation=True,
                max_length=max_len,
            )

        tok    = getattr(processor, "tokenizer", processor)
        lbl    = tok(answer, max_length=64, truncation=True,
                      padding="max_length", return_tensors=None)
        labels = [t if t != tok.pad_token_id else -100 for t in lbl["input_ids"]]

        result = {k: v for k, v in enc.items()}
        result["labels"] = labels
        return result

    def _encode_llava(self, question, answer, image_path, processor, max_len) -> Dict:
        """LLaVA models (llava-hf)."""
        image  = self._load_image(image_path)
        prompt = f"USER: <image>\n{question}\nASSISTANT: {answer}"

        if image is not None:
            enc = processor(
                text=prompt,
                images=image,
                return_tensors=None,
                padding="max_length",
                truncation=True,
                max_length=max_len,
            )
        else:
            tok = getattr(processor, "tokenizer", processor)
            enc = tok(prompt, return_tensors=None,
                       padding="max_length", truncation=True, max_length=max_len)

        result         = {k: v for k, v in enc.items()}
        result["labels"] = result.get("input_ids", [])[:]
        return result

    def _encode_qwen_vl(self, question, answer, image_path, processor, max_len) -> Dict:
        """Qwen2-VL and Qwen2.5-VL models."""
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
            enc = processor(
                text=text,
                images=[image],
                return_tensors=None,
                padding="max_length",
                truncation=True,
                max_length=max_len,
            )
        else:
            tok  = getattr(processor, "tokenizer", processor)
            text = f"Question: {question}\nAnswer: {answer}"
            enc  = tok(text, return_tensors=None,
                        padding="max_length", truncation=True, max_length=max_len)

        result           = {k: v for k, v in enc.items()}
        result["labels"] = result.get("input_ids", [])[:]
        return result

    def _encode_phi_vision(self, question, answer, image_path, processor, max_len) -> Dict:
        """Microsoft Phi-3.5-vision-instruct."""
        image = self._load_image(image_path)

        if image is not None:
            prompt = f"<|user|>\n<|image_1|>\n{question}<|end|>\n<|assistant|>\n{answer}<|end|>"
            enc = processor(
                text=prompt,
                images=[image],
                return_tensors=None,
                padding="max_length",
                truncation=True,
                max_length=max_len,
            )
        else:
            tok  = getattr(processor, "tokenizer", processor)
            text = f"<|user|>\n{question}<|end|>\n<|assistant|>\n{answer}<|end|>"
            enc  = tok(text, return_tensors=None,
                        padding="max_length", truncation=True, max_length=max_len)

        result           = {k: v for k, v in enc.items()}
        result["labels"] = result.get("input_ids", [])[:]
        return result

    def _encode_causal(self, question, answer, tokenizer, max_len) -> Dict:
        """Generic causal LM fallback for any unknown model."""
        text = f"Question: {question}\nAnswer: {answer}"
        tok  = getattr(tokenizer, "tokenizer", tokenizer)
        enc  = tok(
            text,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors=None,
        )
        labels = [
            t if t != (tok.pad_token_id or 0) else -100
            for t in enc["input_ids"]
        ]
        return {
            "input_ids":      enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels":         labels,
        }

    # ── Image loading helper ──────────────────────────────────────────────────

    def _load_image(self, image_path: str):
        """Load PIL image. Returns None if path missing or invalid."""
        if not image_path or not Path(image_path).exists():
            return None
        try:
            from PIL import Image
            return Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.debug(f"[FeatureEng] Image load failed {image_path}: {e}")
            return None

    # ── HuggingFace Dataset ───────────────────────────────────────────────────

    def _build_hf_datasets(self, encoded: List[Dict]):
        """Split encoded records 90/10 into train and val HF Datasets."""
        from datasets import Dataset

        n     = len(encoded)
        split = max(1, int(n * 0.9))
        train_ds = Dataset.from_list(encoded[:split])
        val_ds   = Dataset.from_list(encoded[split:])
        return train_ds, val_ds

    def _save_to_disk(self, train_ds, val_ds, hf_id: str) -> str:
        """Save encoded datasets to disk. Returns base path."""
        safe_name = hf_id.replace("/", "_").replace(" ", "_")
        base_path = FEATURE_DIR / safe_name
        base_path.mkdir(parents=True, exist_ok=True)

        train_ds.save_to_disk(str(base_path / "train"))
        val_ds.save_to_disk(str(base_path / "val"))

        logger.info(f"[FeatureEng] Saved to {base_path}")
        return str(base_path)

    # ── LLM assessment ────────────────────────────────────────────────────────

    def _get_llm_assessment(
        self, hf_id, model_family, is_vision,
        train_n, val_n, columns,
    ) -> Dict:
        prompt = (
            f"Feature engineering completed for Medical VQA.\n"
            f"Model: {hf_id}\n"
            f"Family: {model_family}  Vision: {is_vision}\n"
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


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice : {device}")

    agent = FeatureEngineeringAgent()

    # Test with flan-t5-base (lightest, always works)
    plan = {
        "hf_id":        "google/flan-t5-base",
        "name":         "Flan-T5-Base",
        "architecture": "seq2seq",
        "vision":       False,
        "max_seq_len":  128,
    }

    data_path = "./data/processed/processed_dataset.jsonl"

    if not Path(data_path).exists():
        print(f"\nData file not found: {data_path}")
        print("Run the full pipeline first with --dry-run:")
        print("  python main.py --image image.jpg --question 'Is there pneumonia?' --dry-run")
    else:
        result = agent.engineer_features(data_path, plan, device=device)
        print("\n" + "="*50)
        print("Feature Engineering Result:")
        print(json.dumps(result, indent=2, default=str))
        print("="*50)
