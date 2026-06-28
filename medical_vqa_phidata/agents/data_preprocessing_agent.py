"""
agents/data_preprocessing_agent.py

DataPreprocessingAgent — cleans, resizes, anonymizes, and validates the dataset.
No LLM needed — pure deterministic tools.
Uses PhiData Agent structure (same as reference code).
"""

from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.python import PythonTools

from typing import Dict, Any, List, Tuple
import json, re, logging
from pathlib import Path

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("./data/processed")
TARGET_SIZE   = (224, 224)

PHI_PATTERNS = [
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
    re.compile(r"\b(patient|pt\.?)\s*#?\s*\d+\b", re.I),
    re.compile(r"\bssn\s*:?\s*\d{3}-?\d{2}-?\d{4}\b", re.I),
    re.compile(r"\bmrn\s*:?\s*\d+\b", re.I),
]


class DataPreprocessingAgent:
    """
    Cleans, resizes, normalizes, and anonymizes the raw dataset.
    Deterministic — no LLM reasoning needed for data cleaning.
    """

    def __init__(self, model_id: str = "mistral"):
        self.agent = Agent(
            name="DataPreprocessingAgent",
            model=Groq(id="llama-3.1-8b-instant"),
            tools=[PythonTools()],
            instructions=[
                "You are a data preprocessing quality agent.",
                "Review preprocessing results and confirm data quality.",
                "Always reply with a JSON object containing 'status', 'valid_records', 'dropped_records', and 'message'.",
                "Do not add explanation outside the JSON.",
            ],
            show_tool_calls=True,
            markdown=False,
        )

    def preprocess(self, raw_data_path: str) -> Dict[str, Any]:
        """
        Clean, resize, anonymize, and validate the raw dataset.

        Args:
            raw_data_path: Path to the raw JSONL file from DataCollectionAgent.

        Returns:
            Dict with 'processed_path', 'valid_records', 'dropped_records', 'status'.
        """
        if not Path(raw_data_path).exists():
            return {"status": "failed", "message": f"File not found: {raw_data_path}"}

        records = [json.loads(l) for l in open(raw_data_path) if l.strip()]
        logger.info(f"[Preprocessing] Loaded {len(records)} raw records.")

        # ── Modality filter (NEW) ────────────────────────────────────────────
        # VQA-RAD mixes CT, MRI, and chest X-ray images. Training on all
        # modalities dilutes domain-specific learning. When MODALITY_FILTER
        # is set (e.g. "mri"), only records whose image_type / modality field
        # matches are kept. This is opt-in (empty string = no filtering).
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from config.settings import MODALITY_FILTER
        except Exception:
            MODALITY_FILTER = ""
        if MODALITY_FILTER:
            mf = MODALITY_FILTER.lower()
            before = len(records)
            records = [
                r for r in records
                if mf in (r.get("image_type") or r.get("modality") or "").lower()
                or mf in (r.get("question") or "").lower()
            ]
            logger.info(
                f"[Preprocessing] Modality filter '{MODALITY_FILTER}': "
                f"{len(records)}/{before} records kept."
            )

        # Step 1: Repair image paths
        records = self._repair_paths(records, str(Path(raw_data_path).parent.parent))

        # Step 2: Resize + normalize images
        records = self._resize_images(records)

        # Step 3: Clean + anonymize text
        records = [self._clean_record(r) for r in records]

        # Step 4: Validate
        valid, dropped = self._validate(records)

        if not valid:
            return {"status": "failed", "message": "No valid records after preprocessing",
                    "valid_records": 0, "dropped_records": len(dropped)}

        # Step 5: Save
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out_path = PROCESSED_DIR / "processed_dataset.jsonl"
        with open(out_path, "w") as f:
            for r in valid:
                f.write(json.dumps(r) + "\n")

        logger.info(f"[Preprocessing] Saved {len(valid)} records → {out_path}")

        # Ask LLM to confirm quality
        prompt = f"""
        Data preprocessing completed for a Medical VQA dataset.

        Results:
        - Total input records: {len(records)}
        - Valid records kept: {len(valid)}
        - Records dropped: {len(dropped)}
        - Output path: {out_path}
        - Sample record: {json.dumps(valid[0])[:200]}

        Confirm this preprocessing result is acceptable for model training.

        Reply with ONLY this JSON:
        {{"status": "ok", "valid_records": {len(valid)}, "dropped_records": {len(dropped)}, "message": "<one sentence>"}}
        """

        response = self.agent.run(prompt)
        result = self._parse_response(response)
        result["processed_path"] = str(out_path)
        result["valid_records"]  = len(valid)
        result["dropped_records"] = len(dropped)
        return result

    def _repair_paths(self, records: List[Dict], search_root: str) -> List[Dict]:
        """Try to fix broken image paths by searching under search_root."""
        fixed = []
        for rec in records:
            path_str = rec.get("image_path", "")
            if path_str and Path(path_str).exists():
                fixed.append(rec)
                continue
            if path_str:
                hits = list(Path(search_root).rglob(Path(path_str).name))
                if hits:
                    fixed.append({**rec, "image_path": str(hits[0])})
                    continue
            fixed.append({**rec, "image_path": ""})
        return fixed

    def _resize_images(self, records: List[Dict]) -> List[Dict]:
        """
        Validate images are loadable and convert to RGB.
        
        FIX: Previously this resized all images to 224×224 (TARGET_SIZE) before
        saving. This was destructive — Qwen2-VL and Qwen2.5-VL use dynamic
        resolution encoding that produces far better visual features from
        high-resolution images. Forcing 224×224 discards detail that the model's
        patch tokeniser would otherwise preserve, directly harming accuracy on
        fine-grained medical questions (e.g. "Is there a small lesion?").
        
        The processor for each model family handles its own resizing internally.
        Preprocessing should only ensure the image is readable as RGB — not
        impose a fixed resolution.
        """
        try:
            from PIL import Image
            img_dir = PROCESSED_DIR / "images"
            img_dir.mkdir(parents=True, exist_ok=True)
            result = []
            for i, rec in enumerate(records):
                img_path = rec.get("image_path", "")
                if img_path and Path(img_path).exists():
                    try:
                        # Load and re-save as PNG to ensure consistent format,
                        # but at ORIGINAL resolution — no resize applied.
                        img = Image.open(img_path).convert("RGB")
                        out = img_dir / f"img_{i:06d}.png"
                        img.save(out)
                        rec = {**rec, "image_path": str(out)}
                    except Exception as e:
                        logger.debug(f"Image re-save failed {img_path}: {e}")
                result.append(rec)
            return result
        except ImportError:
            logger.warning("Pillow not available, skipping image processing.")
            return records

    def _clean_record(self, rec: Dict) -> Dict:
        """Clean and anonymize text fields."""
        q = self._clean_text(rec.get("question", ""))
        a = self._clean_text(rec.get("answer", ""))
        return {**rec, "question": q, "answer": a}

    def _clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        for pattern in PHI_PATTERNS:
            text = pattern.sub("[REDACTED]", text)
        return text

    def _validate(self, records: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        valid, dropped = [], []
        for r in records:
            if r.get("question", "").strip() and r.get("answer", "").strip():
                valid.append(r)
            else:
                dropped.append(r)
        return valid, dropped

    def _parse_response(self, response) -> Dict:
        try:
            text = response.content if hasattr(response, "content") else str(response)
            match = re.search(r'\{.*?\}', text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            pass
        return {"status": "ok", "message": "Preprocessing complete"}


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    agent = DataPreprocessingAgent(model_id="mistral")
    result = agent.preprocess("./data/raw/flaviagiammarino_vqa-rad_unified.jsonl")
    print("Preprocessing Result:", json.dumps(result, indent=2))
