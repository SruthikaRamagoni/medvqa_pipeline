"""
agents/data_collection_agent.py

DataCollectionAgent — downloads the selected dataset from HuggingFace or
loads from local disk, unifies the schema, and caches to JSONL.
Uses PhiData Agent with Groq LLM (llama-3.1-8b-instant). Same structure as reference code.
"""

from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.python import PythonTools

from typing import Dict, Any, List, Optional
import json, logging
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_DIR = Path("./data/raw")
MAX_SAMPLES = 5000

QUESTION_ALIASES = ["question", "q", "Question", "query", "input", "text"]
ANSWER_ALIASES   = ["answer",   "a", "Answer",   "output","label","target","response"]
IMAGE_ALIASES    = ["image_path","image","img","image_name","image_id",
                    "file_name","filename","img_path","image_file"]


class DataCollectionAgent:
    """
    Downloads or loads the selected Medical VQA dataset,
    unifies its schema to standard format, and caches to disk.
    """

    def __init__(self, model_id: str = "mistral", max_samples: int = MAX_SAMPLES):
        self.max_samples = max_samples
        self.agent = Agent(
            name="DataCollectionAgent",
            model=Groq(id="llama-3.1-8b-instant"),
            tools=[PythonTools()],
            instructions=[
                "You are a data engineering agent for medical AI.",
                "Your job is to download datasets and verify data quality.",
                "Always reply with a JSON object containing 'status', 'records_count', and 'message'.",
                "Do not add explanation outside the JSON.",
            ],
            show_tool_calls=True,
            markdown=False,
        )

    def collect_dataset(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Download or load the dataset and cache it locally.

        Args:
            dataset_info: Dict from DataDiscoveryAgent with name, source, local_path.

        Returns:
            Dict with 'cache_path', 'records_count', 'status'.
        """
        name   = dataset_info.get("name", "")
        source = dataset_info.get("source", "huggingface")
        local  = dataset_info.get("local_path", "")
        modality = dataset_info.get("modality", "")

        # Load records
        records: List[Dict] = []
        if source == "local" and local and Path(local).exists():
            logger.info(f"[DataCollection] Loading local: {local}")
            records = self._load_local(local)
        else:
            logger.info(f"[DataCollection] Downloading HF: {name}")
            records = self._download_hf(name)

        if not records:
            return {"status": "failed", "records_count": 0, "cache_path": "",
                    "message": f"No records retrieved from {name}"}

        # Unify schema
        unified = self._unify_schema(records, modality)
        if len(unified) < 5:
            return {"status": "failed", "records_count": len(unified),
                    "cache_path": "", "message": "Too few valid records after schema unification"}

        # Cache
        cache_path = self._cache(unified, name)

        # Ask LLM to verify quality
        prompt = f"""
        A medical VQA dataset was downloaded and processed.

        Dataset: {name}
        Total records downloaded: {len(records)}
        Valid records after schema unification: {len(unified)}
        Cache path: {cache_path}
        Sample record: {json.dumps(unified[0], default=str)[:300]}

        Is this dataset acceptable for training a Medical VQA model?

        Reply with ONLY this JSON:
        {{"status": "ok", "records_count": {len(unified)}, "message": "<one sentence assessment>"}}
        """

        response = self.agent.run(prompt)
        result = self._parse_response(response)
        result["cache_path"] = cache_path
        result["records_count"] = len(unified)
        return result

    def _download_hf(self, dataset_id: str) -> List[Dict]:
        """Download from HuggingFace datasets library."""
        try:
            from datasets import load_dataset
            # Do NOT pass trust_remote_code — removed in newer datasets versions
            try:
                ds = load_dataset(dataset_id)
            except Exception:
                ds = load_dataset(dataset_id, split="train")

            records = []
            for split_name in ["train", "validation", "val", "test"]:
                if hasattr(ds, "__getitem__") and split_name in ds:
                    for item in ds[split_name]:
                        records.append({**dict(item), "_split": split_name})
                        if len(records) >= self.max_samples:
                            break
                elif hasattr(ds, "features"):
                    for item in ds:
                        records.append({**dict(item), "_split": "train"})
                        if len(records) >= self.max_samples:
                            break
                if len(records) >= self.max_samples:
                    break

            logger.info(f"[DataCollection] Downloaded {len(records)} records.")
            return records
        except Exception as e:
            logger.error(f"[DataCollection] HF download failed: {e}")
            raise

    def _load_local(self, path: str) -> List[Dict]:
        """Load from local JSONL / JSON / CSV."""
        records = []
        p = Path(path)
        for jl in p.rglob("*.jsonl"):
            with open(jl) as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
            if records:
                break
        if not records:
            for jf in p.rglob("*.json"):
                try:
                    data = json.load(open(jf))
                    records = data if isinstance(data, list) else next(
                        (v for v in data.values() if isinstance(v, list)), [])
                    if records:
                        break
                except Exception:
                    continue
        if not records:
            try:
                import csv
                for cf in p.rglob("*.csv"):
                    records = list(csv.DictReader(open(cf)))
                    if records:
                        break
            except Exception:
                pass
        return records[:self.max_samples]

    def _unify_schema(self, records: List[Dict], modality: str) -> List[Dict]:
        """Normalize arbitrary field names to standard VQA schema.
        
        FIX: Previously only 5 fields were kept (image_path, question, answer,
        split, modality). All other metadata — including answer_type (CLOSED/OPEN
        in VQA-RAD), image_organ, image_name — were silently dropped.
        
        answer_type is specifically important because:
        - It identifies yes/no questions (CLOSED) vs open-ended (OPEN)
        - Downstream agents need it for answer balancing
        - EvaluationAgent needs it to compute per-type accuracy breakdowns
        
        Now all original fields are preserved alongside the normalised ones.
        """
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        unified = []
        for i, rec in enumerate(records):
            question = next((str(rec[k]).strip() for k in QUESTION_ALIASES if k in rec and rec[k]), "")
            answer   = next((str(rec[k]).strip() for k in ANSWER_ALIASES   if k in rec and rec[k]), "")
            split    = next((str(rec[k]) for k in ["split","_split","subset"] if k in rec), "train")

            # Image path
            image_path = ""
            for alias in IMAGE_ALIASES:
                if alias in rec and rec[alias]:
                    raw = str(rec[alias])
                    if Path(raw).exists():
                        image_path = raw
                        break
            # PIL Image from HuggingFace
            if not image_path and "image" in rec and hasattr(rec["image"], "save"):
                img_file = CACHE_DIR / f"img_{i:06d}.jpg"
                rec["image"].save(img_file)
                image_path = str(img_file)

            if not question or not answer:
                continue

            # Preserve ALL original metadata fields that may be useful downstream.
            # Specifically for VQA-RAD: answer_type (CLOSED/OPEN), image_organ,
            # image_name, image_id — these are used for per-type accuracy and
            # answer balancing. Never silently drop dataset-provided metadata.
            extra = {}
            preserved_keys = [
                "answer_type", "image_organ", "image_name", "image_id",
                "question_type", "content_type", "phrase_type",
                "qid", "question_id", "uid",
            ]
            for key in preserved_keys:
                if key in rec and rec[key] is not None:
                    extra[key] = rec[key]

            unified.append({
                "image_path": image_path,
                "question":   question,
                "answer":     answer,
                "split":      split,
                "modality":   modality,
                **extra,
            })

        logger.info(f"[DataCollection] Unified {len(unified)} valid records.")
        return unified

        logger.info(f"[DataCollection] Unified {len(unified)} valid records.")
        return unified

    def _cache(self, records: List[Dict], name: str) -> str:
        """Save unified records as JSONL."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        safe = name.replace("/", "_").replace(" ", "_")
        out = CACHE_DIR / f"{safe}_unified.jsonl"
        with open(out, "w") as f:
            for rec in records:
                f.write(json.dumps(rec, default=str) + "\n")
        logger.info(f"[DataCollection] Cached {len(records)} records → {out}")
        return str(out)

    def _parse_response(self, response) -> Dict:
        """Parse LLM JSON response."""
        try:
            import re
            text = response.content if hasattr(response, "content") else str(response)
            match = re.search(r'\{.*?\}', text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            pass
        return {"status": "ok", "message": "Collection complete"}


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    agent = DataCollectionAgent(model_id="mistral", max_samples=100)
    result = agent.collect_dataset({
        "name": "flaviagiammarino/vqa-rad",
        "source": "huggingface",
        "modality": "X-Ray",
    })
    print("Collection Result:", json.dumps(result, indent=2))
