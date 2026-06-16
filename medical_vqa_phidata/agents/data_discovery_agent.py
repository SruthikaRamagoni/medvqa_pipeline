"""
agents/data_discovery_agent.py

DataDiscoveryAgent — searches HuggingFace and local directories for
the best Medical VQA dataset matching the detected modality.
Uses PhiData Agent with Groq LLM (llama-3.1-8b-instant). Identical structure to reference code.
"""

from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.python import PythonTools

from typing import Dict, Any, List
import json, logging, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import FREE_DATASETS

logger = logging.getLogger(__name__)

LOCAL_SEARCH_ROOTS = ["./data", "/content/data"]


class DataDiscoveryAgent:
    """
    Discovers and selects the best Medical VQA dataset
    for the detected imaging modality.
    """

    def __init__(self, model_id: str = "mistral"):
        self.agent = Agent(
            name="DataDiscoveryAgent",
            model=Groq(id="llama-3.1-8b-instant"),
            tools=[PythonTools()],
            instructions=[
                "You are a medical AI data expert agent.",
                "Your job is to select the best dataset for training a Medical VQA model.",
                "Consider the imaging modality, dataset size, and data quality.",
                "Always reply with a JSON object containing 'selected_dataset' (name) and 'reason'.",
                "Do not add explanation outside the JSON.",
            ],
            show_tool_calls=True,
            markdown=False,
        )

    def discover_dataset(self, modality: str) -> Dict[str, Any]:
        """
        Find the best available Medical VQA dataset for the given modality.

        Args:
            modality: Detected imaging modality (e.g., 'X-Ray', 'CT', 'Pathology').

        Returns:
            Dict with dataset name, source, and selection details.
        """
        local      = self._scan_local(modality)
        all_cands  = local + FREE_DATASETS
        scored     = self._score_candidates(all_cands, modality)
        top3       = scored[:3]

        top3_summary = "\n".join(
            f"{i+1}. {c['name']} | tags={c.get('tags',[])} | size={c.get('size',0)}"
            for i, c in enumerate(top3)
        )

        prompt = f"""
        Select the best dataset for Medical VQA with imaging modality: {modality}

        Top candidates:
        {top3_summary}

        Pick the dataset that best matches modality '{modality}' and has the most samples.

        Reply with ONLY this JSON (no extra text):
        {{"selected_dataset": "<dataset_name>", "reason": "<one sentence>"}}
        """

        response = self.agent.run(prompt)
        llm_pick  = self._parse_response(response)

        best = top3[0] if top3 else FREE_DATASETS[0]
        if llm_pick:
            for c in all_cands:
                if llm_pick.lower() in c["name"].lower():
                    best = c
                    break

        return {
            "name":        best["name"],
            "source":      best.get("source", "huggingface"),
            "local_path":  best.get("local_path", ""),
            "size":        best.get("size", 0),
            "modality":    modality,
            "tags":        best.get("tags", []),
        }

    def _scan_local(self, modality: str) -> List[Dict]:
        found = []
        for root in LOCAL_SEARCH_ROOTS:
            p = Path(root)
            if not p.exists():
                continue
            for child in p.iterdir():
                if not child.is_dir():
                    continue
                n_imgs  = len(list(child.rglob("*.jpg"))) + len(list(child.rglob("*.png")))
                has_ann = any(child.rglob("*.json")) or any(child.rglob("*.csv"))
                if n_imgs > 0 and has_ann:
                    found.append({
                        "name": child.name, "source": "local",
                        "local_path": str(child), "size": n_imgs,
                        "tags": ["local", modality.lower()],
                    })
        return found

    def _score_candidates(self, candidates: List[Dict], modality: str) -> List[Dict]:
        def score(c):
            s = 0.0
            combined = " ".join(c.get("tags", [])) + " " + c["name"].lower()
            if modality.lower() in combined: s += 0.4
            if "medical" in combined or "vqa" in combined: s += 0.3
            s += min(c.get("size", 0) / 50000, 0.2)
            if c.get("source") == "local": s += 0.1
            return s
        return sorted(candidates, key=score, reverse=True)

    def _parse_response(self, response) -> str:
        try:
            text  = response.content if hasattr(response, "content") else str(response)
            match = re.search(r'\{.*?\}', text, re.DOTALL)
            if match:
                return json.loads(match.group()).get("selected_dataset", "")
        except Exception:
            pass
        return ""


if __name__ == "__main__":
    agent  = DataDiscoveryAgent(model_id="mistral")
    result = agent.discover_dataset(modality="X-Ray")
    print("Selected Dataset:", json.dumps(result, indent=2))
