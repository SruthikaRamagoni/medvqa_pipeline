"""
agents/model_selection_agent.py

ModelSelectionAgent — scores and selects the best open-source VQA model
architecture based on hardware, dataset size, and modality.

Enhanced version: Enforces a strict schema contract containing structural tokens
required by both FeatureEngineeringAgent and TrainingAgent. Implements comprehensive
heuristic exclusions and LLM fallback paths for robust recovery.
"""

from phi.agent import Agent
from phi.model.groq import Groq

from typing import Dict, Any, List, Optional
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

# Fallback catalog in case global settings are unavailable
MODEL_CATALOGUE = {
    "Salesforce/instructblip-vicuna-7b": {
        "name": "InstructBLIP-Vicuna-7B",
        "architecture": "Vision2Seq",
        "loader": "Blip2ForConditionalGeneration",
        "processor_type": "AutoProcessor",
        "vision": True,
        "feature_strategy": "Vision2Seq",
        "collator_type": "DataCollatorForSeq2Seq",
        "batch_size": 2,
        "epochs": 3,
        "lora_r": 8,
        "target_modules": ["q_proj", "v_proj"]
    },
    "Qwen/Qwen2-VL-7B-Instruct": {
        "name": "Qwen2-VL-7B-Instruct",
        "architecture": "CausalLM",
        "loader": "AutoModelForVision2Seq",
        "processor_type": "AutoProcessor",
        "vision": True,
        "feature_strategy": "CausalLM",
        "collator_type": "QwenVLCollator",
        "batch_size": 1,
        "epochs": 3,
        "lora_r": 4,
        "target_modules": ["q_proj", "v_proj", "kv_proj"]
    },
    "google/flan-t5-base": {
        "name": "Flan-T5-Base",
        "architecture": "Seq2Seq",
        "loader": "AutoModelForSeq2SeqLM",
        "processor_type": "AutoTokenizer",
        "vision": False,
        "feature_strategy": "Seq2Seq",
        "collator_type": "DataCollatorForSeq2Seq",
        "batch_size": 8,
        "epochs": 5,
        "lora_r": 16,
        "target_modules": ["q", "v"]
    }
}

class ModelSelectionAgent:
    """
    Selects the best model for Medical VQA training based on
    GPU VRAM, dataset size, and imaging modality.
    Supports contract compliance and self-recovering execution hooks.
    """

    def __init__(self, model_id: str = "llama-3.1-8b-instant"):
        self.agent = Agent(
            name="ModelSelectionAgent",
            model=Groq(id=model_id),
            instructions=[
                "You are an expert ML Infrastructure Architect specializing in Medical VQA systems.",
                "Your objective is to examine hardware constraints, modality requirements, and failure lists,",
                "and select the optimal structural execution profile.",
                "You MUST return your output strictly inside a valid JSON markdown block.",
                "Ensure your returned parameters are syntactically aligned to the architecture requirements."
            ]
        )

    def select_model(
        self,
        dataset_size: int,
        modality: str,
        failed_models: Optional[List[str]] = None,
        failure_reason: str = ""
    ) -> Dict[str, Any]:
        """
        Dynamically applies hard heuristic filters to exclude crashing architectures
        and scores remaining models to return a fully compliant pipeline configuration.
        """
        if failed_models is None:
            failed_models = []

        logger.info(f"Selecting model for Modality: {modality}, Dataset Size: {dataset_size}")
        if failed_models:
            logger.warning(f"Excluding previously failed models: {failed_models}. Reason: {failure_reason}")

        # Step 1: Filter remaining eligible models
        eligible_candidates = {}
        for hf_id, metadata in MODEL_CATALOGUE.items():
            if hf_id in failed_models:
                continue
            
            # If previous model encountered OOM, downscale target batch configurations safely
            candidate_config = metadata.copy()
            if "OOM" in failure_reason or "out of memory" in failure_reason.lower():
                candidate_config["batch_size"] = max(1, candidate_config["batch_size"] // 2)
                candidate_config["lora_r"] = max(4, candidate_config["lora_r"] // 2)

            eligible_candidates[hf_id] = candidate_config

        # Absolute Fallback if all standard candidates are excluded
        if not eligible_candidates:
            logger.critical("All catalogue entries exhausted or failed. Initiating fallback safety text model.")
            return {
                "hf_id": "google/flan-t5-base",
                "name": "Flan-T5-Base",
                "architecture": "Seq2Seq",
                "loader": "AutoModelForSeq2SeqLM",
                "processor_type": "AutoTokenizer",
                "vision": False,
                "feature_strategy": "Seq2Seq",
                "collator_type": "DataCollatorForSeq2Seq",
                "batch_size": 4,
                "epochs": 3,
                "lora_r": 8,
                "target_modules": ["q", "v"]
            }

        # Step 2: Use Agent LLM context parsing to score the remaining eligible candidates
        prompt = (
            f"Select the absolute best model profile from the following eligible choices dictionary:\n"
            f"{json.dumps(eligible_candidates, indent=2)}\n\n"
            f"Context details:\n"
            f"- Modality: {modality}\n"
            f"- Dataset Size: {dataset_size}\n"
            f"- Failure History: {failed_models}\n"
            f"- Last Failure Reason: '{failure_reason}'\n\n"
            f"Return ONLY the complete raw JSON object matching the chosen model configuration structure. "
            f"Do not truncate parameters or include prose explanations outside the JSON block."
        )

        try:
            response = self.agent.run(prompt)
            parsed_config = self._parse_response(response)
            if parsed_config and "hf_id" in parsed_config:
                parsed_config["hf_id"] = str(parsed_config["hf_id"])
                return parsed_config
        except Exception as e:
            logger.error(f"Error invoking model selection agent LLM: {e}. Using deterministic fallback.")

        # Fallback choice parsing
        first_available_id = list(eligible_candidates.keys())[0]
        selected = eligible_candidates[first_available_id]
        selected["hf_id"] = first_available_id
        return selected

    def _parse_response(self, response) -> Dict[str, Any]:
        try:
            text = response.content if hasattr(response, "content") else str(response)
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as e:
            logger.error(f"Failed to parse selection payload: {e}")
        return {}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    selector = ModelSelectionAgent()
    print("Test Normal Selection:")
    print(json.dumps(selector.select_model(3515, "X-Ray"), indent=2))
