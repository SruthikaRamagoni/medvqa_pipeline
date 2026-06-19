"""
agents/model_selection_agent.py

ModelSelectionAgent — scores and selects the best open-source VQA model
architecture based on hardware, dataset size, and modality.

Enhanced version: Enforces an explicit configuration contract containing all keys
demanded by FeatureEngineeringAgent and TrainingAgent, while integrating robust
fallback rules when handling previous failure metrics.
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

# System-wide architecture dictionary matching explicit deployment contracts
MODEL_CATALOGUE_DATA = {
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
    Selects the optimal model plan for Medical VQA training based on
    system hardware metrics, dataset specifications, and historical run failures.
    """

    def __init__(self, model_id: str = "llama-3.1-8b-instant"):
        self.agent = Agent(
            name="ModelSelectionAgent",
            model=Groq(id=model_id),
            instructions=[
                "You are an expert ML Systems Architect specializing in Medical VQA setups.",
                "Your objective is to evaluate hardware details, image characteristics, and error arrays,",
                "and select the best matching model execution schema from the candidate database.",
                "You MUST return the schema strictly formatted inside a single JSON markdown wrapper block.",
                "Do not include extra explanations, prose, or unformatted text outside of the JSON payload."
            ]
        )

    def select_model(
        self,
        dataset_size: int,
        modality: str,
        failed_models: Optional[List[str]] = None,
        failure_reason: str = "",
        failure_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Filters out failing architectures, re-scores remaining model candidates,
        and returns a complete compatibility contract block.
        """
        if failed_models is None:
            failed_models = []

        # Handle backward compatibility with older orchestration scripts passing a dict context
        if failure_context and isinstance(failure_context, dict):
            failed_id = failure_context.get("failed_hf_id") or failure_context.get("failed_model")
            if failed_id and failed_id not in failed_models:
                failed_models.append(failed_id)
            if not failure_reason:
                failure_reason = failure_context.get("reason", "")

        logger.info(f"Selecting VQA model layout for Modality: {modality}, Size: {dataset_size}")
        if failed_models:
            logger.warning(f"Excluding known problematic models from execution scope: {failed_models}. Reason: {failure_reason}")

        # Filter out broken pipelines based on explicit parameters
        active_candidates = {}
        for hf_id, properties in MODEL_CATALOGUE_DATA.items():
            if hf_id in failed_models:
                continue
            
            candidate_map = properties.copy()
            # If historical data signals an out of memory limit, systematically dial down parameters safely
            if "OOM" in failure_reason or "out of memory" in failure_reason.lower():
                candidate_map["batch_size"] = max(1, candidate_map["batch_size"] // 2)
                candidate_map["lora_r"] = max(4, candidate_map["lora_r"] // 2)

            active_candidates[hf_id] = candidate_map

        # Absolute fallback defense if all tracking architectures are spent
        if not active_candidates:
            logger.critical("All catalogue candidates are disqualified. Activating baseline Seq2Seq safety configuration.")
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
                "epochs": 2,
                "lora_r": 8,
                "target_modules": ["q", "v"]
            }

        prompt = (
            f"Select the best single model contract from the following eligible candidate matrix:\n"
            f"{json.dumps(active_candidates, indent=2)}\n\n"
            f"Target Environment Constraints:\n"
            f"- Modality Target: {modality}\n"
            f"- Records count: {dataset_size}\n"
            f"- Disqualified list: {failed_models}\n"
            f"- Triggering exception trace: '{failure_reason}'\n\n"
            f"Return the chosen entry with its structural keys completely preserved. Include the model's identifier "
            f"as the value for the 'hf_id' field inside the root of your JSON object response."
        )

        try:
            response = self.agent.run(prompt)
            parsed_block = self._parse_response(response)
            if parsed_block and "hf_id" in parsed_block:
                return parsed_block
        except Exception as e:
            logger.error(f"Error resolving agent LLM recommendations: {e}. Defaulting to deterministic selection.")

        # Backup deterministic fallback selection from remaining dictionary elements
        fallback_id = list(active_candidates.keys())[0]
        selected_plan = active_candidates[fallback_id]
        selected_plan["hf_id"] = fallback_id
        return selected_plan

    def _parse_response(self, response) -> Dict[str, Any]:
        try:
            text = response.content if hasattr(response, "content") else str(response)
            json_pattern = re.search(r'\{.*\}', text, re.DOTALL)
            if json_pattern:
                return json.loads(json_pattern.group())
        except Exception as e:
            logger.error(f"Failed parsing selected model structure string map: {e}")
        return {}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    selector = ModelSelectionAgent()
    print(json.dumps(selector.select_model(2244, "X-Ray"), indent=2))
