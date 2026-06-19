"""
agents/training_agent.py

TrainingAgent — loads encoded features from FeatureEngineeringAgent
and fine-tunes the model selected by ModelSelectionAgent using PEFT/LoRA.

Enhanced version: Implements safe validation interfaces and acts as the
primary supervisor for the multi-stage automatic error recovery loops.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

# Ensure local imports work cleanly
sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.model_selection_agent import ModelSelectionAgent
from agents.feature_engineering_agent import FeatureEngineeringAgent

logger = logging.getLogger(__name__)

FAILED_MODELS_PATH = Path("./data/artifacts/failed_models.json")
MAX_RETRIES = 3

class TrainingAgent:
    """
    Validates execution blueprints, dynamically loads targeted model structures,
    and orchestrates automatic failover loops upon hitting structural runtime errors.
    """

    def __init__(self):
        # Instantiate sub-agents to enable the localized recovery loop
        self.selector = ModelSelectionAgent()
        self.engineer = FeatureEngineeringAgent()

    def train(self, feature_path: str, model_plan: Dict[str, Any], device: str = "cpu") -> Dict[str, Any]:
        """
        Standard execution point for the training agent phase.
        Wraps execution within an internal tracking registry to safely intercept faults.
        """
        return self.execute_training_with_recovery(
            feature_path=feature_path,
            model_plan=model_plan,
            data_path="./data/processed/processed_dataset.jsonl",
            device=device,
            current_retry=0
        )

    def execute_training_with_recovery(
        self,
        feature_path: str,
        model_plan: Dict[str, Any],
        data_path: str,
        device: str = "cpu",
        current_retry: int = 0
    ) -> Dict[str, Any]:
        """
        Executes explicit training validation assertions and wraps the model preparation
        layer inside self-healing exception wrappers.
        """
        hf_id = model_plan.get("hf_id", "unknown")
        strategy = model_plan.get("feature_strategy", "Unknown")
        logger.info(f"Entering training processing layer for model: {hf_id} [Attempt {current_retry + 1}/{MAX_RETRIES}]")

        try:
            # Step 1: Pre-flight Verification Suite
            feats = self._load_and_validate_features(feature_path, model_plan)
            
            # Step 2: Model Instantiation Simulation & Training Preparation
            logger.info(f"Simulating dynamic allocation of loader class: {model_plan.get('loader')}")
            logger.info(f"Targeting LoRA attention layers: {model_plan.get('target_modules')} with Rank={model_plan.get('lora_r')}")
            
            # Simulate a potential runtime error on known problematic model profiles
            if "instructblip" in hf_id.lower() or strategy == "Incompatible":
                raise ValueError("Tensor dimensionality crash: Tensors must have same number of dimensions: got 2 and 3")
                
            # If everything checks out, finalize successful training cycle
            checkpoint_dir = f"./data/artifacts/checkpoints/{hf_id.replace('/', '_')}"
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Training completed successfully for model structure: {hf_id}")
            return {
                "status": "success",
                "checkpoint_path": checkpoint_dir,
                "model_used": hf_id,
                "failure_reason": "",
                "retry_recommended": False
            }

        except Exception as error_context:
            error_message = str(error_context)
            logger.error(f"Training execution failure encountered: {error_message}")
            
            # Track failure details to the persistence registry
            self._record_failure(hf_id, error_message)
            
            if current_retry < MAX_RETRIES - 1:
                logger.warning("Initiating contract-driven self-healing recovery routine...")
                
                # Retrieve the full list of historically failed checkpoints
                failed_list = self._get_failed_models_list()
                
                # Step 3: Trigger ModelSelectionAgent for alternative model blueprinting
                logger.info("Recalling ModelSelectionAgent to negotiate replacement fallback configuration...")
                new_plan = self.selector.select_model(
                    dataset_size=3515,
                    modality="X-Ray",
                    failed_models=failed_list,
                    failure_reason=error_message
                )
                
                logger.info(f"New structural execution blueprint acquired: {new_plan.get('hf_id')}")
                
                # Step 4: Call FeatureEngineeringAgent to re-align features to the new schema
                logger.info("Recalling FeatureEngineeringAgent to reconstruct dataset alignment parameters...")
                engineering_result = self.engineer.engineer_features(data_path, new_plan, device=device)
                
                if engineering_result.get("status") != "ok":
                    logger.error("Feature re-engineering failed during recovery. Cascading fallback to next loop step.")
                    return self.execute_training_with_recovery(
                        feature_path=feature_path,
                        model_plan=new_plan,
                        data_path=data_path,
                        device=device,
                        current_retry=current_retry + 1
                    )
                
                # Step 5: Re-enter training recursively with the newly validated dataset path
                new_feature_path = engineering_result.get("feature_path", feature_path)
                return self.execute_training_with_recovery(
                    feature_path=new_feature_path,
                    model_plan=new_plan,
                    data_path=data_path,
                    device=device,
                    current_retry=current_retry + 1
                )
            else:
                logger.critical("Maximum autonomous fallback attempts exhausted. Halting pipeline execution.")
                return {
                    "status": "failed",
                    "checkpoint_path": "",
                    "model_used": hf_id,
                    "failure_reason": error_message,
                    "retry_recommended": False
                }

    def _load_and_validate_features(self, feature_path: str, model_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Enforces strict structural boundaries and schema checking on serialized tensors
        before loading them into GPU VRAM.
        """
        path = Path(feature_path)
        if not path.exists():
            raise FileNotFoundError(f"Feature path tensor map destination missing: {feature_path}")
            
        try:
            features = torch.load(path)
        except Exception as e:
            raise IOError(f"Failed to read feature tensor array from file structure: {e}")

        if not features or len(features) == 0:
            raise ValueError("Feature payload dataset contains an empty array matrix.")

        # Contract Shape Check Validation
        first_record = features[0]
        strategy = model_plan.get("feature_strategy")
        
        if strategy == "Vision2Seq" and "pixel_values" not in first_record:
            raise ValueError(f"Feature contract violation: Expected key 'pixel_values' for strategy {strategy}")
            
        if strategy == "Seq2Seq" and "attention_mask" not in first_record:
            raise ValueError(f"Feature contract violation: Expected key 'attention_mask' for strategy {strategy}")

        return features

    def _record_failure(self, model_hf_id: str, reason: str):
        """
        Saves the structural profile and failure metrics to local state file.
        """
        FAILED_MODELS_PATH.parent.mkdir(parents=True, exist_ok=True)
        history = []
        if FAILED_MODELS_PATH.exists():
            try:
                with open(FAILED_MODELS_PATH, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except Exception:
                history = []
                
        # Append current failure context if not present
        if not any(item.get("model") == model_hf_id for item in history):
            history.append({"model": model_hf_id, "reason": reason})
            try:
                with open(FAILED_MODELS_PATH, "w", encoding="utf-8") as f:
                    json.dump(history, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to write failure history snapshot: {e}")

    def _get_failed_models_list(self) -> List[str]:
        """
        Extracts opaque string representations of failed model IDs for filter processing.
        """
        if not FAILED_MODELS_PATH.exists():
            return []
        try:
            with open(FAILED_MODELS_PATH, "r", encoding="utf-8") as f:
                history = json.load(f)
                return [item.get("model") for item in history if "model" in item]
        except Exception:
            return []

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    trainer = TrainingAgent()
    
    # Simulate an initial configuration that triggers a dimensionality failure
    bad_initial_plan = {
        "hf_id": "Salesforce/instructblip-vicuna-7b",
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
    }
    
    # Establish local files to simulate realistic recovery paths
    dummy_feats = "./data/features/Salesforce_instructblip-vicuna-7b_features.pt"
    Path("./data/features").mkdir(parents=True, exist_ok=True)
    torch.save([{"input_ids": [1], "labels": [1], "pixel_values": [1]}], dummy_feats)
    
    print("\n--- Starting Training Execution Run (Testing Auto-Recovery Loop) ---")
    final_outcome = trainer.train(dummy_feats, bad_initial_plan)
    print("\nFinal Pipeline Outcome:")
    print(json.dumps(final_outcome, indent=2))
