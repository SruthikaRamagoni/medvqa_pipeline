"""
agents/model_selection_agent.py

ModelSelectionAgent — scores and selects the best open-source VQA model
architecture based on hardware, dataset size, and modality.
Updated to pass loader field so FeatureEngineeringAgent and TrainingAgent
know exactly which HuggingFace class to use.
"""

from phi.agent import Agent
from phi.model.groq import Groq

from typing import Dict, Any, List
import json, logging, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import MODEL_CATALOGUE

logger = logging.getLogger(__name__)


class ModelSelectionAgent:
    """
    Selects the best model for Medical VQA training based on
    GPU VRAM, dataset size, and imaging modality.
    Returns a complete model_plan dict used by both
    FeatureEngineeringAgent and TrainingAgent.
    """

    def __init__(self, model_id: str = "mistral"):
        self.agent = Agent(
            name="ModelSelectionAgent",
            model=Groq(id="llama-3.1-8b-instant"),
            instructions=[
                "You are a machine learning model selection expert.",
                "Select the best vision-language model for Medical VQA fine-tuning.",
                "Prefer vision models when VRAM allows.",
                "Always reply with ONLY a JSON object containing "
                "'selected_model_hf_id', 'model_name', and 'reason'.",
                "Do not write code. Do not add any text outside the JSON.",
            ],
            show_tool_calls=False,
            markdown=False,
        )

    def _detect_resources(self) -> Dict[str, Any]:
        """Auto-detect available GPU and CPU resources at runtime."""
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)

        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                # Subtract already used memory to get truly available VRAM
                reserved_gb = torch.cuda.memory_reserved(0) / (1024 ** 3)
                vram_available_gb = vram_gb - reserved_gb
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"GPU detected: {gpu_name} | Total VRAM: {vram_gb:.1f}GB | Available: {vram_available_gb:.1f}GB")
            else:
                device = "cpu"
                vram_gb = 0.0
                vram_available_gb = 0.0
                logger.info("No GPU detected, using CPU.")
        except Exception as e:
            logger.warning(f"Could not detect GPU: {e}. Falling back to CPU.")
            device = "cpu"
            vram_gb = 0.0
            vram_available_gb = 0.0

        logger.info(f"RAM available: {ram_gb:.1f}GB | Device: {device}")

        return {
            "device":             device,
            "vram_gb":            vram_available_gb,
            "ram_gb":             ram_gb,
        }

    def select_model(
        self,
        dataset_size: int,
        modality:     str,
    ) -> Dict[str, Any]:
        """
        Auto-detect resources, score all candidate models, and return
        a complete model_plan dict.

        Returns:
            Dict with hf_id, name, architecture, vision, loader,
            lora config, batch_size, epochs, learning_rate, precision,
            use_4bit, target_modules, max_seq_len.
        """
        # Auto-detect resources
        resources    = self._detect_resources()
        device       = resources["device"]
        vram_gb      = resources["vram_gb"]
        ram_gb       = resources["ram_gb"]

        scored = self._score_models(vram_gb, device, dataset_size)
        if not scored:
            scored = [m for m in MODEL_CATALOGUE if m["name"] == "Flan-T5-Base"]

        top3_summary = "\n".join(
            f"{i+1}. {m['name']} | hf_id={m['hf_id']} | "
            f"vision={m['vision']} | params={m['params_b']}B | quality={m['quality']}"
            for i, m in enumerate(scored[:3])
        )

        prompt = (
            f"Select the best model for Medical Visual Question Answering.\n"
            f"Hardware: device={device}  VRAM={vram_gb:.1f}GB  RAM={ram_gb:.1f}GB\n"
            f"Dataset:  {dataset_size} samples  modality={modality}\n\n"
            f"Top candidates:\n{top3_summary}\n\n"
            f"Pick the model that best balances quality and hardware fit.\n"
            f"Prefer vision models when VRAM >= 4 GB.\n\n"
            f'Reply with ONLY: {{"selected_model_hf_id": "<hf_id>", '
            f'"model_name": "<name>", "reason": "<one sentence>"}}'
        )

        response  = self.agent.run(prompt)
        llm_hf_id = self._parse_response(response)

        # Use LLM pick if valid, else take top scored
        best = scored[0]
        if llm_hf_id:
            for m in MODEL_CATALOGUE:
                if llm_hf_id.lower() in m["hf_id"].lower():
                    best = m
                    break

        use_4bit = (vram_gb < best["min_vram"]) and (device == "cuda")

        return {
            # Identity
            "hf_id":          best["hf_id"],
            "name":           best["name"],
            "architecture":   best["architecture"],
            "vision":         best["vision"],
            "loader":         best.get("loader", "auto"),

            # Hardware
            "use_4bit":       use_4bit,
            "precision":      "fp32" if device == "cpu" else "fp16",

            # LoRA
            "lora_r":         8  if dataset_size < 1000 else 16,
            "lora_alpha":     16 if dataset_size < 1000 else 32,
            "lora_dropout":   0.05,
            "target_modules": best["target_modules"],

            # Training
            "batch_size":     1 if device == "cpu" else (2 if use_4bit else 4),
            "epochs":         5 if dataset_size < 500 else 3,
            "learning_rate":  2e-4,

            # Feature engineering
            "max_seq_len":    128,
        }

    def _score_models(self, vram_gb: float, device: str, dataset_size: int) -> List[Dict]:
        feasible = []
        for m in MODEL_CATALOGUE:
            ok = (
                True if device == "cpu"
                else (vram_gb >= m["min_vram"] or vram_gb >= m["min_vram_4bit"])
            )
            if not ok:
                continue
            score = m["quality"]
            if dataset_size < 500 and m["params_b"] > 7:
                score *= 0.8
            feasible.append({**m, "_score": score})
        return sorted(feasible, key=lambda x: x["_score"], reverse=True)

    def _parse_response(self, response) -> str:
        try:
            text  = response.content if hasattr(response, "content") else str(response)
            match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group()).get("selected_model_hf_id", "")
        except Exception:
            pass
        return ""


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s][%(levelname)s] %(message)s")
    agent  = ModelSelectionAgent()
    result = agent.select_model(
        dataset_size=3515,
        modality="X-Ray",
    )
    print("Model Plan:")
    print(json.dumps(result, indent=2))
