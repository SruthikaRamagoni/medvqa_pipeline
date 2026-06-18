"""
agents/model_selection_agent.py

ModelSelectionAgent — scores and selects the best open-source VQA model
architecture based on hardware, dataset size, and modality.

Changes from previous version:
  - select_model() now accepts an optional `failure_context` dict so the
    Coordinator can retry selection after a training failure, passing the
    failed hf_id and error reason.  The agent feeds this to the LLM so it
    avoids the same model.
  - _score_models() now hard-filters any model whose known failure
    patterns match the reported error (OOM, missing class, etc.).
  - Added _infer_failure_class() to translate free-text error messages into
    structured exclusion rules (oom / load_error / training_error).
  - All other behaviour (PhiData + Groq pattern, CLASS_MAP, loader field)
    is unchanged.
"""

from phi.agent import Agent
from phi.model.groq import Groq

from typing import Dict, Any, List, Optional
import json, logging, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import MODEL_CATALOGUE

logger = logging.getLogger(__name__)


class ModelSelectionAgent:
    """
    Selects the best model for Medical VQA training based on
    GPU VRAM, dataset size, and imaging modality.

    Supports retry-aware selection: pass `failure_context` on a second call
    so the agent avoids the model that caused the previous training failure.

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
                "If a previous model failed, never pick it again and explain why "
                "the alternative is safer.",
                "Always reply with ONLY a JSON object containing "
                "'selected_model_hf_id', 'model_name', and 'reason'.",
                "Do not write code. Do not add any text outside the JSON.",
            ],
            show_tool_calls=False,
            markdown=False,
        )

    # ── Public ────────────────────────────────────────────────────────────────

    def select_model(
        self,
        dataset_size: int,
        modality:     str,
        failure_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Auto-detect resources, score all candidate models, and return
        a complete model_plan dict.

        Args:
            dataset_size    : Number of training samples.
            modality        : Imaging modality string (e.g. 'X-Ray').
            failure_context : Optional dict from a previous failed training run.
                              Expected keys:
                                failed_hf_id  – hf_id of the model that failed
                                reason        – free-text error message
                              When provided, the failed model is excluded from
                              scoring and the LLM is told what went wrong.

        Returns:
            Dict with hf_id, name, architecture, vision, loader,
            lora config, batch_size, epochs, learning_rate, precision,
            use_4bit, target_modules, max_seq_len.
        """
        resources = self._detect_resources()
        device    = resources["device"]
        vram_gb   = resources["vram_gb"]
        ram_gb    = resources["ram_gb"]

        # Derive structured exclusion rules from the failure context
        failed_hf_id    = ""
        failure_class   = ""
        failure_summary = ""
        if failure_context:
            failed_hf_id    = failure_context.get("failed_hf_id", "")
            raw_reason      = failure_context.get("reason", "")
            failure_class   = self._infer_failure_class(raw_reason)
            failure_summary = (
                f"Previous model '{failed_hf_id}' failed "
                f"[{failure_class}]: {raw_reason[:200]}"
            )
            logger.info(f"[ModelSelection] Retry mode — {failure_summary}")

        scored = self._score_models(
            vram_gb, device, dataset_size,
            exclude_hf_id=failed_hf_id,
            failure_class=failure_class,
        )
        if not scored:
            # Last-resort fallback: cheapest text-only model
            scored = [m for m in MODEL_CATALOGUE if m["name"] == "Flan-T5-Base"]

        top3_summary = "\n".join(
            f"{i+1}. {m['name']} | hf_id={m['hf_id']} | "
            f"vision={m['vision']} | params={m['params_b']}B | quality={m['quality']}"
            for i, m in enumerate(scored[:3])
        )

        failure_clause = (
            f"\nIMPORTANT: {failure_summary}\n"
            f"Do NOT select '{failed_hf_id}'. Choose a safer alternative.\n"
            if failure_context else ""
        )

        prompt = (
            f"Select the best model for Medical Visual Question Answering.\n"
            f"Hardware: device={device}  VRAM={vram_gb:.1f}GB  RAM={ram_gb:.1f}GB\n"
            f"Dataset:  {dataset_size} samples  modality={modality}\n"
            f"{failure_clause}\n"
            f"Top candidates:\n{top3_summary}\n\n"
            f"Pick the model that best balances quality and hardware fit.\n"
            f"Prefer vision models when VRAM >= 4 GB.\n\n"
            f'Reply with ONLY: {{"selected_model_hf_id": "<hf_id>", '
            f'"model_name": "<name>", "reason": "<one sentence>"}}'
        )

        response  = self.agent.run(prompt)
        llm_hf_id = self._parse_response(response)

        # Use LLM pick if valid and not the failed model, else take top scored
        best = scored[0]
        if llm_hf_id and llm_hf_id.lower() != failed_hf_id.lower():
            for m in MODEL_CATALOGUE:
                if llm_hf_id.lower() in m["hf_id"].lower():
                    best = m
                    break

        use_4bit = (vram_gb < best["min_vram"]) and (device == "cuda")

        plan = {
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

            # Metadata for retry tracking
            "_selected_reason": "",   # filled below
        }

        # Attach LLM reason if available
        try:
            text = (
                response.content
                if hasattr(response, "content")
                else str(response)
            )
            match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if match:
                plan["_selected_reason"] = (
                    json.loads(match.group()).get("reason", "")
                )
        except Exception:
            pass

        logger.info(
            f"[ModelSelection] Selected: {plan['hf_id']} "
            f"(4bit={use_4bit}) reason={plan['_selected_reason']}"
        )
        return plan

    # ── Resource detection ────────────────────────────────────────────────────

    def _detect_resources(self) -> Dict[str, Any]:
        """Auto-detect available GPU and CPU resources at runtime."""
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)

        try:
            import torch
            if torch.cuda.is_available():
                device            = "cuda"
                vram_gb           = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                reserved_gb       = torch.cuda.memory_reserved(0) / (1024 ** 3)
                vram_available_gb = vram_gb - reserved_gb
                gpu_name          = torch.cuda.get_device_name(0)
                logger.info(
                    f"GPU detected: {gpu_name} | "
                    f"Total VRAM: {vram_gb:.1f}GB | "
                    f"Available: {vram_available_gb:.1f}GB"
                )
            else:
                device            = "cpu"
                vram_available_gb = 0.0
                logger.info("No GPU detected, using CPU.")
        except Exception as e:
            logger.warning(f"Could not detect GPU: {e}. Falling back to CPU.")
            device            = "cpu"
            vram_available_gb = 0.0

        logger.info(f"RAM available: {ram_gb:.1f}GB | Device: {device}")
        return {"device": device, "vram_gb": vram_available_gb, "ram_gb": ram_gb}

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score_models(
        self,
        vram_gb:        float,
        device:         str,
        dataset_size:   int,
        exclude_hf_id:  str = "",
        failure_class:  str = "",
    ) -> List[Dict]:
        """
        Score and filter models.

        Extra exclusion logic when a failure_class is known:
          oom          → also exclude all models with min_vram > vram_gb * 0.9
          load_error   → exclude the exact failed hf_id only (already done)
          training_err → no extra exclusion; rely on exclude_hf_id
        """
        feasible = []
        for m in MODEL_CATALOGUE:
            # Hard-exclude the model that already failed
            if exclude_hf_id and m["hf_id"].lower() == exclude_hf_id.lower():
                logger.debug(f"[ModelSelection] Excluded (prev failure): {m['hf_id']}")
                continue

            # Hard-exclude models that won't fit in VRAM when OOM was the cause
            if failure_class == "oom" and device == "cuda":
                safe_vram = vram_gb * 0.85  # keep 15 % headroom
                if m["min_vram"] > safe_vram and m["min_vram_4bit"] > safe_vram:
                    logger.debug(
                        f"[ModelSelection] Excluded (OOM headroom): {m['hf_id']}"
                    )
                    continue

            # Normal VRAM feasibility check
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

    # ── Failure classification ────────────────────────────────────────────────

    def _infer_failure_class(self, reason: str) -> str:
        """
        Translate a free-text training error into one of three categories:
          oom           – out-of-memory / CUDA memory error
          load_error    – model failed to load (class not found, weight mismatch …)
          training_err  – model loaded but training loop crashed

        Used by _score_models() to apply appropriate exclusion heuristics.
        """
        r = reason.lower()
        if any(k in r for k in ("out of memory", "cuda error", "oom",
                                 "cudaoutofmemory", "memory")):
            return "oom"
        if any(k in r for k in ("failed to load", "cannot load",
                                 "no module", "not found in transformers",
                                 "weight", "checkpoint")):
            return "load_error"
        return "training_err"

    # ── Response parsing ──────────────────────────────────────────────────────

    def _parse_response(self, response) -> str:
        try:
            text  = response.content if hasattr(response, "content") else str(response)
            match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group()).get("selected_model_hf_id", "")
        except Exception:
            pass
        return ""


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
    )

    agent = ModelSelectionAgent()

    # Normal selection
    print("=== Normal selection ===")
    result = agent.select_model(dataset_size=3515, modality="X-Ray")
    print(json.dumps(result, indent=2))

    # Retry after OOM
    print("\n=== Retry after OOM ===")
    result2 = agent.select_model(
        dataset_size=3515,
        modality="X-Ray",
        failure_context={
            "failed_hf_id": result["hf_id"],
            "reason":       "CUDA out of memory. Tried to allocate 2.50 GiB",
        },
    )
    print(json.dumps(result2, indent=2))
