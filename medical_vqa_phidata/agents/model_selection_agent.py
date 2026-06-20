"""
agents/model_selection_agent.py

ModelSelectionAgent — scores and selects the best open-source VQA model
architecture based on hardware, dataset size, and modality.

RECALL / SELF-HEALING UPDATE
-----------------------------
select_model() now accepts:
    failed_models : List[str]   — hf_ids to permanently exclude (accumulates
                                   across retries; TrainingAgent maintains
                                   this list and passes the full history
                                   each time, not just the latest failure).
    failure_reason: str         — free-text error from the most recent
                                   training attempt. Classified into one of
                                   a fixed set of failure classes and used to
                                   penalize/exclude the *entire family* of
                                   the failed model, not just its exact
                                   hf_id (e.g. a Qwen-VL tensor-shape error
                                   excludes all Qwen-VL candidates, since
                                   the same collator bug would recur).

    Backward compatibility: the old `failure_context={"failed_hf_id":...,
    "reason":...}` calling convention is still accepted and is internally
    merged into failed_models / failure_reason.

Returns a complete compatibility-metadata dict (hf_id, architecture,
loader, processor_type, vision, feature_strategy, collator_type,
batch_size, epochs, lora_r, ...) so FeatureEngineeringAgent and
TrainingAgent never have to re-derive model-family behaviour themselves.
"""

from phi.agent import Agent
from phi.model.groq import Groq

from typing import Dict, Any, List, Optional
import json, logging, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import MODEL_CATALOGUE

logger = logging.getLogger(__name__)


# Canonical failure classes used for family-level penalization.
FAILURE_CLASSES = ("oom", "load_error", "tensor_dim_mismatch",
                    "processor_not_supported", "training_error")

# Families whose collator/processor pipeline is most exposed to a
# tensor-dimension mismatch (ragged pixel_values / stray leading dims).
TENSOR_FRAGILE_FAMILIES = {"qwen_vl", "blip2", "instructblip", "llava", "phi_vision"}


class ModelSelectionAgent:
    """
    Selects the best model for Medical VQA training based on
    GPU VRAM, dataset size, and imaging modality.

    Supports retry-aware selection: pass `failed_models` (+ `failure_reason`)
    on subsequent calls so the agent avoids the models — and model
    *families* — that caused previous training failures.
    """

    def __init__(self, model_id: str = "mistral"):
        self.agent = Agent(
            name="ModelSelectionAgent",
            model=Groq(id="llama-3.1-8b-instant"),
            instructions=[
                "You are a machine learning model selection expert.",
                "Select the best vision-language model for Medical VQA fine-tuning.",
                "Prefer vision models when VRAM allows.",
                "If previous models failed, never pick them or models from the "
                "same architecture family again, and explain why the "
                "alternative is safer.",
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
        failed_models: Optional[List[str]] = None,
        failure_reason: str = "",
        failure_context: Optional[Dict[str, Any]] = None,  # back-compat
    ) -> Dict[str, Any]:
        """
        Auto-detect resources, score all candidate models, and return
        a complete model_plan dict with full compatibility metadata.

        Args:
            dataset_size    : Number of training samples.
            modality        : Imaging modality string (e.g. 'X-Ray').
            failed_models   : hf_ids to exclude (accumulated failure history).
            failure_reason  : Free-text error from the most recent failure.
            failure_context : Deprecated single-failure dict
                               {"failed_hf_id": ..., "reason": ...}.
                               Merged into failed_models/failure_reason.

        Returns:
            model_plan dict — see module docstring.
        """
        failed_models = list(failed_models or [])

        # Merge deprecated single-failure calling convention
        if failure_context:
            fid = failure_context.get("failed_hf_id", "")
            if fid and fid not in failed_models:
                failed_models.append(fid)
            failure_reason = failure_reason or failure_context.get("reason", "")

        resources = self._detect_resources()
        device    = resources["device"]
        vram_gb   = resources["vram_gb"]
        ram_gb    = resources["ram_gb"]

        failure_class = self._infer_failure_class(failure_reason) if failure_reason else ""
        failed_families = {
            self._detect_model_family(fid) for fid in failed_models
        }

        if failure_reason:
            logger.info(
                f"[ModelSelection] Retry mode — excluded={failed_models} "
                f"failure_class={failure_class} failed_families={failed_families}"
            )

        scored = self._score_models(
            vram_gb, device, dataset_size,
            failed_models=failed_models,
            failure_class=failure_class,
            failed_families=failed_families,
        )
        if not scored:
            # Last-resort fallback: cheapest text-only model, even if it
            # technically matches a failed family (better than total halt).
            scored = [m for m in MODEL_CATALOGUE
                      if m["name"] == "Flan-T5-Base"] or [MODEL_CATALOGUE[0]]

        top3_summary = "\n".join(
            f"{i+1}. {m['name']} | hf_id={m['hf_id']} | "
            f"vision={m['vision']} | params={m['params_b']}B | quality={m['quality']}"
            for i, m in enumerate(scored[:3])
        )

        failure_clause = ""
        if failed_models:
            failure_clause = (
                f"\nIMPORTANT: These models already failed and must NOT be "
                f"reselected: {failed_models}\n"
                f"Failure reason: {failure_reason}\n"
                f"Choose a model from a DIFFERENT architecture family if the "
                f"failure looks structural (tensor shape / processor issues).\n"
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

        best = scored[0]
        if llm_hf_id and llm_hf_id.lower() not in [f.lower() for f in failed_models]:
            for m in MODEL_CATALOGUE:
                if llm_hf_id.lower() in m["hf_id"].lower():
                    if self._detect_model_family(m["hf_id"]) not in failed_families:
                        best = m
                    break

        use_4bit = (vram_gb < best["min_vram"]) and (device == "cuda")
        family   = self._detect_model_family(best["hf_id"])
        compat   = self._compat_metadata(family)

        plan = {
            # Identity
            "hf_id":          best["hf_id"],
            "name":           best["name"],
            "architecture":   best["architecture"],
            "vision":         best["vision"],
            "loader":         compat["loader"],

            # Compatibility metadata (consumed by FeatureEngineeringAgent /
            # TrainingAgent so they never have to re-derive family behaviour)
            "model_family":      family,
            "processor_type":    compat["processor_type"],
            "feature_strategy":  compat["feature_strategy"],
            "collator_type":     compat["collator_type"],
            "tensor_schema":     compat["tensor_schema"],

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

            # Retry bookkeeping
            "excluded_models": failed_models,
            "_selected_reason": "",
        }

        try:
            text = response.content if hasattr(response, "content") else str(response)
            match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if match:
                plan["_selected_reason"] = json.loads(match.group()).get("reason", "")
        except Exception:
            pass

        logger.info(
            f"[ModelSelection] Selected: {plan['hf_id']} family={family} "
            f"(4bit={use_4bit}) reason={plan['_selected_reason']}"
        )
        return plan

    # ── Resource detection ────────────────────────────────────────────────────

    def _detect_resources(self) -> Dict[str, Any]:
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
                    f"GPU detected: {gpu_name} | Total VRAM: {vram_gb:.1f}GB | "
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

    # ── Family detection (shared vocabulary with FeatureEngineeringAgent) ─────

    def _detect_model_family(self, hf_id: str) -> str:
        hid = (hf_id or "").lower()
        if "flan-t5" in hid or "flan_t5" in hid:          return "flan_t5"
        if "blip2" in hid or "blip-2" in hid:             return "blip2"
        if "instructblip" in hid:                          return "instructblip"
        if "llava" in hid:                                 return "llava"
        if "qwen2.5-vl" in hid or "qwen2-vl" in hid:      return "qwen_vl"
        if "phi-3.5-vision" in hid or "phi3.5" in hid:    return "phi_vision"
        if "idefics" in hid:                               return "idefics"
        if "t5" in hid:                                    return "seq2seq"
        if "bart" in hid:                                  return "seq2seq"
        return "causal"

    # ── Compatibility metadata per family ──────────────────────────────────────

    def _compat_metadata(self, family: str) -> Dict[str, str]:
        table = {
            "flan_t5":       dict(loader="AutoModelForSeq2SeqLM",
                                   processor_type="AutoTokenizer",
                                   feature_strategy="seq2seq",
                                   collator_type="DataCollatorForSeq2Seq",
                                   tensor_schema="input_ids[b,s], attention_mask[b,s], labels[b,s]"),
            "seq2seq":       dict(loader="AutoModelForSeq2SeqLM",
                                   processor_type="AutoTokenizer",
                                   feature_strategy="seq2seq",
                                   collator_type="DataCollatorForSeq2Seq",
                                   tensor_schema="input_ids[b,s], attention_mask[b,s], labels[b,s]"),
            "causal":        dict(loader="AutoModelForCausalLM",
                                   processor_type="AutoTokenizer",
                                   feature_strategy="causal_lm",
                                   collator_type="default_data_collator",
                                   tensor_schema="input_ids[b,s], labels[b,s]"),
            "blip2":         dict(loader="Blip2ForConditionalGeneration",
                                   processor_type="AutoProcessor",
                                   feature_strategy="vision2seq",
                                   collator_type="default_data_collator",
                                   tensor_schema="pixel_values[b,3,224,224], input_ids[b,s], labels[b,s]"),
            "instructblip":  dict(loader="InstructBlipForConditionalGeneration",
                                   processor_type="AutoProcessor",
                                   feature_strategy="vision2seq",
                                   collator_type="default_data_collator",
                                   tensor_schema="pixel_values[b,3,224,224], input_ids[b,s], labels[b,s]"),
            "llava":         dict(loader="AutoModelForImageTextToText",
                                   processor_type="AutoProcessor",
                                   feature_strategy="vision2seq",
                                   collator_type="default_data_collator",
                                   tensor_schema="pixel_values[b,3,H,W], input_ids[b,s], labels[b,s]"),
            "qwen_vl":       dict(loader="auto",  # resolved by TrainingAgent (Qwen2_5_VL / Qwen2VL)
                                   processor_type="AutoProcessor",
                                   feature_strategy="vision2seq_patchified",
                                   collator_type="qwen_vl_collator",
                                   tensor_schema="pixel_values[total_patches,patch_dim], image_grid_thw[n_img,3], input_ids[b,s], labels[b,s]"),
            "phi_vision":    dict(loader="AutoModelForCausalLM",
                                   processor_type="AutoProcessor",
                                   feature_strategy="vision2seq",
                                   collator_type="default_data_collator",
                                   tensor_schema="pixel_values[b,3,H,W], input_ids[b,s], labels[b,s]"),
            "idefics":       dict(loader="AutoModelForImageTextToText",
                                   processor_type="AutoProcessor",
                                   feature_strategy="vision2seq",
                                   collator_type="default_data_collator",
                                   tensor_schema="pixel_values[b,3,H,W], input_ids[b,s], labels[b,s]"),
        }
        return table.get(family, table["causal"])

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score_models(
        self,
        vram_gb:        float,
        device:         str,
        dataset_size:   int,
        failed_models:  List[str],
        failure_class:  str = "",
        failed_families: Optional[set] = None,
    ) -> List[Dict]:
        failed_models   = [f.lower() for f in failed_models]
        failed_families = failed_families or set()

        feasible = []
        for m in MODEL_CATALOGUE:
            if m["hf_id"].lower() in failed_models:
                logger.debug(f"[ModelSelection] Excluded (prev failure): {m['hf_id']}")
                continue

            m_family = self._detect_model_family(m["hf_id"])

            # Structural failure classes (tensor-shape / processor issues)
            # exclude the WHOLE family, not just the exact failed model,
            # since the bug is architectural, not weight-specific.
            if failure_class in ("tensor_dim_mismatch", "processor_not_supported"):
                if m_family in failed_families and m_family in TENSOR_FRAGILE_FAMILIES:
                    logger.debug(
                        f"[ModelSelection] Excluded (family-level, "
                        f"{failure_class}): {m['hf_id']} [{m_family}]"
                    )
                    continue

            if failure_class == "oom" and device == "cuda":
                safe_vram = vram_gb * 0.85
                if m["min_vram"] > safe_vram and m["min_vram_4bit"] > safe_vram:
                    logger.debug(f"[ModelSelection] Excluded (OOM headroom): {m['hf_id']}")
                    continue

            ok = (
                True if device == "cpu"
                else (vram_gb >= m["min_vram"] or vram_gb >= m["min_vram_4bit"])
            )
            if not ok:
                continue

            score = m["quality"]
            if dataset_size < 500 and m["params_b"] > 7:
                score *= 0.8
            if m_family in failed_families:
                score *= 0.5  # soft penalty even for non-structural failures

            feasible.append({**m, "_score": score})

        return sorted(feasible, key=lambda x: x["_score"], reverse=True)

    # ── Failure classification ────────────────────────────────────────────────

    def _infer_failure_class(self, reason: str) -> str:
        """
        Maps free-text / structured failure_reason strings onto one of:
          oom | load_error | tensor_dim_mismatch | processor_not_supported |
          training_error
        Accepts both human-written errors and the short canonical codes
        TrainingAgent emits (e.g. "tensor_dim_mismatch").
        """
        r = (reason or "").lower()
        if r in FAILURE_CLASSES:
            return r
        if any(k in r for k in ("out of memory", "cuda error", "oom",
                                 "cudaoutofmemory", "memory")):
            return "oom"
        if any(k in r for k in ("number of dimensions", "shape", "unpack",
                                 "tensor_dim_mismatch", "dimension")):
            return "tensor_dim_mismatch"
        if any(k in r for k in ("processor", "autoprocessor", "tokenizer",
                                 "processor_not_supported")):
            return "processor_not_supported"
        if any(k in r for k in ("failed to load", "cannot load", "no module",
                                 "not found in transformers", "weight", "checkpoint")):
            return "load_error"
        return "training_error"

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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")
    agent = ModelSelectionAgent()

    print("=== Normal selection ===")
    result = agent.select_model(dataset_size=3515, modality="X-Ray")
    print(json.dumps(result, indent=2, default=str))

    print("\n=== Retry after tensor_dim_mismatch ===")
    result2 = agent.select_model(
        dataset_size=3515,
        modality="X-Ray",
        failed_models=[result["hf_id"]],
        failure_reason="tensor_dim_mismatch",
    )
    print(json.dumps(result2, indent=2, default=str))
