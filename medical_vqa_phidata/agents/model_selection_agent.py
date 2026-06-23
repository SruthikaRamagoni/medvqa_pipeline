"""
agents/model_selection_agent.py

ModelSelectionAgent — Adaptive model selection for Medical VQA.

COMPATIBILITY CONTRACT
-----------------------
Returns a complete model_plan with full compatibility metadata so that
FeatureEngineeringAgent and TrainingAgent never have to re-derive
model-family behaviour themselves:

    {
        "hf_id", "name", "architecture", "vision",
        "loader",           # exact HF class to use
        "model_family",     # canonical family string
        "processor_type",   # AutoTokenizer | AutoProcessor
        "feature_strategy", # seq2seq | causal_lm | vision2seq | ...
        "collator_type",    # DataCollatorForSeq2Seq | default_data_collator
        "tensor_schema",    # human-readable expected tensor shapes
        "use_4bit", "precision",
        "lora_r", "lora_alpha", "lora_dropout", "target_modules",
        "batch_size", "epochs", "learning_rate", "max_seq_len",
        "excluded_models",  # passed through for bookkeeping
    }

CALLING CONVENTIONS
--------------------
Primary (from main.py — unchanged):
    select_model(vram_gb, ram_gb, dataset_size, modality, device,
                 excluded_models=[], failure_reason="")

Alternative (retry-aware, from TrainingAgent):
    select_model(dataset_size=N, modality="X-Ray",
                 failed_models=[...], failure_reason="...")

Both are fully supported via flexible kwargs.

FAILURE CLASSES
----------------
oom                  → exclude models that still won't fit after cache flush
tensor_dim_mismatch  → exclude entire fragile vision family
processor_not_supported → exclude family
load_error           → exclude exact model only
training_error       → exclude exact model only

CONSTANTS
----------
TEXT_ONLY_FAMILIES          — flan_t5, seq2seq excluded from VQA scoring
                              (no vision encoder)
HARDWARE_INCOMPATIBLE_ON_T4 — InstructBLIP >= 7B excluded on < 16 GB VRAM
TENSOR_FRAGILE_FAMILIES     — families prone to dimension mismatches
"""

from phi.agent import Agent
from phi.model.groq import Groq

from typing import Dict, Any, List, Optional
import json, logging, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import MODEL_CATALOGUE

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

FAILURE_CLASSES = (
    "oom", "load_error", "tensor_dim_mismatch",
    "processor_not_supported", "training_error",
)

TENSOR_FRAGILE_FAMILIES = {
    "qwen_vl", "blip2", "instructblip", "llava", "phi_vision",
}

TEXT_ONLY_FAMILIES = {"flan_t5", "seq2seq"}

HARDWARE_INCOMPATIBLE_ON_T4 = {
    "Salesforce/instructblip-vicuna-7b",
    "Salesforce/instructblip-flan-t5-xxl",
}

# Full compatibility table: family → loader, processor, strategy, collator, schema
_COMPAT_TABLE: Dict[str, Dict[str, str]] = {
    "flan_t5": dict(
        loader="AutoModelForSeq2SeqLM",
        processor_type="AutoTokenizer",
        feature_strategy="seq2seq",
        collator_type="DataCollatorForSeq2Seq",
        tensor_schema="input_ids[b,s], attention_mask[b,s], labels[b,s]",
    ),
    "seq2seq": dict(
        loader="AutoModelForSeq2SeqLM",
        processor_type="AutoTokenizer",
        feature_strategy="seq2seq",
        collator_type="DataCollatorForSeq2Seq",
        tensor_schema="input_ids[b,s], attention_mask[b,s], labels[b,s]",
    ),
    "causal": dict(
        loader="AutoModelForCausalLM",
        processor_type="AutoTokenizer",
        feature_strategy="causal_lm",
        collator_type="default_data_collator",
        tensor_schema="input_ids[b,s], labels[b,s]",
    ),
    "blip2": dict(
        loader="Blip2ForConditionalGeneration",
        processor_type="AutoProcessor",
        feature_strategy="vision2seq",
        collator_type="default_data_collator",
        tensor_schema="pixel_values[b,3,224,224], input_ids[b,s], labels[b,s]",
    ),
    "instructblip": dict(
        loader="InstructBlipForConditionalGeneration",
        processor_type="AutoProcessor",
        feature_strategy="vision2seq",
        collator_type="default_data_collator",
        tensor_schema="pixel_values[b,3,224,224], input_ids[b,s], labels[b,s]",
    ),
    "llava": dict(
        loader="AutoModelForImageTextToText",
        processor_type="AutoProcessor",
        feature_strategy="vision2seq",
        collator_type="default_data_collator",
        tensor_schema="pixel_values[b,3,H,W], input_ids[b,s], labels[b,s]",
    ),
    "qwen_vl": dict(
        loader="AutoModelForVision2Seq",           # resolved per hf_id at runtime
        processor_type="AutoProcessor",
        feature_strategy="vision2seq_patchified",
        collator_type="qwen_vl_collator",
        tensor_schema=(
            "pixel_values[total_patches,patch_dim], "
            "image_grid_thw[n_img,3], input_ids[b,s], labels[b,s]"
        ),
    ),
    "phi_vision": dict(
        loader="AutoModelForCausalLM",
        processor_type="AutoProcessor",
        feature_strategy="vision2seq",
        collator_type="default_data_collator",
        tensor_schema="pixel_values[b,3,H,W], input_ids[b,s], labels[b,s]",
    ),
    "idefics": dict(
        loader="AutoModelForImageTextToText",
        processor_type="AutoProcessor",
        feature_strategy="vision2seq",
        collator_type="default_data_collator",
        tensor_schema="pixel_values[b,3,H,W], input_ids[b,s], labels[b,s]",
    ),
}


class ModelSelectionAgent:
    """
    Selects the best model for Medical VQA training.

    Supports two calling conventions (both accepted):
      1. main.py style:
            select_model(vram_gb=..., ram_gb=..., dataset_size=...,
                         modality=..., device=...,
                         excluded_models=[...], failure_reason="...")
      2. Retry style (from TrainingAgent):
            select_model(dataset_size=..., modality=...,
                         failed_models=[...], failure_reason="...")

    Both return the same complete model_plan dict.
    """

    def __init__(self, model_id: str = "mistral"):
        self.agent = Agent(
            name="ModelSelectionAgent",
            model=Groq(id="llama-3.1-8b-instant"),
            instructions=[
                "You are a machine learning model selection expert for Medical VQA.",
                "Select the best vision-language model given hardware and dataset constraints.",
                "Only choose models that can ACTUALLY run on the given hardware.",
                "On CPU (VRAM=0): only choose text-only models (Flan-T5).",
                "If previous models failed, NEVER pick them or models from the same "
                "architecture family again when the failure looks structural.",
                "Always reply with ONLY a JSON object: "
                '{"selected_model_hf_id": "<hf_id>", '
                '"model_name": "<name>", "reason": "<one sentence>"}',
                "Do not write code. No text outside the JSON.",
            ],
            show_tool_calls=False,
            markdown=False,
        )

    # ── Public interface ──────────────────────────────────────────────────────

    def select_model(
        self,
        # ── main.py calling convention ────────────────────────────────────────
        vram_gb:          Optional[float]      = None,
        ram_gb:           Optional[float]      = None,
        dataset_size:     int                  = 1000,
        modality:         str                  = "",
        device:           Optional[str]        = None,
        # ── retry / recovery calling convention ──────────────────────────────
        failed_models:    Optional[List[str]]  = None,   # alias: excluded_models
        failure_reason:   str                  = "",
        # ── back-compat ───────────────────────────────────────────────────────
        excluded_models:  Optional[List[str]]  = None,   # synonym for failed_models
        failure_context:  Optional[Dict]       = None,   # legacy single-failure dict
    ) -> Dict[str, Any]:
        """
        Select the best feasible model. Auto-detects hardware if vram_gb /
        ram_gb / device are not provided (retry-aware mode).
        """
        # ── Normalise excluded list ───────────────────────────────────────────
        all_excluded: List[str] = []
        if failed_models:
            all_excluded.extend(failed_models)
        if excluded_models:
            all_excluded.extend(excluded_models)
        if failure_context:
            fid = failure_context.get("failed_hf_id", "")
            if fid and fid not in all_excluded:
                all_excluded.append(fid)
            failure_reason = failure_reason or failure_context.get("reason", "")

        failure_class   = self._infer_failure_class(failure_reason)
        failed_families = {self._detect_family(fid) for fid in all_excluded}

        # ── Resolve hardware ──────────────────────────────────────────────────
        flush_cache = (failure_class == "oom")
        hw = self._detect_resources(flush=flush_cache)

        resolved_device  = device  if device  is not None else hw["device"]
        resolved_vram_gb = vram_gb if vram_gb is not None else hw["vram_gb"]
        resolved_ram_gb  = ram_gb  if ram_gb  is not None else hw["ram_gb"]

        logger.info(
            f"[ModelSelection] device={resolved_device}  "
            f"VRAM={resolved_vram_gb:.1f}GB  RAM={resolved_ram_gb:.1f}GB  "
            f"dataset={dataset_size}  excluded={all_excluded}  "
            f"failure_class={failure_class}"
        )

        # ── Score candidates ──────────────────────────────────────────────────
        scored = self._score_models(
            vram_gb=resolved_vram_gb,
            device=resolved_device,
            dataset_size=dataset_size,
            failed_models=all_excluded,
            failure_class=failure_class,
            failed_families=failed_families,
        )

        if not scored:
            logger.warning("[ModelSelection] No feasible models. Using absolute fallback.")
            return self._absolute_fallback(resolved_device, dataset_size, all_excluded)

        # ── Groq LLM confirmation ─────────────────────────────────────────────
        top3_summary = "\n".join(
            f"{i+1}. {m['name']} | hf_id={m['hf_id']} | "
            f"vision={m['vision']} | params={m['params_b']}B | "
            f"score={m['_score']:.2f}"
            for i, m in enumerate(scored[:3])
        )

        failure_clause = ""
        if all_excluded:
            failure_clause = (
                f"\nIMPORTANT: These models already FAILED — do not re-select them: "
                f"{all_excluded}\n"
                f"Failure reason: {failure_reason}\n"
                f"Choose a different architecture family if failure looks structural.\n"
            )

        prompt = (
            f"Select the best model for Medical Visual Question Answering.\n"
            f"Hardware: device={resolved_device}  VRAM={resolved_vram_gb:.1f}GB  "
            f"RAM={resolved_ram_gb:.1f}GB\n"
            f"Dataset: {dataset_size} samples  modality={modality}\n"
            f"{failure_clause}\n"
            f"Top feasible candidates:\n{top3_summary}\n\n"
            f"Rules:\n"
            f"- On CPU (VRAM=0): only pick Flan-T5 (text-only).\n"
            f"- Prefer vision models when VRAM >= 4 GB.\n"
            f"- After tensor-shape or processor failure: pick a different family.\n"
            f'Reply with ONLY: {{"selected_model_hf_id": "<hf_id>", '
            f'"model_name": "<name>", "reason": "<one sentence>"}}'
        )

        response  = self.agent.run(prompt)
        llm_hf_id = self._parse_response(response)

        best = scored[0]  # default to top-scored
        if llm_hf_id and llm_hf_id.lower() not in [f.lower() for f in all_excluded]:
            for m in scored:
                if llm_hf_id.lower() in m["hf_id"].lower():
                    fam = self._detect_family(m["hf_id"])
                    # Guard: never let LLM pick a text-only or HW-incompatible model
                    if (fam not in TEXT_ONLY_FAMILIES
                            and not (m["hf_id"] in HARDWARE_INCOMPATIBLE_ON_T4
                                     and resolved_vram_gb < 16.0)):
                        best = m
                        logger.info(f"[ModelSelection] LLM confirmed: {m['name']}")
                    break

        return self._build_plan(
            best, resolved_vram_gb, resolved_device, dataset_size,
            all_excluded, response
        )

    # ── Backward-compat wrapper for TrainingAgent ─────────────────────────────

    def reselect_after_failure(
        self,
        failed_hf_id:        str,
        failure_reason:      str,
        vram_gb:             float,
        ram_gb:              float,
        dataset_size:        int,
        modality:            str,
        device:              str,
        previously_excluded: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Convenience wrapper — calls select_model with failure context."""
        excluded = list(previously_excluded or [])
        if failed_hf_id not in excluded:
            excluded.append(failed_hf_id)
        return self.select_model(
            vram_gb=vram_gb,
            ram_gb=ram_gb,
            dataset_size=dataset_size,
            modality=modality,
            device=device,
            failed_models=excluded,
            failure_reason=failure_reason,
        )

    # ── Hardware detection ────────────────────────────────────────────────────

    def _detect_resources(self, flush: bool = False) -> Dict[str, Any]:
        """
        flush=True: empty GPU cache first and report total VRAM
        (not total-reserved), so OOM retries see the full budget.
        """
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        try:
            import torch
            if torch.cuda.is_available():
                if flush:
                    torch.cuda.empty_cache()
                props   = torch.cuda.get_device_properties(0)
                total   = props.total_memory / (1024 ** 3)
                reserved= torch.cuda.memory_reserved(0) / (1024 ** 3)
                vram    = total if flush else max(total - reserved, 0.0)
                logger.info(
                    f"[ModelSelection] GPU: {props.name} | "
                    f"Total={total:.1f}GB | Available={vram:.1f}GB"
                )
                return {"device": "cuda", "vram_gb": vram, "ram_gb": ram_gb}
            elif torch.backends.mps.is_available():
                return {"device": "mps", "vram_gb": ram_gb * 0.5, "ram_gb": ram_gb}
        except Exception as e:
            logger.warning(f"[ModelSelection] GPU detection failed: {e}")
        return {"device": "cpu", "vram_gb": 0.0, "ram_gb": ram_gb}

    # ── Family detection ──────────────────────────────────────────────────────

    def _detect_family(self, hf_id: str) -> str:
        """
        Canonical model-family string.
        Shared vocabulary with FeatureEngineeringAgent and TrainingAgent.
        """
        hid = (hf_id or "").lower()
        if "flan-t5" in hid or "flan_t5" in hid:        return "flan_t5"
        if "blip2"   in hid or "blip-2"  in hid:        return "blip2"
        if "instructblip" in hid:                        return "instructblip"
        if "llava"        in hid:                        return "llava"
        if "qwen2.5-vl"   in hid or "qwen2-vl" in hid:  return "qwen_vl"
        if "phi-3.5-vision" in hid or "phi3.5"  in hid: return "phi_vision"
        if "idefics"      in hid:                        return "idefics"
        if "t5"  in hid:                                 return "seq2seq"
        if "bart" in hid:                                return "seq2seq"
        return "causal"

    # ── Compatibility metadata ────────────────────────────────────────────────

    def _compat_metadata(self, family: str, hf_id: str = "") -> Dict[str, str]:
        meta = dict(_COMPAT_TABLE.get(family, _COMPAT_TABLE["causal"]))
        # Resolve exact Qwen loader
        if family == "qwen_vl":
            if "qwen2.5-vl" in hf_id.lower():
                meta["loader"] = "Qwen2_5_VLForConditionalGeneration"
            else:
                meta["loader"] = "Qwen2VLForConditionalGeneration"
        return meta

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score_models(
        self,
        vram_gb:         float,
        device:          str,
        dataset_size:    int,
        failed_models:   List[str],
        failure_class:   str,
        failed_families: set,
    ) -> List[Dict]:
        excluded_lower = {f.lower() for f in failed_models}
        feasible       = []

        for m in MODEL_CATALOGUE:
            # Exact exclusions
            if m["hf_id"].lower() in excluded_lower:
                continue

            family = self._detect_family(m["hf_id"])

            # Never score text-only for vision VQA
            if family in TEXT_ONLY_FAMILIES:
                continue

            # Hardware-incompatible on T4
            if (m["hf_id"] in HARDWARE_INCOMPATIBLE_ON_T4
                    and device == "cuda" and vram_gb < 16.0):
                continue

            # Family-level exclusion for structural failures
            if failure_class in ("tensor_dim_mismatch", "processor_not_supported"):
                if family in failed_families and family in TENSOR_FRAGILE_FAMILIES:
                    continue

            # OOM: apply headroom buffer
            if failure_class == "oom" and device == "cuda":
                safe = vram_gb * 0.85
                if m["min_vram"] > safe and m.get("min_vram_4bit", 0) > safe:
                    continue

            # Hardware feasibility
            if device == "cuda":
                fits = vram_gb >= m["min_vram"] or vram_gb >= m.get("min_vram_4bit", 0)
            elif device == "cpu":
                fits = True
            else:
                fits = True

            if not fits:
                continue

            score = float(m.get("quality", 0.5))
            if dataset_size < 500 and m.get("params_b", 0) > 7:
                score *= 0.8
            if family in failed_families:
                score *= 0.5          # soft-penalise same family

            feasible.append({**m, "_score": score, "_family": family})

        return sorted(feasible, key=lambda x: x["_score"], reverse=True)

    # ── Plan builder ──────────────────────────────────────────────────────────

    def _build_plan(
        self,
        best:         Dict,
        vram_gb:      float,
        device:       str,
        dataset_size: int,
        excluded:     List[str],
        response,
    ) -> Dict[str, Any]:
        family  = self._detect_family(best["hf_id"])
        compat  = self._compat_metadata(family, best["hf_id"])
        params_b= float(best.get("params_b", 0))

        # 4-bit logic: large models on T4, never on CPU
        if device == "cuda" and vram_gb < 16.0:
            use_4bit = params_b > 3.0
        elif device == "cuda":
            use_4bit = vram_gb < best.get("min_vram", 99)
        else:
            use_4bit = False

        # Adaptive hyperparams
        if   dataset_size < 200:  lora_r, lora_a, epochs, batch = 4,  8,  5, 1
        elif dataset_size < 1000: lora_r, lora_a, epochs, batch = 8,  16, 5, 2
        else:                     lora_r, lora_a, epochs, batch = 16, 32, 3, 2 if use_4bit else 4
        if device == "cpu":       batch = 1

        # Parse LLM reason
        reason = ""
        try:
            text  = response.content if hasattr(response, "content") else str(response)
            match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if match:
                reason = json.loads(match.group()).get("reason", "")
        except Exception:
            pass

        plan = {
            # ── Identity ──────────────────────────────────────────────────────
            "hf_id":             best["hf_id"],
            "name":              best["name"],
            "architecture":      best.get("architecture", "causal"),
            "vision":            best.get("vision", False),
            # ── Compatibility contract ────────────────────────────────────────
            "loader":            compat["loader"],
            "model_family":      family,
            "processor_type":    compat["processor_type"],
            "feature_strategy":  compat["feature_strategy"],
            "collator_type":     compat["collator_type"],
            "tensor_schema":     compat["tensor_schema"],
            # ── Hardware ──────────────────────────────────────────────────────
            "use_4bit":          use_4bit,
            "precision":         "fp32" if device == "cpu" else "fp16",
            # ── LoRA ──────────────────────────────────────────────────────────
            "lora_r":            lora_r,
            "lora_alpha":        lora_a,
            "lora_dropout":      0.05,
            "target_modules":    best.get("target_modules", ["q", "v"]),
            # ── Training ──────────────────────────────────────────────────────
            "batch_size":        batch,
            "epochs":            epochs,
            "learning_rate":     2e-4,
            # ── Feature engineering ───────────────────────────────────────────
            "max_seq_len":       128,
            # ── Bookkeeping ───────────────────────────────────────────────────
            "excluded_models":   excluded,
            "_selected_reason":  reason,
        }

        logger.info(
            f"[ModelSelection] -> {plan['hf_id']} | family={family} | "
            f"4bit={use_4bit} | strategy={compat['feature_strategy']} | "
            f"collator={compat['collator_type']}"
        )
        return plan

    # ── Absolute fallback ─────────────────────────────────────────────────────

    def _absolute_fallback(
        self, device: str, dataset_size: int, excluded: List[str]
    ) -> Dict[str, Any]:
        """
        Last resort: try Qwen2-VL-2B first (GPU), then Flan-T5-Base (CPU/any).
        This is the one place where a text-only model is acceptable —
        because having any answer is better than pipeline failure.
        """
        excluded_lower = {e.lower() for e in excluded}

        # Try Qwen2-VL-2B on GPU first
        if device != "cpu":
            for m in MODEL_CATALOGUE:
                if ("qwen2-vl-2b" in m["hf_id"].lower()
                        and m["hf_id"].lower() not in excluded_lower):
                    logger.warning(f"[ModelSelection] Fallback to {m['name']}")
                    return self._build_plan(m, 0.0, device, dataset_size, excluded, "")

        # CPU / last resort: Flan-T5-Base
        for m in MODEL_CATALOGUE:
            if ("flan-t5-base" in m["hf_id"].lower()
                    and m["hf_id"].lower() not in excluded_lower):
                logger.warning(f"[ModelSelection] Absolute fallback to {m['name']}")
                return self._build_plan(m, 0.0, device, dataset_size, excluded, "")

        # If Flan-T5 also excluded, just take first available
        for m in MODEL_CATALOGUE:
            if m["hf_id"].lower() not in excluded_lower:
                return self._build_plan(m, 0.0, device, dataset_size, excluded, "")

        raise RuntimeError("[ModelSelection] All models excluded — cannot continue.")

    # ── Failure classification ────────────────────────────────────────────────

    def _infer_failure_class(self, reason: str) -> str:
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
                                 "not found in transformers", "weight")):
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


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
    )
    import torch

    agent   = ModelSelectionAgent()
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    vram_gb = 0.0
    if device == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
    print(f"Device: {device}  VRAM: {vram_gb:.1f}GB\n")

    print("=== main.py calling convention ===")
    plan = agent.select_model(
        vram_gb=vram_gb, ram_gb=32.0, dataset_size=2244,
        modality="X-Ray", device=device,
    )
    print(json.dumps(plan, indent=2))

    print("\n=== Retry after tensor_dim_mismatch ===")
    plan2 = agent.select_model(
        dataset_size=2244, modality="X-Ray",
        failed_models=[plan["hf_id"]],
        failure_reason="Tensors must have same number of dimensions: got 2 and 3",
    )
    print(json.dumps(plan2, indent=2))

    print("\n=== Retry after OOM ===")
    plan3 = agent.select_model(
        dataset_size=2244, modality="X-Ray",
        failed_models=[plan["hf_id"], plan2["hf_id"]],
        failure_reason="CUDA out of memory",
    )
    print(json.dumps(plan3, indent=2))
