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
        vram_gb   = resources["vram_gb"]        # available (post-crash can be ~0)
        vram_total = resources["vram_total_gb"]  # physical total — use for OOM headroom
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
            vram_gb, vram_total, device, dataset_size,
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
            f"vision={m['vision']} | params={m['params_b']}B | "
            f"min_vram={m['min_vram']}GB | score={m['_score']:.3f}"
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

        # The scorer owns the final model choice. The LLM's role here is only
        # to provide a human-readable reason string for logging/debugging.
        # On retry (failed_models non-empty), the LLM additionally validates
        # that scored[0] is not from a failed family — if it disagrees, its
        # pick is honoured only if it appears in the scored list.
        best = scored[0]

        prompt = (
            f"Explain in one sentence why the following model was selected "
            f"for Medical Visual Question Answering.\n"
            f"Hardware: device={device}  VRAM={vram_total:.1f}GB total  RAM={ram_gb:.1f}GB\n"
            f"Dataset: {dataset_size} samples  modality={modality}\n"
            f"{failure_clause}\n"
            f"Selected model: {best['hf_id']} (score={best.get('_score', 0):.3f})\n"
            f"Other candidates considered:\n{top3_summary}\n\n"
            f'Reply with ONLY: {{"selected_model_hf_id": "{best["hf_id"]}", '
            f'"model_name": "{best["name"]}", "reason": "<one sentence>"}}'
        )

        response  = self.agent.run(prompt)

        # On retry: allow LLM to flag a better alternative from the scored list
        # (e.g. it notices scored[0] shares a family with a failed model).
        # CRITICAL: only accept hf_ids that actually exist in scored — the LLM
        # can hallucinate model names from its training data (e.g. Phi-3.5 even
        # after it was removed from the catalogue). If the LLM pick is not in
        # scored, silently fall back to scored[0].
        if failed_models:
            llm_hf_id = self._parse_response(response)
            scored_hf_ids = {m["hf_id"].lower() for m in scored}
            if llm_hf_id and llm_hf_id.lower() in scored_hf_ids:
                if llm_hf_id.lower() not in [f.lower() for f in failed_models]:
                    for m in scored:
                        if m["hf_id"].lower() == llm_hf_id.lower():
                            if self._detect_model_family(m["hf_id"]) not in failed_families:
                                best = m
                                logger.info(
                                    f"[ModelSelection] Retry: LLM overrode scorer "
                                    f"to '{best['hf_id']}' (avoids failed family)."
                                )
                            break
            elif llm_hf_id:
                logger.info(
                    f"[ModelSelection] Retry: LLM suggested '{llm_hf_id}' but it is "
                    f"not in the current catalogue — ignoring, using scorer's pick: "
                    f"{best['hf_id']}."
                )

        family   = self._detect_model_family(best["hf_id"])
        # Never apply 4-bit to text-only / sub-1B models:
        # (a) bitsandbytes 4-bit + device_map="auto" wraps T4 in DataParallel
        #     which fires CUBLAS_STATUS_EXECUTION_FAILED on sm_75 hardware.
        # (b) flan-t5-base/large fit in fp16 trivially — 4-bit adds zero benefit.
        _text_only_families = {"flan_t5", "seq2seq"}
        _skip_4bit = (family in _text_only_families) or (best.get("params_b", 0) < 1.0)
        # FIX: use vram_total (physical) not vram_gb (available) to decide 4bit.
        # After an OOM crash, available VRAM ≈ 0.2 GB → vram_gb < min_vram is
        # always True → every model gets use_4bit=True including instructblip-7B
        # which still OOMs even in 4bit (needs ~7 GB), and Qwen gets 4bit which
        # quantizes the visual encoder → StopIteration.
        # Physical total VRAM is the right budget: 14.6 GB >= 6.0 GB min_vram
        # for Qwen → use_4bit=False, loads cleanly in fp16.
        use_4bit = (vram_total < best["min_vram"]) and (device == "cuda") and not _skip_4bit
        compat   = self._compat_metadata(family, best["hf_id"])

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
            # batch_size=4 at max_seq_len=1024 for a 3B vision model consumes
            # ~13-14 GB VRAM (weights 6 GB + activations/grads 7-8 GB).
            # That leaves <1 GB headroom — any spike in image patch count
            # causes OOM. Use batch_size=2 for vision models and compensate
            # with gradient_accumulation_steps=4 (effective batch=8).
            "batch_size":     1 if device == "cpu" else (
                2 if (use_4bit or family in ("qwen_vl", "llava", "phi_vision", "idefics"))
                else 4
            ),
            "epochs":         5 if dataset_size < 500 else 3,
            "learning_rate":  2e-4,

            # Feature engineering — family-aware seq len.
            # FIX: vision models MUST be >> 512:
            #   LLaVA-1.5 expands <image> to 576 patch tokens at 336px → need >= 1024
            #   Phi-3.5-vision: with num_crops=1 image tokens ~144 + question ~80 = ~224
            #     prompt tokens → 512 is ample and keeps per-record tensor size small.
            #     (Old value was 2048, based on default num_crops=4 which produced ~750
            #      image tokens alone and caused mass record-skipping + OOM.)
            #   Qwen-VL: dynamic patch count, 1024 is sufficient for most cases.
            "max_seq_len":    {
                "qwen_vl": 1024, "phi_vision": 1024, "llava": 1024,
                "instructblip": 512, "blip2": 512, "idefics": 1024,
                "flan_t5": 128, "seq2seq": 128, "causal": 256,
            }.get(family, 256),

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
                vram_gb           = 0.0
                logger.info("No GPU detected, using CPU.")
        except Exception as e:
            logger.warning(f"Could not detect GPU: {e}. Falling back to CPU.")
            device            = "cpu"
            vram_available_gb = 0.0
            vram_gb           = 0.0

        logger.info(f"RAM available: {ram_gb:.1f}GB | Device: {device}")
        return {"device": device, "vram_gb": vram_available_gb,
                "vram_total_gb": vram_gb, "ram_gb": ram_gb}

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

    def _compat_metadata(self, family: str, hf_id: str = "") -> Dict[str, str]:
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
            "qwen_vl":       dict(loader=(
                                       "Qwen2_5_VLForConditionalGeneration"
                                       if "qwen2.5-vl" in hf_id.lower()
                                       else "Qwen2VLForConditionalGeneration"
                                   ),
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
        vram_total:     float,
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
                # FIX: use physical total VRAM, not available VRAM.
                # Right after an OOM crash, available VRAM ≈ 0 GB because
                # PyTorch hasn't fully released the previous model's memory yet.
                # Using available VRAM here caused ALL vision models to be
                # excluded (their min_vram_4bit >> 0.085 GB), leaving only
                # Flan-T5-Base as the sole survivor — a text-only model that
                # can't do VQA. Total VRAM (14.6 GB on T4) is the right budget.
                safe_vram = vram_total * 0.75  # 75% of physical VRAM is safe headroom
                if m["min_vram"] > safe_vram and m["min_vram_4bit"] > safe_vram:
                    logger.debug(f"[ModelSelection] Excluded (OOM headroom vs total {vram_total:.1f}GB): {m['hf_id']}")
                    continue

            # VRAM feasibility: a model is loadable if EITHER:
            #   (a) safe_vram >= min_vram  → can run in fp16 with headroom
            #   (b) safe_vram >= min_vram_4bit → can run in 4-bit with headroom
            # FIX: also add a hard ceiling — if min_vram > vram_total entirely,
            # the model physically cannot fit even in 4bit loading mode because
            # loading in fp16 fills VRAM before quantization applies, then OOMs.
            # InstructBLIP-7B has min_vram=14.0, T4 has 14.6GB total → it just
            # barely "fits" by min_vram but has 0 headroom for LoRA + activations.
            # The 75% rule (safe_vram=10.95) catches it for fp16 (10.95 < 14.0 ✗)
            # but NOT for 4bit (10.95 >= 7.0 ✓) — so it still passes.
            # Hard ceiling: if min_vram > vram_total * 0.95, exclude entirely.
            hard_ceiling = vram_total * 0.95
            if device == "cuda" and m["min_vram"] > hard_ceiling:
                logger.debug(
                    f"[ModelSelection] Excluded (hard ceiling {hard_ceiling:.1f}GB): "
                    f"{m['hf_id']} needs min_vram={m['min_vram']}GB"
                )
                continue

            safe_vram = vram_total * 0.75
            ok = (
                True if device == "cpu"
                else (safe_vram >= m["min_vram"] or safe_vram >= m["min_vram_4bit"])
            )
            if not ok:
                continue

            score = m["quality"]

            # Dynamic VRAM-efficiency bonus: models that use less of the
            # available VRAM get a proportional bonus. On a 14.6 GB T4:
            #   Qwen2.5-VL-3B: 6.0 GB min → headroom_ratio = (14.6-6.0)/14.6 = 0.59
            #   Phi-3.5-vision: 8.0 GB min → headroom_ratio = (14.6-8.0)/14.6 = 0.45
            # So Qwen gets +0.059 and Phi-3.5 gets +0.045 automatically —
            # no hardcoded numbers, purely derived from the hardware at runtime.
            if device == "cuda" and vram_total > 0:
                headroom = max(0.0, vram_total - m["min_vram"]) / vram_total
                score += headroom * 0.1   # max +0.1 bonus for most VRAM-efficient model

            if dataset_size < 500 and m["params_b"] > 7:
                score *= 0.8
            if m_family in failed_families:
                score *= 0.5  # soft penalty even for non-structural failures

            # Hard-penalise text-only models for VQA tasks: they have no vision
            # encoder and produce all-zero eval metrics on image-grounded tasks.
            if not m.get("vision", False):
                score *= 0.05  # drops flan-t5-base from 0.60 → 0.03

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
