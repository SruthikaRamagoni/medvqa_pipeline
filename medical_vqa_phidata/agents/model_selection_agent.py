"""
model_selection_agent.py  — patched
Fixes:
  1. GPU memory is re-measured AFTER freeing the previous model (stale 0.1 GB bug)
  2. On OOM retry, ONLY vision-capable models are considered for vision tasks (MRI/CT/X-ray)
  3. Qwen2-VL-2B-Instruct added as first-choice T4-safe fallback (~9 GB)
  4. text-only fallback (flan-t5, etc.) is hard-blocked when modality requires vision
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch

logger = logging.getLogger("agents.model_selection_agent")

# ---------------------------------------------------------------------------
# Model catalogue
# Each entry: (hf_id, family, vision, vram_gb_fp16, vram_gb_4bit)
# ---------------------------------------------------------------------------
_CATALOGUE = [
    # vision-capable, T4-friendly
    ("Qwen/Qwen2-VL-2B-Instruct",          "qwen2_vl",    True,  9.0,  5.0),
    ("Salesforce/blip2-opt-2.7b",           "blip2",       True,  8.5,  5.0),
    ("Salesforce/instructblip-vicuna-7b",   "instructblip",True, 18.0, 10.0),
    ("llava-hf/llava-1.5-7b-hf",           "llava",       True, 16.0,  9.0),
    # text-only — ONLY valid when modality does NOT require vision
    ("google/flan-t5-base",                 "flan_t5",     False,  1.0,  1.0),
    ("google/flan-t5-large",                "flan_t5",     False,  2.0,  2.0),
]

VISION_MODALITIES = {"mri", "ct", "x-ray", "xray", "pathology", "fundus", "dermoscopy"}


@dataclass
class ModelCandidate:
    model_id: str
    family: str
    vision: bool
    vram_fp16: float
    vram_4bit: float
    use_4bit: bool = False

    @property
    def required_vram(self) -> float:
        return self.vram_4bit if self.use_4bit else self.vram_fp16


def _free_gpu_memory() -> None:
    """Aggressively release GPU memory before re-querying availability."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _available_vram_gb() -> float:
    """Return FREE VRAM in GB — must be called AFTER _free_gpu_memory()."""
    if not torch.cuda.is_available():
        return 0.0
    free, _ = torch.cuda.mem_info(0)          # bytes
    return free / (1024 ** 3)


def _requires_vision(modality: str) -> bool:
    return modality.lower().strip() in VISION_MODALITIES


class ModelSelectionAgent:
    """
    Selects the best HuggingFace model for the task given hardware constraints.

    Parameters
    ----------
    modality : str
        Data modality, e.g. "MRI", "CT", "text".
    excluded : list[str]
        Model IDs to skip (e.g. previously failed models).
    failure_class : str | None
        "oom" triggers a GPU memory flush before re-measuring.
    failed_families : set[str]
        Families already tried — avoids picking a sibling of a failed model
        unless no other option exists.
    """

    def __init__(
        self,
        modality: str = "MRI",
        excluded: Optional[list[str]] = None,
        failure_class: Optional[str] = None,
        failed_families: Optional[set[str]] = None,
    ):
        self.modality = modality
        self.excluded: list[str] = excluded or []
        self.failure_class = failure_class
        self.failed_families: set[str] = failed_families or set()

    # ------------------------------------------------------------------
    def select(self) -> ModelCandidate:
        is_retry = bool(self.excluded)
        needs_vision = _requires_vision(self.modality)

        # --- 1. Free GPU memory BEFORE measuring (fixes stale 0.1 GB bug) ---
        if is_retry and self.failure_class == "oom":
            logger.info("[ModelSelection] OOM retry — flushing GPU cache before re-measuring VRAM.")
            _free_gpu_memory()

        free_vram = _available_vram_gb()
        logger.info(
            "[ModelSelection] Retry=%s | modality=%s | needs_vision=%s | "
            "free_vram=%.2f GB | excluded=%s | failed_families=%s",
            is_retry, self.modality, needs_vision, free_vram,
            self.excluded, self.failed_families,
        )

        # --- 2. Build candidate list ---
        candidates: list[ModelCandidate] = []
        for (mid, family, vision, fp16, b4) in _CATALOGUE:
            if mid in self.excluded:
                continue
            if family in self.failed_families:
                continue
            # Hard block: vision task → text-only model is not allowed
            if needs_vision and not vision:
                logger.warning(
                    "[ModelSelection] Skipping text-only model %s — "
                    "modality '%s' requires a vision-capable model.",
                    mid, self.modality,
                )
                continue
            candidates.append(ModelCandidate(mid, family, vision, fp16, b4))

        if not candidates:
            raise RuntimeError(
                f"[ModelSelection] No valid candidates for modality='{self.modality}' "
                f"with excluded={self.excluded}. "
                "Cannot fall back to a text-only model for a vision task."
            )

        # --- 3. Pick best fit within available VRAM ---
        # Prefer fp16 first; fall back to 4-bit if needed.
        # Among fitting models, prefer vision-capable & smaller VRAM footprint.
        def _score(c: ModelCandidate) -> tuple:
            fits_fp16 = free_vram >= c.vram_fp16 + 1.0   # +1 GB headroom
            fits_4bit = free_vram >= c.vram_4bit + 1.0
            # (prefer fp16, prefer vision, prefer lower vram)
            return (
                not fits_fp16 and not fits_4bit,  # False = fits = better
                not c.vision,                      # False = vision = better
                c.vram_4bit if not fits_fp16 else c.vram_fp16,
            )

        candidates.sort(key=_score)
        chosen = candidates[0]

        # Decide quantisation
        if free_vram < chosen.vram_fp16 + 1.0 and free_vram >= chosen.vram_4bit + 1.0:
            chosen.use_4bit = True
            logger.info(
                "[ModelSelection] Selected %s (family=%s) in 4-bit quantisation "
                "— fp16 requires %.1f GB but only %.2f GB free.",
                chosen.model_id, chosen.family, chosen.vram_fp16, free_vram,
            )
        elif free_vram < chosen.vram_4bit + 1.0:
            logger.warning(
                "[ModelSelection] WARNING: best candidate %s needs %.1f GB (4-bit) "
                "but only %.2f GB free — will attempt anyway; may OOM.",
                chosen.model_id, chosen.vram_4bit, free_vram,
            )
            chosen.use_4bit = True
        else:
            logger.info(
                "[ModelSelection] Selected %s (family=%s) in fp16 "
                "— %.1f GB required, %.2f GB free.",
                chosen.model_id, chosen.family, chosen.vram_fp16, free_vram,
            )

        return chosen
