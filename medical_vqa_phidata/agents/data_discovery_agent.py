"""
agents/data_discovery_agent.py

DataDiscoveryAgent — searches HuggingFace and local directories for
the best Medical VQA dataset matching the detected modality and the
available compute resources.

REDESIGN NOTES
--------------
The previous version selected `flaviagiammarino/vqa-rad` for almost every
modality (X-Ray, MRI, CT, ...) regardless of what was actually detected.
The root cause was NOT a tuning problem in the scoring weights — it was
that the dataset catalogue itself (config.settings.FREE_DATASETS) tags
vqa-rad with every modality at once:

    "tags": ["medical", "radiology", "vqa", "ct", "mri", "chest", "x-ray"]

Combined with vqa-rad also having by far the largest sample count among
the vision-capable candidates, ANY modality string trivially matched its
tag list, plus a flat "vqa"/"medical" bonus applied to every candidate
indiscriminately, plus a large size bonus. The result mathematically
favored vqa-rad regardless of the real modality, and the LLM step added
no real agency on top of that because it was only ever shown the same
pre-biased top-3 to rubber-stamp.

This revision changes the agent's actual behavior, not just its weights:

1.  MODALITY SPECIFICITY, not blanket tag membership. A dataset that
    lists every modality is treated as a generalist (lower confidence
    match) relative to a dataset that lists ONLY the requested modality
    (a specialist, higher confidence match) — see _modality_specificity().
    This directly neutralizes the catch-all-tagging problem instead of
    re-weighting around it.

2.  RESOURCE-AWARE FILTERING. The agent accepts detected hardware
    (vram_gb, ram_gb) and downweights/filters candidates whose size
    implies a feature-engineering/training memory footprint poor for the
    detected resources, mirroring how ModelSelectionAgent reasons about
    hardware for model choice. This is a soft preference (resource fit),
    not a hard requirement unless nothing fits at all.

3.  REAL LLM AGENCY. The LLM is given the full reasoning (modality
    specificity scores + resource fit) for the top candidates and asked
    to make an actual judgment call with that context, rather than being
    shown an already-collapsed top-3 with no differentiating signal.

4.  TRANSPARENT FALLBACK. If the LLM's pick can't be resolved, or no
    candidate is a plausible modality match at all, this is logged
    explicitly (not silently defaulted to whatever sorted first).
"""

from phi.agent import Agent
from phi.model.groq import Groq

from typing import Dict, Any, List, Optional
import json, logging, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import FREE_DATASETS

logger = logging.getLogger(__name__)

LOCAL_SEARCH_ROOTS = ["./data", "/content/data"]

# Canonical modality vocabulary. Used to detect when a dataset's tag list
# spans many/most modalities at once (a "generalist" entry) vs. genuinely
# specializing in the requested one. Extend this list if new modalities
# are introduced elsewhere in the pipeline (e.g. ModalityDiscoveryAgent).
KNOWN_MODALITIES = {
    "x-ray", "xray", "ct", "mri", "ultrasound", "pathology",
    "histology", "microscopy", "dermatology", "fundus", "endoscopy",
    "mammography", "pet", "chest",
}

# Rough per-1000-sample memory footprint heuristic (GB) for feature
# engineering a vision dataset at typical patch counts. This is
# deliberately coarse — it exists only to break ties / down-rank
# obviously oversized datasets on obviously constrained hardware, not to
# make a precise prediction. ModelSelectionAgent is the source of truth
# for actual model-level VRAM requirements; this is dataset-level only.
EST_GB_PER_1000_SAMPLES = 0.35


class DataDiscoveryAgent:
    """
    Discovers and selects the best Medical VQA dataset for the detected
    imaging modality, reasoning explicitly about (a) how specifically a
    candidate dataset matches the requested modality rather than trusting
    its tag list at face value, and (b) whether the candidate's scale is
    a reasonable fit for the detected compute resources.
    """

    def __init__(self, model_id: str = "mistral"):
        self.agent = Agent(
            name="DataDiscoveryAgent",
            model=Groq(id="llama-3.1-8b-instant"),
            instructions=[
                "You are a medical AI data expert agent.",
                "Your task is to select the single best dataset for training a "
                "Medical Visual Question Answering (Medical VQA) model.",
                "You will be given candidates with a precomputed modality_specificity "
                "score (1.0 = dataset specializes in exactly the requested modality, "
                "lower = the dataset is a generalist covering many modalities or only "
                "loosely related) and a resource_fit assessment for the detected hardware.",
                "Prefer higher modality_specificity over raw sample count — a smaller "
                "dataset that genuinely specializes in the requested modality is a "
                "better choice than a larger generalist dataset that merely happens to "
                "list the modality among many others.",
                "Only deprioritize a high-specificity candidate for resource reasons if "
                "its resource_fit is explicitly marked poor for the detected hardware.",
                "Use only the dataset candidates provided in the prompt.",
                "Do not execute code. Do not call any external tools or functions.",
                "Do not generate scripts, files, or calculations.",
                "Reason internally and return only the final decision.",
                "Return ONLY valid JSON.",
                "Do not include markdown, explanations, comments, code blocks, or extra text.",
                "Output schema exactly:",
                '{"selected_dataset":"<dataset_name>","reason":"<one sentence>"}',
            ],
            show_tool_calls=False,
            markdown=False,
        )

    # ── Public method ─────────────────────────────────────────────────────────

    def discover_dataset(
        self,
        modality: str,
        vram_gb: Optional[float] = None,
        ram_gb: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Find the best available Medical VQA dataset for the given modality
        and (optionally) the detected compute resources.

        Args:
            modality : Detected imaging modality (e.g., 'X-Ray', 'CT', 'MRI', 'Pathology').
            vram_gb  : Detected GPU VRAM in GB, if known (0 or None if CPU-only).
            ram_gb   : Detected system RAM in GB, if known.

        Returns:
            Dict with dataset name, source, modality match diagnostics,
            and selection details — including how the pick was made
            (llm | heuristic_fallback) so callers/logs can tell whether
            the LLM's judgment was actually used.
        """
        if vram_gb is None or ram_gb is None:
            vram_gb, ram_gb = self._detect_hardware(vram_gb, ram_gb)

        local     = self._scan_local(modality)
        all_cands = local + FREE_DATASETS

        scored = self._score_candidates(all_cands, modality, vram_gb, ram_gb)

        if not scored:
            logger.error(
                f"[DataDiscovery] No candidates available at all for "
                f"modality='{modality}'. Falling back to first FREE_DATASETS "
                f"entry as a last resort — this is NOT a modality match."
            )
            fallback = FREE_DATASETS[0]
            return self._build_result(fallback, modality, vram_gb, ram_gb,
                                       pick_method="emergency_fallback",
                                       reason="No scoreable candidates; used first configured dataset as a last resort.")

        top_n = scored[:3]
        best_specificity = top_n[0]["modality_specificity"]

        if best_specificity < 0.34:
            logger.warning(
                f"[DataDiscovery] Best modality_specificity for "
                f"modality='{modality}' is only {best_specificity:.2f} "
                f"(top candidate='{top_n[0]['name']}') — no strong "
                f"modality-specific dataset was found in FREE_DATASETS. "
                f"Proceeding with the best available match, but this "
                f"choice should be reviewed; consider adding a "
                f"modality-specific dataset to config.settings.FREE_DATASETS."
            )

        candidate_block = "\n".join(
            f"{i+1}. {c['name']} | modality_specificity={c['modality_specificity']:.2f} "
            f"| resource_fit={c['resource_fit']} | size={c.get('size', 0)} "
            f"| tags={c.get('tags', [])}"
            for i, c in enumerate(top_n)
        )

        prompt = f"""
        Requested modality: {modality}
        Detected hardware: VRAM={vram_gb:.1f}GB, RAM={ram_gb:.1f}GB

        Top candidates (already filtered/ranked by modality specificity and resource fit):
        {candidate_block}

        Select the best dataset for Medical VQA given the modality and hardware above.

        Reply with ONLY this JSON (no extra text):
        {{"selected_dataset": "<dataset_name>", "reason": "<one sentence>"}}
        """

        llm_pick_name = ""
        try:
            response = self.agent.run(prompt)
            llm_pick_name = self._parse_response(response)
        except Exception as e:
            logger.warning(f"[DataDiscovery] LLM selection call failed: {e}")

        chosen, pick_method = self._resolve_pick(llm_pick_name, top_n, all_cands)

        return self._build_result(
            chosen, modality, vram_gb, ram_gb,
            pick_method=pick_method,
            reason=(
                f"LLM-selected match for modality='{modality}' "
                f"(specificity={chosen.get('modality_specificity', 0):.2f}, "
                f"resource_fit={chosen.get('resource_fit', 'unknown')})."
                if pick_method == "llm" else
                f"Heuristic top match for modality='{modality}' "
                f"(LLM pick unavailable or unresolvable) "
                f"(specificity={chosen.get('modality_specificity', 0):.2f}, "
                f"resource_fit={chosen.get('resource_fit', 'unknown')})."
            ),
        )

    # ── Hardware detection ───────────────────────────────────────────────────

    def _detect_hardware(self, vram_gb: Optional[float], ram_gb: Optional[float]):
        """Best-effort self-detection if the caller didn't already pass
        detected hardware through (main.py typically detects this once
        and could pass it to every agent that wants it)."""
        detected_vram = vram_gb if vram_gb is not None else 0.0
        detected_ram  = ram_gb if ram_gb is not None else 0.0
        try:
            import torch
            if vram_gb is None and torch.cuda.is_available():
                detected_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        except Exception:
            pass
        try:
            if ram_gb is None:
                import psutil
                detected_ram = psutil.virtual_memory().total / (1024 ** 3)
        except Exception:
            pass
        return detected_vram, detected_ram

    # ── Modality specificity (the actual fix) ────────────────────────────────

    def _modality_specificity(self, candidate: Dict, modality: str) -> float:
        """
        Score how SPECIFICALLY a candidate dataset matches the requested
        modality, as opposed to merely whether the modality string
        appears somewhere in its tags.

        Returns 0.0 (no match at all) to 1.0 (dataset's tags name ONLY
        the requested modality among known modality tags — a genuine
        specialist). A dataset tagged with many different modalities at
        once (a generalist / catch-all entry) scores lower than a
        specialist even when both technically "contain" the requested
        modality string, which is the direct fix for vqa-rad's
        ["ct","mri","chest","x-ray", ...] tag list previously causing it
        to outscore everything for every modality.
        """
        modality_norm = modality.lower().replace(" ", "-")
        tags = [t.lower() for t in candidate.get("tags", [])]
        name = candidate["name"].lower()
        combined_tags = set(tags)

        modality_tags_present = combined_tags & KNOWN_MODALITIES
        direct_match = (
            modality_norm in combined_tags
            or modality_norm.replace("-", "") in {t.replace("-", "") for t in combined_tags}
            or modality_norm in name
        )

        if not direct_match:
            return 0.0

        if not modality_tags_present:
            # Matched only via name, no explicit modality tags at all —
            # weak, ambiguous match.
            return 0.3

        # Specificity = how much of the dataset's modality-tag footprint
        # is JUST the requested modality. 1 modality tag matching the
        # request = specialist (1.0). Many modality tags present = the
        # match is diluted across a generalist catalogue entry.
        specificity = 1.0 / len(modality_tags_present)
        return round(min(1.0, max(0.0, specificity)), 3)

    # ── Resource fit ──────────────────────────────────────────────────────────

    def _resource_fit(self, candidate: Dict, vram_gb: float, ram_gb: float) -> str:
        """
        Coarse resource-feasibility label for a candidate dataset, given
        detected hardware. This is intentionally a soft signal (used to
        break ties and warn, not to hard-exclude) since dataset size
        alone is a weak predictor of actual memory needs — model choice
        and batch size matter far more and are ModelSelectionAgent's
        responsibility. Returns one of: "good", "tight", "poor", "unknown".
        """
        size = candidate.get("size", 0)
        if size <= 0:
            return "unknown"

        est_gb = (size / 1000.0) * EST_GB_PER_1000_SAMPLES

        # No GPU detected: lean on system RAM as the constraint, since
        # feature engineering and CPU-mode training both live in RAM.
        effective_budget = ram_gb if ram_gb else 8.0  # assume a modest default if totally unknown

        if est_gb <= effective_budget * 0.25:
            return "good"
        if est_gb <= effective_budget * 0.6:
            return "tight"
        return "poor"

    def _score_candidates(
        self, candidates: List[Dict], modality: str, vram_gb: float, ram_gb: float,
    ) -> List[Dict]:
        scored = []
        for c in candidates:
            specificity = self._modality_specificity(c, modality)
            if specificity <= 0.0:
                # Not a modality match at all — exclude rather than let a
                # flat "medical"/"vqa" bonus drag an irrelevant dataset
                # into contention (this was the second half of the
                # original bug: an indiscriminate +0.3 for any
                # medical/VQA dataset regardless of modality relevance).
                continue

            fit = self._resource_fit(c, vram_gb, ram_gb)
            fit_bonus = {"good": 0.15, "tight": 0.05, "poor": -0.15, "unknown": 0.0}[fit]
            size_bonus = min(c.get("size", 0) / 50000, 0.15)
            local_bonus = 0.05 if c.get("source") == "local" else 0.0

            final_score = (specificity * 0.7) + size_bonus + fit_bonus + local_bonus

            enriched = dict(c)
            enriched["modality_specificity"] = specificity
            enriched["resource_fit"] = fit
            enriched["_score"] = final_score
            scored.append(enriched)

        return sorted(scored, key=lambda c: c["_score"], reverse=True)

    # ── Pick resolution ────────────────────────────────────────────────────────

    def _resolve_pick(self, llm_pick_name: str, top_n: List[Dict], all_cands: List[Dict]):
        """Resolve the LLM's chosen dataset name against the ranked
        candidates, falling back to the top heuristic match with an
        explicit, logged reason if the LLM's output can't be resolved."""
        if llm_pick_name:
            for c in top_n:
                if llm_pick_name.lower() in c["name"].lower():
                    return c, "llm"
            # LLM named something outside the ranked top_n — still honor
            # it if it's a real candidate, but log this since it means
            # the LLM diverged from the precomputed ranking.
            for c in all_cands:
                if llm_pick_name.lower() in c["name"].lower():
                    logger.info(
                        f"[DataDiscovery] LLM selected '{c['name']}', which "
                        f"was outside the precomputed top-3 ranking — "
                        f"honoring the LLM's choice."
                    )
                    enriched = dict(c)
                    enriched.setdefault("modality_specificity", 0.0)
                    enriched.setdefault("resource_fit", "unknown")
                    return enriched, "llm"
            logger.warning(
                f"[DataDiscovery] LLM returned '{llm_pick_name}', which did "
                f"not match any known candidate by name — falling back to "
                f"the top heuristic match."
            )

        if not top_n:
            logger.error("[DataDiscovery] No ranked candidates to fall back to.")
            return dict(all_cands[0]) if all_cands else {}, "emergency_fallback"

        return top_n[0], "heuristic_fallback"

    def _build_result(
        self, best: Dict, modality: str, vram_gb: float, ram_gb: float,
        pick_method: str, reason: str,
    ) -> Dict[str, Any]:
        logger.info(
            f"[DataDiscovery] Selected '{best['name']}' for modality="
            f"'{modality}' via {pick_method} "
            f"(specificity={best.get('modality_specificity', 'n/a')}, "
            f"resource_fit={best.get('resource_fit', 'n/a')})."
        )
        return {
            "name":                 best["name"],
            "source":               best.get("source", "huggingface"),
            "local_path":           best.get("local_path", ""),
            "size":                 best.get("size", 0),
            "modality":             modality,
            "tags":                 best.get("tags", []),
            "modality_specificity": best.get("modality_specificity", 0.0),
            "resource_fit":         best.get("resource_fit", "unknown"),
            "pick_method":          pick_method,
            "vram_gb_detected":     round(vram_gb, 2),
            "ram_gb_detected":      round(ram_gb, 2),
            "selection_reason":     reason,
        }

    # ── Local scan (unchanged behaviour) ─────────────────────────────────────

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
    agent = DataDiscoveryAgent(model_id="mistral")
    for modality in ["X-Ray", "MRI", "CT", "Pathology"]:
        result = agent.discover_dataset(modality=modality, vram_gb=0.0, ram_gb=16.0)
        print(f"\n[{modality}] ->", json.dumps(result, indent=2))
