"""
agents/modality_discovery_agent.py

ModalityDiscoveryAgent — detects medical imaging modality from image + question.
Uses PhiData Agent with Groq LLM (vision-capable model for the fallback path).
Same structure as reference code.

Fixes (this revision):
  - [BUG FIX] is_grayscale now inspects actual per-pixel channel equality for
    RGB-mode images instead of trusting PIL `mode`. Medical images are almost
    always saved as 3-channel RGB even though every pixel is R=G=B, so the old
    `img.mode in ("L","1","LA")` check was False for nearly all real inputs —
    silently disabling the entire pixel-based heuristic.
  - [BUG FIX] CT and MRI no longer receive an identical score bonus for the
    same pixel-brightness bucket. The old tie was always broken by Python
    dict-iteration order, which silently favored CT for every ambiguous
    grayscale image (including MRIs and borderline X-rays). Added an
    edge-sharpness feature (bone/skeletal high-contrast edges are far more
    prominent in X-Ray/CT than in MRI) to separate the two instead of a
    pure tie, and ties that remain return "Unknown" rather than an arbitrary
    pick.
  - [BUG FIX] LLM fallback is now actually multimodal — the image is attached
    to the Groq agent call (vision-capable model) instead of asking a
    text-only model to guess modality from a few numbers (which had also
    been silently fed the wrong is_grayscale value).
  - Heuristic still covers pneumonia, infection, consolidation → X-Ray.
  - Groq prompt is short and direct (no multi-line JSON blocks that confuse LLM).
  - Heuristic result is used directly when confidence >= 0.3 (no LLM needed).
  - LLM called only for low-confidence ambiguous cases.
"""

from phi.agent import Agent
from phi.model.groq import Groq

from typing import Dict, Any, Tuple
import json, re, logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Keyword map for heuristic modality detection ──────────────────────────────
MODALITY_KEYWORDS: Dict[str, list] = {
    "X-Ray": [
        "xray", "x-ray", "x ray", "chest", "radiograph", "cxr",
        "lung", "lungs", "pulmonary", "pneumonia", "pneumothorax",
        "pleural", "effusion", "consolidation", "infiltrate",
        "atelectasis", "cardiomegaly", "fracture", "bone", "rib",
        "clavicle", "trachea", "diaphragm", "mediastinum",
    ],
    "CT": [
        "ct ", "ct-", "computed tomography", "axial", "sagittal",
        "coronal", "hounsfield", "ct scan", "ctscan",
        "abdomen ct", "brain ct", "chest ct", "pelvis ct",
        "nodule", "mass", "tumor", "lesion ct",
    ],
    "MRI": [
        "mri", "magnetic resonance", "t1", "t2", "flair",
        "dwi", "adc", "mrcp", "brain mri", "spine mri",
        "knee mri", "shoulder mri", "signal", "hyperintense",
        "hypointense", "gadolinium",
    ],
    "Pathology": [
        "pathology", "histology", "histopathology", "biopsy",
        "slide", "h&e", "haematoxylin", "eosin", "stain",
        "tissue", "microscop", "cell", "carcinoma", "adenoma",
        "neoplasm", "mitosis", "nucleus", "cytology",
    ],
    "Fundus": [
        "fundus", "retina", "retinal", "optic", "optic disc",
        "macula", "fovea", "diabetic retinopathy", "glaucoma",
        "drusen", "artery", "vein", "cup", "disc",
    ],
    "Ultrasound": [
        "ultrasound", "us ", "echography", "sonograph",
        "doppler", "gallbladder", "liver us", "thyroid us",
        "obstetric", "fetal", "echocardiogram", "cardiac echo",
    ],
    "Dermoscopy": [
        "dermoscop", "dermoscopy", "skin", "lesion", "melanoma",
        "nevus", "mole", "dermatology", "pigment",
    ],
    "Endoscopy": [
        "endoscop", "colonoscop", "gastroscop", "polyp",
        "colon", "intestin", "stomach", "esophag", "barrett",
    ],
}

# Vision-capable Groq model for the multimodal fallback path.
# (llama-3.1-8b-instant is text-only and cannot see the image.)
VISION_MODEL_ID = "llama-3.2-11b-vision-preview"


class ModalityDiscoveryAgent:
    """
    Detects the medical imaging modality (X-Ray, CT, MRI, Pathology, etc.)
    from the input image pixel statistics and question text.

    Strategy:
      1. Extract image pixel stats (true grayscale check, mean, edge sharpness).
      2. Run keyword + pixel heuristic → fast, no API call.
      3. If heuristic confidence >= 0.3 → use heuristic directly.
      4. If confidence < 0.3 → ask a vision-capable Groq LLM, passing the
         actual image, for a final decision.
    """

    def __init__(self, model_id: str = VISION_MODEL_ID):
        self.agent = Agent(
            name="ModalityDiscoveryAgent",
            model=Groq(id=model_id),
            instructions=[
                "You are a medical imaging expert.",
                "You will be shown a medical image plus a clinical question.",
                "Look at the actual image content (texture, contrast, cross-sectional "
                "vs. projection geometry, bone brightness, soft-tissue detail) to decide "
                "the imaging modality. Do not just guess from the question text.",
                "Choose ONLY from: X-Ray, CT, MRI, Pathology, Fundus, Ultrasound, Dermoscopy, Endoscopy, Unknown.",
                "Reply with ONLY a JSON object like: {\"modality\": \"X-Ray\", \"confidence\": 0.9}",
                "Do not add any explanation or text outside the JSON.",
                "Do not write or execute code.",
            ],
            show_tool_calls=False,
            markdown=False,
        )

    # ── Public method ─────────────────────────────────────────────────────────

    def discover_modality(self, image_path: str, question: str) -> Dict[str, Any]:
        """
        Detect imaging modality from image + question.

        Args:
            image_path : Path to the medical image file.
            question   : Clinical question about the image.

        Returns:
            Dict with 'modality' (str) and 'confidence' (float 0-1).
        """
        # Step 1: get image pixel statistics
        stats = self._get_image_stats(image_path)
        logger.info(f"[Modality] Image stats: {stats}")

        # Step 2: keyword + pixel heuristic
        heuristic, conf = self._heuristic_modality(stats, question)
        logger.info(f"[Modality] Heuristic: {heuristic} (conf={conf:.2f})")

        # Step 3: if heuristic is confident enough → use directly, skip LLM
        if conf >= 0.3:
            logger.info(f"[Modality] High confidence heuristic — skipping LLM call.")
            return {"modality": heuristic, "confidence": round(conf, 2)}

        # Step 4: low confidence → ask Groq vision LLM, with the actual image
        logger.info(f"[Modality] Low confidence — asking Groq vision LLM.")
        return self._ask_llm(image_path, stats, question, heuristic, conf)

    # ── Image stats ───────────────────────────────────────────────────────────

    def _get_image_stats(self, image_path: str) -> Dict[str, Any]:
        """Extract pixel statistics from the image, including a *real*
        grayscale check (per-pixel channel equality) and a simple edge-
        sharpness feature used to separate X-Ray/CT (high-contrast bone
        edges) from MRI (smoother soft-tissue contrast)."""
        if not image_path or not Path(image_path).exists():
            return {
                "found": False,
                "is_grayscale": False,
                "mean_pixel": 128,
                "channels": 3,
                "width": 0,
                "height": 0,
                "edge_strength": 0.0,
            }
        try:
            from PIL import Image
            import numpy as np

            img = Image.open(image_path)
            arr = np.array(img)

            # --- Real grayscale detection -------------------------------------
            # PIL `mode` reflects file color space, not visual content. Medical
            # images are very commonly stored as 3-channel RGB with R=G=B per
            # pixel. Detect that case explicitly instead of trusting `mode`.
            if img.mode in ("L", "1", "LA"):
                is_grayscale = True
            elif img.mode == "RGB" and arr.ndim == 3 and arr.shape[-1] == 3:
                # Allow small JPEG-compression tolerance between channels.
                diff_rg = np.abs(arr[..., 0].astype(int) - arr[..., 1].astype(int))
                diff_gb = np.abs(arr[..., 1].astype(int) - arr[..., 2].astype(int))
                is_grayscale = bool(diff_rg.mean() < 2.0 and diff_gb.mean() < 2.0)
            else:
                is_grayscale = False

            # --- Edge-sharpness feature ----------------------------------------
            # Crude but cheap: mean absolute gradient on a single luminance
            # channel. X-Ray/CT typically show sharp, high-contrast bone edges;
            # MRI soft tissue contrast is comparatively smooth. This is used
            # only as a tie-breaker, not a definitive classifier.
            gray = arr.mean(axis=-1) if arr.ndim == 3 else arr.astype(float)
            gx = np.abs(np.diff(gray, axis=1)).mean() if gray.shape[1] > 1 else 0.0
            gy = np.abs(np.diff(gray, axis=0)).mean() if gray.shape[0] > 1 else 0.0
            edge_strength = round(float((gx + gy) / 2), 2)

            return {
                "found":        True,
                "width":        img.width,
                "height":       img.height,
                "mode":         img.mode,
                "channels":     len(img.getbands()),
                "mean_pixel":   round(float(arr.mean()), 2),
                "std_pixel":    round(float(arr.std()),  2),
                "is_grayscale": is_grayscale,
                "edge_strength": edge_strength,
            }
        except Exception as e:
            logger.warning(f"[Modality] Image stats error: {e}")
            return {"found": False, "is_grayscale": False, "mean_pixel": 128, "edge_strength": 0.0}

    # ── Heuristic ─────────────────────────────────────────────────────────────

    def _heuristic_modality(
        self, stats: Dict, question: str
    ) -> Tuple[str, float]:
        """
        Score each modality using keyword matching on the question
        and pixel statistics from the image.
        Returns (best_modality, confidence_score).
        """
        scores: Dict[str, float] = {m: 0.0 for m in MODALITY_KEYWORDS}
        q = question.lower()

        # Keyword matching on question text
        for modality, keywords in MODALITY_KEYWORDS.items():
            for kw in keywords:
                if kw in q:
                    scores[modality] += 0.25

        # Pixel statistics boost
        is_gray       = stats.get("is_grayscale", False)
        mean_px       = stats.get("mean_pixel",   128)
        edge_strength = stats.get("edge_strength", 0.0)

        # Grayscale + dark background → very likely X-Ray
        if is_gray and mean_px < 100:
            scores["X-Ray"] += 0.30

        # Grayscale + medium brightness → CT or MRI; use edge sharpness to
        # split the tie instead of awarding both the same bonus.
        if is_gray and 80 <= mean_px <= 180:
            # Higher edge_strength → sharper high-contrast structures (bone,
            # CT slice boundaries) → favor CT. Lower → smoother soft-tissue
            # contrast typical of MRI.
            if edge_strength >= 12.0:
                scores["CT"] += 0.22
                scores["MRI"] += 0.08
            elif edge_strength <= 6.0:
                scores["MRI"] += 0.22
                scores["CT"] += 0.08
            else:
                # Genuinely ambiguous on pixels alone — small, equal nudge.
                # Combined with keyword scoring this rarely ends in an exact
                # tie; if it still does, _heuristic_modality below returns
                # "Unknown" rather than silently picking one.
                scores["CT"] += 0.10
                scores["MRI"] += 0.10

        # Colour image → likely Pathology, Fundus, or Dermoscopy
        if not is_gray and stats.get("channels", 3) == 3:
            scores["Pathology"]  += 0.05
            scores["Fundus"]     += 0.05
            scores["Dermoscopy"] += 0.05

        # Pick best, but detect unresolved ties among the top score instead
        # of silently relying on dict iteration order (which previously
        # always favored "CT" over "MRI" on exact ties).
        best_val = max(scores.values())
        top = [m for m, v in scores.items() if v == best_val]

        if best_val < 0.1:
            return "Unknown", 0.0

        if len(top) > 1:
            logger.info(f"[Modality] Heuristic tie among {top} at {best_val:.2f} — treating as low confidence.")
            return "Unknown", 0.0

        best = top[0]
        conf = min(best_val, 1.0)
        return best, conf

    # ── LLM call ─────────────────────────────────────────────────────────────

    def _ask_llm(
        self,
        image_path: str,
        stats: Dict,
        question: str,
        heuristic: str,
        heuristic_conf: float,
    ) -> Dict[str, Any]:
        """
        Ask a vision-capable Groq LLM to determine modality when the heuristic
        is not confident. The image itself is attached so the model is
        actually looking at pixel content, not guessing from a text summary.
        """
        stats_line = (
            f"grayscale={stats.get('is_grayscale', False)}, "
            f"mean_pixel={stats.get('mean_pixel', 128)}, "
            f"edge_strength={stats.get('edge_strength', 0.0)}, "
            f"channels={stats.get('channels', 3)}, "
            f"size={stats.get('width', 0)}x{stats.get('height', 0)}"
        )

        prompt = (
            f"Medical imaging modality detection task.\n\n"
            f"Image stats (for reference only — look at the attached image itself "
            f"as the primary evidence): {stats_line}\n"
            f"Clinical question: {question}\n"
            f"Preliminary heuristic guess: {heuristic} (confidence {heuristic_conf:.2f})\n\n"
            f"What is the most likely medical imaging modality?\n"
            f"Choose one: X-Ray, CT, MRI, Pathology, Fundus, Ultrasound, Dermoscopy, Endoscopy, Unknown\n\n"
            f'Reply with ONLY this JSON: {{"modality": "<name>", "confidence": <0.0-1.0>}}'
        )

        try:
            if image_path and Path(image_path).exists():
                # phi Agent.run supports passing images for vision-capable models.
                response = self.agent.run(prompt, images=[image_path])
            else:
                response = self.agent.run(prompt)
            return self._parse_response(response, heuristic, heuristic_conf)
        except Exception as e:
            logger.warning(f"[Modality] LLM call failed: {e}. Using heuristic.")
            return {"modality": heuristic, "confidence": round(heuristic_conf, 2)}

    # ── Response parser ───────────────────────────────────────────────────────

    def _parse_response(
        self, response, fallback: str, fallback_conf: float
    ) -> Dict[str, Any]:
        """Parse Groq JSON response safely."""
        VALID_MODALITIES = {
            "X-Ray", "CT", "MRI", "Pathology",
            "Fundus", "Ultrasound", "Dermoscopy", "Endoscopy", "Unknown",
        }
        try:
            text  = response.content if hasattr(response, "content") else str(response)
            match = re.search(r'\{[^{}]*\}', text)
            if match:
                data     = json.loads(match.group())
                modality = data.get("modality", fallback)
                conf     = float(data.get("confidence", fallback_conf))

                # Validate modality is one of the expected values
                if modality not in VALID_MODALITIES:
                    logger.warning(
                        f"[Modality] LLM returned unexpected modality '{modality}'. "
                        f"Using fallback '{fallback}'."
                    )
                    modality = fallback

                return {"modality": modality, "confidence": round(conf, 2)}
        except Exception as e:
            logger.warning(f"[Modality] Response parse failed: {e}")

        return {"modality": fallback, "confidence": round(fallback_conf, 2)}


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
    )

    agent = ModalityDiscoveryAgent()

    # Test cases
    tests = [
        ("image.jpg",  "Is there pneumonia?"),
        ("image.jpg",  "Is there consolidation in the right lower lobe?"),
        ("image.jpg",  "What organ is visible in this MRI scan?"),
        ("image.jpg",  "Is there a tumor in the CT scan?"),
        ("image.jpg",  "What abnormality is visible in this pathology slide?"),
    ]

    print("\n" + "="*55)
    print("  ModalityDiscoveryAgent — Test Results")
    print("="*55)
    for img, q in tests:
        result = agent.discover_modality(img, q)
        print(f"\n  Q        : {q}")
        print(f"  Modality : {result['modality']}")
        print(f"  Conf     : {result['confidence']:.2f}")
    print("="*55)
