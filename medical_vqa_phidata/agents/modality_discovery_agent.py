"""
agents/modality_discovery_agent.py

ModalityDiscoveryAgent — detects medical imaging modality from image + question.
Uses PhiData Agent with Groq LLM (llama-3.1-8b-instant).
Same structure as reference code.

Fixes:
  - Removed PythonTools (caused Groq 400 tool_use_failed loop)
  - Heuristic now covers pneumonia, infection, consolidation → X-Ray
  - Groq prompt is short and direct (no multi-line JSON blocks that confuse LLM)
  - Heuristic result is used directly when confidence >= 0.3 (no LLM needed)
  - LLM called only for low-confidence ambiguous cases
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


class ModalityDiscoveryAgent:
    """
    Detects the medical imaging modality (X-Ray, CT, MRI, Pathology, etc.)
    from the input image pixel statistics and question text.

    Strategy:
      1. Extract image pixel stats (grayscale, mean, channels).
      2. Run keyword + pixel heuristic → fast, no API call.
      3. If heuristic confidence >= 0.3 → use heuristic directly.
      4. If confidence < 0.3 → ask Groq LLM for final decision.
    """

    def __init__(self, model_id: str = "mistral"):
        self.agent = Agent(
            name="ModalityDiscoveryAgent",
            model=Groq(id="llama-3.1-8b-instant"),
            instructions=[
                "You are a medical imaging expert.",
                "Given image statistics and a clinical question, identify the imaging modality.",
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

        # Step 4: low confidence → ask Groq LLM
        logger.info(f"[Modality] Low confidence — asking Groq LLM.")
        return self._ask_llm(stats, question, heuristic, conf)

    # ── Image stats ───────────────────────────────────────────────────────────

    def _get_image_stats(self, image_path: str) -> Dict[str, Any]:
        """Extract basic pixel statistics from the image."""
        if not image_path or not Path(image_path).exists():
            return {
                "found": False,
                "is_grayscale": False,
                "mean_pixel": 128,
                "channels": 3,
                "width": 0,
                "height": 0,
            }
        try:
            from PIL import Image
            import numpy as np

            img  = Image.open(image_path)
            arr  = np.array(img)
            return {
                "found":       True,
                "width":       img.width,
                "height":      img.height,
                "mode":        img.mode,
                "channels":    len(img.getbands()),
                "mean_pixel":  round(float(arr.mean()), 2),
                "std_pixel":   round(float(arr.std()),  2),
                "is_grayscale":img.mode in ("L", "1", "LA"),
            }
        except Exception as e:
            logger.warning(f"[Modality] Image stats error: {e}")
            return {"found": False, "is_grayscale": False, "mean_pixel": 128}

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
        is_gray   = stats.get("is_grayscale", False)
        mean_px   = stats.get("mean_pixel",   128)
        channels  = stats.get("channels",     3)

        # Grayscale + dark background → very likely X-Ray
        if is_gray and mean_px < 100:
            scores["X-Ray"] += 0.30

        # Grayscale + medium brightness → could be CT or MRI
        if is_gray and 80 <= mean_px <= 180:
            scores["CT"]  += 0.10
            scores["MRI"] += 0.10

        # Colour image → likely Pathology, Fundus, or Dermoscopy
        if not is_gray and channels == 3:
            scores["Pathology"]  += 0.05
            scores["Fundus"]     += 0.05
            scores["Dermoscopy"] += 0.05

        # Pick best
        best = max(scores, key=scores.get)
        conf = min(scores[best], 1.0)

        if conf < 0.1:
            return "Unknown", 0.0

        return best, conf

    # ── LLM call ─────────────────────────────────────────────────────────────

    def _ask_llm(
        self,
        stats: Dict,
        question: str,
        heuristic: str,
        heuristic_conf: float,
    ) -> Dict[str, Any]:
        """
        Ask Groq LLM to determine modality when heuristic is not confident.
        Keeps the prompt short and direct to avoid Groq 400 errors.
        """
        # Build a short, clean stats string (no nested JSON to confuse LLM)
        stats_line = (
            f"grayscale={stats.get('is_grayscale', False)}, "
            f"mean_pixel={stats.get('mean_pixel', 128)}, "
            f"channels={stats.get('channels', 3)}, "
            f"size={stats.get('width', 0)}x{stats.get('height', 0)}"
        )

        prompt = (
            f"Medical imaging modality detection task.\n\n"
            f"Image stats: {stats_line}\n"
            f"Clinical question: {question}\n"
            f"Preliminary guess: {heuristic} (confidence {heuristic_conf:.2f})\n\n"
            f"What is the most likely medical imaging modality?\n"
            f"Choose one: X-Ray, CT, MRI, Pathology, Fundus, Ultrasound, Dermoscopy, Endoscopy, Unknown\n\n"
            f'Reply with ONLY this JSON: {{"modality": "<name>", "confidence": <0.0-1.0>}}'
        )

        try:
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