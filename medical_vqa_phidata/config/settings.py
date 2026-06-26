"""
config/settings.py — Central configuration for Medical VQA PhiData Pipeline.
Updated to include feature engineering settings.
"""
import os

# ── Groq / LLM ────────────────────────────────────────────────────────────────
GROQ_MODEL   = os.getenv("GROQ_MODEL",   "llama-3.1-8b-instant")

# ── Data paths ────────────────────────────────────────────────────────────────
RAW_DATA_DIR       = os.getenv("RAW_DATA_DIR",       "./data/raw")
PROCESSED_DATA_DIR = os.getenv("PROCESSED_DATA_DIR", "./data/processed")
FEATURE_DATA_DIR   = os.getenv("FEATURE_DATA_DIR",   "./data/features")   # ← NEW
MAX_SAMPLES        = int(os.getenv("MAX_SAMPLES",    "5000"))
TARGET_IMAGE_SIZE  = (224, 224)
MAX_SEQ_LEN        = 1024   # Phi-3.5-vision with num_crops=1 still produces ~780 image tokens
                             # on high-res inputs (the crop count controls tiling, not resolution).
                             # 1024 leaves ~244 tokens for the answer — ample for VQA short answers.

# ── Training ──────────────────────────────────────────────────────────────────
CHECKPOINT_DIR     = os.getenv("CHECKPOINT_DIR", "./artifacts/checkpoints")
DEFAULT_LORA_R     = 8
DEFAULT_LORA_ALPHA = 16
DEFAULT_EPOCHS     = 3
DEFAULT_BATCH      = 4
DEFAULT_LR         = 2e-4
MAX_MODEL_RETRIES  = 3
# ── Evaluation ────────────────────────────────────────────────────────────────
EVAL_DIR     = os.getenv("EVAL_DIR",    "./artifacts/evaluation")
EVAL_SAMPLES = int(os.getenv("EVAL_SAMPLES", "200"))

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR   = "./logs"
LOG_FILE  = "./logs/pipeline.log"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ── Verified free HuggingFace datasets (no login required) ────────────────────
FREE_DATASETS = [
    {
        "name":   "flaviagiammarino/vqa-rad",
        "tags":   ["medical", "radiology", "vqa", "ct", "mri", "chest", "x-ray"],
        "size":   3515,
        "source": "huggingface",
    },
    {
        "name":   "flaviagiammarino/path-vqa",
        "tags":   ["medical", "pathology", "vqa", "histology", "microscopy"],
        "size":   32799,
        "source": "huggingface",
    },
    {
        "name":   "mdwiratathya/VQA-rad-en",
        "tags":   ["medical", "radiology", "vqa", "chest"],
        "size":   3000,
        "source": "huggingface",
    },
    {
        "name":   "Multimodal-Fatima/VQA_RAD_test",
        "tags":   ["medical", "radiology", "vqa"],
        "size":   451,
        "source": "huggingface",
    },
]

# ── Model catalogue (all freely accessible, no HF login required) ─────────────
MODEL_CATALOGUE = [
    {
        "name":           "Qwen2.5-VL-3B",
        "hf_id":          "Qwen/Qwen2.5-VL-3B-Instruct",
        "vision":         True,
        "params_b":       3.0,
        "min_vram":       6.0,
        "min_vram_4bit":  3.5,
        "quality":        0.85,  # best VQA fit on T4: dynamic resolution, native vision2seq,
                                  # lower VRAM than Phi-3.5, shorter encoded sequences
        "target_modules": ["q_proj", "v_proj"],
        "architecture":   "causal",
        "loader":         "AutoModelForVision2Seq",
    },
    {
        "name":           "Qwen2-VL-2B",
        "hf_id":          "Qwen/Qwen2-VL-2B-Instruct",
        "vision":         True,
        "params_b":       2.0,
        "min_vram":       5.0,
        "min_vram_4bit":  3.0,
        "quality":        0.74,
        "target_modules": ["q_proj", "v_proj"],
        "architecture":   "causal",
        "loader":         "AutoModelForVision2Seq",
    },
    {
        "name":           "Phi-3.5-vision",
        "hf_id":          "microsoft/Phi-3.5-vision-instruct",
        "vision":         True,
        "params_b":       4.2,
        "min_vram":       8.0,
        "min_vram_4bit":  4.5,
        "quality":        0.80,
        "target_modules": ["q_proj", "v_proj"],
        "architecture":   "causal",
        "loader":         "AutoModelForCausalLM",
    },
    {
        "name":           "BLIP-2-OPT-2.7B",
        "hf_id":          "Salesforce/blip2-opt-2.7b",
        "vision":         True,
        "params_b":       3.7,
        "min_vram":       8.0,
        "min_vram_4bit":  4.0,
        "quality":        0.78,
        "target_modules": ["q_proj", "v_proj"],
        "architecture":   "causal",
        "loader":         "Blip2ForConditionalGeneration",
    },
    {
        "name":           "InstructBLIP-Vicuna-7B",
        "hf_id":          "Salesforce/instructblip-vicuna-7b",
        "vision":         True,
        "params_b":       7.0,
        "min_vram":       15.5,   # FIX: was 14.0 — exactly matched T4's 14.56GB leaving
                                   # zero headroom for activations/LoRA → guaranteed OOM.
                                   # 15.5 correctly excludes this model on a single T4.
        "min_vram_4bit":  7.0,
        "quality":        0.85,
        "target_modules": ["q_proj", "v_proj"],
        "architecture":   "causal",
        "loader":         "AutoModelForVision2Seq",
    },
    {
        "name":           "LLaVA-1.5-7B",
        "hf_id":          "llava-hf/llava-1.5-7b-hf",
        "vision":         True,
        "params_b":       7.0,
        "min_vram":       15.5,   # FIX: was 14.0 — same OOM issue as InstructBLIP on T4.
        "min_vram_4bit":  7.0,
        "quality":        0.84,
        "target_modules": ["q_proj", "v_proj"],
        "architecture":   "causal",
        "loader":         "AutoModelForVision2Seq",
    },
    {
        "name":           "Flan-T5-Large",
        "hf_id":          "google/flan-t5-large",
        "vision":         False,
        "params_b":       0.78,
        "min_vram":       2.0,
        "min_vram_4bit":  1.0,
        "quality":        0.68,
        "target_modules": ["q", "v"],
        "architecture":   "seq2seq",
        "loader":         "AutoModelForSeq2SeqLM",
    },
    {
        "name":           "Flan-T5-Base",
        "hf_id":          "google/flan-t5-base",
        "vision":         False,
        "params_b":       0.25,
        "min_vram":       1.0,
        "min_vram_4bit":  0.5,
        "quality":        0.60,
        "target_modules": ["q", "v"],
        "architecture":   "seq2seq",
        "loader":         "AutoModelForSeq2SeqLM",
    },
]
