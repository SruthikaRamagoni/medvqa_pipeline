"""
core/state.py — Shared pipeline state passed between all agents.
Updated to include feature_data_path used by FeatureEngineeringAgent.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json, logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PipelineState:
    # ── Input ─────────────────────────────────────────────────────────────────
    image_path: str = ""
    question:   str = ""

    # ── Discovered ────────────────────────────────────────────────────────────
    modality:            str = ""
    dataset_name:        str = ""
    dataset_source:      str = ""   # "huggingface" | "local"
    dataset_local_path:  str = ""

    # ── Data paths ────────────────────────────────────────────────────────────
    raw_data_path:       str = ""   # DataCollectionAgent output
    processed_data_path: str = ""   # DataPreprocessingAgent output
    feature_data_path:   str = ""   # FeatureEngineeringAgent output  ← NEW
    checkpoint_path:     str = ""   # TrainingAgent output

    # ── Model info ────────────────────────────────────────────────────────────
    model_hf_id:        str   = ""
    model_name:         str   = ""
    model_architecture: str   = ""
    model_vision:       bool  = False
    lora_r:             int   = 8
    lora_alpha:         int   = 16
    batch_size:         int   = 4
    epochs:             int   = 3
    learning_rate:      float = 2e-4
    precision:          str   = "fp16"
    # Retry tracking
    retry_count:        int = 0
    previous_models:    List[str] = field(default_factory=list)
    failure_reason:     str = ""
    # ── Hardware ──────────────────────────────────────────────────────────────
    device:   str   = "cpu"
    vram_gb:  float = 0.0
    ram_gb:   float = 0.0

    # ── Evaluation metrics ────────────────────────────────────────────────────
    bleu_1:          float = 0.0
    bleu_4:          float = 0.0
    rouge_l:         float = 0.0
    exact_match:     float = 0.0
    medical_accuracy:float = 0.0

    # ── Runtime ───────────────────────────────────────────────────────────────
    logs:    List[str] = field(default_factory=list)
    errors:  List[str] = field(default_factory=list)
    success: bool      = False

    def log(self, msg: str) -> None:
        logger.info(msg)
        self.logs.append(msg)

    def save(self, path: str = "./artifacts/pipeline_state.json") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2, default=str)
        logger.info(f"State saved → {path}")
