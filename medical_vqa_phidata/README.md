# Medical VQA — PhiData Multi-Agent Pipeline

Adaptive deep-learning pipeline for **Medical Visual Question Answering** built
with **official PhiData agents** backed by **local Ollama LLMs**.  
No API keys. No OpenAI. Pure local inference.

---

## Architecture

Every agent follows the exact same PhiData pattern as the reference code:

```python
self.agent = Agent(
    name="...",
    model=Ollama(id=model_id),   # local LLM, no API key
    tools=[PythonTools()],
    instructions=[...],
    show_tool_calls=True,
    markdown=False,
)
response = self.agent.run(prompt)
```

```
main.py
  │
  ├── ModalityDiscoveryAgent   → detects X-Ray / CT / MRI / Pathology …
  ├── DataDiscoveryAgent       → finds best free HuggingFace dataset
  ├── DataCollectionAgent      → downloads, unifies schema, caches JSONL
  ├── DataPreprocessingAgent   → resize, clean text, anonymize PHI
  ├── ModelSelectionAgent      → scores 8 VL models against your GPU
  ├── TrainingAgent            → PEFT/LoRA fine-tuning + checkpointing
  └── EvaluationAgent          → BLEU, ROUGE-L, Exact Match, Medical Acc
```

---

## Folder Structure

```
medical_vqa_phidata/
├── agents/
│   ├── __init__.py
│   ├── modality_discovery_agent.py
│   ├── data_discovery_agent.py
│   ├── data_collection_agent.py
│   ├── data_preprocessing_agent.py
│   ├── model_selection_agent.py
│   ├── training_agent.py
│   └── evaluation_agent.py
├── core/
│   ├── __init__.py
│   └── state.py                  ← shared PipelineState dataclass
├── config/
│   ├── __init__.py
│   └── settings.py               ← all paths, catalogues, hyperparams
├── data/
│   ├── raw/                      ← cached JSONL from HuggingFace
│   ├── processed/                ← cleaned + resized data
│   └── features/                 ← tokenized HF Dataset
├── artifacts/
│   ├── checkpoints/              ← LoRA adapter weights
│   └── evaluation/               ← evaluation_report.json
├── logs/
│   └── pipeline.log
├── main.py                       ← full pipeline entry point
├── inference.py                  ← standalone image+question → answer
├── colab_setup.py                ← Google Colab one-shot install
└── requirements.txt
```

---

## Quick Start

### 1 — Install Ollama (local LLM server)

```bash
# Linux / macOS
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral      # ~4 GB — used by all agents for reasoning

# Windows  →  https://ollama.com/download
```

### 2 — Install Python dependencies

```bash
cd medical_vqa_phidata
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3 — Run the full pipeline

```bash
python main.py --image image.jpg --question "Is there pneumonia?"
```

### 4 — Dry-run (10 samples, 1 epoch — fast test)

```bash
python main.py --image image.jpg --question "Is there pneumonia?" --dry-run
```

### 5 — Change the Ollama model used by agents

```bash
ollama pull phi3
python main.py --image image.jpg --question "..." --ollama-model phi3
```

Supported Ollama models: `mistral` · `phi3` · `llama3.2` · `qwen2.5` · `gemma3`

### 6 — Skip training, evaluate existing checkpoint

```bash
python main.py \
  --image image.jpg \
  --question "Is there pneumonia?" \
  --skip-training \
  --checkpoint ./artifacts/checkpoints/Qwen2.5-VL-3B
```

### 7 — Standalone inference

```bash
python inference.py \
  --image image.jpg \
  --question "What abnormality is visible?" \
  --checkpoint ./artifacts/checkpoints/Qwen2.5-VL-3B
```

---

## Google Colab

```python
# Cell 1
!python colab_setup.py

# Cell 2 — upload your image
from google.colab import files
files.upload()

# Cell 3 — run pipeline
!python main.py \
    --image /content/xray.jpg \
    --question "Is there pneumonia?" \
    --ollama-model mistral

# Cell 4 — inference only
!python inference.py \
    --image /content/xray.jpg \
    --question "Describe the abnormality." \
    --checkpoint ./artifacts/checkpoints/...
```

---

## Python API

```python
# Run individual agent
from agents.modality_discovery_agent import ModalityDiscoveryAgent

agent  = ModalityDiscoveryAgent(model_id="mistral")
result = agent.discover_modality("chest_xray.jpg", "Is there pneumonia?")
print(result)  # {'modality': 'X-Ray', 'confidence': 0.85}

# Run full inference
from inference import run_inference
answer = run_inference(
    image_path="chest_xray.jpg",
    question="Is there cardiomegaly?",
    checkpoint_path="./artifacts/checkpoints/Qwen2.5-VL-3B",
    device="cuda",
)
print(answer)
```

---

## Supported VQA Models (auto-selected)

| Model | Vision | Params | Min VRAM (4-bit) |
|-------|--------|--------|-----------------|
| Qwen2.5-VL-3B | ✅ | 3B | 3.5 GB |
| Qwen2-VL-2B | ✅ | 2B | 3.0 GB |
| Phi-3.5-vision | ✅ | 4.2B | 4.5 GB |
| BLIP-2-OPT-2.7B | ✅ | 3.7B | 4.0 GB |
| InstructBLIP-Vicuna-7B | ✅ | 7B | 7.0 GB |
| LLaVA-1.5-7B | ✅ | 7B | 7.0 GB |
| Flan-T5-Large | ❌ | 0.78B | 1.0 GB |
| Flan-T5-Base | ❌ | 0.25B | 0.5 GB |

All freely accessible — no HuggingFace login required.

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| BLEU-1 | Unigram precision vs reference |
| BLEU-4 | 4-gram precision |
| ROUGE-L | Longest common subsequence F1 |
| Exact Match | Normalized string equality |
| Medical Accuracy | Token overlap weighted toward clinical terms |

---

## Environment Variables

```bash
OLLAMA_MODEL=mistral          # agent reasoning LLM
MAX_SAMPLES=5000              # cap on dataset size
EVAL_SAMPLES=200              # evaluation set size
LOG_LEVEL=INFO
```
