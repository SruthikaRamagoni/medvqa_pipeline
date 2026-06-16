#!/usr/bin/env python3
"""
inference.py — Standalone inference for the trained Medical VQA model.

Loads a fine-tuned LoRA checkpoint and generates a free-text answer
for any image + question pair.

Usage (CLI):
    python inference.py \
        --image path/to/xray.jpg \
        --question "Is there pneumonia?" \
        --checkpoint ./artifacts/checkpoints/Qwen2.5-VL-3B

    python inference.py \
        --image img.png \
        --question "What organ is shown?" \
        --model-id google/flan-t5-base      # use base model, no checkpoint

Python API:
    from inference import run_inference
    answer = run_inference(
        image_path="chest.jpg",
        question="Is there cardiomegaly?",
        checkpoint_path="./artifacts/checkpoints/...",
        model_hf_id="Qwen/Qwen2.5-VL-3B-Instruct",
    )
"""

from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)


# ── Main API ──────────────────────────────────────────────────────────────────

def run_inference(
    image_path: str,
    question: str,
    checkpoint_path: str = "",
    model_hf_id: str = "",
    device: str = "",
    max_new_tokens: int = 128,
    temperature: float = 0.3,
    do_sample: bool = True,
) -> str:
    """
    Generate a natural-language answer for image + question.

    Args:
        image_path:      Path to the medical image (jpg / png / dcm).
        question:        Clinical question string.
        checkpoint_path: Path to fine-tuned LoRA adapter directory (optional).
        model_hf_id:     HuggingFace base model ID (auto-resolved from checkpoint if empty).
        device:          'cuda' | 'cpu' | 'mps' | '' (auto-detect).
        max_new_tokens:  Max tokens to generate.
        temperature:     Sampling temperature.
        do_sample:       Whether to sample; False = greedy decoding.

    Returns:
        Generated answer string.
    """
    import torch

    # ── Device ───────────────────────────────────────────────────────────────
    if not device:
        if torch.cuda.is_available():          device = "cuda"
        elif torch.backends.mps.is_available():device = "mps"
        else:                                  device = "cpu"
    logger.info(f"[Inference] device={device}")

    # ── Resolve model ID ─────────────────────────────────────────────────────
    hf_id = model_hf_id or _resolve_hf_id(checkpoint_path, device)
    logger.info(f"[Inference] base model={hf_id}")

    # ── Load model + processor ────────────────────────────────────────────────
    model, processor = _load_model(hf_id, device)

    # ── Apply LoRA adapters ───────────────────────────────────────────────────
    if checkpoint_path and Path(checkpoint_path).exists():
        model = _apply_lora(model, checkpoint_path)

    model.eval()

    # ── Load image ────────────────────────────────────────────────────────────
    image = None
    if image_path and Path(image_path).exists():
        try:
            from PIL import Image
            image = Image.open(image_path).convert("RGB")
            logger.info(f"[Inference] image loaded: {image_path} {image.size}")
        except Exception as e:
            logger.warning(f"[Inference] Could not load image: {e}")

    # ── Build prompt ──────────────────────────────────────────────────────────
    prompt = _build_prompt(question, hf_id, has_image=image is not None)

    # ── Generate ──────────────────────────────────────────────────────────────
    answer = _generate(model, processor, prompt, image, device,
                        max_new_tokens, temperature, do_sample)
    logger.info(f"[Inference] answer: {answer}")
    return answer


# ── Batch inference ───────────────────────────────────────────────────────────

def run_batch_inference(
    samples: list,
    checkpoint_path: str = "",
    model_hf_id: str = "",
    device: str = "",
    max_new_tokens: int = 128,
) -> list:
    """
    Run inference on a list of {'image_path': str, 'question': str} dicts.
    Loads model once, iterates over samples.
    """
    import torch
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    hf_id = model_hf_id or _resolve_hf_id(checkpoint_path, device)
    model, processor = _load_model(hf_id, device)
    if checkpoint_path and Path(checkpoint_path).exists():
        model = _apply_lora(model, checkpoint_path)
    model.eval()

    results = []
    for s in samples:
        img_path = s.get("image_path", "")
        question = s.get("question", "")
        image = None
        if img_path and Path(img_path).exists():
            from PIL import Image
            image = Image.open(img_path).convert("RGB")

        prompt = _build_prompt(question, hf_id, has_image=image is not None)
        answer = _generate(model, processor, prompt, image, device,
                            max_new_tokens, 0.3, False)
        results.append({"image_path": img_path, "question": question, "answer": answer})

    return results


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_hf_id(checkpoint_path: str, device: str) -> str:
    """Read hf_id from checkpoint model_plan.json, or choose a safe default."""
    if checkpoint_path:
        plan_file = Path(checkpoint_path) / "model_plan.json"
        if plan_file.exists():
            try:
                return json.load(open(plan_file)).get("hf_id", "")
            except Exception:
                pass
    import torch
    vram = 0.0
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    if device == "cuda" and vram >= 6:
        return "Qwen/Qwen2.5-VL-3B-Instruct"
    return "google/flan-t5-base"   # always safe CPU fallback


def _load_model(hf_id: str, device: str):
    import torch
    from transformers import AutoTokenizer
    dtype = torch.float16 if device == "cuda" else torch.float32
    load_kw = dict(pretrained_model_name_or_path=hf_id, trust_remote_code=True, torch_dtype=dtype)
    if device == "cuda":
        load_kw["device_map"] = "auto"

    model = None
    for cls_name in ["AutoModelForVision2Seq", "AutoModelForCausalLM", "AutoModelForSeq2SeqLM"]:
        try:
            import transformers as tf
            cls = getattr(tf, cls_name, None)
            if cls:
                model = cls.from_pretrained(**load_kw)
                logger.info(f"[Inference] Loaded with {cls_name}")
                break
        except Exception:
            continue

    if model is None:
        raise RuntimeError(f"Cannot load model: {hf_id}")

    if device not in ("cuda",):
        try: model = model.to(device)
        except Exception: pass

    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
    except Exception:
        processor = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)

    if hasattr(processor, "pad_token") and processor.pad_token is None:
        processor.pad_token = processor.eos_token

    return model, processor


def _apply_lora(model, checkpoint_path: str):
    try:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, checkpoint_path)
        model = model.merge_and_unload()
        logger.info(f"[Inference] LoRA merged from {checkpoint_path}")
    except ImportError:
        logger.warning("[Inference] PEFT not installed; using base model.")
    except Exception as e:
        logger.warning(f"[Inference] LoRA load failed ({e}); using base model.")
    return model


def _build_prompt(question: str, hf_id: str, has_image: bool) -> str:
    hid = hf_id.lower()
    if "qwen" in hid:
        if has_image:
            return (f"<|im_start|>user\n"
                    f"<|vision_start|><|image_pad|><|vision_end|>"
                    f"{question}<|im_end|>\n<|im_start|>assistant\n")
        return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    if "phi" in hid:
        return f"<|user|>\n{question}<|end|>\n<|assistant|>\n"
    if "llava" in hid or "blip" in hid:
        return f"Question: {question}\nAnswer:"
    if "mistral" in hid:
        return f"[INST] {question} [/INST]"
    return f"Question: {question}\nAnswer:"


def _generate(model, processor, prompt, image, device,
              max_new_tokens, temperature, do_sample) -> str:
    import torch
    with torch.no_grad():
        try:
            if image is not None and hasattr(processor, "image_processor"):
                if hasattr(processor, "apply_chat_template"):
                    msgs = [{"role": "user", "content": [
                        {"type": "image"}, {"type": "text", "text": prompt}
                    ]}]
                    text   = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                    inputs = processor(text=text, images=[image], return_tensors="pt").to(device)
                else:
                    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
                out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                      temperature=temperature if do_sample else 1.0,
                                      do_sample=do_sample)
                gen = out[0][inputs["input_ids"].shape[-1]:]
                tok = getattr(processor, "tokenizer", processor)
                return tok.decode(gen, skip_special_tokens=True).strip()
            else:
                tok    = getattr(processor, "tokenizer", processor)
                inputs = tok(prompt, return_tensors="pt",
                             truncation=True, max_length=512).to(device)
                out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                      temperature=temperature if do_sample else 1.0,
                                      do_sample=do_sample,
                                      pad_token_id=getattr(tok, "eos_token_id", 1))
                gen = out[0][inputs["input_ids"].shape[-1]:]
                return tok.decode(gen, skip_special_tokens=True).strip()
        except Exception as e:
            logger.error(f"[Inference] Generation error: {e}")
            return f"[Error: {e}]"


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Medical VQA Inference")
    p.add_argument("--image",       required=True)
    p.add_argument("--question",    required=True)
    p.add_argument("--checkpoint",  default="")
    p.add_argument("--model-id",    default="")
    p.add_argument("--device",      default="", choices=["","cuda","cpu","mps"])
    p.add_argument("--max-tokens",  type=int,   default=128)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--greedy",      action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    args = _parse_args()
    answer = run_inference(
        image_path=args.image,
        question=args.question,
        checkpoint_path=args.checkpoint,
        model_hf_id=args.model_id,
        device=args.device,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        do_sample=not args.greedy,
    )
    print(f"\n{'='*55}")
    print(f"  Question : {args.question}")
    print(f"  Answer   : {answer}")
    print(f"{'='*55}\n")
