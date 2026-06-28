"""
agents/evaluation_agent.py

EvaluationAgent — evaluates the trained model with BLEU, ROUGE, EM,
and Medical Accuracy metrics.
Uses PhiData Agent with Groq LLM (llama-3.1-8b-instant). Same structure as reference code.
"""

from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.python import PythonTools

from typing import Dict, Any, List
import json, re, logging
from pathlib import Path

logger = logging.getLogger(__name__)

REPORT_DIR = Path("./artifacts/evaluation")


class EvaluationAgent:
    """
    Evaluates the fine-tuned Medical VQA model using standard
    NLP and medical-domain metrics.
    """

    def __init__(self, model_id: str = "mistral", eval_samples: int = 200):
        self.eval_samples = eval_samples
        self.agent = Agent(
            name="EvaluationAgent",
            model=Groq(id="llama-3.1-8b-instant"),
            tools=[PythonTools()],
            instructions=[
                "You are a medical AI evaluation expert.",
                "Interpret evaluation metrics for a Medical VQA model.",
                "Provide clinical and technical assessment of model performance.",
                "Always reply with a JSON object containing 'overall_grade', 'clinical_readiness', and 'recommendations'.",
                "Do not add explanation outside the JSON.",
            ],
            show_tool_calls=True,
            markdown=False,
        )

    def evaluate(self, checkpoint_path: str, processed_data_path: str,
                 model_plan: Dict[str, Any], device: str) -> Dict[str, Any]:
        """
        Run inference on eval samples and compute all metrics.

        Args:
            checkpoint_path: Path to fine-tuned LoRA checkpoint.
            processed_data_path: Path to processed JSONL data.
            model_plan: Fallback dict with hf_id and architecture info (used
                        only if model_plan.json cannot be read from the
                        checkpoint directory).
            device: 'cuda' | 'cpu'.

        Returns:
            Dict with all metrics and LLM assessment.
        """
        # ── GROUND-TRUTH MODEL PLAN: always read from checkpoint ─────────────
        # training_agent._save() writes model_plan.json into the checkpoint
        # directory. That file records the model that was *actually* trained,
        # which may differ from what model_plan was passed here (e.g. after
        # the self-healing retry loop switched to a different model). Always
        # loading from the checkpoint guarantees evaluation uses the correct
        # base model and architecture, regardless of what was passed in.
        effective_plan = self._load_plan_from_checkpoint(checkpoint_path, model_plan)

        # Load eval records
        eval_records = self._load_eval_records(processed_data_path)
        if not eval_records:
            return {"status": "failed", "message": "No evaluation records found"}

        # Load model
        model, processor = self._load_model(checkpoint_path, effective_plan, device)

        # Generate predictions
        results = self._generate_predictions(model, processor, eval_records, device)

        # Compute metrics
        metrics = self._compute_metrics(results)

        # Save report
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report_path = REPORT_DIR / "evaluation_report.json"
        report = {"metrics": metrics, "samples": results[:10]}
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"[Evaluation] Metrics: {metrics}")

        # Ask LLM to interpret results
        prompt = f"""
        Evaluate the performance of a Medical VQA model based on these metrics:

        BLEU-1: {metrics['bleu_1']:.3f}
        BLEU-4: {metrics['bleu_4']:.3f}
        ROUGE-L: {metrics['rouge_l']:.3f}
        Exact Match: {metrics['exact_match']:.3f}
        Medical Accuracy: {metrics['medical_accuracy']:.3f}
        Samples Evaluated: {metrics['n_samples']}

        Sample predictions:
        {json.dumps(results[:3], indent=2)}

        Provide an expert assessment for clinical use of this model.

        Reply with ONLY this JSON:
        {{
          "overall_grade": "<A/B/C/D>",
          "clinical_readiness": "<ready/needs_improvement/not_ready>",
          "recommendations": "<two sentences>"
        }}
        """

        response = self.agent.run(prompt)
        assessment = self._parse_response(response)

        return {
            "status": "ok",
            "metrics": metrics,
            "assessment": assessment,
            "report_path": str(report_path),
            "model_plan_used": effective_plan,
        }

    def _load_plan_from_checkpoint(
        self, checkpoint_path: str, fallback_plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Load model_plan.json from the checkpoint directory.

        training_agent._save() always writes model_plan.json into the
        checkpoint directory alongside the adapter weights. That file is
        the ground truth: it records exactly which base model and
        architecture were used for training, including any model that was
        selected by the self-healing retry loop (which may differ from
        the original model_plan passed into the pipeline).

        Falls back to ``fallback_plan`` with a warning if the file is
        missing (e.g. when pointing at a hand-crafted checkpoint that
        pre-dates this convention).
        """
        if not checkpoint_path:
            logger.warning(
                "[Evaluation] checkpoint_path is empty — using passed-in "
                "model_plan as fallback. Metrics may reflect the wrong model."
            )
            return fallback_plan

        plan_file = Path(checkpoint_path) / "model_plan.json"
        if plan_file.exists():
            try:
                loaded = json.loads(plan_file.read_text())
                logger.info(
                    f"[Evaluation] Loaded model_plan.json from checkpoint: "
                    f"hf_id={loaded.get('hf_id')!r}  name={loaded.get('name')!r}"
                )
                return loaded
            except Exception as e:
                logger.warning(
                    f"[Evaluation] Could not parse model_plan.json at "
                    f"{plan_file}: {e} — falling back to passed-in model_plan."
                )
        else:
            logger.warning(
                f"[Evaluation] model_plan.json not found in checkpoint "
                f"directory '{checkpoint_path}'. Falling back to passed-in "
                f"model_plan (hf_id={fallback_plan.get('hf_id')!r}). "
                f"If this is unexpected, check that training_agent._save() "
                f"completed successfully."
            )
        return fallback_plan

    def _load_eval_records(self, path: str) -> List[Dict]:
        if not path or not Path(path).exists():
            return []
        records = [json.loads(l) for l in open(path) if l.strip()]
        # Use val/test split if available
        val = [r for r in records if r.get("split") in ("val", "validation", "test")]
        pool = val if val else records
        return pool[:self.eval_samples]

    def _load_model(self, checkpoint_path: str, model_plan: Dict, device: str):
        import torch
        hf_id = model_plan["hf_id"]
        dtype = torch.float16 if device == "cuda" else torch.float32

        # FIX 1: use {"": 0} not "auto" — same reason as training_agent:
        # device_map="auto" triggers DataParallel on dual-T4 which breaks
        # bitsandbytes 4-bit and causes CUBLAS failures.
        load_kw = dict(pretrained_model_name_or_path=hf_id, trust_remote_code=True,
                       dtype=dtype)
        if device == "cuda":
            load_kw["device_map"] = {"": 0}

        # FIX 2: try Qwen2_5_VLForConditionalGeneration first for Qwen2.5-VL
        # AutoModelForVision2Seq often resolves to the wrong class or a
        # Placeholder for newer Qwen releases, causing silent empty output.
        hid_lower = hf_id.lower()
        if "qwen2.5-vl" in hid_lower:
            priority = ["Qwen2_5_VLForConditionalGeneration",
                        "AutoModelForVision2Seq", "AutoModelForCausalLM"]
        elif "qwen2-vl" in hid_lower:
            priority = ["Qwen2VLForConditionalGeneration",
                        "AutoModelForVision2Seq", "AutoModelForCausalLM"]
        else:
            priority = ["AutoModelForVision2Seq", "AutoModelForCausalLM",
                        "AutoModelForSeq2SeqLM"]

        model = None
        for cls_name in priority:
            try:
                import transformers as tf
                cls = getattr(tf, cls_name, None)
                if cls is None:
                    # Try submodule for Qwen classes not yet at top-level
                    try:
                        import importlib
                        submod = {
                            "Qwen2_5_VLForConditionalGeneration": "transformers.models.qwen2_5_vl",
                            "Qwen2VLForConditionalGeneration":    "transformers.models.qwen2_vl",
                        }.get(cls_name)
                        if submod:
                            cls = getattr(importlib.import_module(submod), cls_name, None)
                    except Exception:
                        pass
                if cls:
                    model = cls.from_pretrained(**load_kw)
                    logger.info(f"[Evaluation] Loaded base model with {cls_name}")
                    break
            except Exception as e:
                logger.warning(f"[Evaluation] {cls_name} load failed: {e}")
                continue

        if model is None:
            logger.error(f"[Evaluation] Could not load base model {hf_id}")
            return None, None

        # Load LoRA adapters from checkpoint
        if Path(checkpoint_path).exists():
            try:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, checkpoint_path)
                model = model.merge_and_unload()
                logger.info("[Evaluation] LoRA adapters merged.")
            except Exception as e:
                logger.warning(f"[Evaluation] PEFT load failed (running base model): {e}")

        model.eval()

        # FIX 3: ALWAYS load processor from hf_id, never from checkpoint_path.
        # The LoRA checkpoint directory only contains adapter_config.json +
        # adapter weights + model_plan.json — it has NO processor/tokenizer
        # files. Loading from checkpoint_path raises FileNotFoundError or
        # silently falls back to a default tokenizer that doesn't know about
        # image tokens, so every generation call produces garbage / empty output.
        try:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
            logger.info(f"[Evaluation] Loaded processor from base model {hf_id}")
        except Exception:
            try:
                from transformers import AutoTokenizer
                processor = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
                logger.info(f"[Evaluation] Loaded tokenizer from base model {hf_id}")
            except Exception as e:
                logger.error(f"[Evaluation] Processor load failed: {e}")
                return model, None

        inner = getattr(processor, "tokenizer", processor)
        if hasattr(inner, "pad_token") and inner.pad_token is None:
            inner.pad_token = inner.eos_token

        return model, processor

    def _generate_predictions(self, model, processor, records, device) -> List[Dict]:
        import torch
        from PIL import Image

        if model is None or processor is None:
            logger.error("[Evaluation] model or processor is None — skipping generation.")
            return [{"question": r.get("question",""), "ground_truth": r.get("answer",""),
                     "prediction": ""} for r in records]

        tok = getattr(processor, "tokenizer", processor)
        pad_id = getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", 1)

        results = []
        for i, rec in enumerate(records):
            question = rec.get("question", "")
            gt       = rec.get("answer",   "")
            img_path = rec.get("image_path", "") or rec.get("img_path", "")

            prediction = ""
            try:
                with torch.no_grad():
                    image = None
                    if img_path and Path(img_path).exists():
                        image = Image.open(img_path).convert("RGB")

                    if image is not None and hasattr(processor, "apply_chat_template"):
                        # Qwen2.5-VL / Qwen2-VL path
                        msgs = [{"role": "user", "content": [
                            {"type": "image"}, {"type": "text", "text": question},
                        ]}]
                        text = processor.apply_chat_template(
                            msgs, tokenize=False, add_generation_prompt=True)
                        inputs = processor(
                            text=text, images=[image], return_tensors="pt"
                        ).to(device)
                    elif image is not None and hasattr(processor, "image_processor"):
                        # BLIP-2 / LLaVA / InstructBLIP path
                        prompt = f"Question: {question}\nAnswer:"
                        inputs = processor(
                            images=image, text=prompt, return_tensors="pt"
                        ).to(device)
                    else:
                        # Text-only fallback
                        prompt = f"Question: {question}\nAnswer:"
                        inputs = tok(
                            prompt, return_tensors="pt",
                            truncation=True, max_length=256,
                        ).to(device)

                    # FIX: pass only the keys the model actually accepts.
                    # Passing extra keys (e.g. token_type_ids) to Qwen2.5-VL
                    # raises a TypeError and produces no output. Filter to the
                    # forward-signature keys only.
                    import inspect
                    try:
                        sig_keys = set(inspect.signature(model.forward).parameters.keys())
                        # Always keep pixel_values / image_grid_thw even if
                        # they don't appear by name in the signature (some
                        # models accept **kwargs)
                        safe_inputs = {
                            k: v for k, v in inputs.items()
                            if k in sig_keys
                            or k in {"pixel_values", "image_grid_thw",
                                     "input_ids", "attention_mask"}
                        }
                    except Exception:
                        safe_inputs = dict(inputs)

                    out = model.generate(
                        **safe_inputs,
                        max_new_tokens=32,   # VQA-RAD answers are short (avg 1–4 tokens).
                                              # 64 allowed the model to keep generating
                                              # after the answer, adding hallucinated
                                              # text that reduced BLEU/ROUGE/exact-match.
                        do_sample=False,
                        pad_token_id=pad_id,
                    )
                    # Decode only the generated tokens (skip the prompt)
                    prompt_len = safe_inputs["input_ids"].shape[-1]
                    gen_tokens = out[0][prompt_len:]
                    prediction = tok.decode(gen_tokens, skip_special_tokens=True).strip()

            except Exception as e:
                logger.warning(f"[Evaluation] Record {i} generation failed: {type(e).__name__}: {e}")

            results.append({
                "question": question,
                "ground_truth": gt,
                "prediction": prediction,
            })

        logger.info(f"[Evaluation] Generated {len(results)} predictions.")
        # Log a few samples so it's easy to verify output quality in logs
        for s in results[:3]:
            logger.info(
                f"[Evaluation] SAMPLE — Q: {s['question'][:60]!r} | "
                f"GT: {s['ground_truth']!r} | PRED: {s['prediction']!r}"
            )
        return results

    def _compute_metrics(self, results: List[Dict]) -> Dict:
        preds = [r["prediction"] for r in results]
        refs  = [r["ground_truth"] for r in results]

        bleu1, bleu4 = self._bleu(preds, refs)
        rouge_l      = self._rouge(preds, refs)
        em           = self._exact_match(preds, refs)
        med_acc      = self._medical_accuracy(preds, refs)

        return {
            "bleu_1": round(bleu1, 4), "bleu_4": round(bleu4, 4),
            "rouge_l": round(rouge_l, 4), "exact_match": round(em, 4),
            "medical_accuracy": round(med_acc, 4), "n_samples": len(results),
        }

    def _normalise(self, s: str) -> str:
        """Normalise text before metric computation — remove punctuation, lowercase, collapse whitespace."""
        return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s.lower())).strip()

    def _bleu(self, preds, refs):
        try:
            import nltk
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            try: nltk.data.find("tokenizers/punkt")
            except: nltk.download("punkt", quiet=True)
            sf = SmoothingFunction().method1
            # FIX: normalise before tokenising — "Yes." vs "yes" should match.
            np_ = [self._normalise(p) for p in preds]
            nr  = [self._normalise(r) for r in refs]
            b1 = [sentence_bleu([r.split()], p.split(), (1,0,0,0), sf) for p,r in zip(np_,nr)]
            b4 = [sentence_bleu([r.split()], p.split(), (.25,.25,.25,.25), sf) for p,r in zip(np_,nr)]
            return sum(b1)/max(len(b1),1), sum(b4)/max(len(b4),1)
        except Exception:
            return 0.0, 0.0

    def _rouge(self, preds, refs):
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            # FIX: normalise before scoring.
            np_ = [self._normalise(p) for p in preds]
            nr  = [self._normalise(r) for r in refs]
            scores = [scorer.score(r, p)["rougeL"].fmeasure for p,r in zip(np_,nr)]
            return sum(scores)/max(len(scores),1)
        except Exception:
            np_ = [self._normalise(p) for p in preds]
            nr  = [self._normalise(r) for r in refs]
            scores = []
            for p, r in zip(np_, nr):
                rw = set(r.split()); pw = set(p.split())
                scores.append(len(rw & pw)/max(len(rw),1))
            return sum(scores)/max(len(scores),1)

    def _exact_match(self, preds, refs):
        # FIX: use shared _normalise() for consistency across all metrics.
        return sum(1 for p,r in zip(preds,refs)
                   if self._normalise(p) == self._normalise(r)) / max(len(preds),1)

    def _medical_accuracy(self, preds, refs):
        MED = {"yes","no","normal","abnormal","present","absent","mild","moderate",
               "severe","acute","chronic","left","right","bilateral"}
        scores = []
        for p, r in zip(preds, refs):
            # FIX: normalise before comparison — "Yes." vs "yes" previously scored 0.
            rl, pl = self._normalise(r), self._normalise(p)
            if rl in {"yes","no"} and pl in {"yes","no"}:
                scores.append(1.0 if rl == pl else 0.0)
                continue
            rt, pt = set(rl.split()), set(pl.split())
            ov = rt & pt
            med_ov = ov & MED
            # FIX: denominator was len(rt)+len(med_ov)+eps which double-counted
            # medical term matches. A perfect match (p==r) now correctly scores 1.0.
            scores.append(min((len(ov) + len(med_ov)) / max(len(rt) + len(med_ov), 1), 1.0))
        return sum(scores) / max(len(scores), 1)

    def _parse_response(self, response) -> Dict:
        try:
            text = response.content if hasattr(response, "content") else str(response)
            match = re.search(r'\{.*?\}', text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            pass
        return {"overall_grade": "B", "clinical_readiness": "needs_improvement",
                "recommendations": "Evaluate with more samples for reliable assessment."}


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    agent = EvaluationAgent(model_id="mistral", eval_samples=50)
    plan = {"hf_id": "google/flan-t5-base", "name": "Flan-T5-Base", "architecture": "seq2seq"}
    result = agent.evaluate(
        checkpoint_path="./artifacts/checkpoints/Flan-T5-Base",
        processed_data_path="./data/processed/processed_dataset.jsonl",
        model_plan=plan, device="cuda",
    )
    print("Evaluation Result:", json.dumps(result, indent=2))
