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
            model_plan: Dict with hf_id and architecture info.
            device: 'cuda' | 'cpu'.

        Returns:
            Dict with all metrics and LLM assessment.
        """
        # Load eval records
        eval_records = self._load_eval_records(processed_data_path)
        if not eval_records:
            return {"status": "failed", "message": "No evaluation records found"}

        # Load model
        model, processor = self._load_model(checkpoint_path, model_plan, device)

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
        }

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
        load_kw = dict(pretrained_model_name_or_path=hf_id, trust_remote_code=True,
                       torch_dtype=dtype)
        if device == "cuda":
            load_kw["device_map"] = "auto"

        model = None
        for cls_name in ["AutoModelForVision2Seq", "AutoModelForCausalLM", "AutoModelForSeq2SeqLM"]:
            try:
                import transformers as tf
                cls = getattr(tf, cls_name, None)
                if cls:
                    model = cls.from_pretrained(**load_kw)
                    break
            except Exception:
                continue

        # Load LoRA adapters
        if model and Path(checkpoint_path).exists():
            try:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, checkpoint_path)
                model = model.merge_and_unload()
                logger.info("[Evaluation] LoRA adapters merged.")
            except Exception as e:
                logger.warning(f"[Evaluation] PEFT load failed: {e}")

        if model:
            model.eval()

        try:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)
        except Exception:
            try:
                from transformers import AutoTokenizer
                processor = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
            except Exception:
                from transformers import AutoTokenizer
                processor = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)

        return model, processor

    def _generate_predictions(self, model, processor, records, device) -> List[Dict]:
        import torch
        from PIL import Image

        results = []
        for rec in records:
            question = rec.get("question", "")
            gt       = rec.get("answer", "")
            img_path = rec.get("image_path", "")
            prompt   = f"Question: {question}\nAnswer:"

            prediction = ""
            try:
                with torch.no_grad():
                    image = None
                    if img_path and Path(img_path).exists():
                        image = Image.open(img_path).convert("RGB")

                    if image and hasattr(processor, "image_processor"):
                        if hasattr(processor, "apply_chat_template"):
                            msgs = [{"role": "user", "content": [
                                {"type": "image"}, {"type": "text", "text": prompt}
                            ]}]
                            text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                            inputs = processor(text=text, images=[image], return_tensors="pt").to(device)
                        else:
                            inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
                    else:
                        tok = getattr(processor, "tokenizer", processor)
                        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)

                    out = model.generate(**inputs, max_new_tokens=64, do_sample=False,
                                         pad_token_id=getattr(processor, "eos_token_id", 1))
                    gen = out[0][inputs["input_ids"].shape[-1]:]
                    tok = getattr(processor, "tokenizer", processor)
                    prediction = tok.decode(gen, skip_special_tokens=True).strip()
            except Exception as e:
                logger.debug(f"Generation failed: {e}")

            results.append({"question": question, "ground_truth": gt, "prediction": prediction})

        logger.info(f"[Evaluation] Generated {len(results)} predictions.")
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

    def _bleu(self, preds, refs):
        try:
            import nltk
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            try: nltk.data.find("tokenizers/punkt")
            except: nltk.download("punkt", quiet=True)
            sf = SmoothingFunction().method1
            b1 = [sentence_bleu([r.split()], p.split(), (1,0,0,0), sf) for p,r in zip(preds,refs)]
            b4 = [sentence_bleu([r.split()], p.split(), (.25,.25,.25,.25), sf) for p,r in zip(preds,refs)]
            return sum(b1)/max(len(b1),1), sum(b4)/max(len(b4),1)
        except Exception:
            return 0.0, 0.0

    def _rouge(self, preds, refs):
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            scores = [scorer.score(r, p)["rougeL"].fmeasure for p,r in zip(preds,refs)]
            return sum(scores)/max(len(scores),1)
        except Exception:
            scores = []
            for p, r in zip(preds, refs):
                rw = set(r.lower().split()); pw = set(p.lower().split())
                scores.append(len(rw&pw)/max(len(rw),1))
            return sum(scores)/max(len(scores),1)

    def _exact_match(self, preds, refs):
        def norm(s): return re.sub(r"\s+"," ", re.sub(r"[^\w\s]","", s.lower())).strip()
        return sum(1 for p,r in zip(preds,refs) if norm(p)==norm(r)) / max(len(preds),1)

    def _medical_accuracy(self, preds, refs):
        MED = {"yes","no","normal","abnormal","present","absent","mild","moderate",
               "severe","acute","chronic","left","right","bilateral"}
        scores = []
        for p, r in zip(preds, refs):
            rl, pl = r.lower().strip(), p.lower().strip()
            if rl in {"yes","no"} and pl in {"yes","no"}:
                scores.append(1.0 if rl==pl else 0.0); continue
            rt, pt = set(rl.split()), set(pl.split())
            ov = rt & pt; med_ov = ov & MED
            scores.append(min((len(ov)+len(med_ov))/(len(rt)+len(med_ov)+1e-6), 1.0))
        return sum(scores)/max(len(scores),1)

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
