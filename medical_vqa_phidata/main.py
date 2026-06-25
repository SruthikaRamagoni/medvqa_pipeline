#!/usr/bin/env python3
"""
main.py — Medical VQA PhiData Multi-Agent Pipeline

8-stage pipeline:
  1. ModalityDiscoveryAgent    — detect imaging modality
  2. DataDiscoveryAgent        — find best free dataset
  3. DataCollectionAgent       — download + cache
  4. DataPreprocessingAgent    — clean, resize, anonymize
  5. ModelSelectionAgent       — pick best model for GPU
  6. FeatureEngineeringAgent   — encode to model-specific tensors  ← NEW
  7. TrainingAgent             — PEFT/LoRA fine-tuning
  8. EvaluationAgent           — BLEU, ROUGE, EM, Medical Accuracy

Run:
    python main.py --image image.jpg --question "Is there pneumonia?"
    python main.py --image image.jpg --question "..." --dry-run
    python main.py --image image.jpg --question "..." --skip-training --checkpoint ./artifacts/checkpoints/Flan-T5-Base

FIX (this revision):
    main.py previously passed a single `--groq-model` text-reasoning model ID
    (default: llama-3.1-8b-instant) into EVERY agent, including
    ModalityDiscoveryAgent. That agent's low-confidence fallback path needs a
    *vision-capable* Groq model to actually look at the image — but it was
    always being constructed with the shared text-only reasoning model,
    which Groq rejects for image+text payloads with a 400
    ("messages[1].content must be a string"). The agent's internal
    `VISION_MODEL_ID` default was therefore always being silently
    overridden and never took effect.

    Fix: ModalityDiscoveryAgent is now constructed WITHOUT passing `model_id`,
    so it uses its own internal vision-capable default
    (meta-llama/llama-4-scout-17b-16e-instruct as of this writing). A new
    `--vision-model` CLI flag lets you override that independently of
    `--groq-model`, which still controls the text-reasoning model used by
    every other agent.
"""

import argparse, json, logging, sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))
from config.settings import MAX_MODEL_RETRIES
from core.state import PipelineState
from agents.modality_discovery_agent  import ModalityDiscoveryAgent
from agents.data_discovery_agent      import DataDiscoveryAgent
from agents.data_collection_agent     import DataCollectionAgent
from agents.data_preprocessing_agent  import DataPreprocessingAgent
from agents.model_selection_agent     import ModelSelectionAgent
from agents.feature_engineering_agent import FeatureEngineeringAgent
from agents.training_agent            import TrainingAgent
from agents.evaluation_agent          import EvaluationAgent


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(level: str = "INFO") -> None:
    Path("./logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("./logs/pipeline.log", mode="a"),
        ],
    )


# ── Hardware ──────────────────────────────────────────────────────────────────

def detect_hardware() -> dict:
    import psutil
    hw = {
        "device":  "cpu",
        "vram_gb": 0.0,
        "ram_gb":  psutil.virtual_memory().total / 1e9,
    }
    try:
        import torch
        if torch.cuda.is_available():
            hw["device"]  = "cuda"
            hw["vram_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
            hw["gpu_name"]= torch.cuda.get_device_name(0)
        elif torch.backends.mps.is_available():
            hw["device"]  = "mps"
            hw["vram_gb"] = hw["ram_gb"] * 0.5
    except ImportError:
        pass
    return hw


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Medical VQA PhiData Multi-Agent Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--image",         required=True,
                   help="Path to medical image (jpg/png/dcm)")
    p.add_argument("--question",      required=True,
                   help="Clinical question about the image")
    p.add_argument("--groq-model",    default="llama-3.1-8b-instant",
                   help="Groq TEXT-reasoning model ID used by every agent "
                        "EXCEPT ModalityDiscoveryAgent's vision fallback "
                        "(see --vision-model).")
    p.add_argument("--vision-model",  default=None,
                   help="Groq VISION-capable model ID used only by "
                        "ModalityDiscoveryAgent's low-confidence fallback. "
                        "If omitted, the agent's own internal default is "
                        "used. Must be a model that accepts image input — "
                        "passing a text-only model here will 400.")
    p.add_argument("--max-samples",   type=int, default=5000,
                   help="Maximum dataset samples to use")
    p.add_argument("--dry-run",       action="store_true",
                   help="10 samples, 1 epoch — fast end-to-end test")
    p.add_argument("--skip-training", action="store_true",
                   help="Skip training; evaluate an existing checkpoint")
    p.add_argument("--checkpoint",    default="",
                   help="Existing checkpoint path (use with --skip-training)")
    p.add_argument("--log-level",     default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(args) -> PipelineState:
    logger = logging.getLogger("main")
    hw     = detect_hardware()
    m      = args.groq_model

    logger.info(f"Hardware: device={hw['device']}  "
                f"VRAM={hw['vram_gb']:.1f}GB  RAM={hw['ram_gb']:.1f}GB")
    logger.info(f"Groq text-reasoning model for agents: {m}")
    if args.vision_model:
        logger.info(f"Groq vision model override: {args.vision_model}")

    state = PipelineState(
        image_path=args.image,
        question=args.question,
        device=hw["device"],
        vram_gb=hw["vram_gb"],
        ram_gb=hw["ram_gb"],
    )

    max_samples     = 10 if args.dry_run else args.max_samples
    epochs_override = 1  if args.dry_run else 0

    # ── STEP 1 — Modality Discovery ───────────────────────────────────────────
    _banner(logger, "STEP 1 — Modality Discovery")
    # IMPORTANT: do NOT pass the shared text-reasoning model `m` here.
    # ModalityDiscoveryAgent's low-confidence path needs a vision-capable
    # model to actually see the image; the shared `--groq-model` default
    # (llama-3.1-8b-instant) is text-only and will 400 on image+text
    # payloads ("messages[1].content must be a string"). Only pass an
    # explicit override if the user supplied --vision-model; otherwise let
    # the agent use its own internal vision-capable default.
    modality_agent_kwargs = {}
    if args.vision_model:
        modality_agent_kwargs["model_id"] = args.vision_model
    modality_agent  = ModalityDiscoveryAgent(**modality_agent_kwargs)
    modality_result = modality_agent.discover_modality(args.image, args.question)
    state.modality  = modality_result.get("modality", "Unknown")
    state.log(f"Modality: {state.modality}  "
               f"(conf={modality_result.get('confidence', 0):.2f})")
    if modality_result.get("error"):
        # Surface vision-call failures loudly rather than letting them slide
        # by silently as a low-confidence "Unknown" with no trace in the
        # pipeline summary.
        state.errors.append(f"modality_discovery: {modality_result['error']}")
        logger.warning(f"Modality discovery degraded: {modality_result['error']}")

    # ── STEP 2 — Data Discovery ───────────────────────────────────────────────
    _banner(logger, "STEP 2 — Data Discovery")
    discovery_agent = DataDiscoveryAgent(model_id=m)
    dataset_info    = discovery_agent.discover_dataset(state.modality)
    state.dataset_name       = dataset_info["name"]
    state.dataset_source     = dataset_info["source"]
    state.dataset_local_path = dataset_info.get("local_path", "")
    state.log(f"Dataset selected: {state.dataset_name}")

    # ── STEP 3 — Data Collection ──────────────────────────────────────────────
    _banner(logger, "STEP 3 — Data Collection")
    collection_agent  = DataCollectionAgent(model_id=m, max_samples=max_samples)
    collection_result = collection_agent.collect_dataset(dataset_info)
    if collection_result.get("status") == "failed":
        state.errors.append(collection_result.get("message", "Collection failed"))
        logger.error(f"Collection failed: {collection_result['message']}")
        state.save()
        return state
    state.raw_data_path = collection_result["cache_path"]
    state.log(f"Collected {collection_result['records_count']} records → {state.raw_data_path}")

    # ── STEP 4 — Data Preprocessing ───────────────────────────────────────────
    _banner(logger, "STEP 4 — Data Preprocessing")
    preprocess_agent  = DataPreprocessingAgent(model_id=m)
    preprocess_result = preprocess_agent.preprocess(state.raw_data_path)
    if preprocess_result.get("status") == "failed":
        state.errors.append(preprocess_result.get("message", "Preprocessing failed"))
        logger.error(f"Preprocessing failed: {preprocess_result['message']}")
        state.save()
        return state
    state.processed_data_path = preprocess_result["processed_path"]
    state.log(f"Preprocessed: {preprocess_result['valid_records']} valid records")

    # ── STEP 5 — Model Selection ──────────────────────────────────────────────
    _banner(logger, "STEP 5 — Model Selection")
    model_sel_agent = ModelSelectionAgent(model_id=m)
    model_plan      = model_sel_agent.select_model(
        dataset_size=collection_result["records_count"],
        modality=state.modality,
    )
    if epochs_override:
        model_plan["epochs"] = epochs_override

    state.model_hf_id        = model_plan["hf_id"]
    state.model_name         = model_plan["name"]
    state.model_architecture = model_plan["architecture"]
    state.model_vision       = model_plan.get("vision", False)
    state.lora_r             = model_plan["lora_r"]
    state.epochs             = model_plan["epochs"]
    state.batch_size         = model_plan["batch_size"]
    state.log(f"Model: {state.model_name} ({state.model_hf_id})")

    # ── STEP 6 — Feature Engineering ──────────────────────────────────────────
    _banner(logger, "STEP 6 — Feature Engineering")
    fe_agent  = FeatureEngineeringAgent(model_id=m)
    fe_result = fe_agent.engineer_features(
        processed_data_path=state.processed_data_path,
        model_plan=model_plan,
        device=state.device,
    )
    if fe_result.get("status") == "failed":
        state.errors.append(fe_result.get("message", "Feature engineering failed"))
        logger.error(f"Feature engineering failed: {fe_result['message']}")
        state.save()
        return state
    state.feature_data_path = fe_result["feature_path"]
    state.log(
        f"Features encoded: train={fe_result['train_samples']}  "
        f"val={fe_result['val_samples']}  path={state.feature_data_path}"
    )

    # ── STEP 7 — Training ─────────────────────────────────────────────────────
    if args.skip_training and args.checkpoint:
        state.checkpoint_path = args.checkpoint
        state.log(f"Skipping training — checkpoint: {args.checkpoint}")
    else:
        _banner(logger, "STEP 7 — Training")
        training_agent  = TrainingAgent(model_id=m)
        training_result = training_agent.train(
            feature_path=state.feature_data_path,
            model_plan=model_plan,
            device=state.device,
            processed_data_path=state.processed_data_path,
            dataset_size=collection_result["records_count"],
            modality=state.modality,
        )
        if training_result.get("status") == "failed":

            state.retry_count += 1
            state.failure_reason = training_result.get("failure_reason", "")
            state.previous_models.append(
                training_result.get("model_used", "")
            )

            while state.retry_count <= MAX_MODEL_RETRIES:

                logger.warning(
                    f"Retry {state.retry_count}/{MAX_MODEL_RETRIES}"
                )

                retry_plan = model_sel_agent.select_model(
                    dataset_size=collection_result["records_count"],
                    modality=state.modality,
                    failure_context={
                        "failed_hf_id": training_result.get("model_used", ""),
                        "reason": training_result.get("failure_reason", ""),
                    },
                )

                fe_result = fe_agent.engineer_features(
                    processed_data_path=state.processed_data_path,
                    model_plan=retry_plan,
                    device=state.device,
                )

                if fe_result.get("status") == "failed":
                    state.errors.append(
                        fe_result.get("message", "")
                    )
                    break
                training_result = training_agent.train(
                    feature_path=fe_result["feature_path"],
                    model_plan=retry_plan,
                    device=state.device,
                    processed_data_path=state.processed_data_path,
                    dataset_size=collection_result["records_count"],
                    modality=state.modality,
                )

                if training_result.get("status") != "failed":

                    state.model_hf_id = retry_plan["hf_id"]
                    state.model_name = retry_plan["name"]
                    state.model_architecture = retry_plan["architecture"]
                    state.model_vision = retry_plan["vision"]

                    state.checkpoint_path = (
                        training_result["checkpoint_path"]
                    )

                    # FIX: update model_plan to the plan that actually succeeded.
                    # Without this, the original model_plan (e.g. instructblip)
                    # is passed to evaluate() even though a different model
                    # (e.g. flan-t5 / Qwen) was trained — evaluation then tries
                    # to load the wrong model class and fails with
                    # "Unrecognized configuration class".
                    model_plan = retry_plan

                    break

                state.retry_count += 1
                state.previous_models.append(
                    training_result.get("model_used", "")
                )

            if training_result.get("status") == "failed":
                state.errors.append(
                    training_result.get("failure_reason", "")
                )
                state.save()
                return state


        state.checkpoint_path = training_result["checkpoint_path"]
        state.log(
            f"Training done. Loss={training_result.get('train_loss','N/A')}  "
            f"Checkpoint={state.checkpoint_path}"
        )

    # ── STEP 8 — Evaluation ───────────────────────────────────────────────────
    _banner(logger, "STEP 8 — Evaluation")
    eval_agent  = EvaluationAgent(
        model_id=m,
        eval_samples=50 if args.dry_run else 200,
    )
    eval_result = eval_agent.evaluate(
        checkpoint_path=state.checkpoint_path,
        processed_data_path=state.processed_data_path,
        model_plan=model_plan,
        device=state.device,
    )
    if eval_result.get("status") != "failed":
        metrics              = eval_result.get("metrics", {})
        state.bleu_1         = metrics.get("bleu_1",          0.0)
        state.bleu_4         = metrics.get("bleu_4",          0.0)
        state.rouge_l        = metrics.get("rouge_l",         0.0)
        state.exact_match    = metrics.get("exact_match",     0.0)
        state.medical_accuracy= metrics.get("medical_accuracy",0.0)
        state.success        = True
        state.log("Evaluation complete.")
    else:
        state.errors.append(eval_result.get("message", "Evaluation failed"))

    state.save()
    return state


# ── Helpers ───────────────────────────────────────────────────────────────────

def _banner(logger, title: str) -> None:
    logger.info("\n" + "="*55)
    logger.info(f"  {title}")
    logger.info("="*55)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger("main")

    logger.info("="*60)
    logger.info("  Medical VQA — PhiData Multi-Agent Pipeline")
    logger.info("="*60)
    logger.info(f"  Image    : {args.image}")
    logger.info(f"  Question : {args.question}")
    logger.info(f"  Groq LLM (text reasoning) : {args.groq_model}")
    logger.info(f"  Groq LLM (vision, modality discovery) : "
                f"{args.vision_model or '<agent default>'}")
    logger.info(f"  Dry-run  : {args.dry_run}")

    state = run_pipeline(args)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("  PIPELINE SUMMARY")
    logger.info("="*60)
    logger.info(f"  Status      : {'SUCCESS' if state.success else 'FAILED'}")
    logger.info(f"  Modality    : {state.modality}")
    logger.info(f"  Dataset     : {state.dataset_name}")
    logger.info(f"  Model       : {state.model_name}")
    logger.info(f"  Features    : {state.feature_data_path}")
    logger.info(f"  Checkpoint  : {state.checkpoint_path}")
    if state.success:
        logger.info(f"  BLEU-1      : {state.bleu_1:.3f}")
        logger.info(f"  BLEU-4      : {state.bleu_4:.3f}")
        logger.info(f"  ROUGE-L     : {state.rouge_l:.3f}")
        logger.info(f"  Exact Match : {state.exact_match:.3f}")
        logger.info(f"  Medical Acc : {state.medical_accuracy:.3f}")
    if state.errors:
        logger.info(f"  Errors      : {state.errors}")
    logger.info("="*60)

    # ── Inference on input image ───────────────────────────────────────────────
    if state.checkpoint_path and Path(state.checkpoint_path).exists():
        logger.info("\n  Running inference on input image …")
        try:
            from inference import run_inference
            answer = run_inference(
                image_path=args.image,
                question=args.question,
                checkpoint_path=state.checkpoint_path,
                model_hf_id=state.model_hf_id,
                device=state.device,
            )
            logger.info(f"\n  Q : {args.question}")
            logger.info(f"  A : {answer}\n")
        except Exception as e:
            logger.warning(f"  Inference error: {e}")


if __name__ == "__main__":
    main()
