"""
agents/feature_engineering_agent.py

FeatureEngineeringAgent — converts the cleaned processed dataset into
model-specific tensor format ready for training.

CACHE / METADATA UPDATE
------------------------
Every feature directory now carries a metadata.json:
    {
      "model_hf_id": "...", "architecture": "...", "processor_type": "...",
      "feature_strategy": "...", "tensor_schema": "...", "image_enabled": bool
    }
On each call, if a cache for the target hf_id already exists, its
metadata.json is compared against the current model_plan. On any mismatch
(or missing/corrupt metadata) the cache is discarded and regenerated —
this is what lets TrainingAgent's retry loop safely call this agent again
after ModelSelectionAgent picks a different model.

Encoded records are now validated (_validate_entry) before being added to
the dataset: invalid/missing tensor fields or wrong-rank pixel_values are
rejected rather than silently propagated to TrainingAgent.

LABEL-MASKING FIX (this revision)
----------------------------------
ROOT CAUSE of "training loss flatlines, eval metrics all 0.0":
_encode_qwen_vl / _encode_llava / _encode_phi_vision previously did

    result["labels"] = result["input_ids"][:]

i.e. labels were an exact, fully-UNMASKED copy of input_ids. Teacher-forced
LM loss was therefore computed over the ENTIRE sequence — image tokens,
chat-template scaffolding, and the question — not just the answer span.
Those non-answer tokens are trivially predictable (they're scaffolding /
already-seen-in-context tokens), so the loss converges fast to a low
"free" value within ~1 epoch and then has nowhere left to improve, because
the answer tokens (the only part that actually matters) are a small
fraction of total loss and their gradient contribution is diluted away.
The model never learns to produce real answers, hence exact_match / BLEU /
ROUGE / medical_accuracy = 0.0 across the board at eval time, even though
training "looked" like it converged.

FIX: every vision-chat encoder now tokenizes the PROMPT ALONE first (same
image, no answer) to find exactly how many leading tokens belong to the
prompt/image/scaffolding. That length is used to mask labels[:prompt_len]
= -100. Padding tokens are also masked. Only the answer span (+ EOS, if
present before truncation) remains a real loss target — which is the
standard, correct recipe for instruction/VQA fine-tuning.

PROGRESS LOGGING (previous revision)
----------------------------------
_encode_records previously had no logging at all inside its main
per-record loop, so on large datasets (e.g. 2244 records) the only
visible log lines were the max_len probe result and then — after the
ENTIRE loop finished, possibly many minutes later, especially on CPU —
a single summary line. This made a slow-but-healthy run indistinguishable
from a genuine hang. A periodic progress line (every 100 records, plus
the final record) is now emitted with running encoded/skipped/no_image
counts so progress is visible in real time.

STREAMING / OOM FIX (this revision)
----------------------------------
ROOT CAUSE of "tried to allocate more memory than is available" crash
(observed right at the END of encoding, ~2224/2244 records in):
the old _encode_records returned a plain Python `list` that accumulated
EVERY encoded record — including each record's full `pixel_values`
nested list — in RAM simultaneously, for the entire dataset, before a
single byte was written to disk. _build_hf_datasets then called
`Dataset.from_list(encoded)`, which materializes a SECOND full copy of
the same data while building the Arrow table. For Qwen2.5-VL-sized
pixel_values across ~2000+ records, two simultaneous full-dataset copies
in memory (one as raw Python lists, one being converted to Arrow) is
exactly the kind of peak that exhausts RAM right as the final records are
processed/converted — which matches the crash happening at the very end
of the run, not partway through.

FIX: _encode_records is now a *generator* (_encode_records_gen) that
yields one encoded record at a time instead of building a list. It is
fed directly into `datasets.Dataset.from_generator(...)`, which writes
each record to Arrow on disk incrementally (in small writer batches)
as it's produced — so at no point does the full dataset exist twice (or
even once) fully materialized in Python-list form in memory. Only the
already-Arrow-backed (memory-mapped, not RAM-resident) Dataset exists
afterward, which is then split into train/val via `.select(...)` ranges
rather than slicing Python lists.
"""

from phi.agent import Agent
from phi.model.groq import Groq

from typing import Any, Dict, List, Optional, Tuple
import json, logging, re, shutil
from pathlib import Path

logger = logging.getLogger(__name__)

FEATURE_DIR = Path("./data/features")

VISION_FAMILIES = {"blip2", "instructblip", "llava", "qwen_vl", "phi_vision", "idefics"}


class FeatureEngineeringAgent:
    """
    Converts processed JSONL records into model-specific HuggingFace
    Dataset format (input_ids, pixel_values, labels, etc.).

    Adaptive: detects model family from hf_id at runtime, loads the
    correct processor/tokenizer, encodes accordingly, validates the
    result, and caches it with a metadata.json describing the exact
    model/processor/strategy combination used — so stale caches from a
    previously-failed model are never silently reused.
    """

    def __init__(self, model_id: str = "mistral"):
        self.agent = Agent(
            name="FeatureEngineeringAgent",
            model=Groq(id="llama-3.1-8b-instant"),
            instructions=[
                "You are a machine learning feature engineering expert.",
                "You receive dataset statistics and encoding results.",
                "Assess whether the encoded features are ready for model training.",
                "Always reply with ONLY a JSON object like this:",
                '{"status": "ok", "train_samples": <int>, "val_samples": <int>, "message": "<one sentence>"}',
                "Do not write code. Do not add any text outside the JSON.",
            ],
            show_tool_calls=False,
            markdown=False,
        )

    # ── Public method ─────────────────────────────────────────────────────────

    def engineer_features(
        self,
        processed_data_path: str,
        model_plan: Dict[str, Any],
        device: str = "",
        force_regenerate: bool = False,
    ) -> Dict[str, Any]:
        """
        Convert processed JSONL into model-specific encoded HuggingFace Dataset.

        Args:
            processed_data_path : Path to processed JSONL from DataPreprocessingAgent.
            model_plan          : Dict from ModelSelectionAgent with hf_id,
                                   architecture, vision, processor_type,
                                   feature_strategy (if present, used to
                                   build/validate the cache; if absent,
                                   derived locally — backward compatible
                                   with older model_plans).
            device              : 'cuda' | 'cpu' | '' (auto-detect).
            force_regenerate    : Skip cache check entirely (used by
                                   TrainingAgent's retry loop after a
                                   training failure on this exact model
                                   to rule out a corrupt cache).

        Returns:
            Dict with feature_path, train_samples, val_samples, status.
        """
        if not device:
            device = self._detect_device()

        if not Path(processed_data_path).exists():
            return self._fail(f"Processed data not found: {processed_data_path}")

        hf_id        = model_plan.get("hf_id", "google/flan-t5-base")
        architecture = model_plan.get("architecture", "seq2seq")
        is_vision    = model_plan.get("vision", False)
        max_len      = model_plan.get("max_seq_len", 128)
        model_family = model_plan.get("model_family") or self._detect_model_family(hf_id)

        logger.info(f"[FeatureEng] Model: {hf_id}  Family: {model_family}")

        target_meta = {
            "model_hf_id":       hf_id,
            "architecture":      architecture,
            "processor_type":    model_plan.get("processor_type", self._default_processor_type(model_family)),
            "feature_strategy":  model_plan.get("feature_strategy", self._default_feature_strategy(model_family)),
            "tensor_schema":     model_plan.get("tensor_schema", ""),
            "image_enabled":     bool(is_vision),
            # Bumping this whenever the encoding/label-masking recipe, the
            # pixel_values rank-normalization logic, or the effective
            # max_len resolution changes — so a stale cache from before a
            # fix is never silently reused.
            # v2: label masking fix (prompt/padding -> -100).
            # v3: pixel_values multi-layer unwrap fix (InstructBLIP rank-5
            #     bug — raw numpy arrays weren't being unwrapped at all).
            # v4: qwen_vl effective max_len auto-resolution (image-token
            #     expansion could make prompt_len saturate at max_len,
            #     masking 100% of labels when max_seq_len was too small).
            # v5: max_len probe AND _encode_qwen_vl's true_prompt_len /
            #     result["input_ids"] now route through _unwrap_token_ids.
            #     Without it, return_tensors=None processor output wrapped
            #     as [[id,id,...]] made every length check measure the
            #     outer wrapper (=1) instead of the real token count —
            #     v4's probe always reported "1 token", never resized
            #     max_len, and the original all-masked-labels bug
            #     persisted even with v4 in place.
            "label_masking_version": 5,
        }

        feature_path = self._feature_path_for(hf_id, model_family)

        if not force_regenerate and self._cache_is_valid(feature_path, target_meta):
            logger.info(f"[FeatureEng] Reusing valid cache at {feature_path}")
            try:
                from datasets import load_from_disk
                train_ds = load_from_disk(str(Path(feature_path) / "train"))
                val_ds   = load_from_disk(str(Path(feature_path) / "val"))
                assessment = self._get_llm_assessment(
                    hf_id, model_family, is_vision, len(train_ds), len(val_ds),
                    list(train_ds.column_names),
                )
                assessment["feature_path"]  = feature_path
                assessment["train_samples"] = len(train_ds)
                assessment["val_samples"]   = len(val_ds)
                return assessment
            except Exception as e:
                logger.warning(f"[FeatureEng] Cache load failed ({e}) — regenerating.")

        if Path(feature_path).exists():
            logger.info(f"[FeatureEng] Discarding stale cache: {feature_path}")
            shutil.rmtree(feature_path, ignore_errors=True)

        records = self._load_records(processed_data_path)
        if not records:
            return self._fail("No records found in processed data.")

        processor, tokenizer = self._load_processor(hf_id, model_family)
        if processor is None and tokenizer is None:
            return self._fail(f"Could not load processor for {hf_id}")

        # STREAMING PATH: resolve the effective max_len once up front (same
        # as before — this still needs to run before any record is encoded
        # so every record shares one fixed padded length), then hand a
        # *generator* to Dataset.from_generator so records are written to
        # Arrow on disk one at a time instead of being accumulated as a
        # Python list first. See module docstring "STREAMING / OOM FIX".
        max_len = self._resolve_effective_max_len(records, processor, model_family, max_len)
        self._last_encode_diagnostics = {}  # populated by the generator as it runs

        try:
            from datasets import Dataset
            from functools import partial
            # IMPORTANT: do NOT pass `records` (a list) via gen_kwargs.
            # Dataset.from_generator treats any list-valued gen_kwargs entry
            # as a set of per-shard arguments and calls the generator ONCE
            # PER ELEMENT of that list (this is meant for multiprocess
            # sharding, e.g. gen_kwargs={"files": [f1, f2, ...]} -> one
            # call per file). Passing our 2244-record list this way caused
            # the generator to be invoked 2244 separate times, each with
            # a single-record "records" list — exactly what produced the
            # "Starting streaming encode over 1 records" spam repeated
            # thousands of times in the logs instead of one real run.
            # Binding everything via functools.partial (a closure) instead
            # avoids gen_kwargs entirely, so `records` is passed through
            # untouched as one full list to one single generator call.
            gen = partial(
                self._encode_records_gen,
                records=records,
                processor=processor,
                tokenizer=tokenizer,
                model_family=model_family,
                is_vision=is_vision,
                architecture=architecture,
                max_len=max_len,
            )
            full_ds = Dataset.from_generator(gen)
        except Exception as e:
            return self._fail(f"Streaming encode failed: {type(e).__name__}: {e}")

        if len(full_ds) == 0:
            diag = getattr(self, "_last_encode_diagnostics", {})
            detail = ""
            if diag.get("no_image_count"):
                detail = (
                    f" {diag['no_image_count']}/{diag.get('total_records', 0)} records had "
                    f"no usable image — check DataPreprocessingAgent's output schema "
                    f"('image_path' or 'image' field)."
                )
            elif diag.get("error_samples"):
                detail = f" Sample errors: {diag['error_samples']}"
            return self._fail(f"Encoding produced no valid records.{detail}")

        # Split the already Arrow-backed (memory-mapped) dataset via index
        # ranges — .select() does not load the whole dataset into RAM, so
        # this is safe even for large pixel_values columns.
        n = len(full_ds)
        split = max(1, int(n * 0.9))
        train_ds = full_ds.select(range(split))
        val_ds   = full_ds.select(range(split, n)) if split < n else full_ds.select(range(max(0, n - 1), n))

        saved_path = self._save_to_disk(train_ds, val_ds, feature_path, target_meta)
        logger.info(
            f"[FeatureEng] Done. Train={len(train_ds)}  Val={len(val_ds)}  Path={saved_path}"
        )

        assessment = self._get_llm_assessment(
            hf_id, model_family, is_vision,
            len(train_ds), len(val_ds), list(train_ds.column_names),
        )
        assessment["feature_path"]  = saved_path
        assessment["train_samples"] = len(train_ds)
        assessment["val_samples"]   = len(val_ds)
        return assessment

    # ── Failure helper ────────────────────────────────────────────────────────

    def _fail(self, message: str) -> Dict[str, Any]:
        logger.error(f"[FeatureEng] FAILED — {message}")
        return {
            "status": "failed", "message": message, "feature_path": "",
            "train_samples": 0, "val_samples": 0,
        }

    # ── Device detection ──────────────────────────────────────────────────────

    def _detect_device(self) -> str:
        try:
            import torch
            if torch.cuda.is_available():          return "cuda"
            if torch.backends.mps.is_available():  return "mps"
        except ImportError:
            pass
        return "cpu"

    # ── Load records ──────────────────────────────────────────────────────────

    def _load_records(self, path: str) -> List[Dict]:
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        continue
        logger.info(f"[FeatureEng] Loaded {len(records)} records.")
        return records

    # ── Model family detection (shared vocabulary with ModelSelectionAgent) ───

    def _detect_model_family(self, hf_id: str) -> str:
        hid = (hf_id or "").lower()
        if "flan-t5" in hid or "flan_t5" in hid:         return "flan_t5"
        if "blip2" in hid or "blip-2" in hid:             return "blip2"
        if "instructblip" in hid:                          return "instructblip"
        if "llava" in hid:                                 return "llava"
        if "qwen2.5-vl" in hid or "qwen2-vl" in hid:     return "qwen_vl"
        if "phi-3.5-vision" in hid or "phi3.5" in hid:    return "phi_vision"
        if "idefics" in hid:                               return "idefics"
        if "t5" in hid:                                    return "seq2seq"
        if "bart" in hid:                                  return "seq2seq"
        return "causal"

    def _default_processor_type(self, family: str) -> str:
        return "AutoProcessor" if family in VISION_FAMILIES else "AutoTokenizer"

    def _default_feature_strategy(self, family: str) -> str:
        if family == "qwen_vl":
            return "vision2seq_patchified"
        if family in VISION_FAMILIES:
            return "vision2seq"
        if family in ("flan_t5", "seq2seq"):
            return "seq2seq"
        return "causal_lm"

    # ── Cache validation ─────────────────────────────────────────────────────

    def _feature_path_for(self, hf_id: str, model_family: str) -> str:
        safe_name = (hf_id + "_" + model_family).replace("/", "_").replace(" ", "_")
        return str(FEATURE_DIR / safe_name)

    def _cache_is_valid(self, feature_path: str, target_meta: Dict[str, Any]) -> bool:
        meta_file = Path(feature_path) / "metadata.json"
        train_dir = Path(feature_path) / "train"
        val_dir   = Path(feature_path) / "val"
        if not (meta_file.exists() and train_dir.exists() and val_dir.exists()):
            return False
        try:
            existing = json.loads(meta_file.read_text())
        except Exception as e:
            logger.warning(f"[FeatureEng] metadata.json unreadable ({e}) — invalid cache.")
            return False

        compare_keys = ["model_hf_id", "architecture", "processor_type",
                         "feature_strategy", "image_enabled", "label_masking_version"]
        for k in compare_keys:
            if existing.get(k) != target_meta.get(k):
                logger.info(
                    f"[FeatureEng] Cache mismatch on '{k}': "
                    f"cached={existing.get(k)!r} vs target={target_meta.get(k)!r}"
                )
                return False
        return True

    # ── Processor loading ─────────────────────────────────────────────────────

    def _load_processor(self, hf_id: str, model_family: str):
        processor = None
        tokenizer = None
        try:
            if model_family in VISION_FAMILIES:
                from transformers import AutoProcessor
                processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
                tokenizer = getattr(processor, "tokenizer", processor)
                logger.info(f"[FeatureEng] Loaded AutoProcessor for {hf_id}")
            elif model_family in ("flan_t5", "seq2seq"):
                from transformers import T5Tokenizer, AutoTokenizer
                try:
                    tokenizer = T5Tokenizer.from_pretrained(hf_id)
                except Exception:
                    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
                logger.info(f"[FeatureEng] Loaded T5Tokenizer for {hf_id}")
            else:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
                logger.info(f"[FeatureEng] Loaded AutoTokenizer for {hf_id}")
        except Exception as e:
            logger.warning(f"[FeatureEng] Processor load failed for {hf_id}: {e}")
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
                logger.info("[FeatureEng] Fallback to flan-t5-base tokenizer.")
            except Exception as e2:
                logger.error(f"[FeatureEng] All tokenizer loads failed: {e2}")
                return None, None

        tok = tokenizer if tokenizer else processor
        inner = getattr(tok, "tokenizer", tok)
        if hasattr(inner, "pad_token") and inner.pad_token is None:
            inner.pad_token = inner.eos_token

        return processor, tokenizer

    # ── Label masking helper (NEW) ───────────────────────────────────────────

    def _unwrap_token_ids(self, val) -> List[int]:
        """
        Given a processor/tokenizer's `input_ids` output for a SINGLE
        example (with return_tensors=None), return the flat list of real
        token ids — regardless of how many extra batch-of-one wrapper
        layers the processor added around it.
        """
        cur = val
        while isinstance(cur, list) and len(cur) == 1 and isinstance(cur[0], list):
            cur = cur[0]
        if isinstance(cur, list):
            return cur
        try:
            import numpy as np
            arr = np.asarray(cur)
            while arr.ndim > 1 and arr.shape[0] == 1:
                arr = arr[0]
            return arr.tolist()
        except Exception:
            return list(cur) if cur is not None else []

    def _mask_prompt_and_padding(self, input_ids: List[int], prompt_len: int,
                                  pad_token_id: Optional[int]) -> List[int]:
        """
        Build a `labels` list from `input_ids`: mask every token that
        belongs to the prompt/image/scaffolding portion (index < prompt_len)
        AND every padding token, with -100.
        """
        labels = list(input_ids)
        n = len(labels)
        capped_prompt_len = min(prompt_len, n)  # truncation guard
        for i in range(n):
            if i < capped_prompt_len:
                labels[i] = -100
            elif pad_token_id is not None and labels[i] == pad_token_id:
                labels[i] = -100
        return labels

    # ── Encoding ──────────────────────────────────────────────────────────────

    _QWEN_VL_MAX_LEN_CEILING = 2048

    def _resolve_effective_max_len(
        self, records: List[Dict], processor, model_family: str, max_len: int,
        sample_size: int = 8,
    ) -> int:
        if model_family != "qwen_vl":
            return max_len
        if not hasattr(processor, "apply_chat_template"):
            return max_len
        if max_len >= 256:
            logger.info(
                f"[FeatureEng] qwen_vl: configured max_seq_len={max_len} is "
                f"already >= 256 — skipping the max_len probe pass."
            )
            return max_len

        import gc

        min_answer_tokens = 32
        worst_case = 0
        checked = 0

        for rec in records:
            if checked >= sample_size:
                break
            question = (rec.get("question") or "").strip()
            if not question:
                continue
            image = self._resolve_record_image(rec)
            if image is None:
                continue
            try:
                messages = [{"role": "user", "content": [
                    {"type": "image"}, {"type": "text", "text": question},
                ]}]
                prompt_text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                enc = processor(text=prompt_text, images=[image],
                                 return_tensors=None, truncation=False)
                real_tokens = self._unwrap_token_ids(enc["input_ids"])
                worst_case = max(worst_case, len(real_tokens))
                checked += 1
                del enc, real_tokens
            except Exception as e:
                logger.debug(f"[FeatureEng] max_len probe skipped a record: {e}")
            finally:
                if image is not None:
                    try:
                        image.close()
                    except Exception:
                        pass
                del image

        gc.collect()

        if checked == 0:
            logger.warning(
                "[FeatureEng] Could not probe any record to resolve "
                "qwen_vl's effective max_len (no usable image+question "
                f"found in first {sample_size} records) — falling back to "
                f"configured max_seq_len={max_len}."
            )
            return max_len

        needed = worst_case + min_answer_tokens
        if needed <= max_len:
            logger.info(
                f"[FeatureEng] qwen_vl max_len probe: worst-case sampled "
                f"prompt={worst_case} tokens, fits within configured "
                f"max_seq_len={max_len}. No adjustment needed."
            )
            return max_len

        resolved = min(needed, self._QWEN_VL_MAX_LEN_CEILING)
        if resolved < needed:
            logger.warning(
                f"[FeatureEng] qwen_vl max_len probe: worst-case sampled "
                f"prompt={worst_case} tokens needs {needed} total, but "
                f"that exceeds the safety ceiling of "
                f"{self._QWEN_VL_MAX_LEN_CEILING}. Using the ceiling — "
                f"some long-prompt records may still be skipped during "
                f"encoding if their prompt alone exceeds this length."
            )
        else:
            logger.info(
                f"[FeatureEng] qwen_vl max_len probe: worst-case sampled "
                f"prompt={worst_case} tokens exceeds configured "
                f"max_seq_len={max_len}. Auto-raising effective max_len to "
                f"{resolved} for this dataset so answer tokens are never "
                f"truncated away entirely. Consider setting "
                f"model_plan['max_seq_len'] >= {resolved} directly to "
                f"avoid relying on this auto-resolution."
            )
        return resolved

    def _encode_records_gen(
        self, records, processor, tokenizer, model_family, is_vision, architecture, max_len,
    ):
        """
        Generator version of the old _encode_records. Yields one encoded
        (and validated) record dict at a time instead of accumulating a
        Python list — this is what lets Dataset.from_generator write each
        record straight to Arrow on disk as it's produced, so the full
        dataset's pixel_values are never simultaneously resident in RAM as
        plain Python lists (see module docstring "STREAMING / OOM FIX").

        NOTE: `max_len` here is expected to already be the *resolved*
        effective max_len (engineer_features calls
        _resolve_effective_max_len once, up front, before constructing
        this generator) — this method does NOT call it itself, since
        gen_kwargs are re-passed verbatim by datasets internals and the
        probe must only run once per dataset, not once per generator
        invocation/retry.
        """
        import gc, time

        skipped = 0
        no_image_count = 0
        encoded_count = 0
        error_samples: List[str] = []
        seen_errors: set = set()

        GC_EVERY = 200
        PROGRESS_EVERY = 250
        total = len(records)
        t_start = time.time()

        logger.info(f"[FeatureEng] Starting streaming encode over {total} records …")

        for i, rec in enumerate(records):
            question = rec.get("question", "").strip()
            answer   = rec.get("answer",   "").strip()
            image    = self._resolve_record_image(rec)

            try:
                if not question or not answer:
                    skipped += 1
                    continue

                if is_vision and image is None:
                    no_image_count += 1
                    skipped += 1
                    continue

                try:
                    entry = self._encode_single(
                        question, answer, image, processor, tokenizer,
                        model_family, architecture, max_len,
                    )
                    reject_reason = self._validate_entry(entry, model_family) if entry else "encode_single returned None/empty"
                    if entry and not reject_reason:
                        encoded_count += 1
                        yield entry
                    else:
                        if reject_reason and reject_reason not in seen_errors and len(error_samples) < 5:
                            seen_errors.add(reject_reason)
                            error_samples.append(f"validation_rejected: {reject_reason}")
                        skipped += 1
                except Exception as e:
                    msg = f"{type(e).__name__}: {e}"
                    if msg not in seen_errors and len(error_samples) < 5:
                        seen_errors.add(msg)
                        error_samples.append(msg)
                    logger.warning(f"[FeatureEng] Record {i} encoding failed: {msg}")
                    skipped += 1
            finally:
                if image is not None:
                    try:
                        image.close()
                    except Exception:
                        pass
                del image
                if (i + 1) % GC_EVERY == 0:
                    gc.collect()
                if (i + 1) % PROGRESS_EVERY == 0 or (i + 1) == total:
                    elapsed = time.time() - t_start
                    rate = (i + 1) / elapsed if elapsed > 0 else 0.0
                    remaining = total - (i + 1)
                    eta_sec = remaining / rate if rate > 0 else float("inf")
                    eta_str = f"{eta_sec / 60:.1f} min" if eta_sec != float("inf") else "unknown"
                    logger.info(
                        f"[FeatureEng] Progress: {i + 1}/{total} records "
                        f"processed — encoded={encoded_count} skipped={skipped} "
                        f"no_image={no_image_count}  "
                        f"rate={rate:.2f} rec/s  elapsed={elapsed:.1f}s  "
                        f"ETA={eta_str}"
                    )

        gc.collect()
        logger.info(f"[FeatureEng] Encoded {encoded_count} records. Skipped {skipped}.")
        if no_image_count:
            sample_keys = list(records[0].keys()) if records else []
            sample_image_val = records[0].get("image", records[0].get("image_path", "<absent>")) if records else None
            sample_image_type = type(sample_image_val).__name__
            logger.warning(
                f"[FeatureEng] {no_image_count}/{len(records)} records had no "
                f"usable image for a vision model (model_family={model_family}). "
                f"First record keys={sample_keys}  "
                f"'image'/'image_path' field type={sample_image_type}  "
                f"value_preview={str(sample_image_val)[:120]!r}. "
                f"Check DataPreprocessingAgent's JSONL output schema — "
                f"_resolve_record_image() expects 'image_path' (filesystem "
                f"string), or 'image' as a PIL object, raw bytes, "
                f"{{'bytes':..,'path':..}} dict, or numpy array. If the JSONL "
                f"only stores a dataset index / HF dataset row id, the image "
                f"is being dropped during JSON serialization upstream and "
                f"must be re-exported to disk as actual image files."
            )
        if error_samples:
            logger.warning(f"[FeatureEng] Sample encoding/validation errors: {error_samples}")
        self._last_encode_diagnostics = {
            "no_image_count": no_image_count,
            "error_samples": error_samples,
            "total_records": len(records),
        }

    def _validate_entry(self, entry: Dict, model_family: str) -> str:
        """Returns '' if entry is valid, otherwise a short reason string
        explaining why it was rejected (used both to skip bad records and
        to surface a diagnosable cause in engineer_features()'s failure
        message instead of an opaque 'skipped' count)."""
        if "input_ids" not in entry:
            return "missing input_ids"
        if "labels" not in entry:
            return "missing labels"
        if not isinstance(entry["input_ids"], list) or len(entry["input_ids"]) == 0:
            return f"input_ids not a non-empty list (got {type(entry['input_ids']).__name__})"

        labels = entry["labels"]
        if isinstance(labels, list) and labels == entry["input_ids"]:
            return "labels identical to input_ids — masking did not run (would train on prompt/padding tokens)"
        if isinstance(labels, list) and all(t == -100 for t in labels):
            return "labels fully masked (-100 everywhere) — no answer tokens survived, nothing to learn from"

        if model_family in VISION_FAMILIES:
            if "pixel_values" not in entry:
                return "vision model but 'pixel_values' missing from encoded entry (processor call likely returned no image tensor)"
            pv = entry["pixel_values"]
            try:
                import torch
                rank = torch.as_tensor(pv).dim()
            except Exception:
                rank = len(_shape_of(pv))
            if model_family == "qwen_vl":
                if rank not in (2, 3):
                    return f"qwen_vl pixel_values rank={rank} (expected 2 or 3)"
            else:
                if rank not in (3, 4):
                    return f"{model_family} pixel_values rank={rank} (expected 3 or 4)"
        return ""

    def _encode_single(
        self, question, answer, image, processor, tokenizer,
        model_family, architecture, max_len,
    ) -> Optional[Dict]:
        if model_family in ("flan_t5", "seq2seq"):
            entry = self._encode_seq2seq(question, answer, tokenizer, max_len)
        elif model_family == "blip2":
            entry = self._encode_blip2(question, answer, image, processor, max_len)
        elif model_family == "instructblip":
            entry = self._encode_instructblip(question, answer, image, processor, max_len)
        elif model_family == "llava":
            entry = self._encode_llava(question, answer, image, processor, max_len)
        elif model_family == "qwen_vl":
            entry = self._encode_qwen_vl(question, answer, image, processor, max_len)
        elif model_family == "phi_vision":
            entry = self._encode_phi_vision(question, answer, image, processor, max_len)
        else:
            entry = self._encode_causal(question, answer, tokenizer, max_len)

        return self._flatten_sequence_fields(entry) if entry else entry

    _SEQUENCE_FIELD_TARGET_RANK = {
        "input_ids": 1, "attention_mask": 1, "labels": 1,
        "image_grid_thw": 2,
        "qformer_input_ids": 1, "qformer_attention_mask": 1,
    }

    def _flatten_sequence_fields(self, entry: Dict) -> Dict:
        import numpy as np

        def _strip_to_rank(val, target_rank: int):
            arr = np.asarray(val)
            while arr.ndim > target_rank and arr.shape[0] == 1:
                arr = arr[0]
            return arr.tolist()

        for key, target_rank in self._SEQUENCE_FIELD_TARGET_RANK.items():
            if key in entry and entry[key] is not None:
                try:
                    entry[key] = _strip_to_rank(entry[key], target_rank)
                except Exception:
                    pass

        if "pixel_values" in entry and entry["pixel_values"] is not None:
            try:
                arr = np.asarray(entry["pixel_values"])
                while arr.ndim > 4 and arr.shape[0] == 1:
                    arr = arr[0]
                if arr.ndim == 4 and arr.shape[0] == 1:
                    pass
                entry["pixel_values"] = arr.tolist()
            except Exception:
                pass

        return entry

    # ── Model-specific encoders ────────────────────────────────────────────────

    def _encode_seq2seq(self, question, answer, tokenizer, max_len) -> Dict:
        prompt = f"Medical question: {question}"
        inp = tokenizer(prompt, max_length=max_len, padding="max_length",
                         truncation=True, return_tensors=None)
        with tokenizer.as_target_tokenizer():
            tgt = tokenizer(answer, max_length=64, padding="max_length",
                             truncation=True, return_tensors=None)
        labels = [t if t != tokenizer.pad_token_id else -100 for t in tgt["input_ids"]]
        return {"input_ids": inp["input_ids"], "attention_mask": inp["attention_mask"], "labels": labels}

    def _encode_blip2(self, question, answer, image, processor, max_len) -> Dict:
        prompt = f"Question: {question} Answer:"
        if image is not None:
            enc = processor(images=image, text=prompt, return_tensors=None,
                             padding="max_length", truncation=True, max_length=max_len)
        else:
            enc = processor(text=prompt, return_tensors=None,
                             padding="max_length", truncation=True, max_length=max_len)
        tok = getattr(processor, "tokenizer", processor)
        lbl = tok(answer, max_length=64, padding="max_length", truncation=True, return_tensors=None)
        labels = [t if t != tok.pad_token_id else -100 for t in lbl["input_ids"]]
        result = {k: v for k, v in enc.items()}
        result["labels"] = labels
        return result

    def _encode_instructblip(self, question, answer, image, processor, max_len) -> Dict:
        prompt = f"Question: {question}\nAnswer:"
        if image is not None:
            enc = processor(images=image, text=prompt, return_tensors=None,
                             padding="max_length", truncation=True, max_length=max_len)
        else:
            enc = processor(text=prompt, return_tensors=None,
                             padding="max_length", truncation=True, max_length=max_len)
        tok    = getattr(processor, "tokenizer", processor)
        lbl    = tok(answer, max_length=64, truncation=True, padding="max_length", return_tensors=None)
        labels = [t if t != tok.pad_token_id else -100 for t in lbl["input_ids"]]
        result = {k: v for k, v in enc.items()}
        result["labels"] = labels
        return result

    def _encode_llava(self, question, answer, image, processor, max_len) -> Dict:
        prompt_text = f"USER: <image>\n{question}\nASSISTANT:"
        full_text   = f"{prompt_text} {answer}"

        if image is not None:
            prompt_only_enc = processor(text=prompt_text, images=image, return_tensors=None,
                                         truncation=True, max_length=max_len)
            prompt_len = len(prompt_only_enc["input_ids"])
            enc = processor(text=full_text, images=image, return_tensors=None,
                             padding="max_length", truncation=True, max_length=max_len)
            tok = getattr(processor, "tokenizer", processor)
        else:
            tok = getattr(processor, "tokenizer", processor)
            prompt_only_enc = tok(prompt_text, return_tensors=None, truncation=True, max_length=max_len)
            prompt_len = len(prompt_only_enc["input_ids"])
            enc = tok(full_text, return_tensors=None, padding="max_length",
                      truncation=True, max_length=max_len)

        result = {k: v for k, v in enc.items()}
        pad_id = getattr(tok, "pad_token_id", None)
        result["labels"] = self._mask_prompt_and_padding(result["input_ids"], prompt_len, pad_id)
        return result

    def _encode_qwen_vl(self, question, answer, image, processor, max_len) -> Dict:
        tok = getattr(processor, "tokenizer", processor)
        min_answer_tokens = 4

        if image is not None and hasattr(processor, "apply_chat_template"):
            messages = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": question},
            ]}]
            prompt_text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            full_text = prompt_text + answer

            prompt_only_enc = processor(text=prompt_text, images=[image],
                                         return_tensors=None, truncation=False)
            true_prompt_len = len(self._unwrap_token_ids(prompt_only_enc["input_ids"]))

            if true_prompt_len + min_answer_tokens > max_len:
                raise ValueError(
                    f"qwen_vl prompt (image-expanded) is {true_prompt_len} "
                    f"tokens, leaving no room for an answer within "
                    f"max_seq_len={max_len}. Skipping this record. If this "
                    f"happens for most/all records, model_plan['max_seq_len'] "
                    f"is too small for this model's image resolution — it "
                    f"should be resolved automatically by "
                    f"_resolve_effective_max_len() before encoding starts; "
                    f"seeing this error means that resolution step itself "
                    f"needs a higher ceiling or this image is unusually "
                    f"large."
                )

            prompt_len = min(true_prompt_len, max_len)

            enc = processor(text=full_text, images=[image], return_tensors=None,
                             padding="max_length", truncation=True, max_length=max_len)
        else:
            prompt_text = f"Question: {question}\nAnswer:"
            prompt_only_enc = tok(prompt_text, return_tensors=None, truncation=False)
            true_prompt_len = len(self._unwrap_token_ids(prompt_only_enc["input_ids"]))

            if true_prompt_len + min_answer_tokens > max_len:
                raise ValueError(
                    f"Text-only prompt is {true_prompt_len} tokens, leaving "
                    f"no room for an answer within max_seq_len={max_len}. "
                    f"Skipping this record."
                )
            prompt_len = min(true_prompt_len, max_len)

            full_text = f"{prompt_text} {answer}"
            enc = tok(full_text, return_tensors=None, padding="max_length",
                      truncation=True, max_length=max_len)

        result = {k: v for k, v in enc.items()}
        result["input_ids"] = self._unwrap_token_ids(result["input_ids"])
        if "attention_mask" in result:
            result["attention_mask"] = self._unwrap_token_ids(result["attention_mask"])
        pad_id = getattr(tok, "pad_token_id", None)
        result["labels"] = self._mask_prompt_and_padding(result["input_ids"], prompt_len, pad_id)

        if "image_grid_thw" in result:
            g = result["image_grid_thw"]
            if isinstance(g, list) and len(g) == 1 and isinstance(g[0], list) and isinstance(g[0][0], list):
                result["image_grid_thw"] = g[0]
        return result

    def _encode_phi_vision(self, question, answer, image, processor, max_len) -> Dict:
        if image is not None:
            prompt_text = f"<|user|>\n<|image_1|>\n{question}<|end|>\n<|assistant|>\n"
            full_text   = f"{prompt_text}{answer}<|end|>"
            prompt_only_enc = processor(text=prompt_text, images=[image], return_tensors=None,
                                         truncation=True, max_length=max_len)
            prompt_len = len(prompt_only_enc["input_ids"])
            enc = processor(text=full_text, images=[image], return_tensors=None,
                             padding="max_length", truncation=True, max_length=max_len)
            tok = getattr(processor, "tokenizer", processor)
        else:
            tok = getattr(processor, "tokenizer", processor)
            prompt_text = f"<|user|>\n{question}<|end|>\n<|assistant|>\n"
            full_text   = f"{prompt_text}{answer}<|end|>"
            prompt_only_enc = tok(prompt_text, return_tensors=None, truncation=True, max_length=max_len)
            prompt_len = len(prompt_only_enc["input_ids"])
            enc = tok(full_text, return_tensors=None, padding="max_length",
                      truncation=True, max_length=max_len)

        result = {k: v for k, v in enc.items()}
        pad_id = getattr(tok, "pad_token_id", None)
        result["labels"] = self._mask_prompt_and_padding(result["input_ids"], prompt_len, pad_id)
        return result

    def _encode_causal(self, question, answer, tokenizer, max_len) -> Dict:
        text = f"Question: {question}\nAnswer: {answer}"
        tok  = getattr(tokenizer, "tokenizer", tokenizer)
        enc  = tok(text, max_length=max_len, padding="max_length",
                   truncation=True, return_tensors=None)
        labels = [t if t != (tok.pad_token_id or 0) else -100 for t in enc["input_ids"]]
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels}

    # ── Image loading helper ──────────────────────────────────────────────────

    def _load_image(self, image_path: str):
        if not image_path or not Path(image_path).exists():
            return None
        try:
            from PIL import Image
            return Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.debug(f"[FeatureEng] Image load failed {image_path}: {e}")
            return None

    def _resolve_record_image(self, rec: Dict[str, Any]):
        from PIL import Image
        import io

        image_path = (
            rec.get("image_path") or rec.get("img_path")
            or rec.get("image_file") or rec.get("file_path") or ""
        )
        if image_path:
            img = self._load_image(image_path)
            if img is not None:
                return img

        raw = rec.get("image", rec.get("img", rec.get("pixel_data")))
        if raw is None:
            return None

        try:
            if isinstance(raw, Image.Image):
                return raw.convert("RGB")
            if isinstance(raw, dict):
                if raw.get("bytes"):
                    return Image.open(io.BytesIO(raw["bytes"])).convert("RGB")
                if raw.get("path"):
                    return self._load_image(raw["path"])
                return None
            if isinstance(raw, (bytes, bytearray)):
                return Image.open(io.BytesIO(raw)).convert("RGB")
            if isinstance(raw, str):
                import base64
                s = raw.split(",", 1)[-1] if raw.startswith("data:") else raw
                try:
                    decoded = base64.b64decode(s, validate=True)
                    return Image.open(io.BytesIO(decoded)).convert("RGB")
                except Exception:
                    return self._load_image(raw)
            import numpy as np
            if isinstance(raw, np.ndarray):
                return Image.fromarray(raw).convert("RGB")
        except Exception as e:
            logger.debug(f"[FeatureEng] Could not resolve embedded image: {e}")
        return None

    # ── HuggingFace Dataset / cache persistence ────────────────────────────────

    def _save_to_disk(self, train_ds, val_ds, feature_path: str, meta: Dict[str, Any]) -> str:
        import time
        base_path = Path(feature_path)
        base_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"[FeatureEng] Saving to disk — train={len(train_ds)} "
            f"val={len(val_ds)} records (this can take a while for "
            f"vision datasets with pixel_values; no per-shard progress is "
            f"logged by datasets.save_to_disk itself, so a multi-minute "
            f"gap here is expected for larger datasets, not a hang)."
        )
        t0 = time.time()
        train_ds.save_to_disk(str(base_path / "train"))
        logger.info(f"[FeatureEng] train/ saved in {time.time() - t0:.1f}s")

        t1 = time.time()
        val_ds.save_to_disk(str(base_path / "val"))
        logger.info(f"[FeatureEng] val/ saved in {time.time() - t1:.1f}s")

        (base_path / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
        logger.info(
            f"[FeatureEng] Saved to {base_path} (metadata.json written) — "
            f"total save time {time.time() - t0:.1f}s"
        )
        return str(base_path)

    # ── LLM assessment ────────────────────────────────────────────────────────

    def _get_llm_assessment(self, hf_id, model_family, is_vision, train_n, val_n, columns) -> Dict:
        import time
        prompt = (
            f"Feature engineering completed for Medical VQA.\n"
            f"Model: {hf_id}\nFamily: {model_family}  Vision: {is_vision}\n"
            f"Train samples: {train_n}  Val samples: {val_n}\n"
            f"Encoded columns: {columns}\n\n"
            f"Are these features ready for model training?\n"
            f'Reply with ONLY: {{"status": "ok", "train_samples": {train_n}, '
            f'"val_samples": {val_n}, "message": "<one sentence>"}}'
        )
        logger.info("[FeatureEng] Requesting LLM assessment (Groq) …")
        t0 = time.time()
        try:
            response = self.agent.run(prompt)
            logger.info(f"[FeatureEng] LLM assessment returned in {time.time() - t0:.1f}s")
            return self._parse_response(response)
        except Exception as e:
            logger.warning(
                f"[FeatureEng] LLM assessment failed after "
                f"{time.time() - t0:.1f}s: {e}"
            )
            return {"status": "ok", "message": "Feature engineering complete."}

    def _parse_response(self, response) -> Dict:
        try:
            text  = response.content if hasattr(response, "content") else str(response)
            match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            pass
        return {"status": "ok", "message": "Feature engineering complete."}


def _shape_of(x) -> List[int]:
    shape: List[int] = []
    cur = x
    while True:
        if hasattr(cur, "shape"):
            shape.extend(list(cur.shape))
            break
        if isinstance(cur, list):
            shape.append(len(cur))
            cur = cur[0] if cur else None
        else:
            break
    return shape


if __name__ == "__main__":
    import torch
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice : {device}")

    agent = FeatureEngineeringAgent()
    plan = {
        "hf_id": "google/flan-t5-base", "name": "Flan-T5-Base",
        "architecture": "seq2seq", "vision": False, "max_seq_len": 128,
    }
    data_path = "./data/processed/processed_dataset.jsonl"
    if not Path(data_path).exists():
        print(f"\nData file not found: {data_path}")
    else:
        result = agent.engineer_features(data_path, plan, device=device)
        print("\n" + "=" * 50)
        print("Feature Engineering Result:")
        print(json.dumps(result, indent=2, default=str))
        print("=" * 50)
