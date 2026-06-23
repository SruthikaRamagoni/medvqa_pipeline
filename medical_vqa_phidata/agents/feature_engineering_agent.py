"""
PATCH — agents/feature_engineering_agent.py
============================================

TWO BUGS FIXED
--------------

BUG 1 (CRITICAL — causes Encoded=0, Skipped=2244)
    Location : _encode_phi_vision()
    Symptom  : All 2244 records skipped; _validate_entry() rejects every
               record with "labels identical to input_ids — masking did not run"
    Root cause:
        prompt_only_enc = processor(text=..., images=[image],
                                     return_tensors=None, ...)
        prompt_len = len(prompt_only_enc["input_ids"])   # ← BUG

        When return_tensors=None, Phi-3.5-vision's AutoProcessor wraps
        input_ids as [[id, id, ...]] (a batch-of-one list).
        len([[ids...]]) == 1, not the real token count.

        _mask_prompt_and_padding() therefore masks only index 0,
        leaving labels[1:] == input_ids[1:] — i.e. effectively unmasked.
        _validate_entry() catches this and rejects every record → Encoded=0.

    Fix: route prompt_only_enc["input_ids"] through the already-existing
         _unwrap_token_ids() helper (exactly as _encode_qwen_vl does)
         before calling len().

BUG 2 (LATENT — same root cause, would fire for LLaVA models)
    Location : _encode_llava()
    Root cause: identical — prompt_len = len(prompt_only_enc["input_ids"])
                without unwrapping.
    Fix: same — use _unwrap_token_ids().

HOW TO APPLY
------------
Replace the two methods in agents/feature_engineering_agent.py with the
corrected versions below. No other changes needed.
"""


# ─────────────────────────────────────────────────────────────────────────────
# DROP-IN REPLACEMENT for FeatureEngineeringAgent._encode_phi_vision
# ─────────────────────────────────────────────────────────────────────────────

def _encode_phi_vision(self, question, answer, image, processor, max_len):
    """
    Encode a single record for Phi-3.5-vision-instruct.

    FIX (vs previous version):
        prompt_only_enc["input_ids"] is passed through _unwrap_token_ids()
        before len() is called, because Phi's AutoProcessor returns
        [[id, id, ...]] (batch-of-one wrapper) when return_tensors=None.
        Without the unwrap, len() == 1 instead of the real token count,
        which caused _mask_prompt_and_padding to mask only 1 token,
        leaving labels ≈ input_ids and causing _validate_entry to reject
        EVERY record → Encoded=0, Skipped=2244.
    """
    if image is not None:
        prompt_text = f"<|user|>\n<|image_1|>\n{question}<|end|>\n<|assistant|>\n"
        full_text   = f"{prompt_text}{answer}<|end|>"

        prompt_only_enc = processor(
            text=prompt_text, images=[image],
            return_tensors=None, truncation=True, max_length=max_len,
        )
        # ── FIX: unwrap [[ids...]] → [ids...] before measuring length ──
        prompt_ids = self._unwrap_token_ids(prompt_only_enc["input_ids"])
        prompt_len = len(prompt_ids)

        enc = processor(
            text=full_text, images=[image],
            return_tensors=None, padding="max_length",
            truncation=True, max_length=max_len,
        )
        tok = getattr(processor, "tokenizer", processor)
    else:
        tok = getattr(processor, "tokenizer", processor)
        prompt_text = f"<|user|>\n{question}<|end|>\n<|assistant|>\n"
        full_text   = f"{prompt_text}{answer}<|end|>"

        prompt_only_enc = tok(
            prompt_text, return_tensors=None,
            truncation=True, max_length=max_len,
        )
        # ── FIX: unwrap here too for consistency ──
        prompt_ids = self._unwrap_token_ids(prompt_only_enc["input_ids"])
        prompt_len = len(prompt_ids)

        enc = tok(
            full_text, return_tensors=None, padding="max_length",
            truncation=True, max_length=max_len,
        )

    result = {k: v for k, v in enc.items()}
    pad_id = getattr(tok, "pad_token_id", None)
    result["labels"] = self._mask_prompt_and_padding(
        result["input_ids"], prompt_len, pad_id
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# DROP-IN REPLACEMENT for FeatureEngineeringAgent._encode_llava
# (same latent bug, same fix)
# ─────────────────────────────────────────────────────────────────────────────

def _encode_llava(self, question, answer, image, processor, max_len):
    """
    Encode a single record for LLaVA-family models.

    FIX (vs previous version):
        prompt_only_enc["input_ids"] unwrapped via _unwrap_token_ids()
        before len() — same batch-of-one wrapper issue as phi_vision.
    """
    prompt_text = f"USER: <image>\n{question}\nASSISTANT:"
    full_text   = f"{prompt_text} {answer}"

    if image is not None:
        prompt_only_enc = processor(
            text=prompt_text, images=image,
            return_tensors=None, truncation=True, max_length=max_len,
        )
        # ── FIX: unwrap before len() ──
        prompt_ids = self._unwrap_token_ids(prompt_only_enc["input_ids"])
        prompt_len = len(prompt_ids)

        enc = processor(
            text=full_text, images=image,
            return_tensors=None, padding="max_length",
            truncation=True, max_length=max_len,
        )
        tok = getattr(processor, "tokenizer", processor)
    else:
        tok = getattr(processor, "tokenizer", processor)

        prompt_only_enc = tok(
            prompt_text, return_tensors=None,
            truncation=True, max_length=max_len,
        )
        # ── FIX: unwrap before len() ──
        prompt_ids = self._unwrap_token_ids(prompt_only_enc["input_ids"])
        prompt_len = len(prompt_ids)

        enc = tok(
            full_text, return_tensors=None, padding="max_length",
            truncation=True, max_length=max_len,
        )

    result = {k: v for k, v in enc.items()}
    pad_id = getattr(tok, "pad_token_id", None)
    result["labels"] = self._mask_prompt_and_padding(
        result["input_ids"], prompt_len, pad_id
    )
    return result
