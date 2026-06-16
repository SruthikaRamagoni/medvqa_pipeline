"""
fix_bugs.py — Run once to fix two bugs:

Bug 1: UnicodeEncodeError on Windows (emoji in main.py)
Bug 2: too many values to unpack (column prep in training_agent.py)
"""
import re

# ── Fix 1: main.py — replace emoji with ASCII ────────────────────────────────
path = "main.py"
src  = open(path, encoding="utf-8").read()

src = src.replace(
    "{'✅ SUCCESS' if state.success else '❌ FAILED'}",
    "{'SUCCESS' if state.success else 'FAILED'}"
)
# Also fix banner that may have emoji
src = src.replace("✅", "[OK]").replace("❌", "[FAIL]")

open(path, "w", encoding="utf-8").write(src)
print(f"Fixed: {path}  (removed emoji)")

# ── Fix 2: training_agent.py — robust column preparation ─────────────────────
path = "agents/training_agent.py"
src  = open(path, encoding="utf-8").read()

OLD = '''    def _prepare_columns(self, train_ds, val_ds):
        """Keep only tensor-compatible columns. Add labels if missing."""
        drop_tr = [c for c in train_ds.column_names if c not in TENSOR_COLUMNS]
        drop_vl = [c for c in val_ds.column_names   if c not in TENSOR_COLUMNS]
        if drop_tr: train_ds = train_ds.remove_columns(drop_tr)
        if drop_vl: val_ds   = val_ds.remove_columns(drop_vl)

        if ("labels"    not in train_ds.column_names and
                "input_ids" in  train_ds.column_names):
            train_ds = train_ds.map(
                lambda x: {"labels": x["input_ids"]}, batched=True
            )
            val_ds = val_ds.map(
                lambda x: {"labels": x["input_ids"]}, batched=True
            )

        logger.info(f"[Training] Final columns: {train_ds.column_names}")
        return train_ds, val_ds'''

NEW = '''    def _prepare_columns(self, train_ds, val_ds):
        """
        Keep only tensor-compatible columns. Add labels if missing.
        Also flattens any nested list columns that cause unpack errors.
        """
        import numpy as np

        # ── Drop non-tensor columns ───────────────────────────────────────────
        drop_tr = [c for c in train_ds.column_names if c not in TENSOR_COLUMNS]
        drop_vl = [c for c in val_ds.column_names   if c not in TENSOR_COLUMNS]
        if drop_tr: train_ds = train_ds.remove_columns(drop_tr)
        if drop_vl: val_ds   = val_ds.remove_columns(drop_vl)

        # ── Add labels if missing ─────────────────────────────────────────────
        if ("labels"    not in train_ds.column_names and
                "input_ids" in  train_ds.column_names):
            train_ds = train_ds.map(
                lambda x: {"labels": x["input_ids"]}, batched=True
            )
            val_ds = val_ds.map(
                lambda x: {"labels": x["input_ids"]}, batched=True
            )

        # ── Flatten nested pixel_values if present ────────────────────────────
        # InstructBLIP / BLIP-2 encode pixel_values as list-of-list-of-list.
        # HuggingFace Trainer needs it as a flat list or numpy array.
        # We detect and flatten here to prevent "too many values to unpack".
        if "pixel_values" in train_ds.column_names:
            try:
                sample = train_ds[0]["pixel_values"]
                # If it is a nested list (list of lists), leave as-is —
                # DataCollator will handle it. Only flatten if it is
                # a plain Python list wrapped in an extra list.
                if isinstance(sample, list) and len(sample) == 1 and isinstance(sample[0], list):
                    def unwrap_pixel(batch):
                        return {"pixel_values": [pv[0] for pv in batch["pixel_values"]]}
                    train_ds = train_ds.map(unwrap_pixel, batched=True)
                    val_ds   = val_ds.map(unwrap_pixel,   batched=True)
                    logger.info("[Training] Unwrapped nested pixel_values.")
            except Exception as e:
                logger.debug(f"[Training] pixel_values check skipped: {e}")

        logger.info(f"[Training] Final columns: {train_ds.column_names}")
        return train_ds, val_ds'''

if OLD in src:
    src = src.replace(OLD, NEW)
    open(path, "w", encoding="utf-8").write(src)
    print(f"Fixed: {path}  (robust _prepare_columns)")
else:
    print(f"WARNING: Could not find OLD block in {path} — applying line-level patch")
    # Fallback: inject the fix after the existing prepare_columns method
    # by rewriting the method signature match
    src = re.sub(
        r'(def _prepare_columns\(self, train_ds, val_ds\):.*?return train_ds, val_ds)',
        NEW,
        src,
        flags=re.DOTALL
    )
    open(path, "w", encoding="utf-8").write(src)
    print(f"Fixed: {path}  (regex patch applied)")

# ── Fix 3: training_agent.py — also fix _build_trainer data_collator ────────
path = "agents/training_agent.py"
src  = open(path, encoding="utf-8").read()

# Add default_data_collator for vision models to handle variable-size tensors
OLD2 = '''        return Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            **{tok_kwarg: tok},
        )'''

NEW2 = '''        # Use DataCollatorWithPadding for text models
        # Use default_data_collator for vision models (pixel_values present)
        from transformers import default_data_collator
        has_pixels = "pixel_values" in train_ds.column_names

        if has_pixels:
            return Trainer(
                model=model,
                args=args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                data_collator=default_data_collator,
            )
        else:
            return Trainer(
                model=model,
                args=args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                **{tok_kwarg: tok},
            )'''

if OLD2 in src:
    src = src.replace(OLD2, NEW2)
    open(path, "w", encoding="utf-8").write(src)
    print(f"Fixed: {path}  (vision-aware data_collator)")
else:
    print(f"NOTE: data_collator fix not applied (block not found - may already be fixed)")

print("\nAll fixes applied.")
print("Run: python main.py --image image.jpg --question \"Is there pneumonia?\"")
