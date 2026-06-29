#!/usr/bin/env python3
"""process.py — Turn raw data into cached feature vectors.

Steps:
  1. Load + merge raw CSVs.
  2. Stratified 80/10/10 split.
  3. For each split: load each image, run the zoom/white-canvas preprocessing,
     push through the frozen MobileNetV2 backbone, and save the 1280-d feature
     vectors to RAW .npy files immediately (X_<split>_raw.npy). Processed images
     are NOT written to disk — we go straight to features, saving ~2.2 GB.
  4. Build the final train set by resampling the raw train features to balance
     classes (~RESAMPLE_TARGET each), using row indices + a memory-mapped read so
     RAM stays low. val/test final files are copies of their raw features.

Crash-safe & resumable: because raw features are written as soon as a split is
extracted, the expensive backbone pass is never lost. Rerunning skips any split
whose raw features already exist.

This is the long-running step (one forward pass over ~84k images).
Use --limit for a quick smoke run on a stratified subset.

Examples:
  python scripts/process.py
  python scripts/process.py --limit 540     # quick test on a subset
"""
from __future__ import annotations

import argparse
import json
import shutil
import time

import _bootstrap  # noqa: F401
import numpy as np
from tqdm.auto import tqdm

from rakuten_img import backbone, config, data, images


def _features_for_split(df, desc: str):
    """Extract features for every row in df. Returns (X, y, n_skipped).

    Rows whose image fails to load are skipped from BOTH X and y so they stay
    aligned.
    """
    paths = df["imagepath"].tolist()
    labels = df["prdtypecode"].to_numpy()

    feats: list[np.ndarray] = []
    kept_labels: list[int] = []
    skipped = 0

    pbar = tqdm(total=len(paths), desc=desc)
    for batch_idx in backbone.iter_batches(list(range(len(paths))), config.FEATURE_BATCH_SIZE):
        imgs, idxs = [], []
        for i in batch_idx:
            try:
                imgs.append(images.load_and_process(paths[i]))
                idxs.append(i)
            except Exception:
                skipped += 1
        if imgs:
            batch_feats = backbone.extract_features(imgs)
            feats.append(batch_feats)
            kept_labels.extend(int(labels[i]) for i in idxs)
        pbar.update(len(batch_idx))
    pbar.close()

    X = np.vstack(feats) if feats else np.empty((0, config.FEATURE_DIM), np.float32)
    y = np.asarray(kept_labels, dtype=np.int64)
    return X, y, skipped


def _extract_and_save_raw(split_df, name: str) -> int:
    """Extract features for a split and save them to the RAW files immediately.
    Resumable: if the raw files already exist, skip the (expensive) extraction.
    Returns the number of samples in the split.
    """
    rx, ry = config.feature_files_raw(name)
    if rx.exists() and ry.exists():
        n = len(np.load(ry, mmap_mode="r"))
        print(f"⏭️  {name}: raw features already cached ({n:,}) — skipping extraction")
        return n

    X, y, skipped = _features_for_split(split_df, desc=f"features:{name}")
    np.save(rx, X)          # <-- saved to disk NOW, before anything can fail
    np.save(ry, y)
    print(f"💾 {name} raw: X{X.shape} -> {rx.name} (skipped {skipped})")
    del X, y                # free RAM right away
    return int(len(np.load(ry, mmap_mode="r")))


def process(limit: int = 0) -> dict:
    """Extract & cache features for train/val/test, then finalize (resample train,
    copy val/test). Returns the features-metadata dict (identical to what is
    written to features_meta.json) so a caller — the API, or a future orchestrator
    — can consume the run result instead of re-reading the file.

    `limit` mirrors the --limit CLI flag: if >0, use only a stratified subset.
    """
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("📊 Loading raw dataframe...")
    df = data.load_raw_dataframe()
    if limit:
        df = df.groupby("prdtypecode", group_keys=False).head(
            max(2, limit // config.NUM_CLASSES)
        )
        print(f"⚠️  --limit active: using {len(df):,} rows")
    print(f"✅ {len(df):,} products | {df['prdtypecode'].nunique()} classes")

    train_df, val_df, test_df = data.split_dataframe(df)
    print(f"Split -> train {len(train_df):,} | val {len(val_df):,} | test {len(test_df):,}")

    # --- 1) Extraction (resumable, raw features saved to disk immediately) ---
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        _extract_and_save_raw(split_df, name)

    # --- 2) Finalize: resample TRAIN (memory-light via memmap), copy val/test ---
    summary = {}

    print(f"Resampling train to ~{config.RESAMPLE_TARGET}/class (low-memory)...")
    rtx, rty = config.feature_files_raw("train")
    y_train_raw = np.load(rty)
    idx = data.resample_indices(y_train_raw)               # just an index array
    X_train_mm = np.load(rtx, mmap_mode="r")               # not loaded into RAM
    X_train = np.asarray(X_train_mm[idx])                  # single allocation
    y_train = y_train_raw[idx]
    fx, fy = config.feature_files("train")
    np.save(fx, X_train); np.save(fy, y_train)
    summary["train"] = {"samples": int(len(y_train)), "feature_dim": int(X_train.shape[1])}
    print(f"💾 train: X{X_train.shape} -> {fx.name}")
    del X_train, X_train_mm, y_train, y_train_raw

    for name in ("val", "test"):
        rx, ry = config.feature_files_raw(name)
        fx, fy = config.feature_files(name)
        shutil.copyfile(rx, fx)
        shutil.copyfile(ry, fy)
        n = int(len(np.load(fy, mmap_mode="r")))
        summary[name] = {"samples": n, "feature_dim": config.FEATURE_DIM}
        print(f"💾 {name}: {n:,} samples -> {fx.name}")

    meta = {
        "backbone": config.BACKBONE_NAME,
        "feature_dim": config.FEATURE_DIM,
        "resample_target": config.RESAMPLE_TARGET,
        "splits": summary,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (config.PROCESSED_DIR / "features_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"📝 Wrote features_meta.json | total {meta['elapsed_sec']}s")
    print("🎉 process.py done.")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract & cache image features.")
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, use only the first N rows (quick smoke run).")
    args = ap.parse_args()
    process(limit=args.limit)


if __name__ == "__main__":
    main()