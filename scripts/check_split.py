#!/usr/bin/env python3
"""Prove that rakuten_common.split reproduces the image pipeline's split EXACTLY.

This is the gate for step A2. Until it passes on the real data, nothing gets
rewired to use the shared module. It reads the CSVs and computes splits in
memory; it writes nothing, touches no model, and needs neither torch nor images.

    python scripts/check_split.py

Exit code 0 = identical. Non-zero = do not proceed.
"""
from __future__ import annotations

import sys

import _bootstrap  # noqa: F401  (puts src/ on sys.path)

import pandas as pd

from rakuten_common import split as shared
from rakuten_img import config, data


def main() -> int:
    print("Loading via rakuten_img.data.load_raw_dataframe() ...")
    img_df = data.load_raw_dataframe()
    print(f"  image loader : {len(img_df)} rows, index name={img_df.index.name!r}")

    print("Loading via rakuten_common.split.load_labeled_dataframe() ...")
    shared_df = shared.load_labeled_dataframe()
    print(f"  shared loader: {len(shared_df)} rows, index name={shared_df.index.name!r}")

    ok = True

    # 1. Same rows, in the same order, before any splitting.
    same_index = img_df.index.equals(shared_df.index)
    print(f"\n[1] identical row index & order : {same_index}")
    ok &= same_index

    # 2. Same labels attached to those rows.
    same_labels = img_df[shared.LABEL_COLUMN].equals(shared_df[shared.LABEL_COLUMN])
    print(f"[2] identical prdtypecode column: {same_labels}")
    ok &= same_labels

    # 3. The splits themselves.
    i_tr, i_va, i_te = data.split_dataframe(img_df)
    parts = shared.split_labels(shared_df)

    for name, img_part in (("train", i_tr), ("val", i_va), ("test", i_te)):
        shared_ix = parts[name]
        exact = img_part.index.equals(shared_ix)          # order-sensitive
        as_set = set(img_part.index) == set(shared_ix)    # membership only
        print(f"[3] {name:<5} n={len(img_part):>6} | exact order: {exact} | "
              f"same membership: {as_set}")
        ok &= exact

    # 4. Sanity: the partition covers everything exactly once.
    total = sum(len(p) for p in parts.values())
    print(f"\n[4] partition total {total} == {len(shared_df)} rows: "
          f"{total == len(shared_df)}")
    ok &= total == len(shared_df)

    # 5. Stratification actually held (all 27 classes present in every split).
    for name, ix in parts.items():
        n_cls = shared_df.loc[ix, shared.LABEL_COLUMN].nunique()
        print(f"[5] {name:<5} distinct classes: {n_cls} / {config.NUM_CLASSES}")
        ok &= n_cls == config.NUM_CLASSES

    # 6. What the text side has been using until now, for contrast.
    from sklearn.model_selection import train_test_split
    _, text_val = train_test_split(
        shared_df, test_size=0.2, random_state=config.RANDOM_STATE,
        shuffle=True, stratify=shared_df[shared.LABEL_COLUMN],
    )
    overlap = len(set(text_val.index) & set(parts["val"].tolist()))
    print(f"\n[6] old text 80/20 val set: {len(text_val)} rows; overlap with the "
          f"shared val set: {overlap} rows "
          f"({100.0 * overlap / max(len(parts['val']), 1):.1f}% of shared val)")
    print("    (informational -- this is the gap A2 exists to close)")

    print("\n" + ("=" * 62))
    print("RESULT:", "IDENTICAL - safe to rewire" if ok else "MISMATCH - DO NOT PROCEED")
    print("=" * 62)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
