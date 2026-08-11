#!/usr/bin/env python3
"""Measure the fusion weight instead of guessing it.

fusion.weighted_average defaults to text_weight=0.5. Measured separately, the
text model scores ~0.78 weighted F1 and the image head ~0.55, so an even split
almost certainly drags the stronger model down. Now that both modalities are
scored on the SAME held-out products (rakuten_common.split), that weight can be
measured.

METHOD
    Tune on VAL, confirm ONCE on TEST. Sweeping on test and then reporting the
    best test number is choosing a hyperparameter on the test set -- the number
    would be optimistic and the test set spent.

        python scripts/tune_fusion_weight.py                  # sweep on val
        python scripts/tune_fusion_weight.py --weight 0.7 --split test

SANITY CHECK, NOT DECORATION
    weight 0.0 must reproduce the image-only score and weight 1.0 the text-only
    score. If either endpoint is off, the two modalities are misaligned -- the
    products are not lined up row for row -- and every number here is garbage.
    The script says so loudly rather than printing a plausible curve.

Needs no images and no backbone: image probabilities come from the cached
features, text probabilities from the saved vectorizer.
"""
from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from rakuten_common import features as feat
from rakuten_common import split as shared
from rakuten_common.contract import to_canonical
from rakuten_img import classifier, config, fusion
from rakuten_text.predict import TfidfPredictor
from rakuten_text.preprocessing import preparer_dataframe

WEIGHTS = [round(w, 2) for w in np.arange(0.0, 1.001, 0.05)]

_IMAGE_RUN_ID: str | None = None  # set by _probabilities, reported in the JSON


def _probabilities(split: str):
    """(image_proba, text_proba, y_true) for `split`, aligned row for row."""
    df = shared.load_labeled_dataframe()

    # aligned_index re-verifies the cached labels against the shared split and
    # raises if they disagree, so alignment is checked, not assumed.
    ix = feat.aligned_index(split, df=df)
    y_true = df.loc[ix, shared.LABEL_COLUMN].to_numpy()

    print(f"[image] loading cached features for '{split}' ...")
    X = np.load(config.feature_files(split)[0], mmap_mode="r")
    global _IMAGE_RUN_ID
    payload = classifier.load()
    _IMAGE_RUN_ID = payload.get("mlflow_run_id")
    clf = payload["classifier"]
    img_proba = to_canonical(clf.predict_proba(np.asarray(X)), clf.classes_)
    print(f"[image] {img_proba.shape} from model run "
          f"{str(payload.get('mlflow_run_id'))[:8]}")

    print(f"[text ] vectorizing {len(ix):,} products ...")
    df_txt = preparer_dataframe(df.loc[ix])
    predictor = TfidfPredictor().load()
    Xt = predictor.vectorizer.transform(df_txt["texte_complet"].fillna(""))
    txt_proba = to_canonical(predictor.model.predict_proba(Xt),
                             predictor.model.classes_)
    print(f"[text ] {txt_proba.shape}")

    if img_proba.shape != txt_proba.shape:
        raise SystemExit(f"Shape mismatch: {img_proba.shape} vs {txt_proba.shape}")
    return img_proba, txt_proba, y_true


def _score(img_proba, txt_proba, y_true, weight: float) -> dict:
    fused = fusion.weighted_average(img_proba, txt_proba, text_weight=weight)
    y_pred = np.asarray(config.CANONICAL_CLASSES)[fused.argmax(axis=1)]
    return {
        "text_weight": weight,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Measure the fusion weight.")
    ap.add_argument("--split", choices=("val", "test"), default="val")
    ap.add_argument("--weight", type=float, default=None,
                    help="Score a single weight instead of sweeping (use with "
                         "--split test to confirm a weight chosen on val).")
    args = ap.parse_args()

    if args.split == "test" and args.weight is None:
        raise SystemExit(
            "Refusing to sweep on the test set. Sweep on val, then confirm the "
            "chosen weight with --weight W --split test."
        )

    img_proba, txt_proba, y_true = _probabilities(args.split)

    if args.weight is not None:
        row = _score(img_proba, txt_proba, y_true, args.weight)
        print(f"\ntext_weight={row['text_weight']:.2f} on '{args.split}': "
              f"accuracy={row['accuracy']:.4f} f1_weighted={row['f1_weighted']:.4f} "
              f"f1_macro={row['f1_macro']:.4f}")
        return 0

    rows = [_score(img_proba, txt_proba, y_true, w) for w in WEIGHTS]

    print(f"\n{'weight':>7} {'accuracy':>10} {'f1_weighted':>12} {'f1_macro':>10}")
    print("-" * 42)
    best = max(rows, key=lambda r: r["f1_weighted"])
    for r in rows:
        star = " <-- best" if r is best else ""
        print(f"{r['text_weight']:>7.2f} {r['accuracy']:>10.4f} "
              f"{r['f1_weighted']:>12.4f} {r['f1_macro']:>10.4f}{star}")

    image_only, text_only = rows[0], rows[-1]
    print("\nENDPOINT CHECK (these must match the standalone models)")
    print(f"  weight 0.00 (image only): f1_weighted={image_only['f1_weighted']:.4f}")
    print(f"  weight 1.00 (text only) : f1_weighted={text_only['f1_weighted']:.4f}")
    print("  Compare against the image model's val f1_weighted and the text")
    print("  model's val f1_weighted. A mismatch means the rows are NOT aligned")
    print("  and every number above is meaningless.")

    print(f"\nBEST on '{args.split}': text_weight={best['text_weight']:.2f} "
          f"-> f1_weighted={best['f1_weighted']:.4f}")
    gain = best["f1_weighted"] - max(image_only["f1_weighted"], text_only["f1_weighted"])
    print(f"Gain over the better single modality: {gain:+.4f}")
    if gain <= 0:
        print("NOTE: fusion does not beat the best single model here. That is a "
              "real result, not a bug -- report it rather than tuning until it "
              "looks better.")

    out_dir = config.REPORTS_DIR / "fusion"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"weight_sweep_{args.split}.json"
    out.write_text(json.dumps(
        {"split": args.split, "n": int(len(y_true)),
         "image_run_id": _IMAGE_RUN_ID,
         "best": best, "sweep": rows}, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
