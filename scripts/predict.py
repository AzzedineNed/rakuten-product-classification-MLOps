#!/usr/bin/env python3
"""predict.py — Image-only inference.

Image -> zoom/white-canvas preprocessing -> frozen MobileNetV2 -> classifier
-> probability vector in CANONICAL order -> top-k labels.

Exposes reusable functions (used by the API and importable by a fusion layer):
  * predict_proba(pil_image) -> np.ndarray of shape (NUM_CLASSES,) in canonical order
  * predict(image_path, top_k) -> list of {code, label, probability}

CLI:
  python scripts/predict.py --image path/to/product.jpg
  python scripts/predict.py --image a.jpg b.jpg --top-k 3
"""
from __future__ import annotations

import argparse
import json
from functools import lru_cache

import _bootstrap  # noqa: F401
import numpy as np
from PIL import Image

from rakuten_img import backbone, classifier, config, images


@lru_cache(maxsize=1)
def _load_classifier_payload_cached(_mtime_ns: int):
    # Cache keyed on the model file's mtime: a retrain (via /train or the CLI)
    # rewrites the .joblib, changing its mtime, which misses this cache and
    # forces a reload. maxsize=1 keeps only the latest. The _mtime_ns argument
    # is the cache key only; classifier.load() reads the path itself.
    return classifier.load()


def _load_classifier_payload():
    """Load the trained classifier payload, transparently reloading it if the
    model file on disk has changed since it was last cached."""
    try:
        mtime_ns = config.CLASSIFIER_PATH.stat().st_mtime_ns
    except FileNotFoundError:
        mtime_ns = 0  # let classifier.load() raise the usual friendly error
    return _load_classifier_payload_cached(mtime_ns)


def predict_proba(pil_image: Image.Image) -> np.ndarray:
    """PIL image -> probability vector (NUM_CLASSES,) ordered by CANONICAL_CLASSES."""
    payload = _load_classifier_payload()
    clf = payload["classifier"]
    processed = images.process_image(pil_image)
    feats = backbone.extract_one(processed).reshape(1, -1)
    proba = clf.predict_proba(feats)[0]
    return classifier.reorder_to_canonical(proba, payload["classes"])


def predict(image_path: str, top_k: int = 5) -> list[dict]:
    """Predict from a file path; return the top-k classes."""
    with Image.open(image_path) as img:
        img.load()
        proba = predict_proba(img)
    order = np.argsort(proba)[::-1][:top_k]
    return [
        {
            "code": int(config.CANONICAL_CLASSES[i]),
            "label": config.CANONICAL_LABELS[i],
            "probability": float(proba[i]),
        }
        for i in order
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict product type from image(s).")
    ap.add_argument("--image", nargs="+", required=True, help="One or more image paths.")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    results = {path: predict(path, top_k=args.top_k) for path in args.image}
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()