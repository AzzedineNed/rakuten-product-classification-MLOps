#!/usr/bin/env python3
"""predict.py — Image-only inference.

Image -> zoom/white-canvas preprocessing -> frozen MobileNetV2 -> classifier
-> probability vector in CANONICAL order -> top-k labels.

Model loading (registry-first, local fallback):
  1. If MLFLOW_TRACKING_URI is set, the newest version of the registered model
     (config.REGISTERED_MODEL_NAME) is downloaded from the MLflow Model
     Registry ONCE and cached in memory for the process lifetime.
  2. If the registry is unreachable / empty / not configured, we fall back to
     the local .joblib (config.CLASSIFIER_PATH) with the existing mtime-keyed
     cache, so a retrain on disk is still picked up automatically. Serving
     never hard-fails on a network problem.
  A registry failure is remembered for the process lifetime too, so a dead
  tracking server costs ONE failed attempt, not one per request.

Exposes reusable functions (used by the API and importable by a fusion layer):
  * predict_proba(pil_image) -> np.ndarray of shape (NUM_CLASSES,) in canonical order
  * predict(image_path, top_k) -> list of {code, label, probability}
  * model_source() -> where the served model came from (registry/local/none yet)

CLI:
  python scripts/predict.py --image path/to/product.jpg
  python scripts/predict.py --image a.jpg b.jpg --top-k 3
"""
from __future__ import annotations

import argparse
import json
import os
from functools import lru_cache

import _bootstrap  # noqa: F401
import numpy as np
from PIL import Image

from rakuten_img import backbone, classifier, config, images

# Registry cache: resolved once per process. _REGISTRY_FAILED avoids re-hitting
# a dead server on every request once the fallback has kicked in.
_REGISTRY_PAYLOAD: dict | None = None
_REGISTRY_FAILED: bool = False
_SERVING_SOURCE: str = "not-loaded"  # updated by _load_payload()


@lru_cache(maxsize=1)
def _load_classifier_payload_cached(_mtime_ns: int):
    # Cache keyed on the model file's mtime: a retrain (via /train or the CLI)
    # rewrites the .joblib, changing its mtime, which misses this cache and
    # forces a reload. maxsize=1 keeps only the latest. The _mtime_ns argument
    # is the cache key only; classifier.load() reads the path itself.
    return classifier.load()


def _load_local_payload():
    """Load the trained classifier payload from LOCAL disk, transparently
    reloading it if the model file has changed since it was last cached."""
    try:
        mtime_ns = config.CLASSIFIER_PATH.stat().st_mtime_ns
    except FileNotFoundError:
        mtime_ns = 0  # let classifier.load() raise the usual friendly error
    payload = _load_classifier_payload_cached(mtime_ns)
    payload.setdefault("serving_source", f"local:{config.CLASSIFIER_PATH.name}")
    return payload


def _load_payload():
    """Registry first (cached for the process lifetime), local .joblib fallback."""
    global _REGISTRY_PAYLOAD, _REGISTRY_FAILED, _SERVING_SOURCE
    if _REGISTRY_PAYLOAD is not None:
        return _REGISTRY_PAYLOAD
    if not _REGISTRY_FAILED and os.getenv("MLFLOW_TRACKING_URI"):
        try:
            _REGISTRY_PAYLOAD = classifier.load_from_registry()
            _SERVING_SOURCE = _REGISTRY_PAYLOAD.get("serving_source", "registry")
            return _REGISTRY_PAYLOAD
        except Exception as exc:  # noqa: BLE001
            _REGISTRY_FAILED = True
            print(f"⚠️  Registry unavailable ({type(exc).__name__}: {exc}) — "
                  f"falling back to local model.")
    payload = _load_local_payload()
    _SERVING_SOURCE = payload.get("serving_source", "local")
    return payload


def model_source() -> str:
    """Where the currently served model came from ('registry:…', 'local:…'),
    or 'not-loaded' before the first prediction. Used by the API's /health."""
    return _SERVING_SOURCE


def predict_proba(pil_image: Image.Image) -> np.ndarray:
    """PIL image -> probability vector (NUM_CLASSES,) ordered by CANONICAL_CLASSES."""
    payload = _load_payload()
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