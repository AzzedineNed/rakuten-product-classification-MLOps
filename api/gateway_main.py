#!/usr/bin/env python3
"""FastAPI GATEWAY — combines the image and text models into one prediction.

Endpoints:
  GET  /health    liveness of the gateway AND of both upstream services
  POST /predict   optional image file + optional designation/description
                  -> fused top-k + full canonical probability vector

WHAT THIS SERVICE IS
    A coordinator, not a peer. It owns no model and loads no weights: it calls
    the image service and the text service over HTTP and combines the two
    probability vectors with rakuten_common.fusion.weighted_average. No torch,
    no scikit-learn estimator is ever instantiated here.

DEGRADATION IS EXPLICIT, NEVER SILENT
    Send both an image and text -> both models run and the result is fused.
    Send only one -> only that model runs and its vector is returned as-is.
    Either way the response states which modalities actually contributed and
    what weight was used. This matters: the measured fusion score (0.7973
    weighted F1) describes a BOTH-modalities prediction. Returning a
    single-modality answer in an identical-looking envelope would let a caller
    attribute a number to it that was never measured for it.

    If a requested modality fails (upstream down, bad response) the gateway
    returns the other one with "degraded": true and the error recorded, rather
    than failing the whole request. If every requested modality fails, 502.

THE WEIGHT
    text_weight defaults to fusion.DEFAULT_TEXT_WEIGHT (0.85), measured by
    scripts/tune_fusion_weight.py on the shared validation split. It can be
    overridden per request with ?text_weight= for experimentation. The default
    is not a preference; see rakuten_common/fusion.py for the numbers.

Run:
  uvicorn api.gateway_main:app --host 0.0.0.0 --port 8002
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import requests
from fastapi import FastAPI, File, Form, Query, UploadFile
from fastapi.responses import JSONResponse

from rakuten_common import fusion
from rakuten_img import config

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("api-gateway")

IMAGE_API = os.getenv("RAKUTEN_IMAGE_API_URL", "http://api:8000").rstrip("/")
TEXT_API = os.getenv("RAKUTEN_TEXT_API_URL", "http://text-api:8001").rstrip("/")
UPSTREAM_TIMEOUT_S = float(os.getenv("RAKUTEN_UPSTREAM_TIMEOUT_S", "60"))

app = FastAPI(
    title="Rakuten Fusion Gateway",
    description="Combines the image and text classifiers. Probability vectors "
                "follow CANONICAL_CLASSES.",
    version="1.0.0",
)


def _vector_from(payload: dict, source: str) -> np.ndarray:
    """Pull the canonical probability vector out of an upstream response.

    Validates length and the class order the upstream claims. Both services
    return `canonical_classes` precisely so this check is possible: if an
    upstream is ever redeployed with a different order, fusion would otherwise
    combine mismatched classes and report a confident wrong answer.
    """
    proba = payload.get("probabilities")
    if proba is None:
        raise ValueError(f"{source} returned no 'probabilities' field.")
    vec = np.asarray(proba, dtype=np.float64)
    if vec.shape[-1] != config.NUM_CLASSES:
        raise ValueError(
            f"{source} returned {vec.shape[-1]} classes, expected "
            f"{config.NUM_CLASSES}."
        )
    claimed = payload.get("canonical_classes")
    if claimed is not None and [int(c) for c in claimed] != list(config.CANONICAL_CLASSES):
        raise ValueError(
            f"{source} reports a different class order than this gateway's "
            f"CANONICAL_CLASSES. Refusing to fuse mismatched vectors."
        )
    return vec


def _call_image(raw: bytes, filename: str) -> np.ndarray:
    resp = requests.post(
        f"{IMAGE_API}/predict",
        files={"file": (filename or "upload.jpg", raw)},
        timeout=UPSTREAM_TIMEOUT_S,
    )
    resp.raise_for_status()
    return _vector_from(resp.json(), "image service")


def _call_text(designation: str, description: str) -> np.ndarray:
    resp = requests.post(
        f"{TEXT_API}/predict",
        json={"designation": designation, "description": description or ""},
        timeout=UPSTREAM_TIMEOUT_S,
    )
    resp.raise_for_status()
    return _vector_from(resp.json(), "text service")


def _upstream_health(base: str) -> dict:
    try:
        resp = requests.get(f"{base}/health", timeout=10)
        return {"reachable": resp.status_code == 200, "status_code": resp.status_code,
                "body": resp.json() if resp.status_code == 200 else None}
    except requests.RequestException as exc:
        return {"reachable": False, "error": f"{type(exc).__name__}: {exc}"}


@app.get("/health")
def health():
    """The gateway is up if it answers; the upstreams are reported separately.

    Deliberately still 200 when an upstream is down: the gateway itself is
    healthy and can still serve single-modality predictions. A caller reads
    `upstreams` to learn what is actually available.
    """
    image = _upstream_health(IMAGE_API)
    text = _upstream_health(TEXT_API)
    return {
        "status": "ok",
        "service": "fusion-gateway",
        "default_text_weight": fusion.DEFAULT_TEXT_WEIGHT,
        "num_classes": config.NUM_CLASSES,
        "upstreams": {
            "image": {"url": IMAGE_API, **image},
            "text": {"url": TEXT_API, **text},
        },
    }


@app.post("/predict")
async def predict(
    file: Optional[UploadFile] = File(None),
    designation: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    top_k: int = Query(5, ge=1, le=27),
    text_weight: Optional[float] = Query(
        None, ge=0.0, le=1.0,
        description="Override the measured default (0.85). Experimentation only."),
):
    """Predict from an image, from text, or from both (fused)."""
    want_image = file is not None
    want_text = bool(designation and designation.strip())

    if not want_image and not want_text:
        return JSONResponse(status_code=400, content={
            "error": "Provide an image file, a designation, or both.",
        })

    weight = fusion.DEFAULT_TEXT_WEIGHT if text_weight is None else float(text_weight)
    vectors: dict[str, np.ndarray] = {}
    errors: dict[str, str] = {}

    if want_image:
        try:
            raw = await file.read()
            vectors["image"] = _call_image(raw, file.filename)
        except Exception as exc:  # noqa: BLE001
            logger.warning("image upstream failed: %s", exc)
            errors["image"] = f"{type(exc).__name__}: {exc}"

    if want_text:
        try:
            vectors["text"] = _call_text(designation, description or "")
        except Exception as exc:  # noqa: BLE001
            logger.warning("text upstream failed: %s", exc)
            errors["text"] = f"{type(exc).__name__}: {exc}"

    if not vectors:
        return JSONResponse(status_code=502, content={
            "error": "Every requested modality failed.", "errors": errors,
        })

    if "image" in vectors and "text" in vectors:
        fused = fusion.weighted_average(vectors["image"], vectors["text"],
                                        text_weight=weight)
        used_weight = weight
    else:
        # Single modality: return it untouched. NOT fused, and the response
        # says so -- the measured fusion score does not apply here.
        fused = next(iter(vectors.values()))
        used_weight = None

    order = fused.argsort()[::-1][:top_k]
    top = [
        {"prdtypecode": int(config.CANONICAL_CLASSES[i]),
         "label": config.CANONICAL_LABELS[i],
         "probability": float(fused[i])}
        for i in order
    ]

    body = {
        "modalities": sorted(vectors.keys()),
        "fused": len(vectors) > 1,
        "text_weight": used_weight,
        "top_k": top,
        "prediction": top[0],
        "canonical_classes": list(config.CANONICAL_CLASSES),
        "probabilities": [float(p) for p in fused],
        "per_modality": {
            name: {
                "prdtypecode": int(config.CANONICAL_CLASSES[int(vec.argmax())]),
                "label": config.CANONICAL_LABELS[int(vec.argmax())],
                "probability": float(vec.max()),
            }
            for name, vec in vectors.items()
        },
    }
    if errors:
        # A modality was asked for and could not be delivered. Say so loudly in
        # the payload rather than returning a quietly weaker answer.
        body["degraded"] = True
        body["errors"] = errors
    return body
