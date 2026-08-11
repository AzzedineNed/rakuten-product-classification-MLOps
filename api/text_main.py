#!/usr/bin/env python3
"""FastAPI service exposing the TEXT model (TF-IDF + LogisticRegression).

Endpoints:
  GET  /health          liveness + whether the model is loaded
  POST /predict         {designation, description} -> top-k + full canonical proba
  POST /predict/batch   a list of products -> one result each

THE PAYLOAD MIRRORS THE IMAGE SERVICE ON PURPOSE. api/main.py's /predict
returns top_k / prediction / canonical_classes / probabilities, and so does
this. The gateway can therefore treat the two services identically instead of
special-casing each modality, and a probability vector from either can go
straight into fusion.weighted_average.

LOADING: eager, at startup, unlike the image service. The image API imports
torch lazily because torch plus a MobileNetV2 costs hundreds of MB; the text
model is a 0.5 MB vectorizer plus a 2.2 MB estimator, so paying that at boot
buys a fast, predictable first request. A missing model does NOT stop the
service from starting -- /health reports it, mirroring the image side.

Run:
  uvicorn api.text_main:app --host 0.0.0.0 --port 8001
"""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

# Make the src packages importable without install / PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from rakuten_text import config
from rakuten_text.predict import TfidfPredictor

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("api-text")

predictor = TfidfPredictor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model at startup. A failure is logged, not raised: the service
    still starts and /health reports model_loaded=false, so an orchestrator
    gets a truthful answer instead of a container that will not boot."""
    try:
        predictor.load()
        logger.info("Text model loaded — API ready.")
    except FileNotFoundError as exc:
        logger.error("Could not load the text model: %s", exc)
    yield


app = FastAPI(
    title="Rakuten Text Classifier",
    description="Predicts a product type code from its designation and "
                "description. Probability vectors follow CANONICAL_CLASSES.",
    version="1.0.0",
    lifespan=lifespan,
)


class Produit(BaseModel):
    designation: str = Field(..., min_length=1, description="Product title")
    description: str = Field("", description="Longer description (optional)")


class ProduitBatch(BaseModel):
    produits: List[Produit] = Field(..., description="Products to classify")


class TopK(BaseModel):
    prdtypecode: int
    label: str
    probability: float


class Prediction(BaseModel):
    top_k: List[TopK]
    prediction: TopK
    canonical_classes: List[int]
    probabilities: List[float]


class BatchPrediction(BaseModel):
    predictions: List[Prediction]


def _unavailable() -> Optional[JSONResponse]:
    if predictor.is_loaded:
        return None
    return JSONResponse(
        status_code=503,
        content={"error": "Text model not loaded. Run "
                          "`python -m rakuten_text.train` (or `dvc pull`)."},
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "modality": "text",
        "model_loaded": predictor.is_loaded,
        "model_path": str(predictor.model_path),
        "num_classes": config.NUM_CLASSES,
        "registered_model": config.REGISTERED_MODEL_NAME,
    }


@app.post("/predict", response_model=Prediction)
def predict(produit: Produit, top_k: int = 5):
    unavailable = _unavailable()
    if unavailable:
        return unavailable
    top_k = max(1, min(int(top_k), config.NUM_CLASSES))
    try:
        return predictor.predict_detailed(produit.designation,
                                          produit.description, top_k=top_k)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed")
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.post("/predict/batch", response_model=BatchPrediction)
def predict_batch(batch: ProduitBatch, top_k: int = 5):
    unavailable = _unavailable()
    if unavailable:
        return unavailable
    if not batch.produits:
        return JSONResponse(status_code=400,
                            content={"error": "The product list is empty."})
    top_k = max(1, min(int(top_k), config.NUM_CLASSES))
    try:
        results = [
            predictor.predict_detailed(p.designation, p.description, top_k=top_k)
            for p in batch.produits
        ]
        return {"predictions": results}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Batch prediction failed")
        return JSONResponse(status_code=500, content={"error": str(exc)})
