"""
API REST — Classification de produits Rakuten (TF-IDF + Régression Logistique).

Lancement :
    uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

Documentation interactive : http://localhost:8000/docs

Endpoints :
    GET  /health         → état de l'API et du modèle
    POST /predict        → prédiction pour un produit
    POST /predict/batch  → prédiction pour une liste de produits
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .predict import TfidfPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("api")

# Prédicteur chargé une seule fois au démarrage
predictor = TfidfPredictor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle au démarrage de l'API."""
    try:
        predictor.load()
        logger.info("✅ Modèle chargé, API prête.")
    except FileNotFoundError as e:
        # On log l'erreur sans bloquer le démarrage : /health signalera l'état.
        logger.error("Impossible de charger le modèle : %s", e)
    yield


app = FastAPI(
    title="API Classification Produits Rakuten",
    description="Prédit la catégorie (prdtypecode) d'un produit e-commerce "
                "à partir de sa désignation et de sa description.",
    version="1.0.0",
    lifespan=lifespan,
)


# --- Schémas Pydantic ----------------------------------------------------
class Produit(BaseModel):
    designation: str = Field(..., description="Nom/titre du produit", min_length=1)
    description: str = Field("", description="Description détaillée (optionnelle)")


class ProduitBatch(BaseModel):
    produits: List[Produit] = Field(..., description="Liste de produits à classifier")


class Prediction(BaseModel):
    prdtypecode: int = Field(..., description="Code de catégorie prédit")
    confidence: Optional[float] = Field(None, description="Probabilité de la classe prédite")


class BatchPrediction(BaseModel):
    predictions: List[Prediction]


# --- Endpoints -----------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": predictor.is_loaded,
        "model_path": str(predictor.model_path),
    }


@app.post("/predict", response_model=Prediction)
def predict(produit: Produit):
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Modèle non chargé. Lancez `python -m src.train`.")
    try:
        return predictor.predict(produit.designation, produit.description)
    except Exception as e:  # pragma: no cover
        logger.exception("Erreur de prédiction")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPrediction)
def predict_batch(batch: ProduitBatch):
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Modèle non chargé. Lancez `python -m src.train`.")
    if not batch.produits:
        raise HTTPException(status_code=400, detail="La liste de produits est vide.")
    try:
        preds = predictor.predict_batch([p.model_dump() for p in batch.produits])
        return {"predictions": preds}
    except Exception as e:  # pragma: no cover
        logger.exception("Erreur de prédiction batch")
        raise HTTPException(status_code=500, detail=str(e))
