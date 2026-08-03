"""
Prédicteur réutilisable : charge le vectoriseur TF-IDF + le modèle de
Régression Logistique et expose des méthodes de prédiction.

Utilisé à la fois par l'API (api.py) et le script d'évaluation (evaluate.py).
"""
from __future__ import annotations

from typing import List

import joblib
import numpy as np

from . import config
from .preprocessing import construire_texte_complet


class TfidfPredictor:
    """Encapsule vectoriseur + modèle pour prédire la catégorie d'un produit."""

    def __init__(self, vectorizer_path=None, model_path=None):
        self.vectorizer_path = vectorizer_path or config.VECTORIZER_PATH
        self.model_path = model_path or config.MODEL_PATH
        self.vectorizer = None
        self.model = None

    def load(self) -> "TfidfPredictor":
        """Charge les artefacts depuis le disque."""
        if not self.vectorizer_path.exists():
            raise FileNotFoundError(
                f"Vectoriseur introuvable : {self.vectorizer_path}. "
                "Lancez d'abord `python -m src.train`."
            )
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Modèle introuvable : {self.model_path}. "
                "Lancez d'abord `python -m src.train`."
            )
        self.vectorizer = joblib.load(self.vectorizer_path)
        self.model = joblib.load(self.model_path)
        return self

    @property
    def is_loaded(self) -> bool:
        return self.vectorizer is not None and self.model is not None

    def _ensure_loaded(self):
        if not self.is_loaded:
            self.load()

    def predict(self, designation: str, description: str = "") -> dict:
        """Prédit la catégorie d'un seul produit.

        Retourne un dict {prdtypecode, confidence}.
        """
        self._ensure_loaded()
        texte = construire_texte_complet(designation, description)
        X = self.vectorizer.transform([texte])
        code = self.model.predict(X)[0]
        confidence = None
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            confidence = float(np.max(proba))
        return {"prdtypecode": int(code), "confidence": confidence}

    def predict_batch(self, produits: List[dict]) -> List[dict]:
        """Prédit pour une liste de produits [{designation, description}, ...]."""
        self._ensure_loaded()
        textes = [
            construire_texte_complet(p.get("designation", ""), p.get("description", ""))
            for p in produits
        ]
        X = self.vectorizer.transform(textes)
        codes = self.model.predict(X)
        resultats = [{"prdtypecode": int(c), "confidence": None} for c in codes]
        if hasattr(self.model, "predict_proba"):
            probas = self.model.predict_proba(X)
            for r, p in zip(resultats, probas):
                r["confidence"] = float(np.max(p))
        return resultats
