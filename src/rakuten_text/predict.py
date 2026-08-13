"""
Prédicteur réutilisable : charge le vectoriseur TF-IDF + le modèle de
Régression Logistique et expose des méthodes de prédiction.

Utilisé à la fois par l'API (api.py) et le script d'évaluation (evaluate.py).
"""
from __future__ import annotations

from typing import List

import joblib
import numpy as np

from rakuten_common.contract import to_canonical, validate_vector

from . import config
from .preprocessing import construire_texte_complet


class TfidfPredictor:
    """Encapsule vectoriseur + modèle pour prédire la catégorie d'un produit."""

    def __init__(self, vectorizer_path=None, model_path=None):
        self.vectorizer_path = vectorizer_path or config.VECTORIZER_PATH
        self.model_path = model_path or config.MODEL_PATH
        self.vectorizer = None
        self.model = None
        # Where the loaded model actually came from. Mirrors the image side's
        # scripts/predict.py model_source(), including the "not-loaded" value
        # before anything is loaded, so /health can report it identically.
        self.serving_source = "not-loaded"

    def load(self, prefer_registry: bool = False) -> "TfidfPredictor":
        """Charge les artefacts.

        prefer_registry=False (DEFAULT) — local disk only. This is what
        scripts/evaluate_text.py and scripts/tune_fusion_weight need: an
        evaluation must score the artifacts training just wrote, not whatever
        the registry currently serves, or the reported metrics describe a
        different model than the run they are attached to.

        prefer_registry=True — registry first (production alias, else newest
        version), local .pkl files as the fallback. Serving passes this; the
        chain NEVER hard-fails on a registry problem, it degrades to local and
        says so in serving_source.

        The default is opt-IN on purpose: a new caller that forgets the flag
        gets today's behaviour rather than silently reaching for the network.
        Same split as the image modality, where the registry chain lives in the
        serving path (scripts/predict.py) and rakuten_img.classifier.load()
        stays local-only.
        """
        if prefer_registry:
            try:
                from .tracking import load_from_registry

                self.vectorizer, self.model, self.serving_source = load_from_registry()
                return self
            except Exception as exc:  # noqa: BLE001
                # Registry unreachable, unset, empty, or holding a malformed
                # version. None of that is a reason to refuse to serve.
                print(f"⚠️  Registry unavailable ({type(exc).__name__}: {exc}) — "
                      f"falling back to the local model.")

        if not self.vectorizer_path.exists():
            raise FileNotFoundError(
                f"Vectoriseur introuvable : {self.vectorizer_path}. "
                "Lancez d'abord `python scripts/train_text.py`."
            )
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Modèle introuvable : {self.model_path}. "
                "Lancez d'abord `python scripts/train_text.py`."
            )
        self.vectorizer = joblib.load(self.vectorizer_path)
        self.model = joblib.load(self.model_path)
        self.serving_source = f"local:{self.model_path.name}"
        return self

    @property
    def is_loaded(self) -> bool:
        return self.vectorizer is not None and self.model is not None

    def _ensure_loaded(self):
        """Lazy load for CLI/library callers. LOCAL ONLY, deliberately: this
        runs on the prediction path, and retrying a dead registry on every
        request is exactly the failure mode the image side guards against with
        its _REGISTRY_FAILED flag. Serving loads eagerly at startup instead.
        """
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

    # ---------------------------------------------------------------- #
    # THE FUSION INTERFACE
    # ---------------------------------------------------------------- #
    def predict_proba_vector(self, designation: str, description: str = ""):
        """Full probability vector in CANONICAL_CLASSES order (length 27).

        This is what fusion.weighted_average needs and what the original
        predict() threw away: it computed predict_proba and kept only
        np.max(proba), so no caller could ever combine the two modalities.
        """
        self._ensure_loaded()
        texte = construire_texte_complet(designation, description)
        X = self.vectorizer.transform([texte])
        proba = self.model.predict_proba(X)[0]
        vec = to_canonical(proba, self.model.classes_)
        return validate_vector(vec)

    def predict_proba_batch(self, produits: List[dict]):
        """Same, for a list of products -> array of shape (n, 27)."""
        self._ensure_loaded()
        textes = [
            construire_texte_complet(p.get("designation", ""), p.get("description", ""))
            for p in produits
        ]
        X = self.vectorizer.transform(textes)
        proba = self.model.predict_proba(X)
        vecs = to_canonical(proba, self.model.classes_)
        for row in vecs:
            validate_vector(row)
        return vecs

    def predict_detailed(self, designation: str, description: str = "",
                         top_k: int = 5) -> dict:
        """Mirror of the image API's /predict payload, so a gateway can treat
        both modalities identically: top-k with human labels, plus the full
        canonical vector and the class order it follows."""
        vec = self.predict_proba_vector(designation, description)
        order = vec.argsort()[::-1][:top_k]
        top = [
            {"prdtypecode": int(config.CANONICAL_CLASSES[i]),
             "label": config.CANONICAL_LABELS[i],
             "probability": float(vec[i])}
            for i in order
        ]
        return {
            "top_k": top,
            "prediction": top[0],
            "canonical_classes": list(config.CANONICAL_CLASSES),
            "probabilities": [float(p) for p in vec],
        }
