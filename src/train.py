"""
Script d'ENTRAÎNEMENT — TF-IDF + Régression Logistique.

Charge les données, nettoie le texte, vectorise en TF-IDF, entraîne une
Régression Logistique et sauvegarde le vectoriseur, le modèle et les métriques.

Usage :
    python -m src.train
    python -m src.train --test-size 0.2 --max-features 10000
    python -m src.train --full   # réentraîne sur 100% des données (pas de split)
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from . import config
from .preprocessing import preparer_dataframe

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("train")


def charger_donnees() -> pd.DataFrame:
    """Charge et fusionne X_train + Y_train, ajoute la colonne texte_complet."""
    logger.info("Chargement des données...")
    X_train = pd.read_csv(config.X_TRAIN_PATH)
    Y_train = pd.read_csv(config.Y_TRAIN_PATH)
    train_data = pd.merge(X_train, Y_train, left_index=True, right_index=True)
    train_data = preparer_dataframe(train_data)
    logger.info("Données chargées : %d produits, %d classes",
                len(train_data), train_data["prdtypecode"].nunique())
    return train_data


def entrainer(args):
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_data = charger_donnees()
    X_text = train_data["texte_complet"].fillna("")
    y = train_data["prdtypecode"]

    # --- Vectorisation TF-IDF ---
    tfidf_params = dict(config.TFIDF_PARAMS)
    tfidf_params["max_features"] = args.max_features
    logger.info("Vectorisation TF-IDF (%s)...", tfidf_params)
    vectorizer = TfidfVectorizer(**tfidf_params)
    X_tfidf = vectorizer.fit_transform(X_text)
    logger.info("Matrice TF-IDF : %s", X_tfidf.shape)

    # --- Split train/validation ---
    metrics = {}
    if not args.full:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_tfidf, y, test_size=args.test_size,
            random_state=config.RANDOM_STATE, stratify=y,
        )
        logger.info("Split : train=%d, val=%d", X_tr.shape[0], X_val.shape[0])
    else:
        X_tr, y_tr = X_tfidf, y
        X_val = y_val = None
        logger.info("Mode --full : entraînement sur 100%% des données")

    # --- Entraînement ---
    logger.info("Entraînement de la Régression Logistique...")
    model = LogisticRegression(**config.LOGREG_PARAMS)
    t0 = time.time()
    model.fit(X_tr, y_tr)
    duree = time.time() - t0
    logger.info("Entraînement terminé en %.1f s", duree)

    # --- Évaluation sur la validation ---
    if X_val is not None:
        y_pred = model.predict(X_val)
        metrics = {
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "f1_weighted": float(f1_score(y_val, y_pred, average="weighted")),
            "f1_macro": float(f1_score(y_val, y_pred, average="macro")),
            "n_train": int(X_tr.shape[0]),
            "n_val": int(X_val.shape[0]),
            "training_time_sec": round(duree, 1),
        }
        logger.info("Validation → accuracy=%.4f | f1_weighted=%.4f | f1_macro=%.4f",
                    metrics["accuracy"], metrics["f1_weighted"], metrics["f1_macro"])

    # --- Sauvegarde ---
    joblib.dump(vectorizer, config.VECTORIZER_PATH)
    joblib.dump(model, config.MODEL_PATH)
    logger.info("Vectoriseur sauvegardé : %s", config.VECTORIZER_PATH)
    logger.info("Modèle sauvegardé      : %s", config.MODEL_PATH)

    if metrics:
        with open(config.METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        logger.info("Métriques sauvegardées : %s", config.METRICS_PATH)

    logger.info("✅ Entraînement terminé.")


def main():
    parser = argparse.ArgumentParser(description="Entraînement TF-IDF + Régression Logistique")
    parser.add_argument("--test-size", type=float, default=config.TEST_SIZE,
                        help="Proportion pour la validation (défaut : 0.2)")
    parser.add_argument("--max-features", type=int,
                        default=config.TFIDF_PARAMS["max_features"],
                        help="Nombre max de features TF-IDF (défaut : 10000)")
    parser.add_argument("--full", action="store_true",
                        help="Entraîner sur 100%% des données (pas de split ni métriques de validation)")
    args = parser.parse_args()
    entrainer(args)


if __name__ == "__main__":
    main()
