"""
Script d'ÉVALUATION — TF-IDF + Régression Logistique.

Charge le modèle entraîné, reconstruit le même split de validation
(random_state fixe) et produit :
  - rapport de classification (precision / recall / f1 par classe)
  - métriques globales (accuracy, f1 macro/weighted)
  - matrice de confusion (image PNG)

Usage :
    python -m src.evaluate
    python -m src.evaluate --predict-test   # génère aussi les prédictions sur X_test
"""
from __future__ import annotations

import argparse
import json
import logging

import matplotlib
matplotlib.use("Agg")  # backend non interactif (exécution en script)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

from . import config
from .predict import TfidfPredictor
from .preprocessing import preparer_dataframe

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("evaluate")


def charger_validation(vectorizer):
    """Recharge les données et reconstruit le split de validation (identique à train)."""
    X_train = pd.read_csv(config.X_TRAIN_PATH)
    Y_train = pd.read_csv(config.Y_TRAIN_PATH)
    train_data = pd.merge(X_train, Y_train, left_index=True, right_index=True)
    train_data = preparer_dataframe(train_data)

    X_text = train_data["texte_complet"].fillna("")
    y = train_data["prdtypecode"]
    X_tfidf = vectorizer.transform(X_text)

    _, X_val, _, y_val = train_test_split(
        X_tfidf, y, test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE, stratify=y,
    )
    return X_val, y_val


def tracer_matrice_confusion(y_true, y_pred):
    """Trace et sauvegarde la matrice de confusion (top 15 catégories)."""
    top = y_true.value_counts().head(15).index
    mask = y_true.isin(top) & pd.Series(y_pred, index=y_true.index).isin(top)
    cm = confusion_matrix(
        y_true[mask], pd.Series(y_pred, index=y_true.index)[mask], labels=top
    )
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=top, yticklabels=top)
    plt.title("Matrice de confusion - Régression Logistique (Top 15 catégories)")
    plt.xlabel("Prédictions")
    plt.ylabel("Vraies valeurs")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(config.CONFUSION_MATRIX_PATH, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Matrice de confusion sauvegardée : %s", config.CONFUSION_MATRIX_PATH)


def predire_test(predictor: TfidfPredictor):
    """Génère les prédictions sur X_test et les sauvegarde en CSV."""
    logger.info("Prédictions sur X_test...")
    X_test = pd.read_csv(config.X_TEST_PATH)
    X_test = preparer_dataframe(X_test)
    X_tfidf = predictor.vectorizer.transform(X_test["texte_complet"].fillna(""))
    preds = predictor.model.predict(X_tfidf)
    df = pd.DataFrame({"productid": X_test["productid"], "prdtypecode": preds})
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.PREDICTIONS_PATH, index=False)
    logger.info("Prédictions sauvegardées : %s (%d lignes)", config.PREDICTIONS_PATH, len(df))


def evaluer(args):
    predictor = TfidfPredictor().load()
    logger.info("Modèle et vectoriseur chargés.")

    X_val, y_val = charger_validation(predictor.vectorizer)
    logger.info("Validation : %d exemples", X_val.shape[0])

    y_pred = predictor.model.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    f1_w = f1_score(y_val, y_pred, average="weighted")
    f1_m = f1_score(y_val, y_pred, average="macro")

    print("\n" + "=" * 80)
    print("RÉSULTATS SUR LA VALIDATION — Régression Logistique")
    print("=" * 80)
    print(f"Accuracy            : {accuracy:.4f}")
    print(f"F1-score (weighted) : {f1_w:.4f}")
    print(f"F1-score (macro)    : {f1_m:.4f}")
    print("\nRapport de classification détaillé :")
    print(classification_report(y_val, y_pred))

    # Sauvegarde des métriques
    metrics = {
        "accuracy": float(accuracy),
        "f1_weighted": float(f1_w),
        "f1_macro": float(f1_m),
        "n_val": int(X_val.shape[0]),
    }
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info("Métriques sauvegardées : %s", config.METRICS_PATH)

    tracer_matrice_confusion(y_val, y_pred)

    if args.predict_test:
        predire_test(predictor)

    logger.info("✅ Évaluation terminée.")


def main():
    parser = argparse.ArgumentParser(description="Évaluation TF-IDF + Régression Logistique")
    parser.add_argument("--predict-test", action="store_true",
                        help="Générer aussi les prédictions sur X_test_update.csv")
    args = parser.parse_args()
    evaluer(args)


if __name__ == "__main__":
    main()
