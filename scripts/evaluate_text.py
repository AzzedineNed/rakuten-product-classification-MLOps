"""EVALUATION -- TF-IDF + LogisticRegression (text modality).

Scores the trained model on the SHARED held-out split -- the same products the
image pipeline holds out -- so the two modalities are finally comparable and
fusion can be evaluated honestly.

Defaults to the TEST split, mirroring scripts/evaluate.py on the image side:
train.py reports val (model selection), evaluate.py reports test (final).
Pass --split val to re-score validation.

Changes from feat/moussa-rakuten-code:src/evaluate.py:
  * split membership comes from rakuten_common.split, and the fingerprint
    recorded at training time is re-checked, so a changed dataset fails loudly
    instead of scoring against a partition the model never saw;
  * seaborn is gone (it was an undeclared dependency, absent from
    requirements.txt); the heatmap is drawn with matplotlib directly;
  * train and eval metrics go to separate files instead of one overwriting the
    other, and a classification report is written next to them.

Usage:
    python scripts/evaluate_text.py
    python scripts/evaluate_text.py --split val
    python scripts/evaluate_text.py --predict-test
"""
from __future__ import annotations

import argparse
import json
import logging

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (script execution)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

import _bootstrap  # noqa: F401
from rakuten_common import split as shared
from rakuten_common.split import split_fingerprint
from rakuten_text import config, tracking
from rakuten_text.predict import TfidfPredictor
from rakuten_text.preprocessing import preparer_dataframe

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("evaluate-text")


def charger_split_meta() -> dict:
    """Read what training actually did. Refuse to guess."""
    if not config.SPLIT_PATH.exists():
        raise SystemExit(
            f"{config.SPLIT_PATH} introuvable. Ce fichier est écrit par "
            "`python scripts/train_text.py`. Sans lui, le split ne peut pas "
            "être vérifié - relancez l'entraînement."
        )
    with open(config.SPLIT_PATH, encoding="utf-8") as f:
        return json.load(f)


def charger_donnees(meta: dict, split_name: str):
    """Rebuild the shared split and verify it matches what training used."""
    df = shared.load_labeled_dataframe()
    df = preparer_dataframe(df)

    if len(df) != meta["n_rows"]:
        raise SystemExit(
            f"Les données ont changé depuis l'entraînement : {len(df)} lignes "
            f"maintenant, {meta['n_rows']} au moment du fit. Réentraînez."
        )

    parts = shared.split_labels(df)

    # Check EVERY split, not just the one being scored. A change confined to
    # the training rows leaves the test fingerprint untouched, so scoring only
    # 'test' would pass while the model was in fact fit on different data.
    enregistrees = meta.get("fingerprints") or {}
    ecarts = []
    for nom in ("train", "val", "test"):
        attendu = enregistrees.get(nom)
        obtenu = split_fingerprint(parts[nom])
        if attendu and attendu != obtenu:
            ecarts.append(f"  {nom}: {obtenu} != {attendu} (enregistré)")
    if ecarts:
        raise SystemExit(
            "Le partitionnement a changé depuis l'entraînement - toute métrique "
            "calculée ici serait incomparable. Réentraînez.\n" + "\n".join(ecarts)
        )
    logger.info("Empreintes des 3 splits vérifiées (scoring sur '%s' : %s).",
                split_name, split_fingerprint(parts[split_name]))

    ix = parts[split_name]
    return df.loc[ix, "texte_complet"].fillna(""), df.loc[ix, shared.LABEL_COLUMN]


def tracer_matrice_confusion(y_true, y_pred):
    """Plot and save the confusion matrix (top 15 categories)."""
    top = y_true.value_counts().head(15).index
    y_pred_s = pd.Series(y_pred, index=y_true.index)
    mask = y_true.isin(top) & y_pred_s.isin(top)
    cm = confusion_matrix(y_true[mask], y_pred_s[mask], labels=top)

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)

    ticks = np.arange(len(top))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    noms = {c: l for c, l in zip(config.CANONICAL_CLASSES, config.CANONICAL_LABELS)}
    etiquettes = [f"{c} {noms.get(int(c), '?')}" for c in top]
    ax.set_xticklabels(etiquettes, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(etiquettes, rotation=0, fontsize=8)

    seuil = cm.max() / 2.0 if cm.size and cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                    color="white" if cm[i, j] > seuil else "black", fontsize=8)

    ax.set_title("Matrice de confusion - Régression Logistique (Top 15 catégories)")
    ax.set_xlabel("Prédictions")
    ax.set_ylabel("Vraies valeurs")
    fig.tight_layout()
    fig.savefig(config.CONFUSION_MATRIX_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Matrice de confusion sauvegardée : %s", config.CONFUSION_MATRIX_PATH)


def predire_test(predictor: TfidfPredictor):
    """Generate predictions on the unlabeled ENS challenge set, if present."""
    if not config.X_TEST_PATH.exists():
        raise SystemExit(
            f"{config.X_TEST_PATH} introuvable. Le jeu de test non labellisé du "
            "challenge ENS n'est pas téléchargé par collect.py (seuls x-train, "
            "y-train et les images le sont). Rien à prédire."
        )
    logger.info("Prédictions sur X_test...")
    X_test = pd.read_csv(config.X_TEST_PATH)
    X_test = preparer_dataframe(X_test)
    X_tfidf = predictor.vectorizer.transform(X_test["texte_complet"].fillna(""))
    preds = predictor.model.predict(X_tfidf)
    df = pd.DataFrame({"productid": X_test["productid"], "prdtypecode": preds})
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.PREDICTIONS_PATH, index=False)
    logger.info("Prédictions sauvegardées : %s (%d lignes)",
                config.PREDICTIONS_PATH, len(df))


def evaluer(args):
    meta = charger_split_meta()
    predictor = TfidfPredictor().load()
    logger.info("Modèle chargé (split partagé : %s, %d lignes).",
                meta.get("split_source"), meta.get("n_rows", -1))

    textes, y_true = charger_donnees(meta, args.split)
    logger.info("Split '%s' : %d exemples", args.split, len(y_true))

    X = predictor.vectorizer.transform(textes)
    y_pred = predictor.model.predict(X)

    accuracy = accuracy_score(y_true, y_pred)
    f1_w = f1_score(y_true, y_pred, average="weighted")
    f1_m = f1_score(y_true, y_pred, average="macro")

    # Human-readable class names, not bare codes: 'Jeux de table' is
    # actionable in a review, '1180' is not. labels= pins the row order to
    # CANONICAL_CLASSES so the report and target_names cannot drift apart.
    rapport = classification_report(
        y_true, y_pred,
        labels=list(config.CANONICAL_CLASSES),
        target_names=[f"{c} {l}" for c, l in
                      zip(config.CANONICAL_CLASSES, config.CANONICAL_LABELS)],
        zero_division=0,
    )
    print("\n" + "=" * 80)
    print(f"RÉSULTATS: Régression Logistique (texte), split '{args.split}'")
    print("=" * 80)
    print(f"Accuracy            : {accuracy:.4f}")
    print(f"F1-score (weighted) : {f1_w:.4f}")
    print(f"F1-score (macro)    : {f1_m:.4f}")
    print("\nRapport de classification détaillé :")
    print(rapport)

    metrics = {
        "split": args.split,
        "accuracy": float(accuracy),
        "f1_weighted": float(f1_w),
        "f1_macro": float(f1_m),
        "n_eval": int(len(y_true)),
    }
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.EVAL_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(config.CLASSIFICATION_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(rapport)
    logger.info("Métriques sauvegardées : %s", config.EVAL_METRICS_PATH)
    logger.info("Rapport sauvegardé     : %s", config.CLASSIFICATION_REPORT_PATH)

    tracer_matrice_confusion(y_true, y_pred)

    # Reopen the SAME run train.py created, so one registered version has one
    # complete record: params + val metrics from training, test metrics and
    # plots from here. Falls back to a standalone run if there is no id.
    tracking.attach_evaluation(
        meta.get("mlflow_run_id"),
        metrics,
        artifacts={
            "plots": config.CONFUSION_MATRIX_PATH,
            "reports": config.CLASSIFICATION_REPORT_PATH,
        },
    )

    if args.predict_test:
        predire_test(predictor)

    logger.info("Évaluation terminée.")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    """The CLI definition, split out of main(). See the twin in
    scripts/train_text.py for why the /evaluate endpoint goes through the parser
    instead of calling main() or hand-building a Namespace.

    The defaults matter here: parse_args([]) yields split="test" and
    predict_test=False, so the endpoint scores the same split the host command
    scores, and does not generate the Kaggle-style submission predictions.
    """
    parser = argparse.ArgumentParser(
        description="Évaluation TF-IDF + Régression Logistique (modalité texte)")
    parser.add_argument("--split", choices=("val", "test"), default="test",
                        help="Split partagé à scorer (défaut : test)")
    parser.add_argument("--predict-test", action="store_true",
                        help="Générer aussi les prédictions sur X_test_update.csv")
    return parser


def main():
    evaluer(build_parser().parse_args())


if __name__ == "__main__":
    main()
