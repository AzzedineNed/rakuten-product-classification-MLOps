"""TRAINING -- TF-IDF + LogisticRegression (text modality).

Ported from feat/moussa-rakuten-code:src/train.py, then corrected. What changed
relative to the original, and why:

  1. SHARED SPLIT. Train/val/test membership comes from rakuten_common.split,
     the same partition the image pipeline uses, keyed by row label. The
     original held out its own 80/20; measured on the real data that val set
     was exactly the image pipeline's val + test blocks, so the two modalities
     could never be compared and text was scored on image's test rows.

  2. NO LEAKAGE. The vectorizer is fit on the TRAINING ROWS ONLY and then
     applied to val/test. The original ran fit_transform on the full dataset
     before splitting, so IDF statistics and the vocabulary were computed with
     held-out rows in view. Expect the honest score to be slightly LOWER than
     the 0.7773 that setup produced; lower and true beats higher and wrong.

  3. MERGE GUARD. Labels are joined on the ID column and the row count is
     checked (in rakuten_common.split.load_labeled_dataframe). The original
     merged on row position, which is correct only while the two CSVs stay in
     identical order and silently mislabels everything if they ever don't.

  4. SLIM VECTORIZER. TfidfVectorizer.stop_words_ is dropped before pickling.
     It is diagnostic-only, sklearn does not need it to transform, and on this
     dataset it held 1,687,956 terms -- 25 of the artifact's 27 MB, re-pushed
     to DVC on every retrain.

Usage:
    python scripts/train_text.py
    python scripts/train_text.py --max-features 10000

LOCATION: this is an ENTRYPOINT, so it lives in scripts/ like every other
entrypoint in this repo. It used to be `python -m rakuten_text.train` inside the
package, which made the text modality shaped differently from the image one for
no reason other than its porting history. rakuten_text/__init__.py always
claimed "Logic lives here; entrypoints stay thin, mirroring rakuten_img" -- this
move makes that true.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time

import joblib
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

import _bootstrap  # noqa: F401
from rakuten_common import split as shared
from rakuten_common.split import split_fingerprint
from rakuten_text import config, tracking
from rakuten_text.preprocessing import preparer_dataframe

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("train-text")


def charger_donnees() -> pd.DataFrame:
    """Load the canonical labeled dataframe and add the texte_complet column."""
    logger.info("Chargement des données (merge sur la colonne ID, garde-fou actif)...")
    df = shared.load_labeled_dataframe()
    df = preparer_dataframe(df)
    logger.info("Données chargées : %d produits, %d classes",
                len(df), df[shared.LABEL_COLUMN].nunique())
    return df


def entrainer(args):
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = charger_donnees()
    parts = shared.split_labels(df)
    logger.info("Split partagé : train=%d, val=%d, test=%d",
                len(parts["train"]), len(parts["val"]), len(parts["test"]))

    texte = df["texte_complet"].fillna("")
    y = df[shared.LABEL_COLUMN]

    txt_tr, y_tr = texte.loc[parts["train"]], y.loc[parts["train"]]
    txt_val, y_val = texte.loc[parts["val"]], y.loc[parts["val"]]

    # --- TF-IDF: FIT ON TRAIN ONLY -------------------------------------
    tfidf_params = dict(config.TFIDF_PARAMS)
    tfidf_params["max_features"] = args.max_features
    logger.info("Vectorisation TF-IDF, fit sur le TRAIN uniquement (%s)...", tfidf_params)
    vectorizer = TfidfVectorizer(**tfidf_params)
    X_tr = vectorizer.fit_transform(txt_tr)
    X_val = vectorizer.transform(txt_val)
    logger.info("Matrice TF-IDF : train=%s val=%s", X_tr.shape, X_val.shape)

    # --- fit -----------------------------------------------------------
    logger.info("Entraînement de la Régression Logistique...")
    model = LogisticRegression(**config.LOGREG_PARAMS)
    t0 = time.time()
    model.fit(X_tr, y_tr)
    duree = time.time() - t0
    logger.info("Entraînement terminé en %.1f s", duree)

    # The fusion contract, checked at the source rather than assumed.
    classes = [int(c) for c in model.classes_]
    if classes != list(config.CANONICAL_CLASSES):
        raise SystemExit(
            "model.classes_ does not match config.CANONICAL_CLASSES. Every "
            "probability vector this model emits would be misordered, and "
            "fusion would combine mismatched classes silently.\n"
            f"  classes_ : {classes}\n  canonical: {list(config.CANONICAL_CLASSES)}"
        )
    logger.info("Ordre des classes conforme à CANONICAL_CLASSES (%d classes).",
                len(classes))

    # --- validation ----------------------------------------------------
    y_pred = model.predict(X_val)
    metrics = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "f1_weighted": float(f1_score(y_val, y_pred, average="weighted")),
        "f1_macro": float(f1_score(y_val, y_pred, average="macro")),
        "n_train": int(X_tr.shape[0]),
        "n_val": int(X_val.shape[0]),
        "training_time_sec": round(duree, 1),
    }
    logger.info("Validation -> accuracy=%.4f | f1_weighted=%.4f | f1_macro=%.4f",
                metrics["accuracy"], metrics["f1_weighted"], metrics["f1_macro"])

    # --- save ----------------------------------------------------------
    # stop_words_ is introspection-only; sklearn's transform() never reads it.
    n_stop = len(getattr(vectorizer, "stop_words_", None) or [])
    vectorizer.stop_words_ = None
    joblib.dump(vectorizer, config.VECTORIZER_PATH)
    joblib.dump(model, config.MODEL_PATH)
    size_mb = config.VECTORIZER_PATH.stat().st_size / 1e6
    logger.info("Vectoriseur sauvegardé : %s (%.1f MB, stop_words_ écarté : %d termes)",
                config.VECTORIZER_PATH, size_mb, n_stop)
    logger.info("Modèle sauvegardé      : %s", config.MODEL_PATH)

    # Tracking is best-effort and happens AFTER the artifacts are on disk, so
    # a tracking outage can never cost us a trained model.
    run_id = tracking.log_training_run(config.run_params(), metrics)

    split_meta = {
        "mlflow_run_id": run_id,
        "split_source": "rakuten_common.split",
        "random_state": int(config.RANDOM_STATE),
        "n_rows": int(len(df)),
        "n_train": int(len(parts["train"])),
        "n_val": int(len(parts["val"])),
        "n_test": int(len(parts["test"])),
        "fingerprints": {k: split_fingerprint(v) for k, v in parts.items()},
        "max_features": int(args.max_features),
        "vectorizer_fit_on": "train",
        "sklearn_version": sklearn.__version__,
    }
    with open(config.SPLIT_PATH, "w", encoding="utf-8") as f:
        json.dump(split_meta, f, indent=2)
    logger.info("Provenance du split    : %s", config.SPLIT_PATH)

    with open(config.METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info("Métriques sauvegardées : %s", config.METRICS_PATH)

    # Registers a NEW VERSION and stops. The production alias is not touched:
    # promotion stays a deliberate human act, exactly as on the image side.
    tracking.register_model(
        run_id,
        artifacts=[config.VECTORIZER_PATH, config.MODEL_PATH],
        tags={
            "val_f1_weighted": round(metrics["f1_weighted"], 4),
            "val_accuracy": round(metrics["accuracy"], 4),
            "train_samples": metrics["n_train"],
            "max_features": args.max_features,
            "vectorizer_fit_on": "train",
            "split_source": "rakuten_common.split",
            "env": "container" if os.path.exists("/.dockerenv") else "host",
        },
    )

    logger.info("Entraînement terminé. Le test set n'a PAS été touché ici - "
                "utilisez `python scripts/evaluate_text.py`.")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    """The CLI definition, split out of main() so a NON-command-line caller can
    obtain exactly the same defaults.

    api/text_main.py's /train endpoint calls build_parser().parse_args([]) to
    build the Namespace entrainer() expects. It must not call main(): main()
    parses sys.argv, which under uvicorn holds the SERVER's arguments, and
    argparse then raises SystemExit - which `except Exception` does NOT catch,
    wedging the job status on "running" forever. api/image_main.py documents the
    same trap for process.main() and evaluate.main().

    Going through the parser rather than hand-building a namespace means a new
    argument with a default is picked up by the endpoint automatically, instead
    of surfacing as an AttributeError at run time.
    """
    parser = argparse.ArgumentParser(
        description="Entraînement TF-IDF + Régression Logistique (modalité texte)")
    parser.add_argument("--max-features", type=int,
                        default=config.TFIDF_PARAMS["max_features"],
                        help="Nombre max de features TF-IDF (défaut : 10000)")
    return parser


def main():
    entrainer(build_parser().parse_args())


if __name__ == "__main__":
    main()
