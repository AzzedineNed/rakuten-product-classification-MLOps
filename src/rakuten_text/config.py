"""Configuration for the text modality (TF-IDF + LogisticRegression).

Ported from feat/moussa-rakuten-code:src/config.py. Two things changed:

1. PATHS. The original resolved everything relative to its own repo root
   (X_train_update.csv sitting beside src/). Here the CSVs live in data/raw
   under DVC, and artifacts go to models/text and reports/text.

2. SINGLE SOURCE OF TRUTH. Data paths, RANDOM_STATE and the canonical class
   order are imported from rakuten_img.config rather than redeclared, so the
   two modalities cannot drift apart. rakuten_img/__init__.py imports only
   config (pure pathlib/os/dotenv), so this costs no torch import.

   NOTE: this is a deliberate TEMPORARY coupling. When the shared split module
   lands (step A2) and the services are split, the shared bits should move to a
   neutral package that both modalities import. Until then, one source of truth
   in the wrong place beats two sources of truth.
"""
from __future__ import annotations

from rakuten_img import config as _img

# --- Directories ---------------------------------------------------------
PROJECT_ROOT = _img.PROJECT_ROOT
RAW_DIR = _img.RAW_DIR
MODELS_DIR = _img.MODELS_DIR / "text"
REPORTS_DIR = _img.REPORTS_DIR / "text"

# --- Data files (shared with the image modality) -------------------------
X_TRAIN_PATH = _img.X_TRAIN_CSV
Y_TRAIN_PATH = _img.Y_TRAIN_CSV
# The ENS "x-test" archive is the unlabeled challenge test set; collect.py does
# NOT download it. This path is defined so --predict-test can fail with a clear
# message instead of an AttributeError.
X_TEST_PATH = RAW_DIR / "X_test_update.csv"

# --- Model artifacts -----------------------------------------------------
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
MODEL_PATH = MODELS_DIR / "logistic_regression.pkl"
SPLIT_PATH = MODELS_DIR / "split.json"
METRICS_PATH = REPORTS_DIR / "metrics_text_train.json"
EVAL_METRICS_PATH = REPORTS_DIR / "metrics_text_eval.json"
CLASSIFICATION_REPORT_PATH = REPORTS_DIR / "classification_report_text.txt"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix_text.png"
PREDICTIONS_PATH = REPORTS_DIR / "predictions_text.csv"

# --- TF-IDF hyperparameters (unchanged from the original) ----------------
TFIDF_PARAMS = {
    "max_features": 10000,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.95,
    "lowercase": True,
    "strip_accents": "unicode",
    "stop_words": None,
}

# --- LogisticRegression hyperparameters (unchanged) ----------------------
LOGREG_PARAMS = {
    "max_iter": 1000,
    "random_state": 42,
    "class_weight": "balanced",
}

# --- Split ---------------------------------------------------------------
# There is no text-specific split any more. Train/val/test membership comes
# from rakuten_common.split, which both modalities share, so fusion can be
# scored on the same products. RANDOM_STATE is kept here only because the
# classifier reads it; the split's own seed lives in rakuten_img.config.
#
# The former VAL_SIZE (the teammate's own 80/20) is GONE on purpose. Measured
# on the real data, that 20% val set was EXACTLY the image pipeline's val +
# test blocks combined (8492 + 8492 = 16984), so the text model was being
# scored on rows the image pipeline holds out as its test set.
RANDOM_STATE = _img.RANDOM_STATE

# --- MLflow tracking & Model Registry ------------------------------------
# Its OWN experiment and its OWN registered model, so image, text and later
# fusion runs share a tracking server without colliding. PRODUCTION_ALIAS is
# reused from the image side on purpose: one vocabulary for "what is served".
# Training registers a version and NEVER moves the alias.
import os as _os

EXPERIMENT_NAME = _os.getenv("RAKUTEN_TEXT_EXPERIMENT", "rakuten-text")
REGISTERED_MODEL_NAME = _os.getenv("RAKUTEN_TEXT_REGISTERED_MODEL",
                                   "rakuten-text-classifier")
PRODUCTION_ALIAS = _img.PRODUCTION_ALIAS


def run_params() -> dict:
    """Flat, loggable description of a training run (mirrors config.run_params
    on the image side)."""
    return {
        "modality": "text",
        "vectorizer": "tfidf",
        "classifier": "logistic_regression",
        "max_features": TFIDF_PARAMS["max_features"],
        "ngram_range": str(TFIDF_PARAMS["ngram_range"]),
        "min_df": TFIDF_PARAMS["min_df"],
        "max_df": TFIDF_PARAMS["max_df"],
        "strip_accents": TFIDF_PARAMS["strip_accents"],
        "max_iter": LOGREG_PARAMS["max_iter"],
        "class_weight": LOGREG_PARAMS["class_weight"],
        "random_state": RANDOM_STATE,
        "split_source": "rakuten_common.split",
        "num_classes": NUM_CLASSES,
    }


# --- Canonical class order (the fusion contract) -------------------------
CANONICAL_CLASSES = _img.CANONICAL_CLASSES
CANONICAL_LABELS = _img.CANONICAL_LABELS
NUM_CLASSES = _img.NUM_CLASSES