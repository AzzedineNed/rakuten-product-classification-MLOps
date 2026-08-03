"""
Configuration centralisée du projet Rakuten (modèle TF-IDF + Régression Logistique).

Tous les chemins sont calculés relativement à la racine du projet pour que les
scripts fonctionnent quel que soit le répertoire d'exécution.
"""
from pathlib import Path

# --- Répertoires ---------------------------------------------------------
# racine = dossier parent de src/
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR
MODELS_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "output"
FIGURES_DIR = ROOT_DIR / "figures"

# --- Fichiers de données -------------------------------------------------
X_TRAIN_PATH = DATA_DIR / "X_train_update.csv"
Y_TRAIN_PATH = DATA_DIR / "Y_train_CVw08PX.csv"
X_TEST_PATH = DATA_DIR / "X_test_update.csv"

# --- Artefacts du modèle -------------------------------------------------
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
MODEL_PATH = MODELS_DIR / "logistic_regression.pkl"
METRICS_PATH = OUTPUT_DIR / "metrics_logistic_regression.json"
CONFUSION_MATRIX_PATH = FIGURES_DIR / "confusion_matrix_logreg.png"
PREDICTIONS_PATH = OUTPUT_DIR / "predictions_logistic_regression.csv"

# --- Hyperparamètres TF-IDF ---------------------------------------------
TFIDF_PARAMS = {
    "max_features": 10000,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.95,
    "lowercase": True,
    "strip_accents": "unicode",
    "stop_words": None,
}

# --- Hyperparamètres Régression Logistique -------------------------------
LOGREG_PARAMS = {
    "max_iter": 1000,
    "random_state": 42,
    "class_weight": "balanced",  # gère le déséquilibre des classes
}

# --- Divers --------------------------------------------------------------
TEST_SIZE = 0.2
RANDOM_STATE = 42
