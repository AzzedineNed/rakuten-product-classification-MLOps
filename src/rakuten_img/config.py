"""Central configuration: the single source of truth for paths, the canonical
class order, image parameters, and model choices.

Everything else in the package imports from here. In particular CANONICAL_CLASSES
is the contract that keeps the image pipeline, the API, and the (team-level)
fusion layer aligned: every probability vector produced anywhere is ordered to
match this list.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths (all overridable via environment variables for Docker / CI)
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load a local .env (ENS_USERNAME / ENS_PASSWORD, RAKUTEN_* overrides) BEFORE any
# os.getenv() call below, so values in .env are picked up no matter which entry
# point imports config first. Without this, config's module-level env reads run
# before a script's own load_dotenv(), freezing them to defaults. Absolute path
# (not CWD-relative) so it works regardless of where a script is launched from.
# Guarded so config never hard-depends on python-dotenv (e.g. in minimal CI).
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except Exception:
    pass

DATA_DIR = Path(os.getenv("RAKUTEN_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = Path(os.getenv("RAKUTEN_MODELS_DIR", PROJECT_ROOT / "models"))
REPORTS_DIR = Path(os.getenv("RAKUTEN_REPORTS_DIR", PROJECT_ROOT / "reports"))

# Canonical raw layout produced by collect.py
RAW_IMAGES_DIR = RAW_DIR / "images" / "image_train"
X_TRAIN_CSV = RAW_DIR / "X_train_update.csv"
Y_TRAIN_CSV = RAW_DIR / "Y_train_CVw08PX.csv"
MANIFEST_PATH = RAW_DIR / "manifest.json"

# Where collect.py acquires data from. A local .zip, a local directory, or an
# http(s) URL. Empty by default so the user must point it at their own copy.
RAW_SOURCE = os.getenv("RAKUTEN_RAW_SOURCE", "")

# --- ENS Challenge Data (authenticated download mode: collect.py --from-ens) ---
ENS_BASE = os.getenv("ENS_BASE", "https://challengedata.ens.fr")
ENS_CHALLENGE_ID = int(os.getenv("ENS_CHALLENGE_ID", "35"))
ENS_USERNAME = os.getenv("ENS_USERNAME", "")
ENS_PASSWORD = os.getenv("ENS_PASSWORD", "")
# Endpoints we need: train inputs, train labels, and the image archive.
# (x-test is the unlabeled challenge test set; we don't use it.)
ENS_DOWNLOAD_SLUGS = ["x-train", "y-train", "supplementary-files"]

# --------------------------------------------------------------------------- #
# Labels & canonical class order
# --------------------------------------------------------------------------- #
prdtypecode_labels = {
    10: "Livre usagé",
    40: "Jeux vidéo et accessoires tech",
    50: "Accessoires de console",
    60: "Console de jeux vidéo",
    1140: "Statuette / Figurine",
    1160: "Cartes collectionnables",
    1180: "Jeux de table",
    1280: "Jouets enfants et costumes",
    1281: "Jeux de société",
    1300: "Jouets électroniques",
    1301: "Bas et chaussettes",
    1302: "Jeux extérieurs et vêtements",
    1320: "Articles pour bébé",
    1560: "Meubles intérieurs",
    1920: "Mobilier de chambre",
    1940: "Ustensiles de cuisine",
    2060: "Décoration intérieure",
    2220: "Produits pour animaux",
    2280: "Magazines et journaux",
    2403: "Livres, magazines et BD",
    2462: "Jeux d'occasion",
    2522: "Matériel de bureau",
    2582: "Mobilier de jardin",
    2583: "Équipement de piscine",
    2585: "Outillage et bricolage",
    2705: "Livre nouveau",
    2905: "Jeux pour PC",
}

# THE CONTRACT: prdtypecodes sorted numerically. Every predict_proba vector in
# this project is column-ordered to match this list. scikit-learn sorts integer
# labels the same way, so a classifier fit on raw prdtypecodes already aligns;
# we still verify this explicitly at train and predict time.
CANONICAL_CLASSES = sorted(prdtypecode_labels.keys())
CANONICAL_LABELS = [prdtypecode_labels[c] for c in CANONICAL_CLASSES]
NUM_CLASSES = len(CANONICAL_CLASSES)

# --------------------------------------------------------------------------- #
# Image preprocessing (the "zoom / re-center on white canvas" step)
# --------------------------------------------------------------------------- #
WHITE = 255
INNER_RATIO_THRESHOLD = 0.8  # products filling <= 80% of the frame get zoomed

# --------------------------------------------------------------------------- #
# Backbone (frozen feature extractor)
# --------------------------------------------------------------------------- #
BACKBONE_NAME = "mobilenet_v2"
FEATURE_DIM = 1280            # MobileNetV2 pooled feature size
FEATURE_BATCH_SIZE = int(os.getenv("RAKUTEN_FEATURE_BATCH", "64"))

# --------------------------------------------------------------------------- #
# Split & resampling
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TEST_SIZE = 0.2          # 80% train, 20% temp
VAL_FRACTION_OF_TEMP = 0.5  # temp -> 10% val / 10% test
RESAMPLE_TARGET = 4000   # balance every train class to ~this many samples

# --------------------------------------------------------------------------- #
# Classifier head
# --------------------------------------------------------------------------- #
# "mlp"  -> sklearn MLPClassifier, one hidden layer (recommended)
# "logreg" -> multinomial LogisticRegression (faster, slightly lower F1)
CLASSIFIER_TYPE = os.getenv("RAKUTEN_CLASSIFIER", "mlp")
MLP_HIDDEN = (256,)

# --------------------------------------------------------------------------- #
# Cached artifacts
# --------------------------------------------------------------------------- #
def feature_files(split: str) -> tuple[Path, Path]:
    """(X_path, y_path) for a split in {'train','val','test'} — the FINAL files
    used by train/evaluate (train is resampled)."""
    return PROCESSED_DIR / f"X_{split}.npy", PROCESSED_DIR / f"y_{split}.npy"


def feature_files_raw(split: str) -> tuple[Path, Path]:
    """(X_path, y_path) for the RAW (un-resampled) extracted features. These are
    saved immediately after extraction so the expensive backbone pass is never
    lost, and they make process.py resumable."""
    return PROCESSED_DIR / f"X_{split}_raw.npy", PROCESSED_DIR / f"y_{split}_raw.npy"


CLASSIFIER_PATH = MODELS_DIR / "image_classifier.joblib"


# --------------------------------------------------------------------------- #
# Run parameters snapshot
# --------------------------------------------------------------------------- #
def run_params() -> dict:
    """A single, flat snapshot of the parameters that define a run — backbone,
    split, resampling, and classifier settings.

    This is a pure read of the constants above (no new state, no side effects).
    It exists as a single obvious hook: anything that later wants to record what
    a run was configured with — an experiment tracker, a metadata file, a log
    line — can call this instead of reaching into individual constants. Nothing
    in the pipeline depends on it today.
    """
    return {
        "backbone": BACKBONE_NAME,
        "feature_dim": FEATURE_DIM,
        "feature_batch_size": FEATURE_BATCH_SIZE,
        "classifier_type": CLASSIFIER_TYPE,
        "mlp_hidden": list(MLP_HIDDEN),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "val_fraction_of_temp": VAL_FRACTION_OF_TEMP,
        "resample_target": RESAMPLE_TARGET,
        "num_classes": NUM_CLASSES,
    }