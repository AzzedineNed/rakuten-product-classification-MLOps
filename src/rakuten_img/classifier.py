"""The small classifier that sits on top of the cached MobileNetV2 features.

Default is a one-hidden-layer MLP (good accuracy/speed balance, exposes
predict_proba for fusion); logistic regression is a config switch away.

We persist the fitted estimator together with its class order and backbone name
so predict.py / the API can verify alignment with config.CANONICAL_CLASSES
instead of trusting it implicitly.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from . import config


def build_classifier(kind: Optional[str] = None):
    """Instantiate the configured classifier head."""
    kind = (kind or config.CLASSIFIER_TYPE).lower()
    if kind == "logreg":
        return LogisticRegression(
            C=1.0,
            max_iter=1000,
            n_jobs=-1,
            multi_class="multinomial",
            random_state=config.RANDOM_STATE,
        )
    if kind == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=config.MLP_HIDDEN,
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=256,
            max_iter=60,
            early_stopping=True,
            n_iter_no_change=6,
            random_state=config.RANDOM_STATE,
            verbose=True,
        )
    raise ValueError(f"Unknown classifier kind: {kind!r} (use 'mlp' or 'logreg')")


def save(clf, path: Path = config.CLASSIFIER_PATH, extra: Optional[dict] = None) -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "classifier": clf,
        "classes": [int(c) for c in clf.classes_],
        "backbone": config.BACKBONE_NAME,
        "feature_dim": config.FEATURE_DIM,
        "classifier_type": config.CLASSIFIER_TYPE,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "extra": extra or {},
    }
    joblib.dump(payload, path)
    print(f"💾 Saved classifier -> {path}")


def load(path: Path = config.CLASSIFIER_PATH) -> dict:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"No trained classifier at {path}. Run train.py (or POST /train) first."
        )
    return joblib.load(path)


def reorder_to_canonical(proba: np.ndarray, model_classes) -> np.ndarray:
    """Reorder a predict_proba row/matrix so columns follow CANONICAL_CLASSES.

    With integer prdtypecode labels, sklearn already sorts classes numerically
    (== canonical), so this is usually an identity reorder — but we do it
    explicitly so the contract holds even if the label set ever changes.
    """
    index = {int(c): i for i, c in enumerate(model_classes)}
    cols = [index[c] for c in config.CANONICAL_CLASSES]
    return proba[..., cols]
