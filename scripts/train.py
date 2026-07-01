#!/usr/bin/env python3
"""train.py — Train the classifier head on cached features.

Loads the cached train (and val, for a quick check) feature arrays, fits the
configured classifier, verifies its class order matches the canonical contract,
and saves the model with metadata. Fast and fully re-runnable — this is what
the /train API endpoint calls.

Experiment tracking: if an MLflow tracking server is configured (via the
MLFLOW_TRACKING_URI env var, e.g. a DagsHub repo's .mlflow endpoint), this run's
parameters, metrics, and the trained model file are logged to it. If it is NOT
configured, or the server is unreachable, training proceeds and the model is
still saved locally — remote logging is best-effort and never blocks a train.

Examples:
  python scripts/train.py
  RAKUTEN_CLASSIFIER=logreg python scripts/train.py
"""
from __future__ import annotations

import argparse
import os
import time

import _bootstrap  # noqa: F401
import numpy as np
from sklearn.metrics import f1_score

from rakuten_img import classifier, config


def _load(split: str):
    x_path, y_path = config.feature_files(split)
    if not (x_path.exists() and y_path.exists()):
        raise FileNotFoundError(
            f"Missing cached features for '{split}' ({x_path}). Run process.py first."
        )
    return np.load(x_path), np.load(y_path)


def _mlflow_enabled() -> bool:
    """True only if a tracking server is configured. We gate on the URI so that
    a machine with no MLflow setup (a teammate who just cloned, CI, offline work)
    trains normally without any tracking side effects."""
    return bool(os.getenv("MLFLOW_TRACKING_URI"))


def train() -> dict:
    """Fit and save the classifier. Returns a small metrics dict.

    If MLflow tracking is configured, params/metrics/model are logged to it as a
    single run. Tracking failures are caught and warned about, never fatal.
    """
    t0 = time.time()
    X_train, y_train = _load("train")
    print(f"📊 Train features: {X_train.shape}")

    clf = classifier.build_classifier()
    print(f"🏋️  Fitting {config.CLASSIFIER_TYPE} on {len(y_train):,} samples...")
    clf.fit(X_train, y_train)

    # Verify the column order matches the canonical contract.
    model_classes = [int(c) for c in clf.classes_]
    if model_classes != config.CANONICAL_CLASSES:
        print("⚠️  classifier.classes_ != CANONICAL_CLASSES; predictions will be "
              "reordered at inference time via reorder_to_canonical().")
    else:
        print("✅ Class order matches canonical contract.")

    metrics = {"train_samples": int(len(y_train))}
    try:
        X_val, y_val = _load("val")
        val_f1 = f1_score(y_val, clf.predict(X_val), average="weighted", zero_division=0)
        metrics["val_f1_weighted"] = float(val_f1)
        print(f"📈 Validation weighted F1: {val_f1:.4f}")
    except FileNotFoundError:
        print("ℹ️  No val features found — skipping val check.")

    metrics["elapsed_sec"] = round(time.time() - t0, 1)
    classifier.save(clf, extra=metrics)

    _log_to_mlflow(clf, metrics)

    print(f"🎉 train.py done in {metrics['elapsed_sec']}s")
    return metrics


def _log_to_mlflow(clf, metrics: dict) -> None:
    """Best-effort experiment logging. Never raises into the training path."""
    if not _mlflow_enabled():
        print("ℹ️  MLFLOW_TRACKING_URI not set — skipping experiment logging.")
        return
    try:
        import mlflow

        # Experiment name namespaced by modality so text/fusion runs can share
        # the same tracking server later without colliding with image runs.
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "rakuten-image"))
        with mlflow.start_run():
            mlflow.set_tag("modality", "image")
            mlflow.log_params(config.run_params())
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
            if config.CLASSIFIER_PATH.exists():
                mlflow.log_artifact(str(config.CLASSIFIER_PATH), artifact_path="model")
        print("📝 Logged run to MLflow.")
    except Exception as exc:  # noqa: BLE001
        # A tracking outage, bad creds, or missing mlflow must not lose a model
        # that is already saved to disk. Warn and carry on.
        print(f"⚠️  MLflow logging skipped ({type(exc).__name__}: {exc}).")


def main() -> None:
    argparse.ArgumentParser(description="Train the image classifier head.").parse_args()
    train()


if __name__ == "__main__":
    main()