#!/usr/bin/env python3
"""evaluate.py — Evaluate the trained classifier on the cached test features.

Produces (in reports/):
  * metrics.json            weighted/macro F1 + accuracy
  * classification_report.txt
  * confusion_matrix.png

If MLflow tracking is configured, the test metrics, the confusion matrix, the
classification report, and the model are logged. When the loaded model carries
the MLflow run_id from its training run (train.py stores it), these are attached
to that SAME run, so one run holds the full story (train params + val metrics +
test metrics + confusion matrix + model). If there is no stored run_id (e.g. an
older model), a standalone evaluation run is logged instead. Tracking is
best-effort and never blocks evaluation.

Examples:
  python scripts/evaluate.py
"""
from __future__ import annotations

import argparse
import json
import os

import _bootstrap  # noqa: F401
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from rakuten_img import classifier, config


def _load_test():
    x_path, y_path = config.feature_files("test")
    if not (x_path.exists() and y_path.exists()):
        raise FileNotFoundError(f"Missing test features ({x_path}). Run process.py first.")
    return np.load(x_path), np.load(y_path)


def evaluate() -> dict:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = classifier.load()
    clf = payload["classifier"]
    X_test, y_test = _load_test()
    print(f"📊 Test features: {X_test.shape}")

    y_pred = clf.predict(X_test)

    metrics = {
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "test_samples": int(len(y_test)),
        "backbone": payload.get("backbone"),
        "classifier_type": payload.get("classifier_type"),
    }
    print(f"📈 weighted F1 {metrics['f1_weighted']:.4f} | "
          f"macro F1 {metrics['f1_macro']:.4f} | acc {metrics['accuracy']:.4f}")

    # Use the labels actually present so target_names line up.
    present = sorted(set(int(c) for c in np.unique(np.concatenate([y_test, y_pred]))))
    target_names = [config.prdtypecode_labels[c] for c in present]

    report = classification_report(
        y_test, y_pred, labels=present, target_names=target_names,
        digits=4, zero_division=0,
    )
    (config.REPORTS_DIR / "classification_report.txt").write_text(report)
    (config.REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(report)

    cm_path = _plot_confusion(y_test, y_pred, present, target_names)

    _log_to_mlflow(payload, metrics, cm_path)
    print("🎉 evaluate.py done. See reports/.")
    return metrics


def _plot_confusion(y_true, y_pred, labels, target_names):
    """Save an annotated confusion matrix with two panels:
      * left  — raw counts (zeros left blank to cut clutter)
      * right — row-normalized, i.e. each row sums to 1 so the diagonal is the
                per-class recall; this is what actually reveals which classes
                bleed into which.

    Returns the output path so the caller can log it as an MLflow artifact.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # Row-normalize; guard rows that sum to 0 (a label predicted but never true).
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm, row_sums, out=np.zeros(cm.shape, dtype=float), where=row_sums != 0
    )

    n = len(target_names)
    fig, (ax_c, ax_n) = plt.subplots(
        1, 2, figsize=(2 * max(12, n * 0.6), max(11, n * 0.5))
    )

    def _draw(ax, mat, title, fmt, vmax, annotate_above):
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=vmax)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(target_names, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(target_names, fontsize=7)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        thresh = vmax * 0.5  # white text on dark cells, black on light
        for i in range(n):
            for j in range(n):
                v = mat[i, j]
                if v <= annotate_above:
                    continue  # leave near-empty cells blank
                ax.text(j, i, format(v, fmt), ha="center", va="center",
                        fontsize=6, color="white" if v > thresh else "black")

    _draw(ax_c, cm, "Counts", "d",
          vmax=int(cm.max()) if cm.max() else 1, annotate_above=0)
    _draw(ax_n, cm_norm, "Row-normalized (diagonal = recall)", ".2f",
          vmax=1.0, annotate_above=0.005)

    fig.suptitle("Confusion Matrix — image model", fontsize=14)
    fig.tight_layout()
    out = config.REPORTS_DIR / "confusion_matrix.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"🖼️  Saved {out}")
    return out


def _log_to_mlflow(payload, metrics, cm_path):
    """Best-effort MLflow logging. Attaches test metrics + confusion matrix +
    classification report + model to the training run recorded in the model
    payload; falls back to a standalone evaluation run if no run_id is stored.
    Never raises into the evaluation path."""
    if not os.getenv("MLFLOW_TRACKING_URI"):
        print("ℹ️  MLFLOW_TRACKING_URI not set — skipping experiment logging.")
        return
    try:
        import mlflow

        run_id = payload.get("mlflow_run_id")
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "rakuten-image"))
        linked = bool(run_id)
        ctx = mlflow.start_run(run_id=run_id) if linked else mlflow.start_run()
        with ctx:
            mlflow.set_tag("modality", "image")
            if not linked:
                mlflow.set_tag("stage", "evaluate-standalone")
            # Log only numeric metrics (skip the string entries backbone/classifier_type).
            numeric = {k: float(v) for k, v in metrics.items()
                       if isinstance(v, (int, float)) and not isinstance(v, bool)}
            mlflow.log_metrics(numeric)
            mlflow.log_artifact(str(cm_path), artifact_path="plots")
            report_p = config.REPORTS_DIR / "classification_report.txt"
            if report_p.exists():
                mlflow.log_artifact(str(report_p), artifact_path="reports")
            metrics_p = config.REPORTS_DIR / "metrics.json"
            if metrics_p.exists():
                mlflow.log_artifact(str(metrics_p), artifact_path="reports")
            if config.CLASSIFIER_PATH.exists():
                mlflow.log_artifact(str(config.CLASSIFIER_PATH), artifact_path="model")
        if linked:
            print(f"📝 Attached test metrics + confusion matrix to training run ({run_id[:8]}…).")
        else:
            print("📝 Logged a standalone evaluation run (no training run_id found).")
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️  MLflow logging skipped ({type(exc).__name__}: {exc}).")


def main() -> None:
    argparse.ArgumentParser(description="Evaluate on the test set.").parse_args()
    evaluate()


if __name__ == "__main__":
    main()