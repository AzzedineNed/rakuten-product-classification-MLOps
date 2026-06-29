#!/usr/bin/env python3
"""evaluate.py — Evaluate the trained classifier on the cached test features.

Produces (in reports/):
  * metrics.json            weighted/macro F1 + accuracy
  * classification_report.txt
  * confusion_matrix.png

Examples:
  python scripts/evaluate.py
"""
from __future__ import annotations

import argparse
import json

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

    _plot_confusion(y_test, y_pred, present, target_names)
    print("🎉 evaluate.py done. See reports/.")
    return metrics


def _plot_confusion(y_true, y_pred, labels, target_names) -> None:
    """Save an annotated confusion matrix with two panels:
      * left  — raw counts (zeros left blank to cut clutter)
      * right — row-normalized, i.e. each row sums to 1 so the diagonal is the
                per-class recall; this is what actually reveals which classes
                bleed into which.
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


def main() -> None:
    argparse.ArgumentParser(description="Evaluate on the test set.").parse_args()
    evaluate()


if __name__ == "__main__":
    main()