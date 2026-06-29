"""Late-fusion helper — kept here to document the team-level contract, not
because the image pipeline needs it on its own.

The image model and the text model (your teammate's CamemBERT) each emit a
probability vector. Fusion is a weighted average of the two. The ONLY
requirement for it to be correct is that both vectors are ordered the same way.
This project orders everything by config.CANONICAL_CLASSES (prdtypecodes sorted
numerically); the text side must do the same. If both honor that, fusion is the
one-liner below — exactly what the Streamlit demo's get_average_pred did.
"""
from __future__ import annotations

import numpy as np

from . import config


def weighted_average(
    image_proba: np.ndarray,
    text_proba: np.ndarray,
    text_weight: float = 0.5,
) -> np.ndarray:
    """Combine image and text probability vectors (both in canonical order)."""
    image_proba = np.asarray(image_proba, dtype=np.float64)
    text_proba = np.asarray(text_proba, dtype=np.float64)
    if image_proba.shape[-1] != config.NUM_CLASSES or text_proba.shape[-1] != config.NUM_CLASSES:
        raise ValueError(
            f"Both vectors must have {config.NUM_CLASSES} classes in canonical order."
        )
    img_weight = 1.0 - text_weight
    return image_proba * img_weight + text_proba * text_weight
