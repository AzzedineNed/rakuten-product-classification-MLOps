"""Late fusion of the image and text probability vectors.

Moved here from rakuten_img.fusion: fusion belongs to neither modality, and the
gateway service will import it without pulling in the image package. The old
path still works (rakuten_img.fusion re-exports this module) so nothing breaks.

THE ONE REQUIREMENT
    Both vectors must be ordered by config.CANONICAL_CLASSES. This function
    checks the LENGTH and cannot detect a wrong ORDER -- a misordered vector
    produces a confident, plausible, wrong answer. Ordering is enforced at the
    source by rakuten_common.contract.to_canonical and asserted by
    tests/test_contract.py. See test_fusion_cannot_detect_misordering, which
    exists to document exactly this hole.
"""
from __future__ import annotations

import numpy as np

from rakuten_img import config

__all__ = ["weighted_average", "DEFAULT_TEXT_WEIGHT"]

# MEASURED, NOT CHOSEN. scripts/tune_fusion_weight.py swept text_weight over
# the shared validation split (8,492 products scored by both modalities):
#
#     weight 0.00 (image only)   f1_weighted 0.5468
#     weight 0.50 (the old default) 0.6621   <- WORSE than text alone
#     weight 0.85 (best)            0.7945
#     weight 1.00 (text only)       0.7818
#
# Confirmed once on the held-out test split: 0.7973 weighted F1, 0.7990
# accuracy -- +0.0193 over text alone. The curve is flat from 0.80 to 0.90, so
# this is not a knife-edge fit.
#
# The old default of 0.5 was actively harmful: it scored 0.12 BELOW using the
# text model on its own, silently. If either model is retrained in a way that
# changes its quality, re-run the sweep -- this number describes a pair of
# models, not a law of nature.
DEFAULT_TEXT_WEIGHT = 0.85


def weighted_average(
    image_proba: np.ndarray,
    text_proba: np.ndarray,
    text_weight: float = DEFAULT_TEXT_WEIGHT,
) -> np.ndarray:
    """Combine image and text probability vectors (both in canonical order).

    Accepts single vectors of shape (27,) or batches of shape (n, 27).
    """
    image_proba = np.asarray(image_proba, dtype=np.float64)
    text_proba = np.asarray(text_proba, dtype=np.float64)
    if image_proba.shape[-1] != config.NUM_CLASSES or text_proba.shape[-1] != config.NUM_CLASSES:
        raise ValueError(
            f"Both vectors must have {config.NUM_CLASSES} classes in canonical order."
        )
    if not 0.0 <= float(text_weight) <= 1.0:
        raise ValueError(f"text_weight must be in [0, 1], got {text_weight!r}.")
    img_weight = 1.0 - text_weight
    return image_proba * img_weight + text_proba * text_weight
