"""The canonical-order contract, in one place.

Every probability vector that crosses a service boundary in this project is
ordered by config.CANONICAL_CLASSES. fusion.weighted_average checks only that a
vector has 27 entries -- it cannot detect that they are in the WRONG order, and
a misordered vector produces confident, wrong, entirely plausible answers. So
ordering is enforced at the source, by this module, and asserted by tests.
"""
from __future__ import annotations

import numpy as np

from rakuten_img import config

__all__ = ["to_canonical", "validate_vector"]


def to_canonical(proba: np.ndarray, model_classes) -> np.ndarray:
    """Reorder predict_proba output so columns follow CANONICAL_CLASSES.

    Works on a single row or a matrix. For a model whose classes_ already equal
    the canonical order (the normal case for integer prdtypecodes) this is an
    identity permutation -- performed anyway, because "usually identity" is not
    a guarantee and the cost is a fancy-index.
    """
    model_classes = [int(c) for c in model_classes]
    missing = set(config.CANONICAL_CLASSES) - set(model_classes)
    if missing:
        raise ValueError(
            f"Model does not know these classes, so no canonical vector can be "
            f"built: {sorted(missing)}. It was probably trained on data missing "
            f"those product types."
        )
    index = {c: i for i, c in enumerate(model_classes)}
    cols = [index[c] for c in config.CANONICAL_CLASSES]
    return np.asarray(proba)[..., cols]


def validate_vector(vec, *, tol: float = 1e-6) -> np.ndarray:
    """Check a single probability vector against the contract. Returns it.

    Note the tolerance: a real softmax/logreg vector sums to something like
    0.9999999999999998, never exactly 1.0. An equality check here would make
    every caller flaky.
    """
    arr = np.asarray(vec, dtype=np.float64)
    if arr.shape[-1] != config.NUM_CLASSES:
        raise ValueError(
            f"Expected {config.NUM_CLASSES} classes, got {arr.shape[-1]}."
        )
    total = float(arr.sum())
    if not np.isclose(total, 1.0, atol=tol):
        raise ValueError(f"Probabilities sum to {total!r}, not 1.0 (tol={tol}).")
    if (arr < 0).any():
        raise ValueError("Negative probability in vector.")
    return arr
