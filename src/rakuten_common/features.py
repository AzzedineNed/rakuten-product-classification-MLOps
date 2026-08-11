"""Map cached image-feature rows back to the products they came from.

THE PROBLEM THIS SOLVES
    scripts/process.py saves only X (features) and y (labels) per split. Product
    ids are never recorded, so a cached .npy row could not be tied to a product
    -- which makes per-product fusion evaluation impossible: you cannot ask
    "what did the text model say about the SAME item".

WHY NOTHING NEEDS REPROCESSING
    Feature rows are written in dataframe order, and rows are dropped only when
    an image fails to load. Measured on the real dataset: val 8492/8492,
    test 8492/8492, raw train 67932/67932 -- ZERO failures. So the id of feature
    row i is simply the i-th label of the shared split. Verified element-wise
    against the stored y arrays, not assumed: every function here re-checks the
    labels before handing back a mapping, and raises if they disagree.

    If a future extraction ever does skip an image, that check fails loudly and
    process.py must then persist ids properly. Failing loudly is the point.

TRAIN IS DIFFERENT, ON PURPOSE
    X_train.npy is the RESAMPLED set (27 classes x RESAMPLE_TARGET, with
    duplicated rows), so row identity there is genuinely unrecoverable. That is
    fine: fusion is scored on held-out products, never on training rows. Use
    "train_raw" if you need the pre-resampling train features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from rakuten_img import config

from . import split as shared

__all__ = ["aligned_index", "aligned_products", "SUPPORTED_SPLITS"]

SUPPORTED_SPLITS = ("val", "test", "train_raw")


def _paths(split: str):
    if split == "train_raw":
        return config.feature_files_raw("train")
    if split in ("val", "test"):
        return config.feature_files(split)
    raise ValueError(
        f"Split {split!r} is not mappable to products. Supported: "
        f"{SUPPORTED_SPLITS}. ('train' is the resampled set -- row identity is "
        f"lost by design.)"
    )


def aligned_index(split: str, df: pd.DataFrame | None = None) -> pd.Index:
    """Row labels for the cached features of `split`, in feature-row order.

    Raises if the stored labels do not match the shared split element-wise --
    i.e. if the features on disk were produced from a different partition, or
    images were skipped during extraction.
    """
    if df is None:
        df = shared.load_labeled_dataframe()
    parts = shared.split_labels(df)

    key = "train" if split == "train_raw" else split
    ix = parts[key]

    _, y_path = _paths(split)
    if not y_path.exists():
        raise FileNotFoundError(
            f"{y_path} not found -- run scripts/process.py to cache features."
        )
    y_cached = np.load(y_path)

    if len(y_cached) != len(ix):
        raise ValueError(
            f"{split}: {len(y_cached)} cached feature rows but {len(ix)} rows in "
            f"the shared split. Images were skipped during extraction, or the "
            f"features predate the current split. The row->product mapping "
            f"CANNOT be reconstructed; process.py must persist ids explicitly."
        )

    y_split = df.loc[ix, shared.LABEL_COLUMN].to_numpy()
    if not np.array_equal(y_cached, y_split):
        n_bad = int((y_cached != y_split).sum())
        raise ValueError(
            f"{split}: cached labels disagree with the shared split on {n_bad} "
            f"of {len(y_cached)} rows. The cached features do not correspond to "
            f"this partition -- re-run scripts/process.py."
        )
    return ix


def aligned_products(split: str, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per cached feature row, in order.

    Columns: productid, imageid, prdtypecode. Indexed by the dataframe label.
    This is the handle a fusion evaluation needs to line up image predictions
    with text predictions for the SAME product.
    """
    if df is None:
        df = shared.load_labeled_dataframe()
    ix = aligned_index(split, df=df)
    cols = [c for c in ("productid", "imageid", shared.LABEL_COLUMN) if c in df.columns]
    return df.loc[ix, cols].copy()
