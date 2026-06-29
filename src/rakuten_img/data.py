"""Data handling: load and merge the raw CSVs, build image paths, do the
stratified 80/10/10 split, and resample the training set to balance classes.

Resampling is done at the *feature* level (after extraction) rather than the
image level, so we never run the backbone twice on the same (oversampled) image.
This reproduces the notebooks' "balance every class to ~4000" methodology.
An alternative is class_weight="balanced" on the classifier; see classifier.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def load_raw_dataframe() -> pd.DataFrame:
    """Merge X_train + Y_train and attach image filename/path columns."""
    X = pd.read_csv(config.X_TRAIN_CSV, index_col=0)
    y = pd.read_csv(config.Y_TRAIN_CSV, index_col=0)
    df = X.merge(y, left_index=True, right_index=True, how="inner")
    df["imagefile"] = (
        "image_" + df["imageid"].astype(str)
        + "_product_" + df["productid"].astype(str) + ".jpg"
    )
    df["imagepath"] = df["imagefile"].apply(lambda f: str(config.RAW_IMAGES_DIR / f))
    return df


def split_dataframe(df: pd.DataFrame):
    """Stratified split into train (80%) / val (10%) / test (10%)."""
    train_df, temp_df = train_test_split(
        df,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        shuffle=True,
        stratify=df["prdtypecode"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=config.VAL_FRACTION_OF_TEMP,
        random_state=config.RANDOM_STATE,
        shuffle=True,
        stratify=temp_df["prdtypecode"],
    )
    return train_df, val_df, test_df


def resample_features(
    X: np.ndarray,
    y: np.ndarray,
    target: int = config.RESAMPLE_TARGET,
    random_state: int = config.RANDOM_STATE,
):
    """Balance every class to `target` samples.

    Classes above target are undersampled (without replacement); classes below
    are oversampled (with replacement). Returns a shuffled (X, y).
    """
    idx = resample_indices(y, target=target, random_state=random_state)
    return X[idx], y[idx]


def resample_indices(
    y: np.ndarray,
    target: int = config.RESAMPLE_TARGET,
    random_state: int = config.RANDOM_STATE,
) -> np.ndarray:
    """Return a shuffled array of ROW INDICES that balances every class to
    `target` samples (undersample above, oversample below).

    Working with indices instead of copying feature rows keeps memory tiny: the
    caller can then materialize X[idx] exactly once (optionally from a memmap),
    avoiding the multiple full-size array copies that a naive vstack creates.
    """
    rng = np.random.default_rng(random_state)
    parts = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        if len(cls_idx) > target:
            sel = rng.choice(cls_idx, size=target, replace=False)
        elif len(cls_idx) < target:
            sel = rng.choice(cls_idx, size=target, replace=True)
        else:
            sel = cls_idx
        parts.append(sel)
    all_idx = np.concatenate(parts)
    rng.shuffle(all_idx)
    return all_idx
