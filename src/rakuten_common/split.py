"""THE shared train/val/test split. One held-out set for every modality.

WHY THIS EXISTS
    The image pipeline held out an 80/10/10 stratified split; the text pipeline
    held out its own 80/20. Different products. Any fusion score computed
    across those two is meaningless -- and meaningless in the worst way, because
    it still looks like a plausible number. This module is the single source of
    truth so that cannot happen.

WHAT IT GUARANTEES
    * The split is computed from the SAME merged dataframe, with the SAME
      calls, in the SAME order as rakuten_img.data.split_dataframe. Membership
      is byte-for-byte identical to what the image pipeline produces today --
      proven by scripts/check_split.py against the real data, not asserted here.
    * Rows are identified by LABEL (the dataframe index, i.e. the ID column of
      the raw CSVs), never by position. Positional identity breaks the moment
      any step reorders or drops a row; label identity survives it. This is also
      the handle needed to map cached feature .npy rows back to products.

WHAT IT DOES NOT DO
    It does not change rakuten_img.data. The image pipeline keeps calling its
    own split_dataframe exactly as before. This module reproduces that
    behaviour; switching the image side over to it is a separate, later step
    taken only once the equivalence check passes.
"""
from __future__ import annotations

import hashlib

import pandas as pd
from sklearn.model_selection import train_test_split

from rakuten_img import config

__all__ = [
    "load_labeled_dataframe",
    "split_dataframe",
    "split_labels",
    "LABEL_COLUMN",
    "split_fingerprint",
]

LABEL_COLUMN = "prdtypecode"


def load_labeled_dataframe() -> pd.DataFrame:
    """Merge X_train + Y_train into the canonical labeled dataframe.

    Merged on the ID COLUMN (index_col=0), not on row position. The teammate's
    text code merged positionally; that happens to be correct for the current
    files (verified: the two id columns agree on all rows) but is silently
    catastrophic if the files are ever regenerated in a different order.

    Identical to rakuten_img.data.load_raw_dataframe minus the two image path
    columns, which do not affect row order or the split.
    """
    X = pd.read_csv(config.X_TRAIN_CSV, index_col=0)
    y = pd.read_csv(config.Y_TRAIN_CSV, index_col=0)

    df = X.merge(y, left_index=True, right_index=True, how="inner")

    # The merge guard. An inner join that silently drops rows means the two
    # files disagree about which products exist -- fail loudly instead of
    # training on a quietly truncated dataset.
    if len(df) != len(X) or len(df) != len(y):
        raise ValueError(
            f"X/Y merge lost rows: X={len(X)}, Y={len(y)}, merged={len(df)}. "
            f"The two CSVs disagree on their ID column; do not train on this."
        )
    if df.index.has_duplicates:
        raise ValueError(
            "Duplicate ids in the merged dataframe -- the split cannot key on "
            "labels that are not unique."
        )
    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Missing '{LABEL_COLUMN}' column after merge.")
    return df


def split_dataframe(df: pd.DataFrame):
    """Stratified 80/10/10 split -> (train_df, val_df, test_df).

    Deliberately a line-for-line reproduction of
    rakuten_img.data.split_dataframe. Do not "improve" it: any change here
    silently invalidates every cached feature file and every metric ever
    measured against the old split.
    """
    train_df, temp_df = train_test_split(
        df,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        shuffle=True,
        stratify=df[LABEL_COLUMN],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=config.VAL_FRACTION_OF_TEMP,
        random_state=config.RANDOM_STATE,
        shuffle=True,
        stratify=temp_df[LABEL_COLUMN],
    )
    return train_df, val_df, test_df


def split_labels(df: pd.DataFrame | None = None) -> dict[str, pd.Index]:
    """Return {'train': Index, 'val': Index, 'test': Index} of row LABELS.

    This is the interface both modalities should use. Loads the canonical
    dataframe itself when none is passed, so a caller that only needs the
    membership never has to know how the CSVs are merged.
    """
    if df is None:
        df = load_labeled_dataframe()
    train_df, val_df, test_df = split_dataframe(df)

    parts = {"train": train_df.index, "val": val_df.index, "test": test_df.index}

    # Cheap invariants. These cost microseconds and would have caught an
    # entire class of silent evaluation bugs.
    total = sum(len(ix) for ix in parts.values())
    if total != len(df):
        raise ValueError(f"Split lost rows: {total} != {len(df)}")
    if len(parts["train"].union(parts["val"]).union(parts["test"])) != len(df):
        raise ValueError("Split partitions overlap -- a row is in two splits.")
    return parts


def split_fingerprint(labels) -> str:
    """Stable, order-independent hash of a split's membership.

    Recorded at train time and re-checked at evaluate time. If the CSVs are
    regenerated, reordered, or filtered, the fingerprint changes and evaluation
    refuses to run rather than scoring against a partition that no longer
    matches the one the model was fit on.

    Lives HERE, not in an entrypoint. It was originally defined in
    rakuten_text/train.py and imported by rakuten_text/evaluate.py; once both
    of those moved to scripts/ that import would have been script-importing-
    script, which does not work because scripts/ is not a package. It is also
    modality-agnostic by nature, it hashes labels and knows nothing about text,
    so rakuten_common is where it belonged all along.
    """
    joined = ",".join(str(x) for x in sorted(labels))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]
