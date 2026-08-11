"""Rakuten text-classification pipeline (text modality only).

TF-IDF -> LogisticRegression. Ported from the teammate's branch
feat/moussa-rakuten-code (aae7a0d) and adapted to this repo's layout:
data comes from data/raw (DVC-tracked), artifacts go to models/text.

Logic lives here; entrypoints stay thin, mirroring rakuten_img.
"""
from . import config

__all__ = ["config"]
__version__ = "0.1.0"
