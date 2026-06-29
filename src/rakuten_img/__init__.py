"""Rakuten image-classification MLOps pipeline (image modality only).

Frozen MobileNetV2 backbone -> cached feature vectors -> small scikit-learn
classifier. Logic lives here; scripts/ and api/ are thin entrypoints over it.
"""
from . import config

__all__ = ["config"]
__version__ = "1.0.0"
