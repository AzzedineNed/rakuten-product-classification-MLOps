"""Modality-agnostic pieces shared by rakuten_img and rakuten_text.

Nothing here may import torch, sklearn estimators, or anything modality
specific: both pipelines and the future fusion service depend on this package,
so it stays cheap and boring on purpose.
"""

__all__ = ["split"]
__version__ = "0.1.0"
