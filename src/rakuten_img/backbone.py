"""Frozen MobileNetV2 feature extractor.

This is the only module that imports torch, and it's loaded lazily so the rest
of the package (config, images, data, classifier, fusion) and the unit tests
never need torch installed.

The backbone is used in exactly two places — process.py (to cache features for
the whole dataset) and predict.py / the API (for live inference) — and both go
through extract_features() / extract_one(), guaranteeing identical preprocessing
on the training and serving paths.
"""
from __future__ import annotations

from typing import Iterable, List

import numpy as np
from PIL import Image

from . import config

_model = None
_transform = None
_device = None


def _init() -> None:
    """Load MobileNetV2 once, strip its classifier, freeze, move to device."""
    global _model, _transform, _device
    if _model is not None:
        return

    import torch  # local import keeps torch optional for the rest of the package
    from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = MobileNet_V2_Weights.IMAGENET1K_V1
    model = mobilenet_v2(weights=weights)
    # Replace the classifier head with identity -> forward() returns the 1280-d
    # pooled feature vector instead of class logits.
    model.classifier = torch.nn.Identity()
    model.eval()
    model.to(_device)
    for p in model.parameters():
        p.requires_grad_(False)

    _model = model
    _transform = weights.transforms()  # resize 256 -> center-crop 224 -> normalize
    print(f"✅ Backbone '{config.BACKBONE_NAME}' loaded on {_device}")


def device() -> str:
    _init()
    return str(_device)


def extract_features(images: Iterable[Image.Image]) -> np.ndarray:
    """Batch of preprocessed PIL images -> (N, FEATURE_DIM) float32 array."""
    import torch

    _init()
    batch = [img for img in images]
    if not batch:
        return np.empty((0, config.FEATURE_DIM), dtype=np.float32)
    tensor = torch.stack([_transform(img) for img in batch]).to(_device)
    with torch.no_grad():
        feats = _model(tensor)
    return feats.detach().cpu().numpy().astype(np.float32)


def extract_one(img: Image.Image) -> np.ndarray:
    """Single PIL image -> (FEATURE_DIM,) float32 vector."""
    return extract_features([img])[0]


def iter_batches(items: List, batch_size: int = config.FEATURE_BATCH_SIZE):
    """Yield consecutive slices of `items` of length `batch_size`."""
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]
