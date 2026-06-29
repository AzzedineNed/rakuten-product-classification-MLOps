"""Image preprocessing: the "zoom" step from the exploration/preprocessing
notebooks, ported from OpenCV to numpy + Pillow so we don't need opencv.

Rakuten product images are a small product centered on a large white field.
We find the non-white bounding box; if the product fills <= INNER_RATIO_THRESHOLD
of the frame, we crop to it, scale it up preserving aspect ratio, and re-center
it on a fresh white canvas of the original size. Images that already fill the
frame are returned unchanged. This is NOT background removal: the white padding
is intentionally kept so every image stays square and uniformly framed.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from . import config


def to_rgb_array(img: Image.Image) -> np.ndarray:
    """PIL image -> HxWx3 uint8 RGB array."""
    return np.asarray(img.convert("RGB"))


def find_inner_box(arr: np.ndarray, white: int = config.WHITE):
    """Return (top, left, bottom, right) of the non-white region, or None if blank."""
    non_white = np.any(arr != white, axis=2)  # HxW bool
    rows = np.any(non_white, axis=1)
    cols = np.any(non_white, axis=0)
    if not rows.any():
        return None  # fully blank image
    top = int(np.argmax(rows))
    bottom = int(len(rows) - 1 - np.argmax(rows[::-1]))
    left = int(np.argmax(cols))
    right = int(len(cols) - 1 - np.argmax(cols[::-1]))
    return top, left, bottom, right


def inner_ratio(arr: np.ndarray, box) -> float:
    """Fraction of the frame occupied by the inner box."""
    h, w = arr.shape[:2]
    top, left, bottom, right = box
    return ((right - left + 1) * (bottom - top + 1)) / (h * w)


def zoom_to_canvas(arr: np.ndarray, box, white: int = config.WHITE) -> np.ndarray:
    """Crop to box, upscale preserving aspect ratio, re-center on a white canvas."""
    h, w = arr.shape[:2]
    top, left, bottom, right = box
    crop = arr[top : bottom + 1, left : right + 1]
    ch, cw = crop.shape[:2]

    # Scale the longer side to the original dimension (matches notebook logic).
    if cw >= ch:
        new_w = w
        new_h = max(1, int(round(ch * (w / cw))))
    else:
        new_h = h
        new_w = max(1, int(round(cw * (h / ch))))

    resized = np.asarray(
        Image.fromarray(crop).resize((new_w, new_h), Image.BILINEAR)
    )

    canvas = np.full((h, w, 3), white, dtype=np.uint8)
    y0 = (h - new_h) // 2
    x0 = (w - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def process_image(img: Image.Image, threshold: float = config.INNER_RATIO_THRESHOLD) -> Image.Image:
    """Full preprocessing for one image: zoom if the product is small, else keep.

    Returns an RGB PIL image ready to hand to the backbone's transforms.
    """
    arr = to_rgb_array(img)
    box = find_inner_box(arr)
    if box is None:
        return Image.fromarray(arr)  # blank -> leave as-is
    if inner_ratio(arr, box) <= threshold:
        arr = zoom_to_canvas(arr, box)
    return Image.fromarray(arr)


def load_and_process(path: str, threshold: float = config.INNER_RATIO_THRESHOLD) -> Image.Image:
    """Load an image from disk and run process_image on it."""
    with Image.open(path) as img:
        img.load()
        return process_image(img, threshold=threshold)
