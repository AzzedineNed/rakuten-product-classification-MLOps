"""Torch-free smoke tests covering the deterministic logic:
image preprocessing, splitting, resampling, classifier round-trip, fusion, and
the canonical-class-order contract. These run on CPU in seconds and need no
model weights or dataset.
"""
import numpy as np
import pytest
from PIL import Image

from rakuten_img import classifier, config, data, fusion, images


def _img_with_center_box(size=120, box=40, color=(200, 30, 30)):
    """White image with a centered colored square (a small 'product')."""
    arr = np.full((size, size, 3), 255, dtype=np.uint8)
    s = (size - box) // 2
    arr[s : s + box, s : s + box] = color
    return Image.fromarray(arr)


def test_find_inner_box_and_ratio():
    img = _img_with_center_box(size=100, box=20)
    arr = images.to_rgb_array(img)
    box = images.find_inner_box(arr)
    assert box is not None
    top, left, bottom, right = box
    assert (bottom - top + 1) == 20 and (right - left + 1) == 20
    assert images.inner_ratio(arr, box) == pytest.approx(0.04, abs=1e-6)


def test_blank_image_returns_none():
    arr = np.full((50, 50, 3), 255, dtype=np.uint8)
    assert images.find_inner_box(arr) is None


def test_process_image_zooms_small_product():
    img = _img_with_center_box(size=120, box=24)  # ratio 0.04 -> well below 0.8
    out = images.process_image(img)
    out_arr = images.to_rgb_array(out)
    assert out_arr.shape == (120, 120, 3)
    # After zoom the product should fill far more of the frame.
    new_box = images.find_inner_box(out_arr)
    assert images.inner_ratio(out_arr, new_box) > 0.5


def test_process_image_keeps_large_product():
    img = _img_with_center_box(size=100, box=95)  # already fills frame
    out_arr = images.to_rgb_array(images.process_image(img))
    assert out_arr.shape == (100, 100, 3)


def test_resample_balances_classes():
    rng = np.random.default_rng(0)
    X = rng.random((100, 8)).astype(np.float32)
    y = np.array([10] * 90 + [40] * 10)  # imbalanced
    Xr, yr = data.resample_features(X, y, target=50)
    counts = {int(c): int((yr == c).sum()) for c in np.unique(yr)}
    assert counts == {10: 50, 40: 50}
    assert Xr.shape[0] == 100 and Xr.shape[1] == 8


def test_canonical_order_is_numeric_sort():
    assert config.CANONICAL_CLASSES == sorted(config.prdtypecode_labels.keys())
    assert len(config.CANONICAL_CLASSES) == 27


def test_classifier_roundtrip_and_reorder(tmp_path):
    rng = np.random.default_rng(1)
    classes = config.CANONICAL_CLASSES
    # Make a tiny separable-ish dataset over all 27 classes.
    X = rng.random((27 * 6, 16)).astype(np.float32)
    y = np.repeat(classes, 6)

    clf = classifier.build_classifier("logreg")  # fast for tests
    clf.fit(X, y)

    # sklearn sorts integer labels numerically -> equals canonical
    assert [int(c) for c in clf.classes_] == classes

    proba = clf.predict_proba(X[:1])
    reordered = classifier.reorder_to_canonical(proba, [int(c) for c in clf.classes_])
    assert reordered.shape[-1] == config.NUM_CLASSES
    np.testing.assert_allclose(reordered, proba)  # identity reorder here

    path = tmp_path / "clf.joblib"
    classifier.save(clf, path=path)
    loaded = classifier.load(path)
    assert loaded["classes"] == classes
    assert loaded["backbone"] == config.BACKBONE_NAME


def test_fusion_weighted_average():
    a = np.zeros(config.NUM_CLASSES)
    b = np.zeros(config.NUM_CLASSES)
    a[0] = 1.0
    b[1] = 1.0
    out = fusion.weighted_average(a, b, text_weight=0.5)
    assert out[0] == pytest.approx(0.5)
    assert out[1] == pytest.approx(0.5)


def test_fusion_rejects_wrong_shape():
    with pytest.raises(ValueError):
        fusion.weighted_average(np.zeros(3), np.zeros(config.NUM_CLASSES))
