"""THE CONTRACT TEST -- the highest-value test in this project.

Fusion combines an image probability vector with a text probability vector by
weighted average. That is correct if and only if both vectors are ordered by
config.CANONICAL_CLASSES. fusion.weighted_average validates the LENGTH and
nothing else, so a misordered vector does not raise, does not warn, and yields
confident wrong answers that look entirely reasonable. Ordering fails SILENTLY.
These tests are what makes it fail loudly instead.

Torch-free and data-free: the fixtures below train tiny sklearn models on
synthetic text in a fraction of a second. No dataset, no cached features, no
model artifacts.
"""
import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from rakuten_common import fusion
from rakuten_common.contract import to_canonical, validate_vector
from rakuten_img import classifier, config


@pytest.fixture(scope="module")
def tiny_text_model():
    """A TF-IDF + LogisticRegression over all 27 classes, trained in-memory."""
    docs, labels = [], []
    for c in config.CANONICAL_CLASSES:
        for k in range(4):
            docs.append(f"produit categorie{c} terme{c}_{k} description commune")
            labels.append(c)
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2)),
        LogisticRegression(max_iter=500, random_state=42),
    )
    model.fit(docs, labels)
    return model


# --------------------------------------------------------------------------- #
# The order itself
# --------------------------------------------------------------------------- #
def test_canonical_classes_are_unique_sorted_ints():
    cls = config.CANONICAL_CLASSES
    assert len(cls) == 27 == len(set(cls))
    assert all(isinstance(c, int) for c in cls)
    assert cls == sorted(cls)


def test_labels_align_one_to_one_with_classes():
    assert len(config.CANONICAL_LABELS) == len(config.CANONICAL_CLASSES)
    # The pairing is what the API returns to users; a shift here mislabels
    # every prediction while every probability stays "correct".
    for code, label in zip(config.CANONICAL_CLASSES, config.CANONICAL_LABELS):
        assert config.prdtypecode_labels[code] == label


# --------------------------------------------------------------------------- #
# Text modality
# --------------------------------------------------------------------------- #
def test_text_model_classes_match_canonical(tiny_text_model):
    got = [int(c) for c in tiny_text_model[-1].classes_]
    assert got == list(config.CANONICAL_CLASSES)


def test_text_vector_satisfies_the_contract(tiny_text_model):
    proba = tiny_text_model.predict_proba(["produit categorie60 terme60_1"])[0]
    vec = to_canonical(proba, tiny_text_model[-1].classes_)
    validate_vector(vec)  # length 27, sums to ~1, non-negative
    assert vec.shape == (config.NUM_CLASSES,)


def test_text_argmax_maps_to_the_right_code(tiny_text_model):
    """The end-to-end claim: position i of the vector means CANONICAL_CLASSES[i]."""
    for code in (10, 60, 1560, 2905):
        proba = tiny_text_model.predict_proba([f"produit categorie{code} terme{code}_0"])[0]
        vec = to_canonical(proba, tiny_text_model[-1].classes_)
        assert config.CANONICAL_CLASSES[int(vec.argmax())] == \
            int(tiny_text_model.predict([f"produit categorie{code} terme{code}_0"])[0])


# --------------------------------------------------------------------------- #
# Both modalities agree
# --------------------------------------------------------------------------- #
def test_both_reorder_helpers_agree():
    """rakuten_common.contract.to_canonical and rakuten_img.classifier.
    reorder_to_canonical must be interchangeable; two implementations of one
    contract is how drift starts."""
    rng = np.random.default_rng(0)
    proba = rng.random((3, config.NUM_CLASSES))
    proba /= proba.sum(axis=1, keepdims=True)
    shuffled = list(reversed(config.CANONICAL_CLASSES))
    np.testing.assert_allclose(
        to_canonical(proba, shuffled),
        classifier.reorder_to_canonical(proba, shuffled),
    )


def test_reorder_actually_reorders():
    """A permuted model order must be undone, not passed through."""
    reversed_classes = list(reversed(config.CANONICAL_CLASSES))
    proba = np.arange(config.NUM_CLASSES, dtype=float)
    proba = proba / proba.sum()
    out = to_canonical(proba, reversed_classes)
    np.testing.assert_allclose(out, proba[::-1])


def test_to_canonical_rejects_a_model_missing_a_class():
    with pytest.raises(ValueError):
        to_canonical(np.zeros(26), config.CANONICAL_CLASSES[:-1])


# --------------------------------------------------------------------------- #
# Tolerance: real vectors never sum to exactly 1.0
# --------------------------------------------------------------------------- #
def test_validate_accepts_float_noise():
    """Measured on the real text model: 0.9999999999999998. An == 1.0 check
    would reject a perfectly valid vector."""
    vec = np.full(config.NUM_CLASSES, 1.0 / config.NUM_CLASSES)
    vec[0] -= 2e-16
    validate_vector(vec)


def test_validate_rejects_wrong_length_and_bad_sums():
    with pytest.raises(ValueError):
        validate_vector(np.full(26, 1 / 26))
    with pytest.raises(ValueError):
        validate_vector(np.full(config.NUM_CLASSES, 1.0))  # sums to 27


# --------------------------------------------------------------------------- #
# Why this file exists
# --------------------------------------------------------------------------- #
def test_fusion_cannot_detect_misordering():
    """Documents the danger rather than fixing it.

    A reversed text vector passes fusion.weighted_average without complaint and
    changes the answer. Fusion cannot police ordering -- only the tests above
    can, which is precisely why they exist.
    """
    image = np.zeros(config.NUM_CLASSES)
    image[0] = 1.0
    text = np.zeros(config.NUM_CLASSES)
    text[0] = 1.0

    correct = fusion.weighted_average(image, text, text_weight=0.5)
    assert int(correct.argmax()) == 0

    misordered = fusion.weighted_average(image, text[::-1], text_weight=0.5)
    assert misordered.shape[-1] == config.NUM_CLASSES  # no error raised
    assert float(misordered.sum()) == pytest.approx(1.0)  # still "valid"
    assert float(misordered[0]) == pytest.approx(0.5)  # confidence halved, silently


# --------------------------------------------------------------------------- #
# The fusion weight is a measurement, not a preference
# --------------------------------------------------------------------------- #
def test_default_text_weight_is_the_measured_one():
    """Pins 0.85 so nobody quietly restores 0.5.

    Measured by scripts/tune_fusion_weight.py on the shared val split: 0.5
    scores 0.6621 weighted F1, BELOW text alone (0.7818), while 0.85 scores
    0.7945 and confirms at 0.7973 on test. A default that loses to one of its
    own inputs is a bug, and it is invisible without this test.
    """
    assert fusion.DEFAULT_TEXT_WEIGHT == 0.85


def test_default_weight_actually_favours_text():
    """A vector fused with the default must lean text-ward, not 50/50."""
    image = np.zeros(config.NUM_CLASSES)
    image[0] = 1.0
    text = np.zeros(config.NUM_CLASSES)
    text[1] = 1.0
    out = fusion.weighted_average(image, text)  # no explicit weight
    assert out[1] > out[0]
    assert out[1] == pytest.approx(0.85)


def test_fusion_rejects_out_of_range_weight():
    a = np.full(config.NUM_CLASSES, 1.0 / config.NUM_CLASSES)
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError):
            fusion.weighted_average(a, a, text_weight=bad)

# ---------------------------------------------------------------------------
# split_fingerprint
#
# Moved into rakuten_common.split from rakuten_text/train.py when the text
# entrypoints went to scripts/: evaluate imported it from train, and once both
# were scripts that would have been script-importing-script, which does not
# work. It is modality-agnostic anyway - it hashes labels.
#
# The fingerprint is a SAFETY device: train records it, evaluate re-checks it,
# and a mismatch stops evaluation rather than scoring against a partition the
# model was never fit on. So these test the properties that make it safe, not
# just that it returns a string.
# ---------------------------------------------------------------------------

def test_split_fingerprint_ignores_order():
    """Membership is what matters; a reordered CSV is the same split."""
    from rakuten_common.split import split_fingerprint
    assert split_fingerprint([10, 40, 2583, 1180]) == split_fingerprint([2583, 10, 1180, 40])


def test_split_fingerprint_changes_when_membership_changes():
    """THE point of the thing. If this ever stops holding, a regenerated or
    filtered split would sail through evaluation unnoticed."""
    from rakuten_common.split import split_fingerprint
    base = [10, 40, 2583]
    assert split_fingerprint(base) != split_fingerprint(base + [60])      # added
    assert split_fingerprint(base) != split_fingerprint(base[:-1])        # removed
    assert split_fingerprint(base) != split_fingerprint([10, 40, 2584])   # changed


def test_split_fingerprint_is_stable_across_calls_and_types():
    """Recorded in one process at train time and compared in another at
    evaluate time, so it must not depend on hash randomisation or on whether
    the labels arrive as a list or a pandas Index."""
    import pandas as pd
    from rakuten_common.split import split_fingerprint
    labels = [10, 40, 2583, 1180]
    first = split_fingerprint(labels)
    assert first == split_fingerprint(labels)
    assert first == split_fingerprint(pd.Index(labels))
    assert first == split_fingerprint(tuple(labels))


def test_split_fingerprint_is_a_short_hex_digest():
    from rakuten_common.split import split_fingerprint
    fp = split_fingerprint([1, 2, 3])
    assert len(fp) == 16
    assert all(c in "0123456789abcdef" for c in fp)


def test_split_fingerprint_is_exported():
    """It is part of rakuten_common's public surface now, not a private helper
    that happens to be importable."""
    from rakuten_common import split
    assert "split_fingerprint" in split.__all__
