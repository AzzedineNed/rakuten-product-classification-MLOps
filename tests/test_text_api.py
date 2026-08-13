"""Characterization tests for api/text_main.py — the service AS IT IS TODAY.

WHY THIS FILE EXISTS. Until now no test imported api/ at all: the three FastAPI
services were the only untested code in the project, and they are the part users
actually talk to. A training endpoint is about to be added to this service, so
its current behaviour is pinned FIRST, in its own commit, exactly as the image
registry was pinned before rakuten_img/tracking.py was extracted. Every
assertion here describes what the code does now; none of it describes what we
would like it to do.

THESE TESTS NEVER TOUCH THE DISK MODEL OR THE REGISTRY, ON PURPOSE.
api/text_main.py loads eagerly in its lifespan, preferring the MLflow registry.
Run naively, this file would behave differently on three machines: green here
(no artifacts, no credentials), registry-backed on the developer's laptop (where
models/text/*.pkl and MLFLOW_TRACKING_URI both exist), and network-dependent in
CI. A test whose result depends on which machine ran it is not a test. So
`predictor.load` is always replaced, and the two states that matter — loaded and
not-loaded — are produced deliberately rather than found.

COUNTERS ARE ASSERTED AS DELTAS. ServiceMetrics defaults to prometheus_client's
global REGISTRY, and api.text_main builds exactly one instance at import, so
counters accumulate across every test in the process. Reading an absolute value
would pass alone and fail in a full run. Read before, act, read after.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# api/ is a package at the repo ROOT, and conftest.py only puts src/ on the
# path. Same reason tests/test_promote.py appends scripts/ itself.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# fastapi is installed by requirements-ci.txt as of the commit that added this
# file. importorskip keeps the file honest if somebody trims the CI subset
# again: it would skip loudly with a reason rather than vanish.
pytest.importorskip("fastapi", reason="fastapi not installed (CI subset)")

from fastapi.testclient import TestClient  # noqa: E402

import api.text_main as tm  # noqa: E402
from rakuten_text import config  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _fake_detailed(designation: str, description: str = "", top_k: int = 5) -> dict:
    """A predict_detailed payload with the real class order and a known winner.

    Deliberately mirrors the real method's SHAPE rather than importing it: the
    point is to characterize the API layer, not to re-test the predictor.
    """
    classes = list(config.CANONICAL_CLASSES)
    labels = list(config.CANONICAL_LABELS)
    probabilities = [0.0] * len(classes)
    probabilities[0] = 1.0
    top = [
        {"prdtypecode": int(classes[i]), "label": labels[i],
         "probability": float(probabilities[i])}
        for i in range(min(top_k, len(classes)))
    ]
    return {
        "top_k": top,
        "prediction": top[0],
        "canonical_classes": classes,
        "probabilities": probabilities,
    }


def _client(monkeypatch, *, loaded: bool, detailed=None) -> TestClient:
    """A TestClient over the real app with the load outcome forced.

    Used as a context manager by every test, because the eager load lives in
    the lifespan and TestClient only runs lifespan inside `with`.
    """
    if loaded:
        # is_loaded is a read-only property over these two attributes, so the
        # loaded state is produced the same way the real load() produces it.
        def fake_load(prefer_registry: bool = False):
            tm.predictor.vectorizer = object()
            tm.predictor.model = object()
            tm.predictor.serving_source = "local:test.pkl"
            return tm.predictor
        monkeypatch.setattr(tm.predictor, "predict_detailed",
                            detailed or _fake_detailed)
    else:
        def fake_load(prefer_registry: bool = False):
            raise FileNotFoundError("no artifacts in this test")

    monkeypatch.setattr(tm.predictor, "vectorizer", None)
    monkeypatch.setattr(tm.predictor, "model", None)
    monkeypatch.setattr(tm.predictor, "serving_source", "not-loaded")
    monkeypatch.setattr(tm.predictor, "load", fake_load)
    return TestClient(tm.app)


def _sample(name: str, labels: dict) -> float:
    value = tm.METRICS.registry.get_sample_value(name, labels)
    return 0.0 if value is None else value


# --------------------------------------------------------------------------- #
# The service with NO model — the state an orchestrator must be told about
# --------------------------------------------------------------------------- #
def test_startup_survives_a_load_failure(monkeypatch):
    """A missing or corrupt artifact must NOT stop the service booting.

    The lifespan catches Exception rather than FileNotFoundError precisely so a
    broken model lands in /health instead of a container that will not start.
    """
    with _client(monkeypatch, loaded=False) as client:
        assert client.get("/health").status_code == 200


def test_health_reports_not_loaded_truthfully(monkeypatch):
    with _client(monkeypatch, loaded=False) as client:
        body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["modality"] == "text"
    assert body["model_loaded"] is False
    assert body["model_source"] == "not-loaded"
    assert body["num_classes"] == config.NUM_CLASSES == 27
    assert body["registered_model"] == config.REGISTERED_MODEL_NAME
    # model_path is the LOCAL FALLBACK path, not the answer to "what is being
    # served" — it is reported even when nothing loaded.
    assert body["model_path"] == str(config.MODEL_PATH)


def test_model_info_gauge_is_published_even_when_loading_failed(monkeypatch):
    """An absent series is indistinguishable from a dead scrape, so the gauge
    is set in BOTH branches of the lifespan. This pins that."""
    with _client(monkeypatch, loaded=False):
        pass
    assert _sample("rakuten_model_info",
                   {"service": "text", "modality": "text",
                    "source": "not-loaded"}) == 1.0


def test_predict_is_503_not_500_when_the_model_is_missing(monkeypatch):
    with _client(monkeypatch, loaded=False) as client:
        response = client.post("/predict",
                               json={"designation": "chaise", "description": ""})
    assert response.status_code == 503
    assert response.json() == {
        "error": "Text model not loaded. Run "
                 "`python scripts/train_text.py` (or `dvc pull`)."
    }


def test_predict_batch_is_503_when_the_model_is_missing(monkeypatch):
    with _client(monkeypatch, loaded=False) as client:
        response = client.post("/predict/batch",
                               json={"produits": [{"designation": "chaise"}]})
    assert response.status_code == 503
    assert "not loaded" in response.json()["error"]


def test_metrics_is_served_even_with_no_model(monkeypatch):
    """/metrics must work in the degraded state — that is when it matters most."""
    with _client(monkeypatch, loaded=False) as client:
        response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "rakuten_model_info" in response.text


# --------------------------------------------------------------------------- #
# The service WITH a model
# --------------------------------------------------------------------------- #
def test_health_reports_the_serving_source_when_loaded(monkeypatch):
    with _client(monkeypatch, loaded=True) as client:
        body = client.get("/health").json()
    assert body["model_loaded"] is True
    assert body["model_source"] == "local:test.pkl"


def test_predict_returns_the_shared_payload_shape(monkeypatch):
    """The payload mirrors the image service so the gateway need not special-case
    a modality, and so either vector can go straight into fusion."""
    with _client(monkeypatch, loaded=True) as client:
        body = client.post("/predict",
                           json={"designation": "chaise",
                                 "description": "en bois"}).json()

    assert set(body) == {"top_k", "prediction", "canonical_classes", "probabilities"}
    assert body["canonical_classes"] == list(config.CANONICAL_CLASSES)
    assert len(body["probabilities"]) == config.NUM_CLASSES
    assert body["prediction"] == body["top_k"][0]
    assert set(body["prediction"]) == {"prdtypecode", "label", "probability"}


@pytest.mark.parametrize("requested, expected", [
    (0, 1),                 # clamped up
    (-5, 1),                # clamped up
    (1, 1),
    (5, 5),
    (27, 27),
    (99, 27),               # clamped down to NUM_CLASSES
])
def test_top_k_is_clamped_into_range(monkeypatch, requested, expected):
    seen = {}

    def spy(designation, description="", top_k=5):
        seen["top_k"] = top_k
        return _fake_detailed(designation, description, top_k)

    with _client(monkeypatch, loaded=True, detailed=spy) as client:
        body = client.post(f"/predict?top_k={requested}",
                           json={"designation": "chaise"}).json()

    assert seen["top_k"] == expected
    assert len(body["top_k"]) == expected


def test_batch_clamps_top_k_the_same_way(monkeypatch):
    """The batch endpoint clamps independently of /predict — it has its own copy
    of the line. Added because a mutant that broke ONLY the batch clamp survived
    the single-product test above.
    """
    seen = []

    def spy(designation, description="", top_k=5):
        seen.append(top_k)
        return _fake_detailed(designation, description, top_k)

    with _client(monkeypatch, loaded=True, detailed=spy) as client:
        client.post("/predict/batch?top_k=99",
                    json={"produits": [{"designation": "a"}, {"designation": "b"}]})
        client.post("/predict/batch?top_k=0",
                    json={"produits": [{"designation": "c"}]})

    assert seen == [config.NUM_CLASSES, config.NUM_CLASSES, 1]


def test_batch_returns_one_prediction_per_product(monkeypatch):
    produits = [{"designation": f"produit {i}"} for i in range(3)]
    with _client(monkeypatch, loaded=True) as client:
        body = client.post("/predict/batch", json={"produits": produits}).json()
    assert len(body["predictions"]) == 3


def test_empty_batch_is_400(monkeypatch):
    with _client(monkeypatch, loaded=True) as client:
        response = client.post("/predict/batch", json={"produits": []})
    assert response.status_code == 400
    assert response.json() == {"error": "The product list is empty."}


def test_description_is_optional_but_designation_is_not(monkeypatch):
    """Pydantic validation runs BEFORE the handler, so this is 422 rather than
    503 even with no model loaded. Pinning the order, not just the code."""
    with _client(monkeypatch, loaded=False) as client:
        assert client.post("/predict", json={"designation": ""}).status_code == 422
        assert client.post("/predict", json={"description": "x"}).status_code == 422

    with _client(monkeypatch, loaded=True) as client:
        assert client.post("/predict",
                           json={"designation": "chaise"}).status_code == 200


# --------------------------------------------------------------------------- #
# Failure handling
# --------------------------------------------------------------------------- #
def test_a_predictor_exception_becomes_500_with_the_reason(monkeypatch):
    def boom(designation, description="", top_k=5):
        raise ValueError("vectorizer exploded")

    with _client(monkeypatch, loaded=True, detailed=boom) as client:
        response = client.post("/predict", json={"designation": "chaise"})

    assert response.status_code == 500
    assert response.json() == {"error": "vectorizer exploded"}


def test_a_predictor_exception_in_a_batch_becomes_500(monkeypatch):
    def boom(designation, description="", top_k=5):
        raise ValueError("vectorizer exploded")

    with _client(monkeypatch, loaded=True, detailed=boom) as client:
        response = client.post("/predict/batch",
                               json={"produits": [{"designation": "chaise"}]})

    assert response.status_code == 500
    assert response.json() == {"error": "vectorizer exploded"}


# --------------------------------------------------------------------------- #
# Instrumentation
# --------------------------------------------------------------------------- #
def test_one_prediction_is_counted_per_product_not_per_request(monkeypatch):
    """A batch of 3 must move predictions_total by 3. Counting the request once
    would make the metric disagree with how many products were classified."""
    winner = str(int(config.CANONICAL_CLASSES[0]))
    labels = {"service": "text", "prdtypecode": winner}

    with _client(monkeypatch, loaded=True) as client:
        before = _sample("rakuten_predictions_total", labels)
        client.post("/predict", json={"designation": "chaise"})
        after_single = _sample("rakuten_predictions_total", labels)
        client.post("/predict/batch",
                    json={"produits": [{"designation": f"p{i}"} for i in range(3)]})
        after_batch = _sample("rakuten_predictions_total", labels)

    assert after_single - before == 1
    assert after_batch - after_single == 3


def test_a_failed_prediction_is_not_counted(monkeypatch):
    def boom(designation, description="", top_k=5):
        raise ValueError("nope")

    winner = str(int(config.CANONICAL_CLASSES[0]))
    labels = {"service": "text", "prdtypecode": winner}

    with _client(monkeypatch, loaded=True, detailed=boom) as client:
        before = _sample("rakuten_predictions_total", labels)
        client.post("/predict", json={"designation": "chaise"})
        after = _sample("rakuten_predictions_total", labels)

    assert after == before


def test_requests_are_recorded_under_the_route_template(monkeypatch):
    labels = {"service": "text", "method": "GET", "path": "/health", "status": "200"}
    with _client(monkeypatch, loaded=False) as client:
        before = _sample("rakuten_http_requests_total", labels)
        client.get("/health")
        after = _sample("rakuten_http_requests_total", labels)
    assert after - before == 1


def test_an_unmatched_path_does_not_explode_label_cardinality(monkeypatch):
    """The guard that stops a 404 scan minting one time series per URL. Without
    it, /aaa, /bbb, /ccc each become a permanent series."""
    labels = {"service": "text", "method": "GET", "path": "unmatched", "status": "404"}
    with _client(monkeypatch, loaded=False) as client:
        before = _sample("rakuten_http_requests_total", labels)
        assert client.get("/no-such-route").status_code == 404
        assert client.get("/another-missing-one").status_code == 404
        after = _sample("rakuten_http_requests_total", labels)

    assert after - before == 2
    assert _sample("rakuten_http_requests_total",
                   {"service": "text", "method": "GET",
                    "path": "/no-such-route", "status": "404"}) == 0.0


# --------------------------------------------------------------------------- #
# The contract the gateway and fusion depend on
# --------------------------------------------------------------------------- #
def test_the_probability_vector_follows_the_canonical_class_order(monkeypatch):
    """The whole fusion story rests on both modalities emitting vectors in the
    same order. This is the serving-side end of that contract."""
    with _client(monkeypatch, loaded=True) as client:
        body = client.post("/predict", json={"designation": "chaise"}).json()

    classes = body["canonical_classes"]
    assert classes == sorted(classes)
    assert len(classes) == len(body["probabilities"]) == 27
    assert body["prediction"]["prdtypecode"] == classes[0]
