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


# --------------------------------------------------------------------------- #
# Training and evaluation endpoints
#
# The real jobs are never run here: they need the dataset, minutes of CPU and
# the registry. What IS tested is everything around them — the 409 guard, the
# status lifecycle, what reaches the status dict on failure, and the argparse
# trap that would wedge a job on "running" forever.
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _reset_job_state():
    """_TRAIN_STATUS and _EVAL_STATUS are module globals that outlive a request.
    Without this the first test to leave one "running" would 409 every later
    test in the file."""
    idle = {"state": "idle", "detail": None, "metrics": None}
    tm._TRAIN_STATUS.clear(), tm._TRAIN_STATUS.update(idle)
    tm._EVAL_STATUS.clear(), tm._EVAL_STATUS.update(idle)
    yield
    tm._TRAIN_STATUS.clear(), tm._TRAIN_STATUS.update(idle)
    tm._EVAL_STATUS.clear(), tm._EVAL_STATUS.update(idle)


def test_status_endpoints_start_idle(monkeypatch):
    with _client(monkeypatch, loaded=True) as client:
        assert client.get("/train/status").json() == {
            "state": "idle", "detail": None, "metrics": None}
        assert client.get("/evaluate/status").json() == {
            "state": "idle", "detail": None, "metrics": None}


def test_train_accepts_and_returns_where_to_poll(monkeypatch):
    calls = []
    monkeypatch.setattr(tm, "_run_training", lambda: calls.append("train"))
    with _client(monkeypatch, loaded=True) as client:
        response = client.post("/train")
    assert response.status_code == 200
    assert response.json() == {"status": "started", "poll": "/train/status"}
    assert calls == ["train"]


def test_evaluate_accepts_and_returns_where_to_poll(monkeypatch):
    calls = []
    monkeypatch.setattr(tm, "_run_evaluation", lambda: calls.append("evaluate"))
    with _client(monkeypatch, loaded=True) as client:
        response = client.post("/evaluate")
    assert response.status_code == 200
    assert response.json() == {"status": "started", "poll": "/evaluate/status"}
    assert calls == ["evaluate"]


@pytest.mark.parametrize("busy_key, path, blocked", [
    ("_TRAIN_STATUS", "/train", "train"),
    ("_TRAIN_STATUS", "/evaluate", "train"),
    ("_EVAL_STATUS", "/train", "evaluate"),
    ("_EVAL_STATUS", "/evaluate", "evaluate"),
])
def test_the_two_jobs_are_mutually_exclusive(monkeypatch, busy_key, path, blocked):
    """Both jobs are CPU-bound on two cores and both write into models/text and
    reports/text, so a second one must be refused rather than queued."""
    monkeypatch.setattr(tm, "_run_training", lambda: None)
    monkeypatch.setattr(tm, "_run_evaluation", lambda: None)
    getattr(tm, busy_key)["state"] = "running"

    with _client(monkeypatch, loaded=True) as client:
        response = client.post(path)

    assert response.status_code == 409
    assert response.json() == {"error": f"A '{blocked}' job is already running."}


def test_training_records_metrics_on_success(monkeypatch):
    fake = {"f1_weighted": 0.78, "accuracy": 0.77, "n_train": 100}

    class FakeScript:
        @staticmethod
        def build_parser():
            import argparse
            parser = argparse.ArgumentParser()
            parser.add_argument("--max-features", type=int, default=10000)
            return parser

        @staticmethod
        def entrainer(args):
            assert args.max_features == 10000   # defaults, not uvicorn's argv
            return fake

    monkeypatch.setitem(sys.modules, "train_text", FakeScript)
    tm._run_training()

    assert tm._TRAIN_STATUS["state"] == "done"
    assert tm._TRAIN_STATUS["metrics"] == fake
    assert tm._TRAIN_STATUS["detail"] == "completed"


def test_a_training_failure_is_reported_not_swallowed(monkeypatch):
    class FakeScript:
        @staticmethod
        def build_parser():
            import argparse
            return argparse.ArgumentParser()

        @staticmethod
        def entrainer(args):
            raise ValueError("dataset missing")

    monkeypatch.setitem(sys.modules, "train_text", FakeScript)
    tm._run_training()

    assert tm._TRAIN_STATUS["state"] == "failed"
    assert "ValueError: dataset missing" in tm._TRAIN_STATUS["detail"]


@pytest.mark.parametrize("runner, status_name, module", [
    ("_run_training", "_TRAIN_STATUS", "train_text"),
    ("_run_evaluation", "_EVAL_STATUS", "evaluate_text"),
])
def test_a_systemexit_does_not_wedge_the_job_on_running(monkeypatch, runner,
                                                        status_name, module):
    """THE TRAP THIS GUARDS. SystemExit does not derive from Exception, so an
    `except Exception` would let argparse — or any sys.exit — kill the
    background task silently, leaving the status on "running" forever. The DAG
    could then only ever report a timeout, hours later."""
    class FakeScript:
        """Raises from the WORK function, not build_parser: in
        `script.entrainer(script.build_parser().parse_args([]))` Python looks up
        the entrainer attribute before it calls build_parser, so a fake missing
        it fails with AttributeError and never reaches the path under test.
        Learned by running this test rather than reasoning about it."""
        @staticmethod
        def build_parser():
            import argparse
            return argparse.ArgumentParser()

        @staticmethod
        def entrainer(args):
            raise SystemExit(2)

        @staticmethod
        def evaluer(args):
            raise SystemExit(2)

    monkeypatch.setitem(sys.modules, module, FakeScript)
    getattr(tm, runner)()

    status = getattr(tm, status_name)
    assert status["state"] == "failed"
    assert "SystemExit" in status["detail"]


def test_evaluation_records_metrics_on_success(monkeypatch):
    fake = {"f1_weighted": 0.7780, "accuracy": 0.7768}

    class FakeScript:
        @staticmethod
        def build_parser():
            import argparse
            parser = argparse.ArgumentParser()
            parser.add_argument("--split", default="test")
            parser.add_argument("--predict-test", action="store_true")
            return parser

        @staticmethod
        def evaluer(args):
            # The endpoint must score the TEST split and must NOT generate the
            # submission predictions — both are argparse defaults.
            assert args.split == "test"
            assert args.predict_test is False
            return fake

    monkeypatch.setitem(sys.modules, "evaluate_text", FakeScript)
    tm._run_evaluation()

    assert tm._EVAL_STATUS["state"] == "done"
    assert tm._EVAL_STATUS["metrics"] == fake


def test_a_running_job_is_visible_through_the_status_endpoint(monkeypatch):
    """The DAG polls this; it must show "running" before it shows "done"."""
    seen = []

    class FakeScript:
        @staticmethod
        def build_parser():
            import argparse
            return argparse.ArgumentParser()

        @staticmethod
        def entrainer(args):
            seen.append(dict(tm._TRAIN_STATUS))
            return {"f1_weighted": 1.0}

    monkeypatch.setitem(sys.modules, "train_text", FakeScript)
    tm._run_training()

    assert seen[0]["state"] == "running"
    assert seen[0]["detail"] == "tfidf + logreg"
    assert tm._TRAIN_STATUS["state"] == "done"
