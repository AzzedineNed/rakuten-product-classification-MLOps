"""Characterization tests for api/image_main.py - the service AS IT IS TODAY.

WHY THIS FILE EXISTS. With the gateway covered, this was the last untested
module in the project. It is also the service with the most moving parts: a
lazy model resolver, two mutually exclusive background jobs, and a /health
answer that is famously NOT about the model being served.

NOTHING HERE LOADS A MODEL, IMPORTS TORCH, OR REACHES THE REGISTRY. Every test
replaces scripts/predict.py's entry points, and the two background jobs get
fake `train`, `process` and `evaluate` modules injected into sys.modules. Run
naively this file would behave differently on three machines: no artifacts
here and in CI, models/classifier.joblib present on the developer's laptop, and
MLFLOW_TRACKING_URI set there too. So config.CLASSIFIER_PATH is forced in both
directions rather than accepted as found.

WHAT THIS FILE USED TO PIN, AND NO LONGER DOES. Two known warts were asserted
here as characterization first, then fixed once the tests made the current
behaviour explicit: /health's model_loaded, which described the local joblib
rather than the served model, and the background runners' `except Exception`,
which let SystemExit wedge a job on "running". Both now match api/text_main.py.
The remaining asymmetry is deliberate: this service resolves its model LAZILY,
so model_loaded is false until the first /predict.

COUNTERS ARE ASSERTED AS DELTAS. ServiceMetrics lives for the whole process,
so absolute values pass alone and fail in a full run.
"""
from __future__ import annotations

import io
import sys
import types

import numpy as np
import pytest

from conftest import import_service_module

# fastapi is installed by requirements-ci.txt. importorskip keeps this file
# honest if somebody trims the CI subset again: it would skip loudly with a
# reason rather than vanish.
pytest.importorskip("fastapi", reason="fastapi not installed (CI subset)")

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from rakuten_img import config  # noqa: E402

# Imported through the helper because a second api module cannot share the
# global Prometheus registry. See tests/conftest.py for the measurement.
im = import_service_module("api.image_main")

# Importable only because api.image_main put scripts/ on sys.path at import.
import predict as predict_script  # noqa: E402

N = config.NUM_CLASSES


# --------------------------------------------------------------------------- #
# Fixtures and doubles
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _reset_job_status():
    """Both status dicts are module globals shared by every test in the file.

    Reset before AND after: a test that leaves "running" behind would turn the
    next test's 200 into a 409, and the failure would land in the wrong place.
    """
    idle = {"state": "idle", "detail": None, "metrics": None}
    im._TRAIN_STATUS.update(idle)
    im._EVAL_STATUS.update(idle)
    yield
    im._TRAIN_STATUS.update(idle)
    im._EVAL_STATUS.update(idle)


@pytest.fixture(autouse=True)
def _no_real_resolution_at_startup(monkeypatch):
    """Stop the startup lifespan from resolving a model for real.

    Every test here enters the app as a context manager, which is what runs the
    lifespan, and the lifespan now calls predict.load(). Left alone that would
    read whatever .joblib happens to sit in models/ and, if MLFLOW_TRACKING_URI
    is set in the shell, TALK TO DAGSHUB - so the suite's result would depend on
    the developer's environment and the network. The stub reports whatever
    model_source() currently says, which is what each test has already faked.
    """
    monkeypatch.setattr(predict_script, "load", predict_script.model_source)


def _vector(winner: int, peak: float = 0.6) -> np.ndarray:
    """A normalised, NON-degenerate probability vector peaking at `winner`."""
    rest = (1.0 - peak) / (N - 1)
    vec = np.full(N, rest, dtype=np.float64)
    vec[winner] = peak
    return vec


def _png(size=(4, 4), colour=(255, 0, 0)) -> bytes:
    """A real PNG. The endpoint calls Image.open().load(), so the bytes have to
    be a decodable image and not a stand-in."""
    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, format="PNG")
    return buffer.getvalue()


def _stub_predict(monkeypatch, *, proba=None, source="local:test.joblib"):
    """Replace scripts/predict.py's two entry points and record the call."""
    seen: dict = {}

    def fake_predict_proba(pil_image):
        seen["size"] = pil_image.size
        seen["calls"] = seen.get("calls", 0) + 1
        return _vector(1) if proba is None else proba

    monkeypatch.setattr(predict_script, "predict_proba", fake_predict_proba)
    monkeypatch.setattr(predict_script, "model_source", lambda: source)
    return seen


def _fake_script(monkeypatch, module_name: str, func_name: str, calls: list,
                 outcome=None):
    """Inject a fake scripts/<module>.py for a background job to import.

    The fake carries the work function the runner actually calls. A fake
    missing it fails with AttributeError before reaching the code under test,
    which costs a debug cycle and proves nothing.
    """
    module = types.ModuleType(module_name)

    def work(*args, **kwargs):
        calls.append(module_name)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    setattr(module, func_name, work)
    monkeypatch.setitem(sys.modules, module_name, module)
    return module


def _sample(name: str, labels: dict) -> float:
    value = im.METRICS.registry.get_sample_value(name, labels)
    return 0.0 if value is None else value


def _client() -> TestClient:
    return TestClient(im.app)


# --------------------------------------------------------------------------- #
# /health
# --------------------------------------------------------------------------- #
def test_health_reports_the_static_service_facts(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CLASSIFIER_PATH", tmp_path / "absent.joblib")
    monkeypatch.setattr(predict_script, "model_source", lambda: "not-loaded")
    with _client() as client:
        body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["backbone"] == config.BACKBONE_NAME
    assert body["num_classes"] == N


def test_health_says_loaded_once_a_model_is_resolved(monkeypatch, tmp_path):
    artifact = tmp_path / "classifier.joblib"
    artifact.write_bytes(b"not a real model, only its presence is read")
    monkeypatch.setattr(config, "CLASSIFIER_PATH", artifact)
    monkeypatch.setattr(predict_script, "model_source", lambda: "local:classifier.joblib")
    with _client() as client:
        body = client.get("/health").json()

    assert body["model_loaded"] is True
    assert body["model_source"] == "local:classifier.joblib"
    assert body["local_model_present"] is True


def test_health_says_not_loaded_before_anything_is_resolved(monkeypatch, tmp_path):
    """This service resolves lazily, so a fresh container reports false until
    the first /predict. That is the design, not a fault."""
    monkeypatch.setattr(config, "CLASSIFIER_PATH", tmp_path / "gone.joblib")
    monkeypatch.setattr(predict_script, "model_source",
                        lambda: predict_script.NOT_LOADED)
    with _client() as client:
        body = client.get("/health").json()

    assert body["model_loaded"] is False
    assert body["model_source"] == "not-loaded"
    assert body["local_model_present"] is False


def test_model_loaded_describes_the_served_model_not_the_disk(monkeypatch, tmp_path):
    """A container serving from the registry with no local artifact IS loaded.

    Under the old behaviour this reported false, because model_loaded was
    CLASSIFIER_PATH.exists(). It now means what it means on the text service.
    """
    monkeypatch.setattr(config, "CLASSIFIER_PATH", tmp_path / "no-local-copy.joblib")
    monkeypatch.setattr(predict_script, "model_source",
                        lambda: "registry:rakuten-image-classifier/2")
    with _client() as client:
        body = client.get("/health").json()

    assert body["model_source"].startswith("registry:")
    assert body["model_loaded"] is True
    assert body["local_model_present"] is False


def test_a_local_artifact_alone_does_not_mean_a_model_is_serving(monkeypatch, tmp_path):
    """The other half of the same correction: a joblib on disk that nothing has
    loaded, and may never load if it is corrupt, is not a served model."""
    artifact = tmp_path / "classifier.joblib"
    artifact.write_bytes(b"could be anything, nobody has opened it")
    monkeypatch.setattr(config, "CLASSIFIER_PATH", artifact)
    monkeypatch.setattr(predict_script, "model_source",
                        lambda: predict_script.NOT_LOADED)
    with _client() as client:
        body = client.get("/health").json()

    assert body["model_loaded"] is False
    assert body["local_model_present"] is True


def test_health_publishes_which_model_is_serving(monkeypatch, tmp_path):
    """Published here as well as after a prediction, because this service
    resolves its model lazily: without it the gauge would stay empty until the
    first /predict, and a dashboard would show a gap for a healthy service."""
    monkeypatch.setattr(config, "CLASSIFIER_PATH", tmp_path / "absent.joblib")
    monkeypatch.setattr(predict_script, "model_source", lambda: "local:probe.joblib")
    with _client() as client:
        client.get("/health")

    assert _sample("rakuten_model_info", {
        "service": "image", "modality": "image", "source": "local:probe.joblib",
    }) == 1.0


def test_the_model_info_gauge_keeps_one_series_after_the_source_changes(monkeypatch,
                                                                       tmp_path):
    """set_model_info clears first. Two series both reading 1 would leave a
    dashboard unable to say which model is current."""
    monkeypatch.setattr(config, "CLASSIFIER_PATH", tmp_path / "absent.joblib")
    with _client() as client:
        monkeypatch.setattr(predict_script, "model_source", lambda: "local:old.joblib")
        client.get("/health")
        monkeypatch.setattr(predict_script, "model_source", lambda: "local:new.joblib")
        client.get("/health")

    assert _sample("rakuten_model_info", {
        "service": "image", "modality": "image", "source": "local:old.joblib"}) == 0.0
    assert _sample("rakuten_model_info", {
        "service": "image", "modality": "image", "source": "local:new.joblib"}) == 1.0


# --------------------------------------------------------------------------- #
# /predict
# --------------------------------------------------------------------------- #
def test_predict_returns_the_full_canonical_vector(monkeypatch):
    _stub_predict(monkeypatch, proba=_vector(3))
    with _client() as client:
        body = client.post("/predict", files={"file": ("a.png", _png())}).json()

    assert body["canonical_classes"] == list(config.CANONICAL_CLASSES)
    assert len(body["probabilities"]) == N
    assert body["probabilities"] == pytest.approx(list(_vector(3)))


def test_predict_reads_the_uploaded_bytes(monkeypatch):
    """The handler decodes the upload before predicting, so a truncated or
    swapped upload cannot silently become a prediction about something else."""
    seen = _stub_predict(monkeypatch)
    with _client() as client:
        client.post("/predict", files={"file": ("a.png", _png(size=(7, 11)))})

    assert seen["size"] == (7, 11)
    assert seen["calls"] == 1


@pytest.mark.parametrize("top_k", [1, 5, 27])
def test_top_k_is_honoured_and_ordered_by_descending_probability(monkeypatch, top_k):
    _stub_predict(monkeypatch, proba=_vector(2))
    with _client() as client:
        body = client.post("/predict", files={"file": ("a.png", _png())},
                           params={"top_k": top_k}).json()

    probabilities = [entry["probability"] for entry in body["top_k"]]
    assert len(body["top_k"]) == top_k
    assert probabilities == sorted(probabilities, reverse=True)
    assert body["prediction"] == body["top_k"][0]


@pytest.mark.parametrize("top_k", [0, -1, 28, 100])
def test_top_k_outside_the_class_count_is_rejected(monkeypatch, top_k):
    seen = _stub_predict(monkeypatch)
    with _client() as client:
        resp = client.post("/predict", files={"file": ("a.png", _png())},
                           params={"top_k": top_k})

    assert resp.status_code == 422
    # Pydantic validates before the handler, so no model work is done for a
    # request that was never going to be answered.
    assert seen == {}


def test_each_entry_maps_code_to_label_canonically(monkeypatch):
    _stub_predict(monkeypatch, proba=_vector(5))
    with _client() as client:
        body = client.post("/predict", files={"file": ("a.png", _png())}).json()

    assert body["prediction"]["code"] == config.CANONICAL_CLASSES[5]
    for entry in body["top_k"]:
        index = list(config.CANONICAL_CLASSES).index(entry["code"])
        assert entry["label"] == config.CANONICAL_LABELS[index]
        assert body["probabilities"][index] == pytest.approx(entry["probability"])


def test_an_undecodable_upload_is_a_400_not_a_500(monkeypatch):
    seen = _stub_predict(monkeypatch)
    with _client() as client:
        resp = client.post("/predict", files={"file": ("a.png", b"this is not an image")})

    assert resp.status_code == 400
    assert "Invalid image" in resp.json()["error"]
    # The model is never asked about a file that could not be opened.
    assert seen == {}


def test_a_rejected_upload_is_not_counted_as_a_prediction(monkeypatch):
    _stub_predict(monkeypatch)
    labels = {"service": "image"}
    before = _sample("rakuten_prediction_confidence_count", labels)

    with _client() as client:
        client.post("/predict", files={"file": ("a.png", b"garbage")})

    assert _sample("rakuten_prediction_confidence_count", labels) == before


def test_a_prediction_is_counted_with_its_winning_code(monkeypatch):
    _stub_predict(monkeypatch, proba=_vector(4))
    counter = {"service": "image", "prdtypecode": str(config.CANONICAL_CLASSES[4])}
    confidence = {"service": "image"}
    before = _sample("rakuten_predictions_total", counter)
    before_confidence = _sample("rakuten_prediction_confidence_count", confidence)

    with _client() as client:
        client.post("/predict", files={"file": ("a.png", _png())})

    assert _sample("rakuten_predictions_total", counter) - before == 1.0
    assert _sample("rakuten_prediction_confidence_count", confidence) - before_confidence == 1.0


def test_predict_republishes_the_model_source(monkeypatch):
    """The first /predict is the earliest point at which model_source() is
    truthful for this service, because the model is resolved lazily."""
    _stub_predict(monkeypatch, source="registry:rakuten-image-classifier/7")
    with _client() as client:
        client.post("/predict", files={"file": ("a.png", _png())})

    assert _sample("rakuten_model_info", {
        "service": "image", "modality": "image",
        "source": "registry:rakuten-image-classifier/7"}) == 1.0


def test_predict_does_not_need_a_local_model_file(monkeypatch, tmp_path):
    """Serving from the registry with no local .joblib is a supported state,
    whatever /health says about it."""
    monkeypatch.setattr(config, "CLASSIFIER_PATH", tmp_path / "absent.joblib")
    _stub_predict(monkeypatch, source="registry:rakuten-image-classifier/2")
    with _client() as client:
        resp = client.post("/predict", files={"file": ("a.png", _png())})

    assert resp.status_code == 200


# --------------------------------------------------------------------------- #
# /train
# --------------------------------------------------------------------------- #
def test_train_returns_immediately_and_says_where_to_poll(monkeypatch):
    calls: list[str] = []
    _fake_script(monkeypatch, "train", "train", calls, outcome={"val_f1_weighted": 0.5468})
    with _client() as client:
        body = client.post("/train").json()

    assert body["status"] == "started"
    assert body["poll"] == "/train/status"
    assert body["reprocess"] is False


def test_a_finished_training_run_publishes_its_metrics(monkeypatch):
    calls: list[str] = []
    _fake_script(monkeypatch, "train", "train", calls,
                 outcome={"val_f1_weighted": 0.5468})
    with _client() as client:
        client.post("/train")
        status = client.get("/train/status").json()

    assert calls == ["train"]
    assert status["state"] == "done"
    assert status["detail"] == "completed"
    assert status["metrics"] == {"val_f1_weighted": 0.5468}


def test_reprocess_re_extracts_features_before_training(monkeypatch):
    """Order matters: training on features the same request is about to
    overwrite would score the previous run's cache."""
    calls: list[str] = []
    _fake_script(monkeypatch, "process", "process", calls)
    _fake_script(monkeypatch, "train", "train", calls, outcome={})
    with _client() as client:
        body = client.post("/train", params={"reprocess": True}).json()
        status = client.get("/train/status").json()

    assert body["reprocess"] is True
    assert calls == ["process", "train"]
    assert status["state"] == "done"


def test_the_default_run_does_not_re_extract_features(monkeypatch):
    """The default has to stay cheap: reprocessing is the multi-hour step."""
    calls: list[str] = []
    _fake_script(monkeypatch, "process", "process", calls)
    _fake_script(monkeypatch, "train", "train", calls, outcome={})
    with _client() as client:
        client.post("/train")

    assert calls == ["train"]


def test_a_failed_training_run_is_reported_not_swallowed(monkeypatch):
    """The DAG can only see what /train/status says. A job that dies quietly
    reads as a timeout hours later."""
    calls: list[str] = []
    _fake_script(monkeypatch, "train", "train", calls,
                 outcome=RuntimeError("cached features missing"))
    with _client() as client:
        client.post("/train")
        status = client.get("/train/status").json()

    assert status["state"] == "failed"
    assert "cached features missing" in status["detail"]
    assert status["metrics"] is None


def test_train_status_starts_idle():
    with _client() as client:
        assert client.get("/train/status").json() == {
            "state": "idle", "detail": None, "metrics": None}


def test_a_running_job_clears_the_previous_metrics(monkeypatch):
    """Otherwise a poller that catches the window sees the LAST run's numbers
    labelled as the current one."""
    seen: dict = {}
    calls: list[str] = []

    def capture():
        seen.update(im._TRAIN_STATUS)
        return {"val_f1_weighted": 0.9}

    im._TRAIN_STATUS.update(state="done", detail="completed",
                            metrics={"val_f1_weighted": 0.1})
    module = _fake_script(monkeypatch, "train", "train", calls)
    module.train = capture
    with _client() as client:
        client.post("/train")

    assert seen["state"] == "running"
    assert seen["metrics"] is None
    assert seen["detail"] == "classifier-only"


# --------------------------------------------------------------------------- #
# /evaluate
# --------------------------------------------------------------------------- #
def test_evaluate_returns_immediately_and_says_where_to_poll(monkeypatch):
    calls: list[str] = []
    _fake_script(monkeypatch, "evaluate", "evaluate", calls, outcome={})
    with _client() as client:
        body = client.post("/evaluate").json()

    assert body["status"] == "started"
    assert body["poll"] == "/evaluate/status"


def test_a_finished_evaluation_publishes_its_metrics(monkeypatch):
    calls: list[str] = []
    metrics = {"accuracy": 0.5446, "f1_weighted": 0.5470, "f1_macro": 0.5012}
    _fake_script(monkeypatch, "evaluate", "evaluate", calls, outcome=metrics)
    with _client() as client:
        client.post("/evaluate")
        status = client.get("/evaluate/status").json()

    assert calls == ["evaluate"]
    assert status["state"] == "done"
    assert status["metrics"] == metrics


def test_a_failed_evaluation_is_reported_not_swallowed(monkeypatch):
    calls: list[str] = []
    _fake_script(monkeypatch, "evaluate", "evaluate", calls,
                 outcome=FileNotFoundError("no test features"))
    with _client() as client:
        client.post("/evaluate")
        status = client.get("/evaluate/status").json()

    assert status["state"] == "failed"
    assert "no test features" in status["detail"]


def test_evaluate_status_starts_idle():
    with _client() as client:
        assert client.get("/evaluate/status").json() == {
            "state": "idle", "detail": None, "metrics": None}


# --------------------------------------------------------------------------- #
# Mutual exclusion - /train rewrites the file /evaluate reads
# --------------------------------------------------------------------------- #
def test_training_while_training_is_a_409():
    im._TRAIN_STATUS.update(state="running")
    with _client() as client:
        resp = client.post("/train")

    assert resp.status_code == 409
    assert "train" in resp.json()["error"]


def test_evaluating_while_training_is_a_409():
    """Scoring a half-written .joblib would produce metrics for a model that
    never existed."""
    im._TRAIN_STATUS.update(state="running")
    with _client() as client:
        resp = client.post("/evaluate")

    assert resp.status_code == 409
    assert "train" in resp.json()["error"]


def test_training_while_evaluating_is_a_409():
    im._EVAL_STATUS.update(state="running")
    with _client() as client:
        resp = client.post("/train")

    assert resp.status_code == 409
    assert "evaluate" in resp.json()["error"]


def test_evaluating_while_evaluating_is_a_409():
    im._EVAL_STATUS.update(state="running")
    with _client() as client:
        resp = client.post("/evaluate")

    assert resp.status_code == 409
    assert "evaluate" in resp.json()["error"]


def test_a_rejected_job_does_not_disturb_the_running_one(monkeypatch):
    im._TRAIN_STATUS.update(state="running", detail="classifier-only", metrics=None)
    with _client() as client:
        client.post("/train")
        client.post("/evaluate")
        status = client.get("/train/status").json()

    assert status["state"] == "running"
    assert status["detail"] == "classifier-only"


def test_a_finished_job_does_not_block_the_next_one(monkeypatch):
    """Only "running" blocks. "done" and "failed" must not, or the service
    would accept exactly one job per container lifetime."""
    calls: list[str] = []
    _fake_script(monkeypatch, "evaluate", "evaluate", calls, outcome={})
    im._TRAIN_STATUS.update(state="done", detail="completed", metrics={})
    im._EVAL_STATUS.update(state="failed", detail="boom", metrics=None)
    with _client() as client:
        assert client.post("/evaluate").status_code == 200


# --------------------------------------------------------------------------- #
# SystemExit does not derive from Exception, so it is caught by name
# --------------------------------------------------------------------------- #
def test_systemexit_in_an_evaluation_job_is_reported_not_fatal(monkeypatch):
    """A background job that calls sys.exit must land in the status dict.

    SystemExit inherits from BaseException, not Exception, so a bare
    `except Exception` lets it through: the task dies, the status stays
    "running" forever, and the DAG's poller can only report a timeout hours
    later. Unreachable today because the runners call the library functions
    rather than main(), which keeps argparse out of the path, but it is one
    sys.exit away at any time. api/text_main.py has caught it explicitly since
    session 11 and this service now matches.

    Called directly rather than through the client, because an exception raised
    in a background task would otherwise propagate out of TestClient itself.
    """
    calls: list[str] = []
    _fake_script(monkeypatch, "evaluate", "evaluate", calls, outcome=SystemExit(2))

    im._run_evaluation()

    assert im._EVAL_STATUS["state"] == "failed"
    assert "SystemExit" in im._EVAL_STATUS["detail"]


def test_systemexit_in_a_training_job_is_reported_not_fatal(monkeypatch):
    calls: list[str] = []
    _fake_script(monkeypatch, "train", "train", calls, outcome=SystemExit(2))

    im._run_training(False)

    assert im._TRAIN_STATUS["state"] == "failed"
    assert "SystemExit" in im._TRAIN_STATUS["detail"]


def test_a_plain_exception_is_caught_too(monkeypatch):
    calls: list[str] = []
    _fake_script(monkeypatch, "train", "train", calls, outcome=ValueError("bad shape"))

    im._run_training(False)

    assert im._TRAIN_STATUS["state"] == "failed"
    assert "bad shape" in im._TRAIN_STATUS["detail"]


def test_the_failure_detail_names_the_exception_type(monkeypatch):
    """Both services format the detail the same way now. "no test features" on
    its own does not say whether the job crashed or exited."""
    calls: list[str] = []
    _fake_script(monkeypatch, "evaluate", "evaluate", calls,
                 outcome=FileNotFoundError("no test features"))

    im._run_evaluation()

    assert im._EVAL_STATUS["detail"].startswith("FileNotFoundError: ")


# --------------------------------------------------------------------------- #
# Metrics plumbing
# --------------------------------------------------------------------------- #
def test_metrics_endpoint_renders_the_prometheus_exposition():
    with _client() as client:
        resp = client.get("/metrics")

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    assert "rakuten_http_requests_total" in resp.text


def test_requests_are_counted_under_the_route_template(monkeypatch, tmp_path):
    """Never the raw path: labelling by raw URL would let a caller mint
    unbounded label values and kill the Prometheus server."""
    monkeypatch.setattr(config, "CLASSIFIER_PATH", tmp_path / "absent.joblib")
    monkeypatch.setattr(predict_script, "model_source", lambda: "not-loaded")
    labels = {"service": "image", "method": "GET", "path": "/health", "status": "200"}
    before = _sample("rakuten_http_requests_total", labels)

    with _client() as client:
        client.get("/health")

    assert _sample("rakuten_http_requests_total", labels) - before == 1.0


def test_an_unmatched_route_does_not_create_a_new_label_value():
    labels = {"service": "image", "method": "GET", "path": "unmatched",
              "status": "404"}
    before = _sample("rakuten_http_requests_total", labels)

    with _client() as client:
        assert client.get("/no-such-route-98765").status_code == 404

    assert _sample("rakuten_http_requests_total", labels) - before == 1.0


# --------------------------------------------------------------------------- #
# Startup resolution (the lifespan)
#
# This service used to resolve its model on the FIRST /predict. A healthy
# container therefore answered model_loaded=false and model_source=not-loaded
# until traffic arrived, which from the outside is indistinguishable from a
# container that cannot load its model at all, and it meant the two modalities
# answered the same question in two different ways. These tests pin the eager
# behaviour and, just as importantly, that a failure to resolve still lets the
# service boot.
# --------------------------------------------------------------------------- #
def _startup(monkeypatch, *, source=None, exc=None):
    """Force the outcome of the startup resolution. Returns the call counter.

    Overrides the autouse stub, so the lifespan runs against this instead.
    """
    calls = []

    def fake_load():
        calls.append(True)
        if exc is not None:
            raise exc
        monkeypatch.setattr(predict_script, "model_source", lambda: source)
        return source

    monkeypatch.setattr(predict_script, "model_source",
                        lambda: predict_script.NOT_LOADED)
    monkeypatch.setattr(predict_script, "load", fake_load)
    return calls


def test_health_reports_a_model_from_boot_with_no_prediction(monkeypatch):
    """THE WART THIS CLOSES. No /predict is sent anywhere in this test."""
    _startup(monkeypatch, source="registry:m@production/v2")

    with _client() as client:
        body = client.get("/health").json()

    assert body["model_loaded"] is True
    assert body["model_source"] == "registry:m@production/v2"


def test_startup_survives_a_resolution_failure(monkeypatch):
    """A missing or corrupt artifact must NOT stop the service booting. The
    lifespan catches Exception so a broken model lands in /health instead of a
    container that will not start."""
    _startup(monkeypatch, exc=FileNotFoundError("no artifacts in this test"))

    with _client() as client:
        body = client.get("/health").json()

    assert body["model_loaded"] is False
    assert body["model_source"] == predict_script.NOT_LOADED


def test_model_info_gauge_is_published_even_when_resolution_failed(monkeypatch):
    """An absent series is indistinguishable from a dead scrape, so the gauge
    is set in BOTH branches of the lifespan."""
    _startup(monkeypatch, exc=RuntimeError("registry and disk both gone"))

    with _client():
        pass

    assert _sample("rakuten_model_info", {
        "service": "image", "modality": "image",
        "source": predict_script.NOT_LOADED}) == 1.0


def test_the_gauge_carries_the_real_source_before_any_traffic(monkeypatch):
    _startup(monkeypatch, source="registry:m@production/v2")

    with _client():
        pass

    assert _sample("rakuten_model_info", {
        "service": "image", "modality": "image",
        "source": "registry:m@production/v2"}) == 1.0


def test_the_registry_is_consulted_once_at_startup_and_not_per_request(monkeypatch):
    """The promise that lets a registry outage leave live traffic alone: the
    resolution happens exactly once, at boot, never on the request path."""
    calls = _startup(monkeypatch, source="registry:m@production/v2")
    _stub_predict(monkeypatch, source="registry:m@production/v2")

    with _client() as client:
        client.get("/health")
        client.post("/predict", files={"file": ("p.png", _png(), "image/png")})
        client.get("/health")

    assert len(calls) == 1
