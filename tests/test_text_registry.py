"""Registry-aware TEXT serving: which text model is actually loaded, and what
happens when the registry is not there.

The text side cannot reuse the image loader: a text version's source is a
DIRECTORY holding tfidf_vectorizer.pkl + logistic_regression.pkl, not one
joblib payload. That difference is the whole reason this file exists, so the
real-MLflow layer below registers REAL two-file directories rather than
asserting against a fake that could not tell the two shapes apart.

Two layers, mirroring tests/test_registry.py:

  * Fast tests with no mlflow at all, so CI (requirements-ci.txt omits mlflow)
    still covers the fallback policy - which is the part that decides whether
    the service serves or 503s.
  * The same expectations against a REAL MLflow registry (SQLite + a local
    artifact store inside tmp_path), skipped when mlflow is absent.

Every expectation here was OBSERVED against real MLflow 3.14.0 before being
written down.
"""
from __future__ import annotations

import inspect
import os

import joblib
import pytest

from rakuten_text import config, tracking
from rakuten_text.predict import TfidfPredictor


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _write_local_artifacts(directory, vectorizer=None, model=None):
    """Write the two .pkl files the local fallback expects. Plain dicts stand in
    for the real estimators: nothing under test calls fit or transform, and a
    dict makes an identity check trivial."""
    directory.mkdir(parents=True, exist_ok=True)
    vec_path = directory / config.VECTORIZER_PATH.name
    mdl_path = directory / config.MODEL_PATH.name
    joblib.dump(vectorizer if vectorizer is not None else {"src": "local-vec"}, vec_path)
    joblib.dump(model if model is not None else {"src": "local-model"}, mdl_path)
    return vec_path, mdl_path


def _predictor(directory):
    vec_path, mdl_path = _write_local_artifacts(directory)
    return TfidfPredictor(vectorizer_path=vec_path, model_path=mdl_path)


# --------------------------------------------------------------------------
# the fallback policy (no mlflow needed)
# --------------------------------------------------------------------------
def test_serving_source_is_not_loaded_before_any_load(tmp_path):
    """Mirrors the image side's model_source(), which reports 'not-loaded'
    until something is actually loaded (wart 3)."""
    assert _predictor(tmp_path).serving_source == "not-loaded"


def test_local_load_reports_a_local_source(tmp_path):
    predictor = _predictor(tmp_path).load()
    assert predictor.is_loaded
    assert predictor.serving_source == f"local:{config.MODEL_PATH.name}"


def test_default_load_never_touches_the_registry(tmp_path, monkeypatch):
    """The DEFAULT must stay local-only. scripts/evaluate_text.py and
    scripts/tune_fusion_weight call load() with no arguments; if that ever
    started resolving the registry, an evaluation would silently score a
    different model than the one training just wrote.

    NOTE: this records the call rather than raising from the stub. Raising does
    NOT work here - load()'s fallback catches Exception, AssertionError is an
    Exception, so a raising stub is swallowed and the test passes even when the
    registry WAS consulted. Found by mutation-testing this very test.
    """
    calls = []

    def _record(alias=None):
        calls.append(alias)
        raise RuntimeError("registry must not be consulted by default")

    monkeypatch.setattr(tracking, "load_from_registry", _record)
    predictor = _predictor(tmp_path).load()
    assert calls == []
    assert predictor.serving_source.startswith("local:")


def test_prefer_registry_defaults_to_false():
    """Pinned as a contract, not a detail: opt-in fails closed, opt-out fails
    open (a future caller that forgets the flag would reach the network)."""
    assert inspect.signature(TfidfPredictor.load).parameters["prefer_registry"].default is False


def test_registry_success_wins_and_disk_is_not_read(tmp_path, monkeypatch):
    """When the registry answers, its objects are served - proven by pointing
    the local paths at files that do not exist."""
    marker = {"src": "registry-model"}
    monkeypatch.setattr(
        tracking, "load_from_registry",
        lambda alias=None: ({"src": "registry-vec"}, marker,
                            "registry:rakuten-text-classifier@production/v1"),
    )
    predictor = TfidfPredictor(vectorizer_path=tmp_path / "nope-vec.pkl",
                               model_path=tmp_path / "nope-model.pkl")
    predictor.load(prefer_registry=True)
    assert predictor.model is marker
    assert predictor.serving_source == "registry:rakuten-text-classifier@production/v1"


def test_registry_failure_falls_back_to_local_and_says_so(tmp_path, monkeypatch, capsys):
    """A dead registry must not take the service down with it."""
    def _explode(alias=None):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(tracking, "load_from_registry", _explode)
    predictor = _predictor(tmp_path).load(prefer_registry=True)

    assert predictor.is_loaded
    assert predictor.serving_source == f"local:{config.MODEL_PATH.name}"
    assert "falling back to the local model" in capsys.readouterr().out


def test_no_tracking_uri_falls_back_to_local(tmp_path, monkeypatch):
    """The real code path, not a stub: tracking.enabled() is False without
    MLFLOW_TRACKING_URI, so load_from_registry raises before importing mlflow.
    This is exactly the situation inside the text container TODAY (no env_file
    on text-api), and it must degrade quietly."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    predictor = _predictor(tmp_path).load(prefer_registry=True)
    assert predictor.serving_source.startswith("local:")


def test_a_registry_load_with_no_local_files_still_serves(tmp_path, monkeypatch):
    """The container mounts models/ read-only and may not have the .pkl files at
    all once serving is registry-driven. Registry-only must work."""
    monkeypatch.setattr(
        tracking, "load_from_registry",
        lambda alias=None: ({"src": "v"}, {"src": "m"}, "registry:x/v3 (unpromoted)"),
    )
    predictor = TfidfPredictor(vectorizer_path=tmp_path / "absent-vec.pkl",
                               model_path=tmp_path / "absent-model.pkl")
    predictor.load(prefer_registry=True)
    assert predictor.is_loaded


def test_both_local_and_registry_missing_raises_the_usual_error(tmp_path, monkeypatch):
    """No model anywhere is a real failure. It must surface as the same
    FileNotFoundError the API already turns into a 503, not as something new."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    predictor = TfidfPredictor(vectorizer_path=tmp_path / "absent-vec.pkl",
                               model_path=tmp_path / "absent-model.pkl")
    with pytest.raises(FileNotFoundError):
        predictor.load(prefer_registry=True)


def test_serving_http_limits_are_small_enough_to_boot(tmp_path, monkeypatch):
    """A dead registry must not hold a container's startup hostage.

    MEASURED against an unreachable tracking URI: MLflow's shipped defaults (7
    retries, 120s timeout) blocked for OVER 170 SECONDS before being killed,
    and resolving an alias costs two requests, so the real figure is worse. With
    these limits the same call gave up in 10.4s. compose's healthcheck allows
    30s start_period + 5x30s, and the gateway waits on text-api being healthy,
    so the unbounded behaviour would take the whole stack down during a DagsHub
    outage instead of degrading to the local model.
    """
    assert int(tracking._SERVING_HTTP_LIMITS["MLFLOW_HTTP_REQUEST_MAX_RETRIES"]) <= 3
    assert int(tracking._SERVING_HTTP_LIMITS["MLFLOW_HTTP_REQUEST_TIMEOUT"]) <= 30


@pytest.mark.slow
def test_serving_http_limits_are_applied(tmp_path, monkeypatch):
    """Verified as applied at CALL time, not import time: MLflow reads these on
    every request, so the bound holds even in a process that imported mlflow
    earlier (measured: 9.0s with mlflow pre-imported)."""
    pytest.importorskip("mlflow", reason="mlflow not installed (CI subset)")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    monkeypatch.delenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", raising=False)
    monkeypatch.setattr(config, "REGISTERED_MODEL_NAME", "never-registered")

    with pytest.raises(Exception):
        tracking.load_from_registry()
    assert os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] == \
        tracking._SERVING_HTTP_LIMITS["MLFLOW_HTTP_REQUEST_MAX_RETRIES"]


@pytest.mark.slow
def test_operator_can_raise_the_serving_http_limits(tmp_path, monkeypatch):
    """setdefault, not assignment: an operator who wants MLflow's patient
    retry behaviour back must be able to have it."""
    pytest.importorskip("mlflow", reason="mlflow not installed (CI subset)")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "9")
    monkeypatch.setattr(config, "REGISTERED_MODEL_NAME", "never-registered")

    with pytest.raises(Exception):
        tracking.load_from_registry()
    assert os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] == "9"


# --------------------------------------------------------------------------
# the same expectations against a real registry
# --------------------------------------------------------------------------
def _real_registry(tmp_path, monkeypatch, versions, alias_on=None, drop_file=None):
    """Stand up a real SQLite registry holding `versions` model versions, each a
    DIRECTORY of two .pkl files, and point rakuten_text at it.

    artifact_location is forced inside tmp_path: the default would create
    ./mlruns in the repo root (wart 11).
    """
    mlflow = pytest.importorskip("mlflow", reason="mlflow not installed (CI subset)")
    from mlflow import MlflowClient

    uri = f"sqlite:///{tmp_path}/mlflow.db"
    mlflow.set_tracking_uri(uri)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)

    name = "rakuten-text-classifier"
    monkeypatch.setattr(config, "REGISTERED_MODEL_NAME", name)

    experiment_id = mlflow.create_experiment(
        "text-registry-test", artifact_location=str(tmp_path / "artifacts"))

    client = MlflowClient()
    client.create_registered_model(name)

    created = []
    for tag in versions:
        staging = tmp_path / f"stage-{tag}"
        staging.mkdir()
        _write_local_artifacts(staging, vectorizer={"src": f"vec-{tag}"},
                               model={"src": f"model-{tag}"})
        with mlflow.start_run(experiment_id=experiment_id) as run:
            for pkl in sorted(staging.iterdir()):
                if drop_file and pkl.name == drop_file:
                    continue
                client.log_artifact(run.info.run_id, str(pkl), artifact_path="model")
            mv = client.create_model_version(
                name=name,
                source=f"{run.info.artifact_uri}/model",
                run_id=run.info.run_id,
            )
            created.append(mv.version)

    if alias_on is not None:
        client.set_registered_model_alias(name, config.PRODUCTION_ALIAS,
                                          created[alias_on])
    return name, created


@pytest.mark.slow
def test_real_registry_loads_both_files_from_a_directory_source(tmp_path, monkeypatch):
    """The shape difference from the image modality, exercised for real."""
    name, created = _real_registry(tmp_path, monkeypatch, versions=["a"])
    vectorizer, model, source = tracking.load_from_registry()

    assert vectorizer == {"src": "vec-a"}
    assert model == {"src": "model-a"}
    assert source == f"registry:{name}/v{created[0]} (unpromoted)"


@pytest.mark.slow
def test_real_registry_alias_beats_the_newest_version(tmp_path, monkeypatch):
    """A promoted v1 must keep serving after v2 is registered. This is the
    behaviour the whole registry item exists for."""
    name, created = _real_registry(tmp_path, monkeypatch,
                                   versions=["a", "b"], alias_on=0)
    vectorizer, model, source = tracking.load_from_registry()

    assert model == {"src": "model-a"}
    assert source == f"registry:{name}@{config.PRODUCTION_ALIAS}/v{created[0]}"


@pytest.mark.slow
def test_real_registry_without_an_alias_serves_the_newest(tmp_path, monkeypatch):
    """Today's live state for rakuten-text-classifier: registered, no alias."""
    _, created = _real_registry(tmp_path, monkeypatch, versions=["a", "b"])
    _, model, source = tracking.load_from_registry()

    assert model == {"src": "model-b"}
    assert f"/v{created[-1]} (unpromoted)" in source


@pytest.mark.slow
def test_real_registry_version_missing_a_file_is_rejected_by_name(tmp_path, monkeypatch):
    """A half-registered version must fail loudly with the missing filename, not
    load a vectorizer with no estimator.

    The assertions below deliberately check the GUARD's own wording, not just
    the filename: joblib.load raises FileNotFoundError naming the file all by
    itself, so a filename-only assertion passes even with the guard deleted
    (verified by mutation). What the guard adds - and what is worth keeping - is
    naming every missing file at once and listing what the version DOES contain,
    which is what you need to diagnose a bad version in a remote registry.
    """
    _real_registry(tmp_path, monkeypatch, versions=["a"],
                   drop_file=config.MODEL_PATH.name)
    with pytest.raises(FileNotFoundError) as excinfo:
        tracking.load_from_registry()
    message = str(excinfo.value)
    assert config.MODEL_PATH.name in message
    assert "needs BOTH files" in message
    assert config.VECTORIZER_PATH.name in message  # the "found:" listing


@pytest.mark.slow
def test_real_registry_end_to_end_through_the_predictor(tmp_path, monkeypatch):
    """What /health will report once step 3 surfaces serving_source."""
    _real_registry(tmp_path, monkeypatch, versions=["a"], alias_on=0)
    predictor = TfidfPredictor(vectorizer_path=tmp_path / "absent-vec.pkl",
                               model_path=tmp_path / "absent-model.pkl")
    predictor.load(prefer_registry=True)

    assert predictor.is_loaded
    assert predictor.serving_source.startswith("registry:")
    assert predictor.model == {"src": "model-a"}


@pytest.mark.slow
def test_registry_reachable_but_model_unregistered_falls_back(tmp_path, monkeypatch):
    """Tracking configured, nothing registered yet - the first deploy. Serving
    must come up on the local artifacts instead of refusing."""
    mlflow = pytest.importorskip("mlflow", reason="mlflow not installed (CI subset)")
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    mlflow.set_tracking_uri(uri)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr(config, "REGISTERED_MODEL_NAME", "never-registered")

    local = tmp_path / "local"
    predictor = _predictor(local).load(prefer_registry=True)
    assert predictor.serving_source == f"local:{config.MODEL_PATH.name}"
