"""Registry-aware IMAGE serving: which image model is actually loaded, what
happens when the registry is not there, and what publishing a version does.

WHY THIS FILE EXISTS. The image modality's registry code had NO direct test
coverage. tests/test_registry.py covers rakuten_common.registry
.resolve_registry_version and nothing else; tests/test_pipeline.py covers
build/save/load/reorder. Nothing exercised classifier.register_in_mlflow,
classifier.load_from_registry, or the registry-first/local-fallback policy in
scripts/predict.py — which is the code that decides whether the image service
serves or 503s, and the code with the most exposure to a DagsHub outage.

These are CHARACTERIZATION tests: they pin the behaviour as it stands TODAY, so
that moving this code into rakuten_img/tracking.py can be proved not to change
it. They must pass unchanged before and after the move. Mirrors
tests/test_text_registry.py, which did the same job for the text side.

Two layers, same as the text file:
  * Fast tests with no mlflow at all, so CI job 1 (requirements-ci.txt omits
    mlflow) still covers the fallback policy.
  * The same expectations against a REAL MLflow registry (SQLite + an artifact
    store inside tmp_path), skipped when mlflow is absent.

THE SHAPE DIFFERENCE FROM TEXT, exercised for real below: an image version's
source is a single joblib FILE, not a directory of two .pkl files. That is why
the two modalities have separate loaders.
"""
from __future__ import annotations

import pathlib
import sys

import joblib
import pytest

from rakuten_img import classifier, config

# scripts/ is not a package and conftest.py only puts src/ on the path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

# Imported directly rather than via importorskip: scripts/predict.py is the live
# serving path and its dependencies (numpy, pillow) are in requirements-ci.txt,
# so an import failure here is a real breakage and must fail rather than skip.
# Verified: it imports with neither torch nor mlflow installed (torch is a lazy
# import inside rakuten_img.backbone).
import predict as predict_script  # noqa: E402


PAYLOAD_TEMPLATE = {
    "classes": list(config.CANONICAL_CLASSES),
    "backbone": config.BACKBONE_NAME,
}


def _payload(marker: str) -> dict:
    """A stand-in for the real joblib payload. Nothing under test calls
    predict_proba, so a dict with an identifiable marker is enough and keeps
    sklearn out of these tests."""
    return dict(PAYLOAD_TEMPLATE, classifier={"src": marker})


@pytest.fixture
def serving(monkeypatch, tmp_path):
    """A clean scripts/predict.py pointed at a throwaway local model.

    predict.py caches the resolved payload, the registry-failure latch, the
    serving source AND an lru_cache for the whole process lifetime, so without
    this reset a test would inherit the previous test's decisions.

    NOTE the __defaults__ patch. classifier.load() and classifier.save() capture
    config.CLASSIFIER_PATH as a DEFAULT ARGUMENT, evaluated once at import, so
    monkeypatching config.CLASSIFIER_PATH alone does NOT redirect them — see
    test_default_model_path_is_bound_at_import_not_call_time. Both have to move
    together or the test would read the developer's real models/ directory.
    """
    local_path = tmp_path / "image_classifier.joblib"
    monkeypatch.setattr(config, "CLASSIFIER_PATH", local_path)
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(classifier.load, "__defaults__", (local_path,))

    monkeypatch.setattr(predict_script, "_REGISTRY_PAYLOAD", None)
    monkeypatch.setattr(predict_script, "_REGISTRY_FAILED", False)
    monkeypatch.setattr(predict_script, "_SERVING_SOURCE", "not-loaded")
    predict_script._load_classifier_payload_cached.cache_clear()

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    yield predict_script
    predict_script._load_classifier_payload_cached.cache_clear()


def _write_local(payload: dict = None) -> dict:
    payload = payload if payload is not None else _payload("local")
    joblib.dump(payload, config.CLASSIFIER_PATH)
    return payload


# --------------------------------------------------------------------------
# the serving fallback policy (no mlflow needed)
# --------------------------------------------------------------------------
def test_model_source_is_not_loaded_before_any_prediction(serving):
    """Wart 3, pinned deliberately rather than fixed: /health reports
    'not-loaded' until the first real /predict, because the payload resolves
    lazily. A change here changes what the Grafana model table shows."""
    assert serving.model_source() == "not-loaded"


def test_registry_success_is_served_and_local_disk_is_not_read(serving, monkeypatch):
    """When the registry answers, its payload is served — proven by leaving the
    local .joblib absent entirely."""
    served = _payload("registry")
    served["serving_source"] = "registry:rakuten-image-classifier@production/v2"
    monkeypatch.setattr(classifier, "load_from_registry",
                        lambda alias=None: served)

    assert not config.CLASSIFIER_PATH.exists()
    assert serving._load_payload() is served
    assert serving.model_source() == \
        "registry:rakuten-image-classifier@production/v2"


def test_registry_failure_falls_back_to_local_and_says_so(serving, monkeypatch, capsys):
    """A dead registry must not take the image service down with it."""
    def _explode(alias=None):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(classifier, "load_from_registry", _explode)
    local = _write_local()

    payload = serving._load_payload()
    assert payload["classifier"] == local["classifier"]
    assert serving.model_source() == f"local:{config.CLASSIFIER_PATH.name}"
    assert "falling back to local model" in capsys.readouterr().out


def test_a_dead_registry_is_consulted_once_not_once_per_request(serving, monkeypatch):
    """The _REGISTRY_FAILED latch. Without it every single prediction would pay
    the full registry timeout while DagsHub is down."""
    calls = []

    def _explode(alias=None):
        calls.append(alias)
        raise RuntimeError("connection refused")

    monkeypatch.setattr(classifier, "load_from_registry", _explode)
    _write_local()

    for _ in range(3):
        serving._load_payload()
    assert len(calls) == 1


def test_a_successful_registry_load_is_cached_for_the_process(serving, monkeypatch):
    """The model is downloaded once, not per request."""
    calls = []
    served = _payload("registry")

    def _count(alias=None):
        calls.append(alias)
        return served

    monkeypatch.setattr(classifier, "load_from_registry", _count)
    for _ in range(3):
        assert serving._load_payload() is served
    assert len(calls) == 1


def test_registry_is_not_consulted_without_a_tracking_uri(serving, monkeypatch):
    """A teammate who just cloned, or CI, must serve the local model without
    ever reaching for the network.

    The stub RECORDS rather than raises: _load_payload catches Exception, so a
    raising stub would be swallowed and the test would pass even if the registry
    HAD been consulted. (The same trap the text side documented.)
    """
    calls = []

    def _record(alias=None):
        calls.append(alias)
        raise RuntimeError("registry must not be consulted")

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(classifier, "load_from_registry", _record)
    _write_local()

    serving._load_payload()
    assert calls == []
    assert serving.model_source() == f"local:{config.CLASSIFIER_PATH.name}"


def test_a_local_payload_gets_a_local_serving_source(serving, monkeypatch):
    """The local loader stamps its own source so /health can distinguish it from
    a registry load. setdefault, so a payload that already carries one keeps
    it."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    _write_local()
    payload = serving._load_local_payload()
    assert payload["serving_source"] == f"local:{config.CLASSIFIER_PATH.name}"


def test_no_model_anywhere_raises_the_friendly_filenotfound(serving, monkeypatch):
    """No registry and no local file is a real failure. It must surface as the
    FileNotFoundError the API already turns into a 503, naming the fix."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with pytest.raises(FileNotFoundError) as excinfo:
        serving._load_payload()
    assert "train.py" in str(excinfo.value)


def test_load_from_registry_raises_without_a_tracking_uri(monkeypatch):
    """Real code path, no stub: the CALLER decides the fallback, so this must
    raise rather than return None. Checked before mlflow is imported, so it
    holds in the CI subset too."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with pytest.raises(RuntimeError, match="MLFLOW_TRACKING_URI"):
        classifier.load_from_registry()


def test_register_in_mlflow_is_a_no_op_without_a_tracking_uri(monkeypatch):
    """The opposite policy to load_from_registry, on purpose: registration is
    best-effort and must never raise into a training run that has already
    saved a model to disk."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert classifier.register_in_mlflow("some-run-id") is None


def test_register_in_mlflow_is_a_no_op_without_a_run_id(monkeypatch, tmp_path, capsys):
    """Training with tracking on but logging failed: there is no run to attach
    the artifact to, so registration is skipped and said out loud."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    assert classifier.register_in_mlflow(None) is None
    assert "skipping model registration" in capsys.readouterr().out


def test_default_model_path_is_bound_at_import_not_call_time(tmp_path, monkeypatch):
    """A trap for anyone moving this code, pinned so it is discovered by a test
    rather than by a wrong-directory read.

    classifier.load()/save() take config.CLASSIFIER_PATH as a DEFAULT ARGUMENT,
    evaluated once at import. scripts/predict.py reads config.CLASSIFIER_PATH
    dynamically for its mtime cache. So patching the config attribute redirects
    the mtime check but NOT the actual load, and the two can silently disagree
    about which file is the model.
    """
    monkeypatch.setattr(config, "CLASSIFIER_PATH", tmp_path / "elsewhere.joblib")
    assert classifier.load.__defaults__[0] != config.CLASSIFIER_PATH


# --------------------------------------------------------------------------
# the same expectations against a real registry
# --------------------------------------------------------------------------
def _real_registry(tmp_path, monkeypatch, markers, alias_on=None, tags=None):
    """Stand up a real SQLite registry and publish one version per entry in
    `markers` THROUGH classifier.register_in_mlflow — the function under test,
    so the write side is exercised rather than simulated.

    artifact_location is forced inside tmp_path: the default would create
    ./mlruns in the repo root (wart 11).
    """
    mlflow = pytest.importorskip("mlflow", reason="mlflow not installed (CI subset)")

    uri = f"sqlite:///{tmp_path}/mlflow.db"
    mlflow.set_tracking_uri(uri)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)

    name = "rakuten-image-classifier-test"
    monkeypatch.setattr(config, "REGISTERED_MODEL_NAME", name)

    experiment_id = mlflow.create_experiment(
        "image-registry-test", artifact_location=str(tmp_path / "artifacts"))

    created = []
    for marker in markers:
        model_file = tmp_path / f"stage-{marker}.joblib"
        joblib.dump(_payload(marker), model_file)
        with mlflow.start_run(experiment_id=experiment_id) as run:
            version = classifier.register_in_mlflow(
                run.info.run_id, path=model_file, tags=tags)
        created.append(version)

    if alias_on is not None:
        from mlflow import MlflowClient
        MlflowClient().set_registered_model_alias(
            name, config.PRODUCTION_ALIAS, created[alias_on])
    return name, created


@pytest.mark.slow
def test_real_registry_round_trip_through_a_single_file_source(tmp_path, monkeypatch):
    """Register, then pull back: the image version's source is one joblib FILE,
    which is the whole reason the image loader is not the text loader."""
    name, created = _real_registry(tmp_path, monkeypatch, markers=["a"])
    payload = classifier.load_from_registry()

    assert payload["classifier"] == {"src": "a"}
    assert payload["serving_source"] == f"registry:{name}/v{created[0]} (unpromoted)"


@pytest.mark.slow
def test_registered_version_is_returned_as_a_string(tmp_path, monkeypatch):
    """MLflow 3.14.0 returns ModelVersion.version as an INT against a local
    backend and a STRING through DagsHub's REST layer. register_in_mlflow
    normalises with str() so callers never have to care."""
    _, created = _real_registry(tmp_path, monkeypatch, markers=["a"])
    assert created == ["1"]


@pytest.mark.slow
def test_real_registry_alias_beats_the_newest_version(tmp_path, monkeypatch):
    """A promoted v1 keeps serving after v2 is registered. This is exactly
    today's live state: rakuten-image-classifier has six versions and serves
    the aliased one, not the newest."""
    name, created = _real_registry(tmp_path, monkeypatch,
                                   markers=["a", "b"], alias_on=0)
    payload = classifier.load_from_registry()

    assert payload["classifier"] == {"src": "a"}
    assert payload["serving_source"] == \
        f"registry:{name}@{config.PRODUCTION_ALIAS}/v{created[0]}"


@pytest.mark.slow
def test_registering_twice_creates_a_second_version_not_an_error(tmp_path, monkeypatch):
    """create_registered_model is called every time and its 'already exists'
    exception is swallowed on purpose, so registration is idempotent."""
    _, created = _real_registry(tmp_path, monkeypatch, markers=["a", "b"])
    assert created == ["1", "2"]


@pytest.mark.slow
def test_descriptive_tags_land_on_the_version(tmp_path, monkeypatch):
    """Tags are how a human tells six versions apart in the registry UI. They
    are stringified on the way in."""
    pytest.importorskip("mlflow", reason="mlflow not installed (CI subset)")
    name, _ = _real_registry(tmp_path, monkeypatch, markers=["a"],
                             tags={"val_f1_weighted": 0.5468, "env": "host"})
    from mlflow import MlflowClient

    version = MlflowClient().get_model_version(name, "1")
    assert version.tags["val_f1_weighted"] == "0.5468"
    assert version.tags["env"] == "host"


@pytest.mark.slow
def test_a_tag_that_cannot_be_set_does_not_lose_the_version(tmp_path, monkeypatch, capsys):
    """Each tag is independently best-effort: the version already exists by the
    time tagging starts, and a rejected tag must not turn success into None."""
    mlflow = pytest.importorskip("mlflow", reason="mlflow not installed (CI subset)")
    from mlflow import MlflowClient

    def _boom(self, *args, **kwargs):
        raise RuntimeError("tag rejected")

    monkeypatch.setattr(MlflowClient, "set_model_version_tag", _boom)

    uri = f"sqlite:///{tmp_path}/mlflow.db"
    mlflow.set_tracking_uri(uri)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr(config, "REGISTERED_MODEL_NAME", "tagging-test")
    experiment_id = mlflow.create_experiment(
        "tagging-test", artifact_location=str(tmp_path / "artifacts"))

    model_file = tmp_path / "model.joblib"
    joblib.dump(_payload("a"), model_file)
    with mlflow.start_run(experiment_id=experiment_id) as run:
        version = classifier.register_in_mlflow(
            run.info.run_id, path=model_file, tags={"env": "host"})

    assert version == "1"
    assert "Could not tag" in capsys.readouterr().out


@pytest.mark.slow
def test_a_bad_run_id_returns_none_instead_of_raising(tmp_path, monkeypatch, capsys):
    """The model is already on disk by the time this is called, so a registry
    problem must cost a warning, not the training run."""
    mlflow = pytest.importorskip("mlflow", reason="mlflow not installed (CI subset)")
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    mlflow.set_tracking_uri(uri)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)

    model_file = tmp_path / "model.joblib"
    joblib.dump(_payload("a"), model_file)

    assert classifier.register_in_mlflow("no-such-run", path=model_file) is None
    assert "Model registration skipped" in capsys.readouterr().out


@pytest.mark.slow
def test_nothing_registered_yet_raises_lookuperror(tmp_path, monkeypatch):
    """The first deploy: tracking configured, registry empty. load_from_registry
    raises so the caller can fall back — see the next test."""
    mlflow = pytest.importorskip("mlflow", reason="mlflow not installed (CI subset)")
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    mlflow.set_tracking_uri(uri)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr(config, "REGISTERED_MODEL_NAME", "never-registered")

    with pytest.raises(LookupError):
        classifier.load_from_registry()


@pytest.mark.slow
def test_end_to_end_serving_prefers_the_registry_over_the_local_file(
        tmp_path, monkeypatch, serving):
    """The real thing, no stubs: a real registered version beats a real local
    .joblib holding a different model."""
    _real_registry(tmp_path, monkeypatch, markers=["registered"])
    _write_local(_payload("local"))

    payload = serving._load_payload()
    assert payload["classifier"] == {"src": "registered"}
    assert serving.model_source().startswith("registry:")


@pytest.mark.slow
def test_end_to_end_serving_falls_back_when_the_registry_is_empty(
        tmp_path, monkeypatch, serving):
    """Tracking reachable, nothing registered: serve the local model rather than
    refusing."""
    mlflow = pytest.importorskip("mlflow", reason="mlflow not installed (CI subset)")
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    mlflow.set_tracking_uri(uri)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr(config, "REGISTERED_MODEL_NAME", "never-registered")
    _write_local()

    serving._load_payload()
    assert serving.model_source() == f"local:{config.CLASSIFIER_PATH.name}"
