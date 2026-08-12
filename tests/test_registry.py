"""The registry resolution rule: which registered version actually serves.

This is the rule that answers "which model is used for prediction". It had NO
test coverage until now, which is why these tests were written BEFORE the
function moved to rakuten_common — they pin the existing behaviour so the move
can be proved not to change it.

Two layers, deliberately:

  * A fake client for the fast path. The resolver only ever calls two client
    methods, so a fake is enough, and it keeps these tests runnable in CI
    without installing mlflow (see requirements-ci.txt).
  * The same expectations against a REAL MLflow registry (SQLite + local
    artifact store), skipped when mlflow is not installed. This is what proves
    the fake is faithful rather than merely convenient.

Every expectation below was OBSERVED against real MLflow 3.14.0 before being
written down; none of it is assumed.
"""
from __future__ import annotations

import pytest

from rakuten_common import registry
from rakuten_img import config


# --------------------------------------------------------------------------
# fake client
# --------------------------------------------------------------------------
class _Version:
    """MLflow 3.14.0 returns ModelVersion.version as an INT on all three paths
    (create_model_version, search_model_versions, get_model_version_by_alias).
    Verified directly, not assumed. Older MLflow returned a string, so the
    resolver must tolerate both — it does, via int() and f-strings — and these
    tests coerce with str() so they do not encode either choice.
    """

    def __init__(self, version):
        self.version = int(version)


class FakeClient:
    """Minimal stand-in. Real MLflow raises MlflowException for a missing alias;
    the resolver catches bare Exception, so the exact type does not matter — but
    it MUST raise rather than return None, which is what this fake reproduces.
    """

    def __init__(self, versions, aliases=None):
        self._versions = [_Version(v) for v in versions]
        self._aliases = dict(aliases or {})
        self.alias_calls = []

    def get_model_version_by_alias(self, name, alias):
        self.alias_calls.append((name, alias))
        if alias not in self._aliases:
            raise RuntimeError(f"alias {alias!r} not found on {name!r}")
        return _Version(self._aliases[alias])

    def search_model_versions(self, filter_string):
        return list(self._versions)


# --------------------------------------------------------------------------
# the selection rule
# --------------------------------------------------------------------------
def test_alias_wins_even_when_it_is_not_the_newest_version():
    """The whole point of an alias: promotion is deliberate, not automatic."""
    client = FakeClient(versions=[1, 2, 3], aliases={"production": 1})
    version, source = registry.resolve_registry_version(client, "m")
    assert str(version.version) == "1"
    assert source == "registry:m@production/v1"


def test_missing_alias_falls_back_to_the_newest_version():
    client = FakeClient(versions=[1, 2, 3])
    version, source = registry.resolve_registry_version(client, "m")
    assert str(version.version) == "3"
    assert "(unpromoted)" in source


def test_unpromoted_marker_is_present_so_operators_can_see_it():
    """A fallback that looks identical to a promotion is a trap."""
    client = FakeClient(versions=[7])
    _, source = registry.resolve_registry_version(client, "m")
    assert "(unpromoted)" in source
    assert "@" not in source


def test_newest_is_chosen_numerically_not_lexicographically():
    """v10 must beat v9. String ordering would pick v9 and serve a stale model."""
    client = FakeClient(versions=[9, 10])
    version, _ = registry.resolve_registry_version(client, "m")
    assert str(version.version) == "10"


def test_explicit_alias_argument_overrides_the_default():
    client = FakeClient(versions=[1, 2], aliases={"staging": 2})
    version, source = registry.resolve_registry_version(client, "m", "staging")
    assert str(version.version) == "2"
    assert source == "registry:m@staging/v2"


def test_default_alias_comes_from_config():
    client = FakeClient(versions=[1])
    registry.resolve_registry_version(client, "m")
    assert client.alias_calls == [("m", config.PRODUCTION_ALIAS)]


def test_no_versions_at_all_raises_lookuperror():
    """The caller decides the fallback (local joblib); the resolver must not
    invent one."""
    client = FakeClient(versions=[])
    with pytest.raises(LookupError):
        registry.resolve_registry_version(client, "m")


def test_resolver_is_modality_agnostic():
    """Same rule, whichever registered model name is passed."""
    client = FakeClient(versions=[1, 2], aliases={"production": 2})
    for name in ("rakuten-image-classifier", "rakuten-text-classifier"):
        version, source = registry.resolve_registry_version(client, name)
        assert str(version.version) == "2"
        assert source == f"registry:{name}@production/v2"


# --------------------------------------------------------------------------
# the same expectations against a real registry
# --------------------------------------------------------------------------
@pytest.mark.parametrize("scenario", ["no_alias", "alias_set", "alias_deleted"])
def test_against_a_real_mlflow_registry(tmp_path, monkeypatch, scenario):
    mlflow = pytest.importorskip("mlflow", reason="mlflow not installed (CI subset)")
    from mlflow import MlflowClient

    uri = f"sqlite:///{tmp_path}/mlflow.db"
    mlflow.set_tracking_uri(uri)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)

    name = f"real-{scenario}"
    client = MlflowClient()
    client.create_registered_model(name)

    versions = []
    for i in range(2):
        with mlflow.start_run() as run:
            blob = tmp_path / f"m{i}.joblib"
            blob.write_bytes(b"x")
            mlflow.log_artifact(str(blob), artifact_path="model")
            mv = client.create_model_version(
                name=name,
                source=f"{run.info.artifact_uri}/model",
                run_id=run.info.run_id,
            )
            versions.append(mv.version)

    if scenario in ("alias_set", "alias_deleted"):
        client.set_registered_model_alias(name, config.PRODUCTION_ALIAS, versions[0])
    if scenario == "alias_deleted":
        client.delete_registered_model_alias(name, config.PRODUCTION_ALIAS)

    version, source = registry.resolve_registry_version(client, name)

    if scenario == "alias_set":
        assert str(version.version) == str(versions[0])
        assert source == f"registry:{name}@{config.PRODUCTION_ALIAS}/v{versions[0]}"
    else:
        assert str(version.version) == str(versions[-1])
        assert "(unpromoted)" in source


def test_real_registry_with_no_versions_raises_lookuperror(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow", reason="mlflow not installed (CI subset)")
    from mlflow import MlflowClient

    uri = f"sqlite:///{tmp_path}/mlflow.db"
    mlflow.set_tracking_uri(uri)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)

    client = MlflowClient()
    client.create_registered_model("empty")
    with pytest.raises(LookupError):
        registry.resolve_registry_version(client, "empty")
