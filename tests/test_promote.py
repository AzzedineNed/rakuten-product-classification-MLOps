"""The promotion audit trail: who moved the serving alias, when, and from what.

WHY THIS MATTERS. Verified directly against MLflow 3.14.0: re-pointing an alias
is a MOVE, not a copy - the previous version's alias list comes back empty, and
MLflow keeps no history of the change. So "which version was in production last
Tuesday, and who promoted it?" is unanswerable from the registry alone. The tags
written by scripts/promote.py ARE the record; these tests are what stop them
quietly disappearing.

Everything here runs against a real SQLite registry, because the behaviour under
test is what the registry actually stores, which a fake could only restate.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

# scripts/ is not a package and conftest.py only puts src/ on the path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

promote_script = pytest.importorskip("promote", reason="needs scripts/promote.py")


def _registry(tmp_path, monkeypatch, n_versions=2):
    mlflow = pytest.importorskip("mlflow", reason="mlflow not installed (CI subset)")
    from mlflow import MlflowClient

    uri = f"sqlite:///{tmp_path}/mlflow.db"
    mlflow.set_tracking_uri(uri)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)

    name = "audit-model"
    client = MlflowClient()
    client.create_registered_model(name)
    experiment_id = mlflow.create_experiment(
        "audit", artifact_location=str(tmp_path / "artifacts"))

    versions = []
    for i in range(n_versions):
        blob = tmp_path / f"model-{i}.joblib"
        blob.write_bytes(b"x")
        with mlflow.start_run(experiment_id=experiment_id) as run:
            client.log_artifact(run.info.run_id, str(blob), artifact_path="model")
            mv = client.create_model_version(
                name, f"{run.info.artifact_uri}/model", run_id=run.info.run_id)
            versions.append(str(mv.version))
    return client, name, versions


def _tags(client, name, version):
    return client.get_model_version(name, str(version)).tags


@pytest.mark.slow
def test_first_promotion_records_who_when_and_that_nothing_preceded_it(
        tmp_path, monkeypatch, capsys):
    client, name, versions = _registry(tmp_path, monkeypatch, n_versions=1)
    monkeypatch.setenv("RAKUTEN_PROMOTED_BY", "alice")

    promote_script.promote(client, name, "production", versions[0])
    tags = _tags(client, name, versions[0])

    assert tags["promoted_by"] == "alice"
    assert tags["promoted_from"] == "none"
    assert tags["promoted_alias"] == "production"
    assert tags["promoted_at"].endswith("Z") and tags["promoted_at"][4] == "-"


@pytest.mark.slow
def test_repointing_records_both_sides_of_the_move(tmp_path, monkeypatch):
    """The destructive part: v1 loses the alias with no trace of its own, so the
    outgoing version must be stamped too or the timeline has a hole."""
    client, name, versions = _registry(tmp_path, monkeypatch)
    monkeypatch.setenv("RAKUTEN_PROMOTED_BY", "bob")

    promote_script.promote(client, name, "production", versions[0])
    promote_script.promote(client, name, "production", versions[1])

    old, new = _tags(client, name, versions[0]), _tags(client, name, versions[1])
    assert new["promoted_from"] == f"v{versions[0]}"
    assert old["demoted_to"] == f"v{versions[1]}"
    assert old["demoted_by"] == "bob"
    # and MLflow itself really did take the alias away from the old version
    assert client.get_model_version(name, versions[0]).aliases == []


@pytest.mark.slow
def test_demote_closes_the_record(tmp_path, monkeypatch):
    """A promotion with no end looks, in the tags, like it is still in force."""
    client, name, versions = _registry(tmp_path, monkeypatch, n_versions=1)
    monkeypatch.setenv("RAKUTEN_PROMOTED_BY", "carol")

    promote_script.promote(client, name, "production", versions[0])
    promote_script.demote(client, name, "production")

    tags = _tags(client, name, versions[0])
    assert tags["demoted_to"] == "(alias removed)"
    assert tags["demoted_by"] == "carol"


@pytest.mark.slow
def test_a_rejected_tag_never_undoes_a_completed_promotion(tmp_path, monkeypatch, capsys):
    """The alias moves BEFORE the tags are written. If a backend rejects a tag
    (DagsHub has been fussy about tagging before), the promotion has already
    happened - reporting failure would tell the operator the opposite of the
    truth."""
    client, name, versions = _registry(tmp_path, monkeypatch, n_versions=1)

    def _refuse(*a, **k):
        raise RuntimeError("backend says no")

    monkeypatch.setattr(client, "set_model_version_tag", _refuse)
    promote_script.promote(client, name, "production", versions[0])   # must not raise

    assert client.get_model_version_by_alias(name, "production").version == int(versions[0])
    assert "the alias DID move" in capsys.readouterr().out


@pytest.mark.slow
def test_promoting_the_same_version_twice_does_not_self_demote(tmp_path, monkeypatch):
    """Re-promoting the version that is already serving is a no-op in MLflow's
    eyes; stamping it as demoted would invent an outage that never happened."""
    client, name, versions = _registry(tmp_path, monkeypatch, n_versions=1)

    promote_script.promote(client, name, "production", versions[0])
    promote_script.promote(client, name, "production", versions[0])

    tags = _tags(client, name, versions[0])
    assert "demoted_at" not in tags
    assert tags["promoted_from"] == f"v{versions[0]}"


def test_actor_falls_back_to_the_os_user(tmp_path, monkeypatch):
    """No override configured is the normal case for a human at a terminal."""
    monkeypatch.delenv("RAKUTEN_PROMOTED_BY", raising=False)
    assert promote_script._actor()
    assert promote_script._actor() != ""
