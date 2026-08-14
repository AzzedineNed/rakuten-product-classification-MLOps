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
    eyes; stamping it as demoted would invent an outage that never happened.

    promoted_from used to come out as the version's OWN number here, which
    claims it was promoted from itself. It now says "(unchanged)", which is
    what actually happened: the alias did not move and the tags were restamped.
    """
    client, name, versions = _registry(tmp_path, monkeypatch, n_versions=1)

    promote_script.promote(client, name, "production", versions[0])
    promote_script.promote(client, name, "production", versions[0])

    tags = _tags(client, name, versions[0])
    assert "demoted_at" not in tags
    assert tags["promoted_from"] == "(unchanged)"


@pytest.mark.slow
def test_a_self_promotion_does_not_report_a_move_or_ask_for_a_restart(
        tmp_path, monkeypatch, capsys):
    """Two cosmetics that both mislead: "v1 -> v1" reads as a move that did not
    happen, and the restart hint sends an operator to bounce a service for a
    change that was not made."""
    client, name, versions = _registry(tmp_path, monkeypatch, n_versions=1)
    promote_script.promote(client, name, "production", versions[0])
    capsys.readouterr()

    promote_script.promote(client, name, "production", versions[0])

    out = capsys.readouterr().out
    assert f"v{versions[0]} -> v{versions[0]}" not in out
    assert "already on" in out
    assert "Restart the API process" not in out


@pytest.mark.slow
def test_the_restart_hint_still_prints_on_a_real_move(tmp_path, monkeypatch, capsys):
    """The guard above must not swallow the hint when it matters."""
    client, name, versions = _registry(tmp_path, monkeypatch)
    promote_script.promote(client, name, "production", versions[0])
    capsys.readouterr()

    promote_script.promote(client, name, "production", versions[1])

    assert "Restart the API process" in capsys.readouterr().out


def test_actor_falls_back_to_the_os_user(tmp_path, monkeypatch):
    """No override configured is the normal case for a human at a terminal."""
    monkeypatch.delenv("RAKUTEN_PROMOTED_BY", raising=False)
    assert promote_script._actor()
    assert promote_script._actor() != ""


# ---------------------------------------------------------------------------
# "UNSET" AND "COULD NOT READ" ARE NOT THE SAME ANSWER
#
# The old helper caught every exception and returned None, so a registry that
# did not answer looked exactly like a registry where nothing was promoted.
# The serving path got this fixed in 8e50577; this is the same bug in the tool
# an operator reaches for when something already looks wrong. It is worse here
# than a misleading line of output, because promote() uses the answer to decide
# what to stamp on the version that LOSES the alias, and MLflow keeps no
# history to rebuild that from.
#
# The error codes below were MEASURED on MLflow 3.14.0 (see the module
# docstring of rakuten_common.registry); the resolver keys off error_code
# alone, so a plain Exception carrying the right code is a faithful stand-in.
# ---------------------------------------------------------------------------


class _AliasUnreachable(Exception):
    """A request that never completed. MEASURED: error_code INTERNAL_ERROR,
    because MLflow builds the timeout client-side with no explicit code."""
    error_code = "INTERNAL_ERROR"


@pytest.fixture
def no_sleep(monkeypatch):
    """Record the backoff delays instead of sleeping them."""
    slept = []
    monkeypatch.setattr(promote_script.time, "sleep", slept.append)
    return slept


def _second_model(client, monkeypatch, image_name):
    """Point both config defaults at models that exist in THIS test registry.

    The fixture registers one model under its own name, so main()'s defaults
    would otherwise reach for the real rakuten-* names and find nothing.
    """
    # NOT derived from image_name: "<image_name>-text" CONTAINS image_name, so
    # an "image model absent from the output" assertion would pass on the text
    # model's own line and prove nothing.
    text_name = "text-side-model"
    client.create_registered_model(text_name)
    monkeypatch.setattr(promote_script.config, "REGISTERED_MODEL_NAME", image_name)
    monkeypatch.setattr(promote_script.text_config, "REGISTERED_MODEL_NAME", text_name)
    monkeypatch.setattr(promote_script, "_client", lambda: client)
    return text_name


def _alias_truth(client, name, alias="production"):
    """What the registry ACTUALLY holds, read through a fresh client so the
    assertion cannot be answered by whatever the test has patched."""
    from mlflow import MlflowClient
    try:
        return str(MlflowClient().get_model_version_by_alias(name, alias).version)
    except Exception:  # noqa: BLE001
        return None


def _break_alias_reads(client, monkeypatch, failures=99):
    """Make the next `failures` alias reads fail as an unreachable registry.

    Wraps the real client, so everything else in the promotion path still runs
    against the real SQLite backend. Returns the call counter.
    """
    real = client.get_model_version_by_alias
    calls = []

    def _flaky(name, alias):
        calls.append(alias)
        if len(calls) <= failures:
            raise _AliasUnreachable("HTTPSConnectionPool: read timed out")
        return real(name, alias)

    monkeypatch.setattr(client, "get_model_version_by_alias", _flaky)
    return calls


@pytest.mark.slow
def test_an_absent_alias_is_answered_in_one_call_by_a_real_registry(
        tmp_path, monkeypatch, no_sleep):
    """Guards the half of the rule that costs nothing: MLflow's own "not found"
    must be recognised as an absence, not retried. If this regresses, every
    first promotion pays the full retry budget for an answer it already had."""
    client, name, _ = _registry(tmp_path, monkeypatch, n_versions=1)
    calls = []
    real = client.get_model_version_by_alias
    monkeypatch.setattr(client, "get_model_version_by_alias",
                        lambda n, a: (calls.append(a), real(n, a))[1])

    assert promote_script._alias_state(client, name, "production") == (None, "absent")
    assert len(calls) == 1
    assert no_sleep == []


@pytest.mark.slow
def test_a_transient_read_does_not_cost_the_outgoing_version_its_record(
        tmp_path, monkeypatch, no_sleep):
    """THE BUG THIS COMMIT EXISTS FOR. One blip while reading the alias used to
    make promote() believe nothing was promoted: the new version got
    promoted_from=none and the version actually losing the alias got no
    demotion stamp at all, silently and permanently."""
    client, name, versions = _registry(tmp_path, monkeypatch)
    monkeypatch.setenv("RAKUTEN_PROMOTED_BY", "dave")
    promote_script.promote(client, name, "production", versions[0])

    _break_alias_reads(client, monkeypatch, failures=1)
    promote_script.promote(client, name, "production", versions[1])

    old, new = _tags(client, name, versions[0]), _tags(client, name, versions[1])
    assert new["promoted_from"] == f"v{versions[0]}"
    assert old["demoted_to"] == f"v{versions[1]}"
    assert no_sleep == [promote_script.ALIAS_RETRY_DELAYS[0]]


@pytest.mark.slow
def test_promote_refuses_to_move_an_alias_it_cannot_read(
        tmp_path, monkeypatch, no_sleep):
    """Moving it would probably succeed, which is the problem: the outgoing
    version could never be named, so its demotion could never be recorded."""
    client, name, versions = _registry(tmp_path, monkeypatch)
    promote_script.promote(client, name, "production", versions[0])
    calls = _break_alias_reads(client, monkeypatch)

    with pytest.raises(SystemExit) as exc:
        promote_script.promote(client, name, "production", versions[1])

    assert "Refusing to move" in str(exc.value)
    assert len(calls) == promote_script.ALIAS_LOOKUP_ATTEMPTS
    # one delay BETWEEN each pair of attempts, and none after the last
    assert no_sleep == list(
        promote_script.ALIAS_RETRY_DELAYS[:promote_script.ALIAS_LOOKUP_ATTEMPTS - 1])
    # and the promotion that was already in force is untouched
    assert _alias_truth(client, name) == versions[0]
    assert "demoted_at" not in _tags(client, name, versions[0])


@pytest.mark.slow
def test_a_failed_readback_is_not_reported_as_a_failed_move(
        tmp_path, monkeypatch, no_sleep, capsys):
    """The readback catches a write that silently did not land. When the
    readback itself cannot complete it knows nothing, so it must not claim the
    move failed - and above all must not exit before the tags are written,
    which would leave the exact hole this commit closes."""
    client, name, versions = _registry(tmp_path, monkeypatch, n_versions=1)
    real = client.get_model_version_by_alias
    calls = []

    def _fails_only_after_the_write(n, a):
        # Call 1 is the "what is promoted now" lookup, call 2 is the readback.
        calls.append(a)
        if len(calls) > 1:
            raise _AliasUnreachable("HTTPSConnectionPool: read timed out")
        return real(n, a)

    monkeypatch.setattr(client, "get_model_version_by_alias",
                        _fails_only_after_the_write)
    promote_script.promote(client, name, "production", versions[0])  # must not raise

    assert "most likely succeeded" in capsys.readouterr().out
    assert _tags(client, name, versions[0])["promoted_by"]


@pytest.mark.slow
def test_demote_refuses_when_it_cannot_name_the_version_losing_the_alias(
        tmp_path, monkeypatch, no_sleep):
    client, name, versions = _registry(tmp_path, monkeypatch, n_versions=1)
    promote_script.promote(client, name, "production", versions[0])
    _break_alias_reads(client, monkeypatch)

    with pytest.raises(SystemExit) as exc:
        promote_script.demote(client, name, "production")

    assert "Refusing to remove" in str(exc.value)
    assert _alias_truth(client, name) == versions[0]


@pytest.mark.slow
def test_list_says_unset_only_when_the_registry_said_so(
        tmp_path, monkeypatch, capsys):
    client, name, _ = _registry(tmp_path, monkeypatch, n_versions=1)
    promote_script.list_versions(client, name, "production")
    out = capsys.readouterr().out
    assert "(unset)" in out
    assert "SERVING" not in out


@pytest.mark.slow
def test_list_does_not_call_an_unreadable_alias_unset(
        tmp_path, monkeypatch, no_sleep, capsys):
    """An operator reading this is usually already suspicious. Reporting
    "(unset)" here points them at the wrong problem entirely."""
    client, name, versions = _registry(tmp_path, monkeypatch, n_versions=1)
    promote_script.promote(client, name, "production", versions[0])
    _break_alias_reads(client, monkeypatch)
    capsys.readouterr()

    promote_script.list_versions(client, name, "production")

    out = capsys.readouterr().out
    assert "(unset)" not in out
    assert "UNKNOWN" in out
    assert "Re-run when the registry answers" in out
    # v1 really is serving, but this run has no right to say so
    assert "<-- SERVING" not in out


@pytest.mark.slow
def test_list_with_no_model_covers_both_modalities(tmp_path, monkeypatch, capsys):
    """--model defaulted to the IMAGE model, so "promote.py --list" answered
    "what is promoted?" for half the system while looking like the whole
    answer. Listing is read-only, so both cost nothing."""
    client, image_name, versions = _registry(tmp_path, monkeypatch, n_versions=1)
    text_name = _second_model(client, monkeypatch, image_name)
    monkeypatch.setattr(sys, "argv", ["promote.py", "--list"])

    promote_script.main()

    out = capsys.readouterr().out
    assert image_name in out
    assert text_name in out


@pytest.mark.slow
def test_an_explicit_model_lists_only_that_one(tmp_path, monkeypatch, capsys):
    client, image_name, _ = _registry(tmp_path, monkeypatch, n_versions=1)
    text_name = _second_model(client, monkeypatch, image_name)
    monkeypatch.setattr(sys, "argv", ["promote.py", "--list", "--model", text_name])

    promote_script.main()

    out = capsys.readouterr().out
    assert text_name in out
    assert image_name not in out


@pytest.mark.slow
def test_a_write_still_defaults_to_one_model_rather_than_guessing(tmp_path,
                                                                 monkeypatch):
    """--list got wider; promote and demote deliberately did not. Writing to a
    model the user did not name is a different class of surprise."""
    client, image_name, versions = _registry(tmp_path, monkeypatch, n_versions=1)
    _second_model(client, monkeypatch, image_name)
    monkeypatch.setattr(sys, "argv",
                        ["promote.py", "--version", str(versions[0])])

    promote_script.main()

    assert _alias_truth(client, image_name) == versions[0]
