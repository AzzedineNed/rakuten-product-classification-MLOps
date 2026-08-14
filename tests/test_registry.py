"""The registry resolution rule: which registered version actually serves.

This is the rule that answers "which model is used for prediction". It had NO
test coverage until now, which is why these tests were written BEFORE the
function moved to rakuten_common - they pin the existing behaviour so the move
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
    resolver must tolerate both - it does, via int() and f-strings - and these
    tests coerce with str() so they do not encode either choice.
    """

    def __init__(self, version):
        self.version = int(version)


class _AliasAbsent(Exception):
    """What a missing alias looks like on the wire.

    MEASURED on MLflow 3.14.0 against a real registry: MlflowException with
    error_code 'INVALID_PARAMETER_VALUE' and the message "Registered model
    alias production not found." The resolver keys off error_code alone, so
    this fake carries the code rather than importing mlflow, which would make
    these tests unrunnable in the CI subset (see requirements-ci.txt).

    THIS USED TO BE A BARE RuntimeError. That was fine while every failure took
    the same path, and became wrong in session 13 when the resolver started
    retrying failures that are NOT an absence: a bare RuntimeError models an
    unreachable registry, not a missing alias, so the fake was quietly testing
    the wrong branch.
    """

    error_code = "INVALID_PARAMETER_VALUE"


class _AliasUnreachable(Exception):
    """What a timed-out alias lookup looks like on the wire.

    MEASURED: MlflowException with error_code 'INTERNAL_ERROR', because the
    timeout is constructed client-side in mlflow.utils.rest_utils with no
    explicit code. This is the case that MUST be retried.
    """

    error_code = "INTERNAL_ERROR"


class FakeClient:
    """Minimal stand-in. Real MLflow raises MlflowException for a missing alias;
    the resolver keys off error_code, not the exception type, so this fake
    reproduces the CODE - and it MUST raise rather than return None.
    """

    def __init__(self, versions, aliases=None):
        self._versions = [_Version(v) for v in versions]
        self._aliases = dict(aliases or {})
        self.alias_calls = []

    def get_model_version_by_alias(self, name, alias):
        self.alias_calls.append((name, alias))
        if alias not in self._aliases:
            raise _AliasAbsent(f"Registered model alias {alias} not found.")
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
@pytest.mark.slow
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


@pytest.mark.slow
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


# --------------------------------------------------------------------------
# retrying a transient alias failure (session 13)
#
# WHY THIS EXISTS. On 2026-08-14 the image container booted serving unpromoted
# v7 while the production alias pointed at v2. Nothing was broken: the alias
# lookup timed out once, the resolver silently downgraded to newest-version,
# load_from_registry returned successfully, and /health reported a healthy
# registry load. MEASURED in the same window: 7 of 12 alias calls failed at a
# 10s ceiling and 4 of 8 at a 60s ceiling, while every success returned in
# under 0.7s. The cause never reproduced and is recorded as unknown; these
# tests are about the RESPONSE to a failure, not its cause.
#
# time.sleep is patched out in every test here. The delays are real in
# production and would only make the suite slow.
# --------------------------------------------------------------------------
class FlakyAliasClient(FakeClient):
    """Fails the alias lookup `failures` times, then behaves normally."""

    def __init__(self, versions, aliases=None, failures=0,
                 exc_factory=_AliasUnreachable):
        super().__init__(versions, aliases)
        self._remaining = failures
        self._exc_factory = exc_factory

    def get_model_version_by_alias(self, name, alias):
        self.alias_calls.append((name, alias))
        if self._remaining > 0:
            self._remaining -= 1
            raise self._exc_factory("API request failed with timeout exception")
        if alias not in self._aliases:
            raise _AliasAbsent(f"Registered model alias {alias} not found.")
        return _Version(self._aliases[alias])


@pytest.fixture
def no_sleep(monkeypatch):
    """Record the backoff delays instead of sleeping them."""
    slept = []
    monkeypatch.setattr(registry.time, "sleep", slept.append)
    return slept


def test_a_transient_alias_failure_is_retried_and_the_promotion_is_honoured(no_sleep):
    """The bug this whole section exists for: one blip must not silently
    change which model serves."""
    client = FlakyAliasClient(versions=[1, 2, 7], aliases={"production": 2},
                              failures=1)
    version, source = registry.resolve_registry_version(client, "m")
    assert str(version.version) == "2"
    assert source == "registry:m@production/v2"
    assert len(client.alias_calls) == 2


def test_retries_continue_up_to_the_attempt_budget(no_sleep):
    client = FlakyAliasClient(versions=[1, 2], aliases={"production": 1},
                              failures=registry.ALIAS_LOOKUP_ATTEMPTS - 1)
    version, source = registry.resolve_registry_version(client, "m")
    assert str(version.version) == "1"
    assert source == "registry:m@production/v1"
    assert len(client.alias_calls) == registry.ALIAS_LOOKUP_ATTEMPTS


def test_exhausting_the_budget_falls_back_but_says_so(no_sleep):
    """The fallback is unchanged - serving must not stop - but the source
    string has to distinguish this from an alias nobody ever set."""
    client = FlakyAliasClient(versions=[1, 2, 7], aliases={"production": 2},
                              failures=99)
    version, source = registry.resolve_registry_version(client, "m")
    assert str(version.version) == "7"
    assert "alias lookup failed" in source
    assert len(client.alias_calls) == registry.ALIAS_LOOKUP_ATTEMPTS


def test_the_two_fallback_reasons_produce_different_strings(no_sleep):
    """An operator reading /health must be able to tell 'nothing was ever
    promoted' from 'a promotion exists and is not being honoured'."""
    absent = FakeClient(versions=[1, 2])
    _, absent_source = registry.resolve_registry_version(absent, "m")

    broken = FlakyAliasClient(versions=[1, 2], aliases={"production": 1},
                              failures=99)
    _, broken_source = registry.resolve_registry_version(broken, "m")

    assert absent_source != broken_source
    assert "alias lookup failed" not in absent_source
    assert "alias lookup failed" in broken_source


def test_an_absent_alias_is_not_retried(no_sleep):
    """Retrying a definite answer only makes a fresh registry slower. This is
    the asymmetry the whole design rests on, so it is asserted, not assumed."""
    client = FakeClient(versions=[1, 2])
    registry.resolve_registry_version(client, "m")
    assert len(client.alias_calls) == 1
    assert no_sleep == []


@pytest.mark.parametrize("code", sorted(registry.ALIAS_ABSENT_ERROR_CODES))
def test_every_absent_error_code_short_circuits(no_sleep, code):
    exc_type = type("_Absent", (Exception,), {"error_code": code})
    client = FlakyAliasClient(versions=[1, 2], failures=99, exc_factory=exc_type)
    _, source = registry.resolve_registry_version(client, "m")
    assert len(client.alias_calls) == 1
    assert "alias lookup failed" not in source


def test_an_exception_with_no_error_code_is_retried(no_sleep):
    """The safe default. An unrecognised client (older MLflow, a raw requests
    exception, a test double) must not be mistaken for a definite absence -
    that would lose a promotion silently, which is the original bug."""
    client = FlakyAliasClient(versions=[1, 2], aliases={"production": 1},
                              failures=1, exc_factory=RuntimeError)
    version, source = registry.resolve_registry_version(client, "m")
    assert str(version.version) == "1"
    assert source == "registry:m@production/v1"
    assert len(client.alias_calls) == 2


def test_backoff_delays_are_applied_in_order(no_sleep):
    client = FlakyAliasClient(versions=[1], aliases={"production": 1},
                              failures=registry.ALIAS_LOOKUP_ATTEMPTS - 1)
    registry.resolve_registry_version(client, "m")
    expected = list(registry.ALIAS_RETRY_DELAYS[:registry.ALIAS_LOOKUP_ATTEMPTS - 1])
    assert no_sleep == expected


def test_no_sleep_after_the_final_attempt(no_sleep):
    """A budget of N attempts costs N-1 waits, not N. Sleeping after the last
    failure would add dead time to every degraded startup."""
    client = FlakyAliasClient(versions=[1], failures=99,
                              exc_factory=_AliasUnreachable)
    registry.resolve_registry_version(client, "m")
    assert len(no_sleep) == registry.ALIAS_LOOKUP_ATTEMPTS - 1


def test_there_are_enough_delays_for_the_attempt_budget():
    """A config guard, not a behaviour test: raising ALIAS_LOOKUP_ATTEMPTS
    without adding a delay would IndexError on the retry path, which is the
    path that only runs when something is already wrong."""
    assert len(registry.ALIAS_RETRY_DELAYS) >= registry.ALIAS_LOOKUP_ATTEMPTS - 1


def test_a_failed_lookup_still_raises_lookuperror_when_there_is_nothing_to_serve(no_sleep):
    """Exhausted retries must not invent a version. The caller falls back to
    the local artifact on LookupError; swallowing it would serve nothing."""
    client = FlakyAliasClient(versions=[], failures=99)
    with pytest.raises(LookupError):
        registry.resolve_registry_version(client, "m")
