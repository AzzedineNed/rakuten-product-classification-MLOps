"""Which registered model version actually serves traffic.

MOVED VERBATIM from rakuten_img.classifier (where it lived as
resolve_registry_version) because the TEXT modality needs exactly the same rule.
The logic is unchanged; only its home is. Characterization tests were written
against the original behaviour first (tests/test_registry.py) so the move could
be proved not to change anything.

The rule, in one line: an alias beats a version number, and a missing alias
falls back to the newest version rather than refusing to serve.

Why this is not modality-specific: it takes a client, a registered model name
and an alias, and returns a version. It knows nothing about images, text,
torch, or sklearn - which is exactly the contract of this package.

WHAT THIS MODULE DELIBERATELY DOES NOT DO: download artifacts or load a model.
Those differ per modality (the image side is one joblib payload, the text side
is a directory of two .pkl files), and mixing the selection rule with the
loading mechanics is what would make this untestable without a network.

WHY THE ALIAS LOOKUP IS RETRIED (session 13)
--------------------------------------------
MEASURED against DagsHub from inside the api container: the alias lookup
intermittently hangs at the socket level and fails on the request timeout. In a
bad window 7 of 12 calls failed at a 10s ceiling, and 4 of 8 still failed at a
60s ceiling, while every success came back in under 0.7s. Raising the timeout
does nothing; the endpoint hangs rather than lags. Retrying works: 5 of 5
succeeded with MLFLOW_HTTP_REQUEST_MAX_RETRIES=2. The same window also took out
search_model_versions and get_registered_model, so the cause is NOT
alias-specific and is NOT explained - it did not reproduce afterwards from
either the host or the container. Recorded as unknown.

The alias call still gets its own retry because of what its failure COSTS.
Every other registry call fails loudly: the exception leaves this module, the
caller falls back to the local artifact, and /health says so. An alias failure
used to be swallowed here and silently downgraded to newest-version, so the
service reported a healthy registry load while serving an unpromoted model.
That is exactly what happened on 2026-08-14: the image container booted pinned
to v7 while the production alias pointed at v2, and stayed that way, because
the resolved payload is cached for the process lifetime by design.

A GENUINELY ABSENT ALIAS IS NOT RETRIED. MEASURED on MLflow 3.14.0: a missing
alias raises MlflowException with error_code INVALID_PARAMETER_VALUE, while a
request timeout raises MlflowException with error_code INTERNAL_ERROR. The
timeout is constructed client-side in mlflow.utils.rest_utils with no explicit
code, so it is INTERNAL_ERROR against every backend including DagsHub. That
asymmetry is what makes this safe: an absence is never mistaken for a blip, so
a fresh registry still resolves in a single call.

WHAT THIS MODULE DOES NOT DO ABOUT IT: re-resolve later. Both services consult
the registry ONCE at load time and never per request, so a registry outage can
never slow or break traffic that is already being served. Retrying at load time
keeps that promise; re-resolving on a degraded payload would break it.
"""
from __future__ import annotations

import time
from typing import Optional

from rakuten_img import config

__all__ = ["resolve_registry_version"]


# Attempts INCLUDING the first, so 4 means one call plus three retries. The
# delays are consumed in order between attempts, so ALIAS_RETRY_DELAYS must
# hold at least ALIAS_LOOKUP_ATTEMPTS - 1 entries.
#
# Budget: in the fully-broken case this adds three more timeouts plus 3.5s of
# sleeping. At the measured 10s ceiling that is about 35s of extra startup,
# against a compose start_period of 30s followed by five healthcheck retries at
# 30s intervals. It fits, with room to spare.
ALIAS_LOOKUP_ATTEMPTS = 4
ALIAS_RETRY_DELAYS = (0.5, 1.0, 2.0)

# Error codes that mean "this alias is not set", as opposed to "the request did
# not get through". See the module docstring; both were measured, not assumed.
ALIAS_ABSENT_ERROR_CODES = frozenset({
    "INVALID_PARAMETER_VALUE",
    "RESOURCE_DOES_NOT_EXIST",
})


def _alias_is_absent(exc: BaseException) -> bool:
    """True when `exc` says the alias does not exist, rather than that the call
    failed to complete.

    getattr with a default because the resolver must keep working against any
    client that raises something without an error_code: a fake in a test, an
    older MLflow, a backend that raises requests' own exceptions. Anything
    unrecognised is treated as retryable, which costs a few fast calls in the
    worst case and never loses a promotion.
    """
    return getattr(exc, "error_code", None) in ALIAS_ABSENT_ERROR_CODES


def _lookup_alias(client, name: str, alias: str):
    """Resolve `alias`, retrying transient failures.

    Returns (version_object, "resolved"), (None, "absent") or (None, "failed").
    Raises nothing: the caller decides what to do, and needs to tell the two
    None cases apart because they mean different things to an operator.
    """
    last_exc: Optional[BaseException] = None
    for attempt in range(1, ALIAS_LOOKUP_ATTEMPTS + 1):
        try:
            return client.get_model_version_by_alias(name, alias), "resolved"
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if _alias_is_absent(exc):
                # A fresh registry, or an alias nobody has set yet. Not an
                # error, and retrying it would only be slower.
                print(f"ℹ️  No '{alias}' alias on '{name}' - "
                      f"falling back to the newest version.")
                return None, "absent"
            if attempt < ALIAS_LOOKUP_ATTEMPTS:
                delay = ALIAS_RETRY_DELAYS[attempt - 1]
                print(f"⚠️  Alias lookup for '{name}' failed "
                      f"(attempt {attempt}/{ALIAS_LOOKUP_ATTEMPTS}, "
                      f"{type(exc).__name__}: {exc}) - retrying in {delay}s.")
                time.sleep(delay)
    # The message, not just the type: session 13 spent six round trips
    # rediscovering a timeout that the old log line had thrown away.
    print(f"❌ Alias lookup for '{name}' FAILED after {ALIAS_LOOKUP_ATTEMPTS} "
          f"attempts ({type(last_exc).__name__}: {last_exc}). Serving the "
          f"newest version instead, which may NOT be the promoted one.")
    return None, "failed"


def resolve_registry_version(client, name: str, alias: Optional[str] = None):
    """Pick which registered version to serve: the one carrying `alias` if that
    alias resolves, otherwise the highest version number.

    Returns (version_object, source_string). Raises LookupError only when the
    registered model has no versions at all - the CALLER decides what to do
    then (both services fall back to a local artifact so serving never
    hard-fails).

    The source string is surfaced by /health and logged at load time. There are
    now THREE outcomes rather than two, because they need different responses:

      registry:name@production/v2
          promoted, serving as intended.
      registry:name/v6 (unpromoted)
          no alias is set. Expected on a fresh registry; nothing to do.
      registry:name/v6 (unpromoted, alias lookup failed)
          the alias could not be read. A promotion may exist and is NOT being
          honoured. Restart the service once the registry is reachable.
    """
    alias = alias or config.PRODUCTION_ALIAS
    version, outcome = _lookup_alias(client, name, alias)
    if version is not None:
        return version, f"registry:{name}@{alias}/v{version.version}"

    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        raise LookupError(f"No versions of '{name}' in the MLflow registry.")
    latest = max(versions, key=lambda v: int(v.version))
    suffix = ("(unpromoted)" if outcome == "absent"
              else "(unpromoted, alias lookup failed)")
    return latest, f"registry:{name}/v{latest.version} {suffix}"
