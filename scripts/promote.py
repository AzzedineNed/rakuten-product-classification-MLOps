#!/usr/bin/env python3
"""promote.py - Decide which registered model version serves traffic.

Training registers a new version every time it runs and tags it with what it
is (val F1, classifier type, host/container). It deliberately does NOT promote
anything: if "newest" automatically meant "served", a bad or accidental run
would silently take over serving. Promotion is this script - an explicit,
human, auditable step that moves the `production` alias
(config.PRODUCTION_ALIAS) onto a version you have actually looked at.

predict.py resolves that alias first and falls back to the newest version if no
alias is set, so serving keeps working before anything has ever been promoted.

Usage:
  python scripts/promote.py --list             # what exists, what is promoted
  python scripts/promote.py --version 2        # promote v2 to 'production'
  python scripts/promote.py --version 2 --alias staging
  python scripts/promote.py --demote           # remove the alias entirely

Requires MLFLOW_TRACKING_URI (+ credentials) in the environment / .env.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import _bootstrap  # noqa: F401

from rakuten_img import config
from rakuten_text import config as text_config
from rakuten_common.registry import (
    ALIAS_LOOKUP_ATTEMPTS,
    ALIAS_RETRY_DELAYS,
    _alias_is_absent,
)


def _client():
    if not os.getenv("MLFLOW_TRACKING_URI"):
        sys.exit("❌ MLFLOW_TRACKING_URI is not set - nothing to promote against.")
    from mlflow import MlflowClient
    return MlflowClient()


def _alias_state(client, name: str, alias: str):
    """Which version carries `alias`, and whether we actually know.

    Returns (version, "resolved"), (None, "absent") or (None, "failed").

    THE THREE-STATE ANSWER IS THE POINT. The helper this replaces caught every
    exception and returned None, so "nobody has promoted anything" and "the
    registry did not answer" came back identical. That is how the image alias
    read as unset for three round trips in session 13 while it was in fact on
    v2 the whole time, and it is the same bug that 8e50577 fixed in the serving
    path. Here it is worse than a wrong line of output: promote() uses the
    answer to decide what to stamp on the OUTGOING version, and MLflow keeps no
    history to rebuild that from afterwards.

    The classification rule and the retry budget are imported rather than
    restated. Both were measured (see rakuten_common.registry), and a second
    copy of a measured constant is a copy that goes stale. The loop is local
    because the messages have to differ: this script is not serving traffic and
    nothing here falls back to a newest version.
    """
    last_exc: BaseException | None = None
    for attempt in range(1, ALIAS_LOOKUP_ATTEMPTS + 1):
        try:
            return client.get_model_version_by_alias(name, alias).version, "resolved"
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if _alias_is_absent(exc):
                return None, "absent"
            if attempt < ALIAS_LOOKUP_ATTEMPTS:
                delay = ALIAS_RETRY_DELAYS[attempt - 1]
                print(f"⚠️  Reading alias '{alias}' failed (attempt "
                      f"{attempt}/{ALIAS_LOOKUP_ATTEMPTS}, "
                      f"{type(exc).__name__}: {exc}) - retrying in {delay}s.")
                time.sleep(delay)
    print(f"❌ Could not read alias '{alias}' on '{name}' after "
          f"{ALIAS_LOOKUP_ATTEMPTS} attempts "
          f"({type(last_exc).__name__}: {last_exc}).")
    return None, "failed"


def list_versions(client, name: str, alias: str) -> None:
    versions = sorted(client.search_model_versions(f"name='{name}'"),
                      key=lambda v: int(v.version))
    if not versions:
        print(f"(no versions of '{name}' registered yet)")
        return
    promoted, outcome = _alias_state(client, name, alias)
    current = {"resolved": f"v{promoted}",
               "absent": "(unset)"}.get(outcome, "(UNKNOWN - see above)")
    print(f"Registered model: {name}   alias '{alias}' -> {current}\n")
    for v in versions:
        serving = outcome == "resolved" and str(v.version) == str(promoted)
        marker = "  <-- SERVING" if serving else ""
        tags = ", ".join(f"{k}={val}" for k, val in sorted(v.tags.items())) or "-"
        print(f"  v{v.version}  run={str(v.run_id)[:8]}  {tags}{marker}")
    if outcome == "failed":
        # Without this the listing looks exactly like a registry where nothing
        # has ever been promoted, which is the reading that cost session 13
        # three round trips.
        print(f"\n⚠️  Nothing is marked SERVING because the alias could not be "
              f"READ, not because nothing is promoted. Re-run when the "
              f"registry answers.")


def _actor() -> str:
    """Who is doing this. RAKUTEN_PROMOTED_BY wins so a CI job or a DAG can
    name itself instead of reporting whatever the container's user happens to
    be."""
    override = os.getenv("RAKUTEN_PROMOTED_BY")
    if override:
        return override
    try:
        import getpass
        return getpass.getuser()
    except Exception:  # noqa: BLE001
        return "unknown"


def _tag(client, name: str, version: str, tags: dict) -> None:
    """Best-effort tagging, ONE TAG AT A TIME.

    Called only AFTER the alias has already moved. A rejected tag must never
    turn a completed promotion into a failure - the serving change is real
    whether or not the audit tag stuck, and exiting here would leave the
    operator believing nothing happened. Failures are printed, not raised.
    """
    for key, value in tags.items():
        try:
            client.set_model_version_tag(name, version, key, str(value))
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️  Could not tag v{version} {key}={value} "
                  f"({type(exc).__name__}) - the alias DID move.")


def _utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def promote(client, name: str, alias: str, version: str) -> None:
    # Confirm the version exists before touching the alias.
    try:
        mv = client.get_model_version(name, version)
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"❌ No version {version} of '{name}' ({type(exc).__name__}).")

    # Confirm the artifact is actually retrievable. Promoting a dangling
    # registry row would point serving at something it cannot download, and the
    # failure would only surface on the next API restart.
    if mv.run_id:
        try:
            files = [a.path for a in client.list_artifacts(mv.run_id, "model")]
        except Exception as exc:  # noqa: BLE001
            sys.exit(f"❌ Could not list artifacts for run {mv.run_id[:8]} "
                     f"({type(exc).__name__}: {exc}).")
        if not files:
            sys.exit(f"❌ v{version} has no files under 'model/' - refusing to "
                     f"promote a version with no downloadable artifact.")
        print(f"✅ v{version} artifact present: {', '.join(files)}")
    else:
        print(f"⚠️  v{version} has no run_id - skipping the artifact check.")

    previous, outcome = _alias_state(client, name, alias)
    if outcome == "failed":
        # Moving it anyway would very likely SUCCEED, and that is the problem:
        # the outgoing version could not be named, so its demotion could never
        # be stamped, and MLflow keeps no history to recover the answer from.
        # A promotion is cheap to repeat and a hole in the audit trail is
        # permanent, so this refuses before touching anything.
        sys.exit(f"❌ Refusing to move '{alias}': the registry would not say "
                 f"which version holds it. Re-run when it answers.")
    client.set_registered_model_alias(name, alias, version)
    # Read back rather than trusting the write: an alias that silently did not
    # move would mean serving stays on the old model while we believe otherwise.
    now, readback = _alias_state(client, name, alias)
    # str() on BOTH sides: MLflow 3.14.0 returns ModelVersion.version as an int
    # against a local backend, while DagsHub's REST layer returns a string. A
    # bare `!=` therefore aborts here AFTER the alias has already moved,
    # telling the operator the promotion failed when it succeeded. Caught by
    # tests/test_promote.py running against a real local registry.
    if readback == "failed":
        # The write did not raise, so it most likely landed. Exiting here would
        # skip the tags below and leave exactly the hole this commit is about,
        # so say what is and is not known and carry on.
        print(f"⚠️  Could not read '{alias}' back after moving it. The write "
              f"did not raise, so it most likely succeeded - confirm with "
              f"--list once the registry answers.")
    elif str(now) != str(version):
        sys.exit(f"❌ Alias did not move (reads back as {now!r}).")
    unchanged = previous is not None and str(previous) == str(version)
    if unchanged:
        # Re-running the same promotion is harmless and sometimes deliberate
        # (re-stamping who and when), but printing "v1 -> v1" reads like a
        # move that did not happen.
        print(f"🏷️  '{alias}' was already on v{version} of '{name}'. "
              f"Re-stamped, nothing moved.")
    else:
        moved = f"v{previous} -> v{version}" if previous else f"-> v{version}"
        print(f"🏷️  '{alias}' {moved} on '{name}'.")

    # THE AUDIT TRAIL. Moving an alias is DESTRUCTIVE: MLflow removes it from
    # the old version and keeps no history, so without these tags "what was in
    # production last Tuesday, and who put it there?" is unanswerable. Verified
    # directly against MLflow 3.14.0: re-pointing an alias leaves the previous
    # version's alias list empty.
    when, who = _utc_now(), _actor()
    _tag(client, name, str(version), {
        "promoted_at": when,
        "promoted_by": who,
        # "v1" on v1 itself would be a lie about where it came from.
        "promoted_from": "(unchanged)" if unchanged
                         else (f"v{previous}" if previous else "none"),
        "promoted_alias": alias,
    })
    if previous and str(previous) != str(version):
        _tag(client, name, str(previous), {
            "demoted_at": when,
            "demoted_by": who,
            "demoted_to": f"v{version}",
        })

    if not unchanged:
        # Only worth saying when something actually changed. Both services
        # resolve once at startup and cache for the process lifetime.
        print("ℹ️  Restart the API process for the change to take effect "
              "(the served model is cached for the process lifetime).")


def demote(client, name: str, alias: str) -> None:
    losing, outcome = _alias_state(client, name, alias)
    if outcome == "failed":
        # Same reasoning as promote(): the deletion could not be recorded on
        # the version that loses the alias, and nothing else records it either.
        sys.exit(f"❌ Refusing to remove '{alias}': the registry would not say "
                 f"which version holds it. Re-run when it answers.")
    if losing is None:
        sys.exit(f"❌ Alias '{alias}' is not set on '{name}'.")
    client.delete_registered_model_alias(name, alias)
    print(f"🏷️  Removed alias '{alias}' from '{name}'. Serving falls back to "
          f"the newest version on the next API restart.")
    # Same reasoning as promote(): the deletion leaves no trace on the version,
    # so without this the record would show a promotion that never ended.
    _tag(client, name, str(losing), {
        "demoted_at": _utc_now(),
        "demoted_by": _actor(),
        "demoted_to": "(alias removed)",
    })


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Promote a registered model version to the serving alias.")
    ap.add_argument("--version", help="Version number to promote, e.g. 2.")
    ap.add_argument("--alias", default=config.PRODUCTION_ALIAS,
                    help=f"Alias to move (default: {config.PRODUCTION_ALIAS}).")
    ap.add_argument("--model", default=None,
                    help=f"Registered model (default: {config.REGISTERED_MODEL_NAME}; "
                         f"--list with no --model shows BOTH modalities).")
    ap.add_argument("--list", action="store_true", help="List versions and exit.")
    ap.add_argument("--demote", action="store_true", help="Remove the alias.")
    args = ap.parse_args()

    client = _client()
    # --list defaulted to the IMAGE model, so "promote.py --list" answered
    # "what is promoted?" for half the system and looked like the whole
    # answer. Listing is read-only, so showing both costs nothing. A write
    # still defaults to one model rather than guessing.
    default_model = config.REGISTERED_MODEL_NAME
    if args.list:
        names = [args.model] if args.model else [
            default_model, text_config.REGISTERED_MODEL_NAME]
        for index, name in enumerate(names):
            if index:
                print()
            list_versions(client, name, args.alias)
    elif args.demote:
        demote(client, args.model or default_model, args.alias)
    elif args.version:
        promote(client, args.model or default_model, args.alias, str(args.version))
    else:
        ap.error("give one of --list, --version N, or --demote")


if __name__ == "__main__":
    main()
