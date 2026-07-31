#!/usr/bin/env python3
"""promote.py — Decide which registered model version serves traffic.

Training registers a new version every time it runs and tags it with what it
is (val F1, classifier type, host/container). It deliberately does NOT promote
anything: if "newest" automatically meant "served", a bad or accidental run
would silently take over serving. Promotion is this script — an explicit,
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

import _bootstrap  # noqa: F401

from rakuten_img import config


def _client():
    if not os.getenv("MLFLOW_TRACKING_URI"):
        sys.exit("❌ MLFLOW_TRACKING_URI is not set — nothing to promote against.")
    from mlflow import MlflowClient
    return MlflowClient()


def _current_alias_version(client, name: str, alias: str):
    """Version string currently carrying `alias`, or None if unset/unsupported."""
    try:
        return client.get_model_version_by_alias(name, alias).version
    except Exception:  # noqa: BLE001
        return None


def list_versions(client, name: str, alias: str) -> None:
    versions = sorted(client.search_model_versions(f"name='{name}'"),
                      key=lambda v: int(v.version))
    if not versions:
        print(f"(no versions of '{name}' registered yet)")
        return
    promoted = _current_alias_version(client, name, alias)
    print(f"Registered model: {name}   alias '{alias}' -> "
          f"{'v' + promoted if promoted else '(unset)'}\n")
    for v in versions:
        marker = "  <-- SERVING" if v.version == promoted else ""
        tags = ", ".join(f"{k}={val}" for k, val in sorted(v.tags.items())) or "-"
        print(f"  v{v.version}  run={str(v.run_id)[:8]}  {tags}{marker}")


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
            sys.exit(f"❌ v{version} has no files under 'model/' — refusing to "
                     f"promote a version with no downloadable artifact.")
        print(f"✅ v{version} artifact present: {', '.join(files)}")
    else:
        print(f"⚠️  v{version} has no run_id — skipping the artifact check.")

    previous = _current_alias_version(client, name, alias)
    client.set_registered_model_alias(name, alias, version)
    # Read back rather than trusting the write: an alias that silently did not
    # move would mean serving stays on the old model while we believe otherwise.
    now = _current_alias_version(client, name, alias)
    if now != str(version):
        sys.exit(f"❌ Alias did not move (reads back as {now!r}).")
    moved = f"v{previous} -> v{version}" if previous else f"-> v{version}"
    print(f"🏷️  '{alias}' {moved} on '{name}'.")
    print("ℹ️  Restart the API process for the change to take effect "
          "(the served model is cached for the process lifetime).")


def demote(client, name: str, alias: str) -> None:
    if _current_alias_version(client, name, alias) is None:
        sys.exit(f"❌ Alias '{alias}' is not set on '{name}'.")
    client.delete_registered_model_alias(name, alias)
    print(f"🏷️  Removed alias '{alias}' from '{name}'. Serving falls back to "
          f"the newest version on the next API restart.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Promote a registered model version to the serving alias.")
    ap.add_argument("--version", help="Version number to promote, e.g. 2.")
    ap.add_argument("--alias", default=config.PRODUCTION_ALIAS,
                    help=f"Alias to move (default: {config.PRODUCTION_ALIAS}).")
    ap.add_argument("--model", default=config.REGISTERED_MODEL_NAME,
                    help=f"Registered model (default: {config.REGISTERED_MODEL_NAME}).")
    ap.add_argument("--list", action="store_true", help="List versions and exit.")
    ap.add_argument("--demote", action="store_true", help="Remove the alias.")
    args = ap.parse_args()

    client = _client()
    if args.list:
        list_versions(client, args.model, args.alias)
    elif args.demote:
        demote(client, args.model, args.alias)
    elif args.version:
        promote(client, args.model, args.alias, str(args.version))
    else:
        ap.error("give one of --list, --version N, or --demote")


if __name__ == "__main__":
    main()