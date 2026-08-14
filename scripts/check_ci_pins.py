#!/usr/bin/env python
"""Fail if requirements-ci.txt has drifted from requirements.txt.

requirements-ci.txt is a deliberate subset of requirements.txt: CI installs it
instead of the full file so the test job does not pull mlflow / dagshub / dvc /
the API stack, none of which the test suite imports.

A subset is only trustworthy if it cannot drift. If somebody bumps a version in
requirements.txt and forgets the CI file, CI would go on testing a version the
project no longer installs - a green build that means nothing. This script makes
that failure loud and immediate.

Checks:
  1. every package pinned in requirements-ci.txt also exists in requirements.txt
  2. the two versions are identical

It deliberately does NOT require the reverse: requirements.txt is allowed to pin
packages that CI skips. That is the whole point of the subset.

Usage:  python scripts/check_ci_pins.py
Exit:   0 = consistent, 1 = drift (message on stdout)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FULL_PATH = ROOT / "requirements.txt"
CI_PATH = ROOT / "requirements-ci.txt"

# name, optional [extras], ==, version   e.g. uvicorn[standard]==0.30.1
PIN_RE = re.compile(r"^([A-Za-z0-9_.\-]+)(\[[^\]]*\])?==([^\s#]+)")


def normalise(name: str) -> str:
    """PEP 503 style: case-insensitive, - and _ and . are equivalent."""
    return re.sub(r"[-_.]+", "-", name).lower()


def parse(path: Path) -> dict[str, str]:
    if not path.exists():
        print(f"ERROR: {path} does not exist")
        sys.exit(1)

    pins: dict[str, str] = {}
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = PIN_RE.match(line)
        if match is None:
            print(f"ERROR: {path.name}:{lineno}: not a '==' pin: {raw.strip()!r}")
            sys.exit(1)
        pins[normalise(match.group(1))] = match.group(3)
    return pins


def main() -> int:
    full = parse(FULL_PATH)
    ci = parse(CI_PATH)

    problems: list[str] = []
    for name, version in sorted(ci.items()):
        if name not in full:
            problems.append(
                f"  {name}=={version} is pinned in requirements-ci.txt "
                f"but absent from requirements.txt"
            )
        elif full[name] != version:
            problems.append(
                f"  {name}: requirements-ci.txt pins {version}, "
                f"requirements.txt pins {full[name]}"
            )

    if problems:
        print("CI PIN DRIFT - requirements-ci.txt disagrees with requirements.txt:")
        print("\n".join(problems))
        print(
            "\nFix: make the two files agree. requirements.txt is the source of "
            "truth for versions; requirements-ci.txt only chooses a subset."
        )
        return 1

    skipped = sorted(set(full) - set(ci))
    print(f"OK: {len(ci)} CI pins agree with requirements.txt ({len(full)} pins total).")
    if skipped:
        print(f"Not installed in CI ({len(skipped)}): {', '.join(skipped)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
