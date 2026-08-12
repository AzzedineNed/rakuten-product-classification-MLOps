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
torch, or sklearn — which is exactly the contract of this package.

WHAT THIS MODULE DELIBERATELY DOES NOT DO: download artifacts or load a model.
Those differ per modality (the image side is one joblib payload, the text side
is a directory of two .pkl files), and mixing the selection rule with the
loading mechanics is what would make this untestable without a network.
"""
from __future__ import annotations

from typing import Optional

from rakuten_img import config

__all__ = ["resolve_registry_version"]


def resolve_registry_version(client, name: str, alias: Optional[str] = None):
    """Pick which registered version to serve: the one carrying `alias` if that
    alias resolves, otherwise the highest version number.

    Returns (version_object, source_string). Raises LookupError only when the
    registered model has no versions at all — the CALLER decides what to do
    then (both services fall back to a local artifact so serving never
    hard-fails).

    The source string is meant to be surfaced by /health and logged at load
    time, so an operator can tell a deliberate promotion
    ("registry:name@production/v2") from a silent fallback
    ("registry:name/v6 (unpromoted)") at a glance.
    """
    alias = alias or config.PRODUCTION_ALIAS
    try:
        mv = client.get_model_version_by_alias(name, alias)
        return mv, f"registry:{name}@{alias}/v{mv.version}"
    except Exception as exc:  # noqa: BLE001
        # No alias set yet (fresh registry), a typo'd alias, or a backend that
        # does not implement aliases. All three mean the same thing here: fall
        # back to newest-version behaviour rather than failing to serve.
        print(f"ℹ️  No '{alias}' alias on '{name}' ({type(exc).__name__}) — "
              f"falling back to the newest version.")
    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        raise LookupError(f"No versions of '{name}' in the MLflow registry.")
    latest = max(versions, key=lambda v: int(v.version))
    return latest, f"registry:{name}/v{latest.version} (unpromoted)"
