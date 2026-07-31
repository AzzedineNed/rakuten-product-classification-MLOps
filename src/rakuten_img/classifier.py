"""The small classifier that sits on top of the cached MobileNetV2 features.

Default is a one-hidden-layer MLP (good accuracy/speed balance, exposes
predict_proba for fusion); logistic regression is a config switch away.

We persist the fitted estimator together with its class order and backbone name
so predict.py / the API can verify alignment with config.CANONICAL_CLASSES
instead of trusting it implicitly.

Model Registry: register_in_mlflow() publishes a saved model file as a new
version of the registered model (config.REGISTERED_MODEL_NAME) on the MLflow
server; load_from_registry() pulls the newest version back. The registered
artifact is the SAME joblib payload as the local file, so serving code handles
both sources identically.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from . import config


def build_classifier(kind: Optional[str] = None):
    """Instantiate the configured classifier head."""
    kind = (kind or config.CLASSIFIER_TYPE).lower()
    if kind == "logreg":
        return LogisticRegression(
            C=1.0,
            max_iter=1000,
            n_jobs=-1,
            multi_class="multinomial",
            random_state=config.RANDOM_STATE,
        )
    if kind == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=config.MLP_HIDDEN,
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=256,
            max_iter=60,
            early_stopping=True,
            n_iter_no_change=6,
            random_state=config.RANDOM_STATE,
            verbose=True,
        )
    raise ValueError(f"Unknown classifier kind: {kind!r} (use 'mlp' or 'logreg')")


def save(clf, path: Path = config.CLASSIFIER_PATH, extra: Optional[dict] = None,
         run_id: Optional[str] = None) -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "classifier": clf,
        "classes": [int(c) for c in clf.classes_],
        "backbone": config.BACKBONE_NAME,
        "feature_dim": config.FEATURE_DIM,
        "classifier_type": config.CLASSIFIER_TYPE,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "mlflow_run_id": run_id,
        "extra": extra or {},
    }
    joblib.dump(payload, path)
    print(f"💾 Saved classifier -> {path}")


def load(path: Path = config.CLASSIFIER_PATH) -> dict:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"No trained classifier at {path}. Run train.py (or POST /train) first."
        )
    return joblib.load(path)


def register_in_mlflow(run_id: Optional[str],
                       path: Path = config.CLASSIFIER_PATH,
                       tags: Optional[dict] = None) -> Optional[str]:
    """Best-effort: attach the saved model file to the given MLflow run (under
    artifact path 'model/') and register it as a new version of
    config.REGISTERED_MODEL_NAME. Returns the new version string, or None if
    tracking is off, there is no run to attach to, or anything fails. Never
    raises into the training path.

    `tags` are DESCRIPTIVE metadata attached to the new version (val F1,
    classifier type, host vs container, ...) so a human comparing versions in
    the registry UI can tell them apart. They confer no privilege: nothing is
    served because of a tag. Promotion is the separate, explicit alias step in
    scripts/promote.py. Tagging failures are logged and ignored — a tag is not
    worth losing a registered version over.
    """
    if not os.getenv("MLFLOW_TRACKING_URI"):
        return None
    if not run_id:
        print("ℹ️  No MLflow run_id — skipping model registration.")
        return None
    try:
        from mlflow import MlflowClient
        from mlflow.exceptions import MlflowException

        client = MlflowClient()
        # 1) The model file becomes an artifact of the training run.
        client.log_artifact(run_id, str(path), artifact_path="model")
        # 2) Ensure the registered model exists (idempotent).
        name = config.REGISTERED_MODEL_NAME
        try:
            client.create_registered_model(
                name, description="Rakuten IMAGE modality classifier "
                                  "(frozen MobileNetV2 features + sklearn head). "
                                  "Artifact is a joblib payload dict.")
        except MlflowException:
            pass  # already exists
        # 3) New version pointing at that artifact.
        source = f"{client.get_run(run_id).info.artifact_uri}/model/{path.name}"
        mv = client.create_model_version(name=name, source=source, run_id=run_id)
        print(f"📦 Registered '{name}' version {mv.version} (run {run_id[:8]}…).")
        # 4) Descriptive tags, set one at a time (the call we verified works on
        #    DagsHub). Each is independently best-effort: the version is already
        #    created and a missing tag must not turn success into failure.
        for key, value in (tags or {}).items():
            try:
                client.set_model_version_tag(name, mv.version, key, str(value))
            except Exception as exc:  # noqa: BLE001
                print(f"⚠️  Could not tag {key}={value} ({type(exc).__name__}).")
        return str(mv.version)
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️  Model registration skipped ({type(exc).__name__}: {exc}).")
        return None


def resolve_registry_version(client, name: str, alias: Optional[str] = None):
    """Pick which registered version to serve: the one carrying `alias` if that
    alias resolves, otherwise the highest version number.

    Returns (version_object, source_string). Raises LookupError only when the
    registered model has no versions at all.

    Split out from load_from_registry() so the selection rule — the part with
    the actual decision in it — is testable without a network, a real MLflow
    server, or a 4 MB artifact download.
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


def load_from_registry(alias: Optional[str] = None) -> dict:
    """Download and load the registered model version that should serve traffic:
    the one aliased `config.PRODUCTION_ALIAS`, else the newest version.

    Raises on any failure (no tracking URI, no registered versions, network
    down, bad artifact) — the CALLER decides the fallback; see predict.py,
    which falls back to the local .joblib so serving never hard-fails.
    Returns the usual payload dict plus a 'serving_source' key for visibility.
    """
    if not os.getenv("MLFLOW_TRACKING_URI"):
        raise RuntimeError("MLFLOW_TRACKING_URI not set — registry unavailable.")
    import mlflow
    from mlflow import MlflowClient

    name = config.REGISTERED_MODEL_NAME
    client = MlflowClient()
    version, source = resolve_registry_version(client, name, alias)
    local_path = mlflow.artifacts.download_artifacts(version.source)
    payload = joblib.load(local_path)
    payload["serving_source"] = source
    print(f"📦 Loaded '{name}' v{version.version} from the MLflow registry.")
    return payload


def reorder_to_canonical(proba: np.ndarray, model_classes) -> np.ndarray:
    """Reorder a predict_proba row/matrix so columns follow CANONICAL_CLASSES.

    With integer prdtypecode labels, sklearn already sorts classes numerically
    (== canonical), so this is usually an identity reorder — but we do it
    explicitly so the contract holds even if the label set ever changes.
    """
    index = {int(c): i for i, c in enumerate(model_classes)}
    cols = [index[c] for c in config.CANONICAL_CLASSES]
    return proba[..., cols]