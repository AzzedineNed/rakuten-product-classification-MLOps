"""MLflow tracking and Model Registry for the TEXT modality.

Mirrors the image pipeline's contract deliberately, so both modalities behave
identically from an operator's point of view:

  * BEST-EFFORT ALWAYS. If MLFLOW_TRACKING_URI is unset, unreachable, or the
    credentials are wrong, training still finishes and the model is still saved
    to disk. Nothing in here may raise into the pipeline.
  * TRAINING NEVER PROMOTES. A run registers a new version and stops. Serving
    follows the production alias, which only a human moves (scripts/promote.py).
    Tags are descriptive -- they say what a version IS so a human can compare
    candidates; they confer no privilege.
  * ONE RUN PER MODEL. train.py opens the run and stores its id; evaluate.py
    reopens the SAME run to attach test metrics and plots, so a version's full
    record lives in one place.

Text registers under its own model name and its own experiment, so image, text
and (later) fusion runs share a tracking server without colliding.

ONE DELIBERATE EXCEPTION to "nothing in here may raise": load_from_registry.
It is a SERVING-side loader, not a training step, and it must be able to say
"I could not get the registered model" so its caller can fall back to the local
artifact. Mirrors rakuten_img.classifier.load_from_registry, which raises for
exactly the same reason.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

from . import config

__all__ = [
    "enabled",
    "log_training_run",
    "register_model",
    "load_from_registry",
    "attach_evaluation",
]


def enabled() -> bool:
    """Gate on the tracking URI, so a fresh clone / CI / offline work trains
    normally with no tracking side effects."""
    return bool(os.getenv("MLFLOW_TRACKING_URI"))


# How long a SERVING process may block on an unreachable registry before giving
# up and using the local artifacts. MLflow's shipped defaults are tuned for
# batch jobs riding out rate limits: 7 retries with an exponential backoff and a
# 120s per-request timeout, which its own source comments as "~4 minutes" — and
# resolving an alias costs TWO requests when the alias lookup fails, so a dead
# tracking server can block a container's startup for the better part of ten
# minutes. That turns "degrade to the local model" into a boot hang, and with
# docker-compose's `depends_on: service_healthy` it would hold the whole stack
# down. These are setdefault, not assignment: an operator can still raise them.
_SERVING_HTTP_LIMITS = {
    "MLFLOW_HTTP_REQUEST_MAX_RETRIES": "2",
    "MLFLOW_HTTP_REQUEST_TIMEOUT": "10",
}


def _set_experiment(mlflow) -> None:
    mlflow.set_experiment(config.EXPERIMENT_NAME)


def log_training_run(params: dict, metrics: dict) -> Optional[str]:
    """Log params + train/val metrics. Returns the run_id, or None. Never raises."""
    if not enabled():
        print("ℹ️  MLFLOW_TRACKING_URI not set — skipping experiment logging.")
        return None
    try:
        import mlflow

        _set_experiment(mlflow)
        with mlflow.start_run() as run:
            mlflow.set_tag("modality", "text")
            mlflow.set_tag("stage", "train")
            mlflow.log_params(params)
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()
                                if isinstance(v, (int, float))})
            run_id = run.info.run_id
        print(f"📝 Logged training run to MLflow (run_id={run_id[:8]}…).")
        return run_id
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️  MLflow logging skipped ({type(exc).__name__}: {exc}).")
        return None


def register_model(run_id: Optional[str],
                   artifacts: Iterable[Path],
                   tags: Optional[dict] = None) -> Optional[str]:
    """Attach the artifacts to `run_id` and register a new model version.

    The text model is TWO files (vectorizer + estimator) rather than the image
    side's single joblib payload, so both are logged under the 'model/'
    artifact path and the version's source is that DIRECTORY. Anything loading
    this version needs both files.

    Returns the new version string, or None. Never raises into training.
    """
    if not enabled():
        return None
    if not run_id:
        print("ℹ️  No MLflow run_id — skipping model registration.")
        return None
    try:
        from mlflow import MlflowClient
        from mlflow.exceptions import MlflowException

        client = MlflowClient()

        for path in artifacts:
            client.log_artifact(run_id, str(path), artifact_path="model")

        name = config.REGISTERED_MODEL_NAME
        try:
            client.create_registered_model(
                name,
                description="Rakuten TEXT modality classifier "
                            "(TF-IDF + LogisticRegression). Artifacts: "
                            "tfidf_vectorizer.pkl + logistic_regression.pkl. "
                            "Probability vectors follow CANONICAL_CLASSES.",
            )
        except MlflowException:
            pass  # already exists

        source = f"{client.get_run(run_id).info.artifact_uri}/model"
        mv = client.create_model_version(name=name, source=source, run_id=run_id)
        print(f"📦 Registered '{name}' version {mv.version} (run {run_id[:8]}…).")

        # One tag at a time: the version already exists and a rejected tag must
        # not turn a success into a failure. (Same approach the image side uses
        # because it is what DagsHub was verified to accept.)
        for key, value in (tags or {}).items():
            try:
                client.set_model_version_tag(name, mv.version, key, str(value))
            except Exception as exc:  # noqa: BLE001
                print(f"⚠️  Could not tag {key}={value} ({type(exc).__name__}).")
        return str(mv.version)
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️  Model registration skipped ({type(exc).__name__}: {exc}).")
        return None


def load_from_registry(alias: Optional[str] = None):
    """Download the registered version that should serve traffic and load BOTH
    artifacts from it.

    Returns (vectorizer, model, source_string), where source_string is the one
    produced by rakuten_common.registry — "registry:NAME@alias/vN" for a
    deliberate promotion, "registry:NAME/vN (unpromoted)" for a fallback to the
    newest version — so /health can show an operator which it is.

    RAISES on every failure (no tracking URI, no versions, network down, a
    version whose artifacts are the wrong shape). The CALLER decides the
    fallback; see TfidfPredictor.load(prefer_registry=True), which falls back to
    the local .pkl files so serving never hard-fails.

    WHY THIS IS NOT rakuten_img.classifier.load_from_registry: a text version's
    source is a DIRECTORY holding tfidf_vectorizer.pkl + logistic_regression.pkl
    (see register_model above), not a single joblib payload. Verified against
    real MLflow 3.14.0: download_artifacts() on such a source returns a
    directory containing both files.
    """
    if not enabled():
        raise RuntimeError("MLFLOW_TRACKING_URI not set — registry unavailable.")

    # Must happen BEFORE the mlflow import/first request; see _SERVING_HTTP_LIMITS.
    for key, value in _SERVING_HTTP_LIMITS.items():
        os.environ.setdefault(key, value)

    import joblib
    import mlflow
    from mlflow import MlflowClient

    from rakuten_common.registry import resolve_registry_version

    name = config.REGISTERED_MODEL_NAME
    client = MlflowClient()
    version, source = resolve_registry_version(client, name, alias)

    local_dir = Path(mlflow.artifacts.download_artifacts(version.source))
    if not local_dir.is_dir():
        raise NotADirectoryError(
            f"Version {version.version} of '{name}' points at {local_dir}, which "
            "is not a directory. A text version must be the 'model/' directory "
            "holding both .pkl files."
        )

    # Filenames come from config, so a rename cannot silently desynchronise the
    # writer (train.py) from this reader.
    vectorizer_file = local_dir / config.VECTORIZER_PATH.name
    model_file = local_dir / config.MODEL_PATH.name
    missing = [p.name for p in (vectorizer_file, model_file) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Version {version.version} of '{name}' is missing {missing} in "
            f"{local_dir} (found: {sorted(p.name for p in local_dir.iterdir())}). "
            "A text version needs BOTH files."
        )

    vectorizer = joblib.load(vectorizer_file)
    model = joblib.load(model_file)
    print(f"📦 Loaded '{name}' v{version.version} from the MLflow registry.")
    return vectorizer, model, source


def attach_evaluation(run_id: Optional[str], metrics: dict,
                      artifacts: Optional[dict] = None) -> None:
    """Attach test metrics and report artifacts to the training run.

    Falls back to a standalone run when no run_id is available, so an
    evaluation is never silently unrecorded. `artifacts` maps an mlflow
    artifact_path to a local file. Never raises.
    """
    if not enabled():
        return
    try:
        import mlflow

        _set_experiment(mlflow)
        linked = bool(run_id)
        ctx = mlflow.start_run(run_id=run_id) if linked else mlflow.start_run()
        with ctx:
            mlflow.set_tag("modality", "text")
            if not linked:
                # Only tag stage on a standalone run. When reopening the
                # training run we leave stage="train" intact -- overwriting it
                # would erase how the run started. Mirrors scripts/evaluate.py.
                mlflow.set_tag("stage", "evaluate-standalone")
            numeric = {f"test_{k}": float(v) for k, v in metrics.items()
                       if isinstance(v, (int, float))}
            mlflow.log_metrics(numeric)
            for artifact_path, local in (artifacts or {}).items():
                p = Path(local)
                if p.exists():
                    mlflow.log_artifact(str(p), artifact_path=artifact_path)
        if linked:
            print(f"📝 Attached test metrics to the training run ({run_id[:8]}…).")
        else:
            print("📝 Logged a standalone evaluation run (no training run_id found).")
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️  MLflow evaluation logging skipped ({type(exc).__name__}: {exc}).")
