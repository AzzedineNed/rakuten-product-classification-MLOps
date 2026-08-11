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
    "attach_evaluation",
]


def enabled() -> bool:
    """Gate on the tracking URI, so a fresh clone / CI / offline work trains
    normally with no tracking side effects."""
    return bool(os.getenv("MLFLOW_TRACKING_URI"))


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
