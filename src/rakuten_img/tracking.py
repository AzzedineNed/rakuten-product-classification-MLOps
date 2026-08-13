"""MLflow tracking and Model Registry for the IMAGE modality.

The image counterpart of rakuten_text.tracking, with the same five entry points
and the same contract, so both modalities behave identically from an operator's
point of view:

  * BEST-EFFORT ALWAYS. If MLFLOW_TRACKING_URI is unset, unreachable, or the
    credentials are wrong, training still finishes and the model is still saved
    to disk. Nothing in here may raise into the pipeline.
  * TRAINING NEVER PROMOTES. A run registers a new version and stops. Serving
    follows the production alias, which only a human moves (scripts/promote.py).
    Tags are descriptive -- they say what a version IS so a human can compare
    candidates; they confer no privilege.
  * ONE RUN PER MODEL. train.py opens the run and stores its id in the model
    payload; evaluate.py reopens the SAME run to attach test metrics and plots,
    so a version's full record lives in one place.

ONE DELIBERATE EXCEPTION to "nothing in here may raise": load_from_registry.
It is a SERVING-side loader, not a training step, and it must be able to say
"I could not get the registered model" so its caller can fall back to the local
artifact. rakuten_text.tracking.load_from_registry raises for the same reason.

WHERE THIS CODE CAME FROM. Nothing here is new. register_model and
load_from_registry were rakuten_img.classifier.register_in_mlflow and
.load_from_registry; log_training_run and attach_evaluation were the private
_log_to_mlflow functions inlined in scripts/train.py and scripts/evaluate.py.
Gathering them here is what the README's layout convention asks for -- packages
hold the logic, scripts/ holds thin entrypoints, one tracking.py per modality --
and it is what makes the image side's MLflow behaviour readable in one place
instead of three. tests/test_img_registry.py was written against the previous
locations FIRST, so the move could be proved to change nothing.

WHY THE IMAGE LOADER IS NOT THE TEXT LOADER: an image version's source is a
single joblib payload FILE holding the estimator, its class order and its
metadata. A text version's source is a DIRECTORY of two .pkl files. The
selection rule they share lives in rakuten_common.registry; only the download
and load differ, and that is the difference these two modules exist to hold.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional, Tuple

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
# up and using the local artifact. MLflow's shipped defaults are tuned for batch
# jobs riding out rate limits: 7 retries with an exponential backoff and a 120s
# per-request timeout, which its own source comments as "~4 minutes" — and
# resolving an alias costs TWO requests when the alias lookup fails, so a dead
# tracking server can block for the better part of ten minutes. On the image
# side that lands on the FIRST /predict rather than at startup (the payload
# resolves lazily), so the cost is a request that appears to hang instead of a
# boot that appears to hang — the same outage, worn differently. These are
# setdefault, not assignment: an operator can still raise them.
#
# Copied deliberately from rakuten_text.tracking, which measured the numbers.
_SERVING_HTTP_LIMITS = {
    "MLFLOW_HTTP_REQUEST_MAX_RETRIES": "2",
    "MLFLOW_HTTP_REQUEST_TIMEOUT": "10",
}


def _set_experiment(mlflow) -> None:
    mlflow.set_experiment(config.EXPERIMENT_NAME)


def log_training_run(params: dict, metrics: dict) -> Optional[str]:
    """Log params + train/val metrics. Returns the run_id, or None. Never raises.

    The run_id is the thread that ties everything together: train.py persists it
    into the model payload, register_model attaches the artifact to it, and
    attach_evaluation reopens it. Losing it costs the link, not the model.
    """
    if not enabled():
        print("ℹ️  MLFLOW_TRACKING_URI not set — skipping experiment logging.")
        return None
    try:
        import mlflow

        _set_experiment(mlflow)
        with mlflow.start_run() as run:
            mlflow.set_tag("modality", "image")
            mlflow.set_tag("stage", "train")
            mlflow.log_params(params)
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
            run_id = run.info.run_id
        print(f"📝 Logged training run to MLflow (run_id={run_id[:8]}…).")
        return run_id
    except Exception as exc:  # noqa: BLE001
        # A tracking outage, bad creds, or missing mlflow must not lose a model
        # that will still be saved to disk. Warn and carry on.
        print(f"⚠️  MLflow logging skipped ({type(exc).__name__}: {exc}).")
        return None


def register_model(run_id: Optional[str],
                   path: Path = config.CLASSIFIER_PATH,
                   tags: Optional[dict] = None) -> Optional[str]:
    """Attach the saved model file to `run_id` (under artifact path 'model/')
    and register it as a new version of config.REGISTERED_MODEL_NAME.

    Returns the new version string, or None if tracking is off, there is no run
    to attach to, or anything fails. Never raises into the training path.

    `tags` are DESCRIPTIVE metadata attached to the new version (val F1,
    classifier type, host vs container, ...) so a human comparing versions in
    the registry UI can tell them apart. They confer no privilege: nothing is
    served because of a tag. Promotion is the separate, explicit alias step in
    scripts/promote.py. Tagging failures are logged and ignored — a tag is not
    worth losing a registered version over.
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
        # str() because MLflow 3.14.0 returns an INT against a local backend and
        # a STRING through DagsHub's REST layer. Callers compare, so normalise.
        return str(mv.version)
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️  Model registration skipped ({type(exc).__name__}: {exc}).")
        return None


def load_from_registry(alias: Optional[str] = None) -> dict:
    """Download and load the registered model version that should serve traffic:
    the one aliased `config.PRODUCTION_ALIAS`, else the newest version.

    Raises on any failure (no tracking URI, no registered versions, network
    down, bad artifact) — the CALLER decides the fallback; see
    scripts/predict.py, which falls back to the local .joblib so serving never
    hard-fails. Returns the usual payload dict plus a 'serving_source' key,
    which /health surfaces so an operator can tell a deliberate promotion
    ("registry:name@production/v2") from a fallback ("registry:name/v6
    (unpromoted)") at a glance.
    """
    if not enabled():
        raise RuntimeError("MLFLOW_TRACKING_URI not set — registry unavailable.")

    # Must happen BEFORE the mlflow import / first request; see
    # _SERVING_HTTP_LIMITS. MLflow reads these per request, so setting them here
    # binds even in a process that imported mlflow earlier.
    for key, value in _SERVING_HTTP_LIMITS.items():
        os.environ.setdefault(key, value)

    import joblib
    import mlflow
    from mlflow import MlflowClient

    from rakuten_common.registry import resolve_registry_version

    name = config.REGISTERED_MODEL_NAME
    client = MlflowClient()
    version, source = resolve_registry_version(client, name, alias)
    local_path = mlflow.artifacts.download_artifacts(version.source)
    payload = joblib.load(local_path)
    payload["serving_source"] = source
    print(f"📦 Loaded '{name}' v{version.version} from the MLflow registry.")
    return payload


def attach_evaluation(run_id: Optional[str], metrics: dict,
                      artifacts: Optional[Iterable[Tuple[str, Path]]] = None) -> None:
    """Attach test metrics and report artifacts to the training run.

    Falls back to a standalone run when no run_id is available, so an evaluation
    is never silently unrecorded. Never raises.

    `artifacts` is a sequence of (mlflow_artifact_path, local_file) PAIRS rather
    than the dict rakuten_text.tracking uses, because the image side logs two
    different files — the classification report and metrics.json — under the
    same 'reports' path, which a dict keyed by artifact path cannot express.
    Missing files are skipped rather than raised on.
    """
    if not enabled():
        print("ℹ️  MLFLOW_TRACKING_URI not set — skipping experiment logging.")
        return
    try:
        import mlflow

        _set_experiment(mlflow)
        linked = bool(run_id)
        ctx = mlflow.start_run(run_id=run_id) if linked else mlflow.start_run()
        with ctx:
            mlflow.set_tag("modality", "image")
            if not linked:
                # Only tag stage on a standalone run. When reopening the
                # training run we leave stage="train" intact — overwriting it
                # would erase how the run started.
                mlflow.set_tag("stage", "evaluate-standalone")
            # Log only numeric metrics (skip the string entries backbone /
            # classifier_type that evaluate.py carries in the same dict).
            numeric = {k: float(v) for k, v in metrics.items()
                       if isinstance(v, (int, float)) and not isinstance(v, bool)}
            mlflow.log_metrics(numeric)
            for artifact_path, local in (artifacts or ()):
                p = Path(local)
                if p.exists():
                    mlflow.log_artifact(str(p), artifact_path=artifact_path)
        if linked:
            print(f"📝 Attached test metrics + confusion matrix to training run "
                  f"({run_id[:8]}…).")
        else:
            print("📝 Logged a standalone evaluation run (no training run_id found).")
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️  MLflow logging skipped ({type(exc).__name__}: {exc}).")
