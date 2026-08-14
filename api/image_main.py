#!/usr/bin/env python3
"""FastAPI service exposing the IMAGE model.

Endpoints:
  GET  /health          liveness + whether a model is loaded
  POST /predict         multipart image upload -> top-k + full canonical proba
  POST /train           retrain the classifier head from cached features
                        (background task; optional ?reprocess=true to re-extract
                        features first — the heavy step)
  GET  /train/status    status + metrics of the last /train job
  POST /evaluate        score the current local model on the cached test
                        features and write reports/ (background task)
  GET  /evaluate/status status + metrics of the last /evaluate job

/train and /evaluate are both long-running background jobs with the same
polling contract, so an orchestrator (Airflow) drives them the same way. They
are mutually exclusive: /train rewrites the classifier .joblib that /evaluate
reads, so starting either while the other runs would score a half-written file.
Whichever is asked for second gets a 409.

Run:
  uvicorn api.image_main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

# Make the src package importable without install / PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fastapi import BackgroundTasks, FastAPI, File, Query, Response, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from rakuten_common.observability import PrometheusMiddleware, ServiceMetrics
from rakuten_img import config

app = FastAPI(title="Rakuten Image Classifier", version="1.0.0")

METRICS = ServiceMetrics("image")
app.add_middleware(PrometheusMiddleware, metrics=METRICS)

_TRAIN_STATUS: dict = {"state": "idle", "detail": None, "metrics": None}
_EVAL_STATUS: dict = {"state": "idle", "detail": None, "metrics": None}


def _busy() -> str | None:
    """Return the name of the job currently running, or None.

    Single point of mutual exclusion for the two long jobs. Both mutate or read
    config.CLASSIFIER_PATH, so they must never overlap.
    """
    if _TRAIN_STATUS["state"] == "running":
        return "train"
    if _EVAL_STATUS["state"] == "running":
        return "evaluate"
    return None


@app.get("/metrics")
def metrics():
    """Prometheus exposition. Deliberately NOT routed through nginx: it is
    scraped over the compose network, so it is not reachable from outside."""
    body, content_type = METRICS.render()
    return Response(content=body, media_type=content_type)


@app.get("/health")
def health():
    import predict as predict_script  # scripts/predict.py

    model_present = config.CLASSIFIER_PATH.exists()
    source = predict_script.model_source()
    # Published here as well as after a prediction because this service
    # resolves its model lazily (model_source() reads "not-loaded" until the
    # first /predict — a documented wart). Whichever happens first, the gauge
    # stops being empty.
    METRICS.set_model_info("image", source)
    return {"status": "ok", "model_loaded": model_present,
            "model_source": source,
            "backbone": config.BACKBONE_NAME, "num_classes": config.NUM_CLASSES}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...), top_k: int = Query(5, ge=1, le=27)):
    """Predict a product type code from an uploaded product image."""
    import predict as predict_script  # scripts/predict.py

    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(status_code=400, content={"error": f"Invalid image: {exc}"})

    proba = predict_script.predict_proba(img)
    order = proba.argsort()[::-1][:top_k]
    top = [
        {"code": int(config.CANONICAL_CLASSES[i]),
         "label": config.CANONICAL_LABELS[i],
         "probability": float(proba[i])}
        for i in order
    ]
    METRICS.observe_prediction(top[0]["code"], top[0]["probability"])
    # The model is only actually resolved on the first /predict, so this is the
    # earliest point at which model_source() is truthful for this service.
    METRICS.set_model_info("image", predict_script.model_source())

    # Full vector + class order so a fusion layer can consume it directly.
    return {
        "top_k": top,
        "prediction": top[0],
        "canonical_classes": config.CANONICAL_CLASSES,
        "probabilities": [float(p) for p in proba],
    }


def _run_training(reprocess: bool) -> None:
    _TRAIN_STATUS.update(state="running", detail="reprocess" if reprocess else "classifier-only",
                         metrics=None)
    try:
        if reprocess:
            import process as process_script
            # Call process() (the library function), NOT main(): main() runs
            # argparse on sys.argv, which inside uvicorn holds the SERVER's
            # arguments -> argparse raises SystemExit, which `except Exception`
            # does not catch, killing the task and wedging _TRAIN_STATUS on
            # "running" forever.
            process_script.process()  # re-extract features (heavy)
        import train as train_script
        metrics = train_script.train()
        _TRAIN_STATUS.update(state="done", metrics=metrics, detail="completed")
    except (Exception, SystemExit) as exc:  # noqa: BLE001
        # SystemExit is caught EXPLICITLY because it does not derive from
        # Exception. Calling the library functions rather than main() keeps
        # argparse out of this path, but any sys.exit anywhere in the call tree
        # would otherwise kill the background task silently and wedge the
        # status on "running" forever, which the DAG can only report as a
        # timeout hours later. Same guard as api/text_main.py.
        _TRAIN_STATUS.update(state="failed", detail=f"{type(exc).__name__}: {exc}")


@app.post("/train")
def train_endpoint(background_tasks: BackgroundTasks,
                   reprocess: bool = Query(False,
                       description="Re-extract features before training (slow).")):
    """Kick off retraining in the background and return immediately."""
    running = _busy()
    if running:
        return JSONResponse(status_code=409,
                            content={"error": f"A '{running}' job is already running."})
    background_tasks.add_task(_run_training, reprocess)
    return {"status": "started", "reprocess": reprocess,
            "poll": "/train/status"}


@app.get("/train/status")
def train_status():
    return _TRAIN_STATUS


def _run_evaluation() -> None:
    _EVAL_STATUS.update(state="running", detail="scoring test features", metrics=None)
    try:
        # Imported here, not at module level, for the same reason train/predict
        # are: keep the idle container's memory profile small (no numpy/sklearn/
        # matplotlib until something actually asks for them).
        import evaluate as evaluate_script  # scripts/evaluate.py

        # Call evaluate() (the library function), NOT main(): main() runs
        # argparse on sys.argv, which under uvicorn holds the SERVER's args ->
        # SystemExit, which `except Exception` does not catch, wedging the
        # status on "running" forever. Same trap as process.main() above.
        metrics = evaluate_script.evaluate()
        _EVAL_STATUS.update(state="done", metrics=metrics, detail="completed")
    except (Exception, SystemExit) as exc:  # noqa: BLE001 - see _run_training
        _EVAL_STATUS.update(state="failed", detail=f"{type(exc).__name__}: {exc}")


@app.post("/evaluate")
def evaluate_endpoint(background_tasks: BackgroundTasks):
    """Score the current local model on the cached test features.

    Evaluates whatever classifier.load() returns — the LOCAL .joblib, i.e. the
    model train.py last wrote — not the registry's production alias. That is
    what an orchestrated train -> evaluate chain needs: the metrics describe the
    model that was just trained, not the one currently being served.
    """
    running = _busy()
    if running:
        return JSONResponse(status_code=409,
                            content={"error": f"A '{running}' job is already running."})
    background_tasks.add_task(_run_evaluation)
    return {"status": "started", "poll": "/evaluate/status"}


@app.get("/evaluate/status")
def evaluate_status():
    return _EVAL_STATUS
