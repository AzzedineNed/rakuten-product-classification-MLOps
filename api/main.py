#!/usr/bin/env python3
"""FastAPI service exposing the image model.

Endpoints:
  GET  /health          liveness + whether a model is loaded
  POST /predict         multipart image upload -> top-k + full canonical proba
  POST /train           retrain the classifier head from cached features
                        (background task; optional ?reprocess=true to re-extract
                        features first — the heavy step)
  GET  /train/status    status + metrics of the last /train job

Run:
  uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

# Make the src package importable without install / PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fastapi import BackgroundTasks, FastAPI, File, Query, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from rakuten_img import config

app = FastAPI(title="Rakuten Image Classifier", version="1.0.0")

_TRAIN_STATUS: dict = {"state": "idle", "detail": None, "metrics": None}


@app.get("/health")
def health():
    import predict as predict_script  # scripts/predict.py

    model_present = config.CLASSIFIER_PATH.exists()
    return {"status": "ok", "model_loaded": model_present,
            "model_source": predict_script.model_source(),
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
    except Exception as exc:  # noqa: BLE001
        _TRAIN_STATUS.update(state="failed", detail=str(exc))


@app.post("/train")
def train_endpoint(background_tasks: BackgroundTasks,
                   reprocess: bool = Query(False,
                       description="Re-extract features before training (slow).")):
    """Kick off retraining in the background and return immediately."""
    if _TRAIN_STATUS["state"] == "running":
        return JSONResponse(status_code=409, content={"error": "Training already running."})
    background_tasks.add_task(_run_training, reprocess)
    return {"status": "started", "reprocess": reprocess,
            "poll": "/train/status"}


@app.get("/train/status")
def train_status():
    return _TRAIN_STATUS