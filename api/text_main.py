#!/usr/bin/env python3
"""FastAPI service exposing the TEXT model (TF-IDF + LogisticRegression).

Endpoints:
  GET  /health          liveness + whether the model is loaded
  POST /predict         {designation, description} -> top-k + full canonical proba
  POST /predict/batch   a list of products -> one result each

THE PAYLOAD MIRRORS THE IMAGE SERVICE ON PURPOSE. api/main.py's /predict
returns top_k / prediction / canonical_classes / probabilities, and so does
this. The gateway can therefore treat the two services identically instead of
special-casing each modality, and a probability vector from either can go
straight into fusion.weighted_average.

LOADING: eager, at startup, unlike the image service. The image API imports
torch lazily because torch plus a MobileNetV2 costs hundreds of MB; the text
model is a 0.5 MB vectorizer plus a 2.2 MB estimator, so paying that at boot
buys a fast, predictable first request. A missing model does NOT stop the
service from starting -- /health reports it, mirroring the image side.

WHERE THE MODEL COMES FROM: the MLflow registry when it is reachable (the
version carrying the production alias, else the newest one), the local .pkl
files otherwise. Resolved ONCE at startup and reported by /health as
model_source, so "which model is serving?" has an answer that is read from the
running process rather than inferred from what is on disk.

Run:
  uvicorn api.text_main:app --host 0.0.0.0 --port 8001
"""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

# Make the src packages importable without install / PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# scripts/ too, for the training and evaluation entry points the endpoints
# below drive. Same two lines api/image_main.py uses.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fastapi import BackgroundTasks, FastAPI, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from rakuten_common.observability import PrometheusMiddleware, ServiceMetrics
from rakuten_text import config
from rakuten_text.predict import TfidfPredictor

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("api-text")

predictor = TfidfPredictor()
METRICS = ServiceMetrics("text")

# Job state for /train and /evaluate. IN-MEMORY AND PER-PROCESS, exactly like
# the image service's: an API restart resets these to "idle", and the Airflow
# polling loop treats "idle" while polling as an error. That is a known wart on
# the image side and it is duplicated here DELIBERATELY - two services behaving
# identically when both are wrong is easier to fix once than two services
# behaving differently. Fixing it means persisting job state somewhere both
# services read, which is a bigger change than this one.
_TRAIN_STATUS: dict = {"state": "idle", "detail": None, "metrics": None}
_EVAL_STATUS: dict = {"state": "idle", "detail": None, "metrics": None}


def _busy() -> str | None:
    """Which job is running, if any.

    /train and /evaluate are mutually exclusive: both are CPU-bound on a 2-core
    laptop and both write into models/text and reports/text. Note this guard is
    PER SERVICE - nothing here prevents an image retrain running at the same
    time as a text retrain.
    """
    if _TRAIN_STATUS["state"] == "running":
        return "train"
    if _EVAL_STATUS["state"] == "running":
        return "evaluate"
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model at startup. A failure is logged, not raised: the service
    still starts and /health reports model_loaded=false, so an orchestrator
    gets a truthful answer instead of a container that will not boot.

    prefer_registry=True is what makes this service registry-aware: the version
    carrying the production alias, else the newest registered version, else the
    local .pkl files. The fallback is inside load(), so a missing or unreachable
    registry degrades to local serving instead of failing to boot.

    EAGER AND ONCE. The registry is consulted exactly here, at startup - never
    per request. A promotion therefore takes effect on restart, which is the
    same deal the image service offers, and means a registry outage cannot slow
    down or break traffic that is already being served.

    Catching Exception, not just FileNotFoundError: a corrupt artifact used to
    take the container down at boot, which contradicts the promise in the first
    paragraph. Now every load failure lands in /health where it can be seen.
    """
    try:
        predictor.load(prefer_registry=True)
        logger.info("Text model loaded from %s - API ready.", predictor.serving_source)
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not load the text model: %s: %s", type(exc).__name__, exc)
    # Published in BOTH cases on purpose. On failure serving_source is still
    # "not-loaded", and a gauge that says so is far more useful on a dashboard
    # than an absent series, which is indistinguishable from a dead scrape.
    METRICS.set_model_info("text", predictor.serving_source)
    yield


app = FastAPI(
    title="Rakuten Text Classifier",
    description="Predicts a product type code from its designation and "
                "description. Probability vectors follow CANONICAL_CLASSES.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(PrometheusMiddleware, metrics=METRICS)


class Produit(BaseModel):
    designation: str = Field(..., min_length=1, description="Product title")
    description: str = Field("", description="Longer description (optional)")


class ProduitBatch(BaseModel):
    produits: List[Produit] = Field(..., description="Products to classify")


class TopK(BaseModel):
    prdtypecode: int
    label: str
    probability: float


class Prediction(BaseModel):
    top_k: List[TopK]
    prediction: TopK
    canonical_classes: List[int]
    probabilities: List[float]


class BatchPrediction(BaseModel):
    predictions: List[Prediction]


def _unavailable() -> Optional[JSONResponse]:
    if predictor.is_loaded:
        return None
    return JSONResponse(
        status_code=503,
        content={"error": "Text model not loaded. Run "
                          "`python scripts/train_text.py` (or `dvc pull`)."},
    )


@app.get("/metrics")
def metrics():
    """Prometheus exposition. Scraped over the compose network only; nginx
    does not route it, so it is not publicly reachable."""
    body, content_type = METRICS.render()
    return Response(content=body, media_type=content_type)


@app.get("/health")
def health():
    """model_source is the AUTHORITATIVE answer to "which model is being
    served": "registry:NAME@production/vN" for a deliberate promotion,
    "registry:NAME/vN (unpromoted)" when no alias is set, "local:<file>" when
    the registry was unavailable, "not-loaded" if nothing loaded at all.

    model_path is kept for continuity but is NOT that answer - it is only the
    local path this service would fall back to. Read model_source.
    """
    return {
        "status": "ok",
        "modality": "text",
        "model_loaded": predictor.is_loaded,
        "model_source": predictor.serving_source,
        "model_path": str(predictor.model_path),
        "num_classes": config.NUM_CLASSES,
        "registered_model": config.REGISTERED_MODEL_NAME,
    }


@app.post("/predict", response_model=Prediction)
def predict(produit: Produit, top_k: int = 5):
    unavailable = _unavailable()
    if unavailable:
        return unavailable
    top_k = max(1, min(int(top_k), config.NUM_CLASSES))
    try:
        result = predictor.predict_detailed(produit.designation,
                                            produit.description, top_k=top_k)
        METRICS.observe_prediction(result["prediction"]["prdtypecode"],
                                   result["prediction"]["probability"])
        return result
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed")
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.post("/predict/batch", response_model=BatchPrediction)
def predict_batch(batch: ProduitBatch, top_k: int = 5):
    unavailable = _unavailable()
    if unavailable:
        return unavailable
    if not batch.produits:
        return JSONResponse(status_code=400,
                            content={"error": "The product list is empty."})
    top_k = max(1, min(int(top_k), config.NUM_CLASSES))
    try:
        results = [
            predictor.predict_detailed(p.designation, p.description, top_k=top_k)
            for p in batch.produits
        ]
        # One observation per PRODUCT, not per request: a batch of 50 is 50
        # predictions. Counting it once would make predictions_total disagree
        # with how many products were actually classified.
        for result in results:
            METRICS.observe_prediction(result["prediction"]["prdtypecode"],
                                       result["prediction"]["probability"])
        return {"predictions": results}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Batch prediction failed")
        return JSONResponse(status_code=500, content={"error": str(exc)})


# --------------------------------------------------------------------------- #
# Training and evaluation
#
# WHY THESE LIVE ON THE SERVING PROCESS AT ALL. They do not belong here: a
# production system runs training as a job on a worker, not as a route on the
# process answering user traffic. They are here because Airflow is deliberately
# kept away from the Docker daemon (no docker.sock, no sklearn in the Airflow
# image - see the image DAG's docstring), so HTTP is the only channel the
# orchestrator has. This is a considered trade-off on a laptop, not a pattern
# to copy. The consequence to remember: a retrain competes for memory with live
# prediction inside one uvicorn worker.
# --------------------------------------------------------------------------- #
def _run_training() -> None:
    _TRAIN_STATUS.update(state="running", detail="tfidf + logreg", metrics=None)
    try:
        # Imported here, not at module level: the idle container should not pay
        # for pandas and scikit-learn until something actually asks to train.
        import train_text as train_script  # scripts/train_text.py

        # parse_args([]) - NOT main(), and NOT a hand-built namespace. main()
        # would parse uvicorn's argv and raise SystemExit; a hand-built
        # namespace would rot the moment the script grew an argument.
        metrics = train_script.entrainer(train_script.build_parser().parse_args([]))
        _TRAIN_STATUS.update(state="done", metrics=metrics, detail="completed")
    except (Exception, SystemExit) as exc:  # noqa: BLE001
        # SystemExit is caught EXPLICITLY because it does not derive from
        # Exception. Anything that reaches argparse or calls sys.exit would
        # otherwise kill the background task silently and wedge the status on
        # "running" forever, which the DAG can only report as a timeout.
        _TRAIN_STATUS.update(state="failed", detail=f"{type(exc).__name__}: {exc}")


@app.post("/train")
def train_endpoint(background_tasks: BackgroundTasks):
    """Retrain the text model in the background and return immediately.

    Takes no parameters on purpose. scripts/train_text.py exposes
    --max-features, but exposing a knob on an authenticated, publicly routed
    endpoint that nothing in the pipeline sets is surface without a caller. The
    hyperparameters live in rakuten_text.config; change them there.

    Registers a NEW VERSION and stops. The production alias is never moved
    here. Promotion stays a deliberate human act, the same deal the image side
    offers, and the reason the registry work exists at all.
    """
    running = _busy()
    if running:
        return JSONResponse(status_code=409,
                            content={"error": f"A '{running}' job is already running."})
    background_tasks.add_task(_run_training)
    return {"status": "started", "poll": "/train/status"}


@app.get("/train/status")
def train_status():
    return _TRAIN_STATUS


def _run_evaluation() -> None:
    _EVAL_STATUS.update(state="running", detail="scoring the test split", metrics=None)
    try:
        import evaluate_text as evaluate_script  # scripts/evaluate_text.py

        metrics = evaluate_script.evaluer(
            evaluate_script.build_parser().parse_args([]))
        _EVAL_STATUS.update(state="done", metrics=metrics, detail="completed")
    except (Exception, SystemExit) as exc:  # noqa: BLE001 - see _run_training
        _EVAL_STATUS.update(state="failed", detail=f"{type(exc).__name__}: {exc}")


@app.post("/evaluate")
def evaluate_endpoint(background_tasks: BackgroundTasks):
    """Score the LOCAL text model on the shared held-out test split.

    Scores what scripts/train_text.py last wrote to models/text, not whatever
    the registry's production alias points at. That is what an orchestrated
    train -> evaluate chain needs: metrics that describe the model just trained,
    not the one currently being served.
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
