"""Rakuten image pipeline — end-to-end training orchestration.

    collect_guard --> process_guard --> train --> evaluate

WHAT THIS DAG DOES NOT DO: promote.
    It registers a new model version and stops. The production alias is moved
    only by a human running scripts/promote.py. Auto-promotion would undo the
    entire point of the model registry work: a retrain that automatically takes
    over serving is exactly the behaviour the alias was introduced to prevent.
    The evaluate task prints a comparison so a human can decide.

HOW IT TALKS TO THE MODEL:
    Over HTTP, through nginx, with basic auth — as an ordinary external client.
    No docker.sock, no shared network with the serving stack. The Airflow image
    stays vanilla: no torch, no scikit-learn, no dataset processing here.

WHY THE GUARDS ARE NOT ShortCircuitOperator:
    ShortCircuitOperator skips ALL downstream tasks when it returns False. That
    is wrong here: "features are already cached, so skip re-extraction" must
    still run train and evaluate. So the guards are plain PythonOperators that
    inspect state, log what they found, and push a decision to XCom. The
    "skipping" of the expensive feature-extraction step happens where it
    actually lives — in the reprocess flag passed to POST /train.

DATA VISIBILITY:
    ./data is mounted READ-ONLY at /opt/airflow/rakuten-data. The guards stat()
    paths and nothing more. Airflow can look; it cannot touch the dataset.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests
from airflow.sdk import DAG
from airflow.exceptions import AirflowException
from airflow.providers.standard.operators.python import PythonOperator

# --------------------------------------------------------------------------- #
# Configuration — all from the environment (airflow/.env.airflow)
# --------------------------------------------------------------------------- #
API_BASE = os.getenv("RAKUTEN_API_BASE_URL", "http://host.docker.internal").rstrip("/")
API_USER = os.getenv("RAKUTEN_API_USER", "admin")
API_PASSWORD = os.getenv("RAKUTEN_API_PASSWORD", "")

DATA_MOUNT = Path(os.getenv("RAKUTEN_DATA_MOUNT", "/opt/airflow/rakuten-data"))
RAW_DIR = DATA_MOUNT / "raw"
PROCESSED_DIR = DATA_MOUNT / "processed"

# Mirrors config.feature_files(): the FINAL files train/evaluate consume.
SPLITS = ("train", "val", "test")
FEATURE_FILES = [PROCESSED_DIR / f"{p}_{s}.npy" for s in SPLITS for p in ("X", "y")]

# Training with reprocess=true re-extracts features for ~84k images on CPU and
# takes hours. Cached-feature training measured at 380s in-container. The
# ceiling below covers the slow path; polling stops as soon as the job reports.
TRAIN_TIMEOUT_S = int(os.getenv("RAKUTEN_TRAIN_TIMEOUT_S", "10800"))  # 3h
EVALUATE_TIMEOUT_S = int(os.getenv("RAKUTEN_EVALUATE_TIMEOUT_S", "900"))  # 15m
POLL_INTERVAL_S = int(os.getenv("RAKUTEN_POLL_INTERVAL_S", "15"))


def _auth() -> tuple[str, str]:
    if not API_PASSWORD:
        raise AirflowException(
            "RAKUTEN_API_PASSWORD is empty. Set it in airflow/.env.airflow to the "
            "PLAINTEXT nginx basic-auth password (not the apr1 hash from "
            "nginx/.htpasswd), then recreate the Airflow containers."
        )
    return (API_USER, API_PASSWORD)


def _post(path: str, params: dict | None = None) -> dict:
    """POST to the API, turning the failure modes into readable messages."""
    url = f"{API_BASE}{path}"
    try:
        resp = requests.post(url, params=params or {}, auth=_auth(), timeout=30)
    except requests.RequestException as exc:
        raise AirflowException(
            f"Could not reach {url}: {exc}. Is the serving stack up "
            f"(docker compose ps) and is host.docker.internal mapped?"
        ) from exc

    if resp.status_code == 401:
        raise AirflowException(
            f"401 from {url}: nginx rejected the basic-auth credentials. Check "
            f"RAKUTEN_API_USER / RAKUTEN_API_PASSWORD against nginx/.htpasswd."
        )
    if resp.status_code == 409:
        raise AirflowException(
            f"409 from {url}: {resp.text}. /train and /evaluate are mutually "
            f"exclusive — another job is already running on the API."
        )
    if resp.status_code >= 400:
        raise AirflowException(f"{resp.status_code} from {url}: {resp.text}")
    return resp.json()


def _poll(status_path: str, timeout_s: int, label: str) -> dict:
    """Poll a status endpoint until the job leaves the 'running' state."""
    url = f"{API_BASE}{status_path}"
    deadline = time.monotonic() + timeout_s
    last_state = None

    while time.monotonic() < deadline:
        try:
            resp = requests.get(url, auth=_auth(), timeout=30)
            resp.raise_for_status()
            status = resp.json()
        except requests.RequestException as exc:
            # A transient blip during a long job should not kill the task; the
            # deadline is the real guard. Log and retry on the next tick.
            print(f"[{label}] status poll failed ({exc}); retrying")
            time.sleep(POLL_INTERVAL_S)
            continue

        state = status.get("state")
        if state != last_state:
            print(f"[{label}] state={state} detail={status.get('detail')}")
            last_state = state

        if state == "done":
            return status
        if state == "failed":
            raise AirflowException(f"{label} failed on the API: {status.get('detail')}")
        if state == "idle":
            # The job finished and something reset it, or it never started.
            raise AirflowException(
                f"{label} reported 'idle' while being polled — the job did not run. "
                f"Check the api container logs."
            )
        time.sleep(POLL_INTERVAL_S)

    raise AirflowException(
        f"{label} did not finish within {timeout_s}s. The job may still be running "
        f"on the API; check GET {status_path} before re-triggering."
    )


# --------------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------------- #
def collect_guard(**_) -> bool:
    """Is the raw dataset present? Never downloads anything.

    Collection needs ENS credentials and pulls ~2.2 GB. That is a deliberate
    human action (python scripts/collect.py), not something an orchestrator
    should trigger on a schedule. This task reports state and fails loudly only
    when NOTHING downstream could work.
    """
    raw_present = RAW_DIR.exists() and any(RAW_DIR.iterdir())
    features_present = all(f.exists() for f in FEATURE_FILES)

    print(f"raw dir        : {RAW_DIR} -> {'present' if raw_present else 'MISSING'}")
    print(f"cached features: {'present' if features_present else 'MISSING'}")

    if not raw_present and not features_present:
        raise AirflowException(
            f"Neither the raw dataset ({RAW_DIR}) nor cached features exist, so "
            f"there is nothing to train on. Run `python scripts/collect.py` (or "
            f"`dvc pull`) on the host first — this DAG will not download 2.2 GB "
            f"for you."
        )
    if not raw_present:
        print("Raw data absent but cached features exist — training can proceed.")
    return raw_present


def process_guard(ti=None, **_) -> bool:
    """Decide whether feature extraction has to run.

    Returns the reprocess flag consumed by the train task. Does NOT extract
    anything: the read-only mount makes that impossible by construction.
    """
    missing = [f.name for f in FEATURE_FILES if not f.exists()]

    if not missing:
        print(f"All {len(FEATURE_FILES)} feature files present in {PROCESSED_DIR}.")
        print("Skipping extraction: train will run on cached features (~380s).")
        return False

    print(f"Missing feature files: {missing}")
    print(
        "Train will be called with reprocess=true. WARNING: this re-extracts "
        "MobileNetV2 features for the whole dataset on CPU and takes hours, not "
        "minutes."
    )
    if not (RAW_DIR.exists() and any(RAW_DIR.iterdir())):
        raise AirflowException(
            f"Features are missing and the raw dataset is not there either "
            f"({RAW_DIR}) — reprocessing is impossible."
        )
    return True


def train(ti=None, **_) -> dict:
    """POST /train and poll until it finishes. Registers a version; promotes nothing."""
    reprocess = ti.xcom_pull(task_ids="process_guard")
    print(f"Starting training (reprocess={reprocess}) against {API_BASE}")

    started = _post("/train", {"reprocess": str(bool(reprocess)).lower()})
    print(f"API accepted the job: {started}")

    timeout = TRAIN_TIMEOUT_S if reprocess else min(TRAIN_TIMEOUT_S, 3600)
    status = _poll("/train/status", timeout, "train")

    metrics = status.get("metrics") or {}
    print(f"Training metrics: {metrics}")
    return metrics


def evaluate(ti=None, **_) -> dict:
    """POST /evaluate, poll, and report — without promoting anything."""
    print(f"Starting evaluation against {API_BASE}")
    started = _post("/evaluate")
    print(f"API accepted the job: {started}")

    status = _poll("/evaluate/status", EVALUATE_TIMEOUT_S, "evaluate")
    metrics = status.get("metrics") or {}

    train_metrics = ti.xcom_pull(task_ids="train") or {}
    print("=" * 70)
    print("RUN COMPLETE — a new model version was registered and NOT promoted.")
    print(f"  train/val metrics : {train_metrics}")
    print(f"  test metrics      : {metrics}")
    print("")
    print("The production alias still points wherever it pointed before this run.")
    print("To inspect versions and promote deliberately, on the host:")
    print("    python scripts/promote.py --list")
    print("    python scripts/promote.py --version N")
    print("    docker compose restart api      # serving picks up the change")
    print("=" * 70)
    return metrics


# --------------------------------------------------------------------------- #
# DAG
# --------------------------------------------------------------------------- #
with DAG(
    dag_id="rakuten_image_pipeline",
    description="End-to-end image pipeline: guards -> train -> evaluate (no promotion)",
    start_date=datetime(2026, 1, 1),
    schedule=None,  # manual trigger; this is a laptop, not a cluster
    catchup=False,
    max_active_runs=1,  # /train and /evaluate are mutually exclusive on the API
    tags=["rakuten", "image", "mlops"],
    default_args={
        # No automatic retries: a retry would re-POST a job that may still be
        # running and collide with the API's 409 guard. Failures here need a
        # human to look at them.
        "retries": 0,
        "execution_timeout": timedelta(hours=4),
    },
    doc_md=__doc__,
) as dag:
    t_collect = PythonOperator(task_id="collect_guard", python_callable=collect_guard)
    t_process = PythonOperator(task_id="process_guard", python_callable=process_guard)
    t_train = PythonOperator(task_id="train", python_callable=train)
    t_evaluate = PythonOperator(task_id="evaluate", python_callable=evaluate)

    t_collect >> t_process >> t_train >> t_evaluate