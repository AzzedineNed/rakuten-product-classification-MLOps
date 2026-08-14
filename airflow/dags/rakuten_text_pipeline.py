"""Rakuten text pipeline — end-to-end training orchestration for the TEXT model.

    data_guard --> train --> evaluate

A SEPARATE DAG, NOT AN EXTENSION OF rakuten_image_pipeline. The two models have
separate registries, separate artifacts and separate failure modes, and each DAG
carries max_active_runs=1 while the API refuses a second job with 409. Folding
text into the image DAG would serialize a 20-second text evaluation behind a
possible 3-hour image reprocess, and a text failure would surface to a reader as
"the pipeline failed". The image DAG's guards are also feature-cache-specific and
mean nothing here: text has no cached-feature step, so there is no reprocess
decision to make and no second guard.

WHAT THIS DAG DOES NOT DO: promote.
    Same deal as the image DAG. It registers a new version of
    rakuten-text-classifier and stops; the production alias is moved only by a
    human running scripts/promote.py --model rakuten-text-classifier.

HOW IT TALKS TO THE MODEL:
    Over HTTP through nginx with basic auth, as an ordinary external client. No
    docker.sock, no shared network with the serving stack, no scikit-learn in
    the Airflow image. /text/train and /text/evaluate are behind the same
    htpasswd file as the image /train and /evaluate.

THE HELPERS BELOW ARE DUPLICATED FROM rakuten_image_pipeline ON PURPOSE.
    _auth, _post and _poll are byte-identical there. They are copied rather than
    extracted into a shared module because the image DAG is a working, untested
    1015-second pipeline: sharing code would mean editing it, and re-proving it
    costs a full run. Airflow's safe-mode discovery also scans every .py in
    dags/ for a DAG, so a helper module there is parsed and found wanting on
    every refresh. IF A THIRD MODALITY APPEARS, EXTRACT THEM THEN — two copies
    is a judgement call, three is a mistake.

CONCURRENCY, THE ONE THING max_active_runs DOES NOT COVER:
    The API's 409 guard is per service. It stops text-train colliding with
    text-evaluate, but nothing stops an image retrain and a text retrain running
    at the same time — different containers, different DAGs. Measured peaks are
    1.48 GiB (image) and 841.8 MiB (text) against 4.807 GiB of RAM, so it fits,
    but on two cores they will crawl. Trigger them one at a time.

DATA VISIBILITY:
    ./data is mounted READ-ONLY at /opt/airflow/rakuten-data. The guard stats
    paths and nothing more.
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

# The text service is behind the /text/ prefix on the same nginx. nginx rewrites
# /text/train -> /train before it reaches uvicorn.
TEXT_PREFIX = os.getenv("RAKUTEN_TEXT_PREFIX", "/text")

DATA_MOUNT = Path(os.getenv("RAKUTEN_DATA_MOUNT", "/opt/airflow/rakuten-data"))
RAW_DIR = DATA_MOUNT / "raw"

# Text training reads the raw CSVs directly — there is no cached-feature stage,
# which is why this DAG has one guard where the image DAG has two.
X_TRAIN_CSV = RAW_DIR / "X_train_update.csv"
Y_TRAIN_CSV = RAW_DIR / "Y_train_CVw08PX.csv"

# MEASURED in-container: 30s to load and merge, 78.2s to fit, ~2 min total.
# Evaluation measured at well under a minute. The ceilings are generous because
# a host under load stretches both, and polling stops the moment a job reports.
TRAIN_TIMEOUT_S = int(os.getenv("RAKUTEN_TEXT_TRAIN_TIMEOUT_S", "1800"))  # 30m
EVALUATE_TIMEOUT_S = int(os.getenv("RAKUTEN_TEXT_EVALUATE_TIMEOUT_S", "600"))  # 10m
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
    if resp.status_code == 404:
        raise AirflowException(
            f"404 from {url}: the route does not exist. The text training routes "
            f"live behind {TEXT_PREFIX}/ and were added in the same commit as the "
            f"endpoints — check that nginx has the current nginx/default.conf "
            f"(it does NOT reload a bind-mounted config; restart it)."
        )
    if resp.status_code == 409:
        raise AirflowException(
            f"409 from {url}: {resp.text}. /train and /evaluate are mutually "
            f"exclusive — another job is already running on the text API."
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
            # Either the job never started, or the container restarted mid-run:
            # the status dicts are in-memory and reset to "idle" on boot.
            raise AirflowException(
                f"{label} reported 'idle' while being polled — the job did not run, "
                f"or the text-api container restarted while it was running. Check "
                f"`docker compose logs text-api`."
            )
        time.sleep(POLL_INTERVAL_S)

    raise AirflowException(
        f"{label} did not finish within {timeout_s}s. The job may still be running "
        f"on the API; check GET {status_path} before re-triggering."
    )


# --------------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------------- #
def data_guard(**_) -> bool:
    """Are the raw CSVs the text model trains on present?

    Text training needs only the two raw CSVs — the designation/description
    text and the labels. It never touches the images, so this DAG can run on a
    machine where the 2.2 GB of image data was never downloaded.
    """
    x_present = X_TRAIN_CSV.exists()
    y_present = Y_TRAIN_CSV.exists()

    print(f"{X_TRAIN_CSV} -> {'present' if x_present else 'MISSING'}")
    print(f"{Y_TRAIN_CSV} -> {'present' if y_present else 'MISSING'}")

    if not (x_present and y_present):
        raise AirflowException(
            f"The raw text CSVs are not in {RAW_DIR}, so there is nothing to "
            f"train on. Run `dvc pull` (or `python scripts/collect.py`) on the "
            f"host first — this DAG will not download the dataset for you."
        )

    print("Raw text data present. Note that training re-reads and re-splits it; "
          "the split is deterministic and shared with the image modality.")
    return True


def train(**_) -> dict:
    """POST /text/train and poll. Registers a version; promotes nothing."""
    print(f"Starting text training against {API_BASE}{TEXT_PREFIX}")

    started = _post(f"{TEXT_PREFIX}/train")
    print(f"API accepted the job: {started}")

    status = _poll(f"{TEXT_PREFIX}/train/status", TRAIN_TIMEOUT_S, "train")

    metrics = status.get("metrics") or {}
    print(f"Training metrics (validation split): {metrics}")
    return metrics


def evaluate(ti=None, **_) -> dict:
    """POST /text/evaluate, poll, and report — without promoting anything."""
    print(f"Starting text evaluation against {API_BASE}{TEXT_PREFIX}")
    started = _post(f"{TEXT_PREFIX}/evaluate")
    print(f"API accepted the job: {started}")

    status = _poll(f"{TEXT_PREFIX}/evaluate/status", EVALUATE_TIMEOUT_S, "evaluate")
    metrics = status.get("metrics") or {}

    train_metrics = ti.xcom_pull(task_ids="train") or {}
    print("=" * 70)
    print("RUN COMPLETE — a new text model version was registered and NOT promoted.")
    print(f"  train/val metrics : {train_metrics}")
    print(f"  test metrics      : {metrics}")
    print("")
    print("The production alias still points wherever it pointed before this run.")
    print("To inspect versions and promote deliberately, on the host:")
    print("    python scripts/promote.py --model rakuten-text-classifier --list")
    print("    python scripts/promote.py --model rakuten-text-classifier --version N")
    print("    docker compose restart text-api   # serving picks up the change")
    print("=" * 70)
    return metrics


# --------------------------------------------------------------------------- #
# DAG
# --------------------------------------------------------------------------- #
with DAG(
    dag_id="rakuten_text_pipeline",
    description="End-to-end text pipeline: guard -> train -> evaluate (no promotion)",
    start_date=datetime(2026, 1, 1),
    schedule=None,  # manual trigger; this is a laptop, not a cluster
    catchup=False,
    max_active_runs=1,  # /train and /evaluate are mutually exclusive on the API
    tags=["rakuten", "text", "mlops"],
    default_args={
        # No automatic retries: a retry would re-POST a job that may still be
        # running and collide with the API's 409 guard. Failures here need a
        # human to look at them.
        "retries": 0,
        # Generous against the ~2 min measured run, because the ceiling exists to
        # catch a wedged job, not to pace a healthy one.
        "execution_timeout": timedelta(hours=1),
    },
    doc_md=__doc__,
) as dag:
    t_guard = PythonOperator(task_id="data_guard", python_callable=data_guard)
    t_train = PythonOperator(task_id="train", python_callable=train)
    t_evaluate = PythonOperator(task_id="evaluate", python_callable=evaluate)

    t_guard >> t_train >> t_evaluate
