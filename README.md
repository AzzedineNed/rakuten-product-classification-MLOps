# Rakuten Product Classification — MLOps (image + text + fusion)

This repository holds the **MLOps phase** of our group's *Rakuten France
Multimodal Product Data Classification* project (ENS Challenge Data #35). It
predicts a product `prdtypecode` (27 classes) from a product **image**, from its
**text** (designation + description), or from **both** via late fusion.

![Architecture](docs/architecture.png)

*Source: [`docs/architecture.excalidraw`](docs/architecture.excalidraw) — open it
at [excalidraw.com](https://excalidraw.com) and re-export the PNG after any
change.*

This is a deliberately **lighter re-implementation** of the modelisation work
(which trained LeNet / VGG16 / EfficientNetB3 on a rented H100). Instead of
training a heavy CNN it uses a **frozen MobileNetV2 backbone as a feature
extractor** with a small scikit-learn head, and a **TF-IDF + LogisticRegression**
text model. That choice is what makes the whole thing run on a modest laptop
(2 cores, 8 GB RAM) and makes the `/train` endpoint return in seconds.

The deliverable is a **working, reproducible, observable pipeline** — not a new
SOTA score.

---

## Results (measured, not estimated)

All figures come from the shared held-out **test** split (8492 products the
models never saw), except the image number, which is the val figure recorded in
the registry.

| model | accuracy | f1_weighted | f1_macro |
|---|---|---|---|
| image (MobileNetV2 + MLP head) | – | **0.5468** *(val)* | – |
| text (TF-IDF + LogReg) | 0.7768 | **0.7780** | 0.7564 |
| **fusion @ text_weight 0.85** | **0.7990** | **0.7973** | 0.7794 |

Fusion beats text alone by **+0.0193 weighted F1**.

### The fusion weight was measured, not guessed

`scripts/tune_fusion_weight.py` sweeps the weight on the **val** split:

| text_weight | f1_weighted | |
|---|---|---|
| 0.00 | 0.5468 | endpoint check — must equal image alone |
| 0.50 | 0.6621 | the old default: **worse than text alone**, silently |
| **0.85** | **0.7945** | chosen |
| 1.00 | 0.7818 | endpoint check — must equal text alone |

The curve is flat between 0.80 and 0.90, so 0.85 is not a knife-edge fit, and
test came out *above* val, so it did not overfit val. The two endpoint checks are
printed on every sweep: if weight 0.0 does not reproduce the image model exactly
and 1.0 does not reproduce the text model exactly, the fusion code is wrong and
the curve means nothing.

Weakest text classes by F1: `1180 Jeux de table` 0.40, `10 Livre usagé` 0.50,
`1280 Jouets enfants` 0.57. Strongest: `2905 Jeux pour PC` 1.00,
`2583 Équipement de piscine` 0.95, `1920 Mobilier` 0.91.

---

## The shared split, and why it matters

Both modalities are trained and scored on **one** 80/10/10 stratified split,
produced by `src/rakuten_common/split.py` and keyed by **row label**, not
position:

```
train 67932   val 8492   test 8492
```

This is not a detail. Before it existed, the text model's 80/20 validation set
was **exactly the image pipeline's val + test blocks combined** (8492 + 8492 =
16984 rows) — text was being scored on rows the image pipeline held out, so the
two numbers were never comparable and fusion could not be evaluated honestly.

Guarantees now enforced in code:

- **No leakage.** The TF-IDF vectorizer is fit on the 67932 training rows only.
- **Guarded merge.** Modalities are merged on the ID column with a row-count
  guard and a duplicate-id check, never by row position.
- **Fingerprints.** `models/text/split.json` stores a SHA-256 of all three
  splits; `rakuten_text/evaluate.py` verifies **all three** before scoring, so a
  change confined to training rows cannot pass unnoticed.
- **Proof, not assertion.** `scripts/check_split.py` demonstrates on the real
  data that the shared module reproduces `rakuten_img.data.split_dataframe`
  exactly — identical index, order and membership, 27 classes in each split.

### Canonical class order (the fusion contract)

> Every probability vector produced anywhere in this repo is ordered by
> `config.CANONICAL_CLASSES` — the 27 `prdtypecode`s sorted **numerically**.

Verified at train time (training aborts if `classes_` disagrees) and at predict
time. The fusion gateway **refuses to fuse** if an upstream reports a different
order. `tests/test_contract.py` covers this.

---

## Project structure

```
rakuten-image-mlops/
├── data/raw/                  # collect.py lands raw data here (gitignored, DVC-tracked)
├── data/processed/            # cached feature .npy + meta (gitignored)
├── models/                    # image_classifier.joblib + models/text/ (DVC-tracked)
├── reports/                   # metrics, classification reports, confusion matrices
├── docs/architecture.*        # the diagram above (source + exported PNG)
│
├── src/rakuten_common/        # modality-agnostic: no torch, no estimators
│   ├── split.py               # THE split: load, split, split_labels
│   ├── contract.py            # to_canonical(), validate_vector()
│   ├── features.py            # cached-feature row -> productid mapping, re-verified
│   └── fusion.py              # weighted_average(), DEFAULT_TEXT_WEIGHT = 0.85
├── src/rakuten_img/           # image modality
│   ├── config.py  images.py  data.py
│   ├── backbone.py            # frozen MobileNetV2 (the only module importing torch)
│   ├── classifier.py          # build/save/load head, registry publish/pull, reorder
│   └── ens_download.py
├── src/rakuten_text/          # text modality
│   ├── config.py  preprocessing.py  predict.py
│   ├── train.py  evaluate.py  tracking.py
│
├── scripts/                   # thin CLI entrypoints over the packages
│   ├── collect.py  process.py  train.py  evaluate.py  predict.py
│   ├── promote.py             # the ONLY thing that moves the production alias
│   ├── check_split.py         # proves the shared split matches the image split
│   ├── tune_fusion_weight.py  # measures the fusion weight
│   └── check_ci_pins.py       # requirements-ci.txt must not drift
│
├── api/image_main.py          # FastAPI :8000  /predict /train /evaluate /health
├── api/text_main.py           # FastAPI :8001  /predict /predict/batch /health
├── api/gateway_main.py        # FastAPI :8002  /predict (fusion) /health
├── nginx/default.conf         # the only public entrypoint
├── airflow/dags/              # rakuten_image_pipeline
├── .github/workflows/tests.yml
├── tests/                     # 57 tests, torch-free and data-free
└── requirements.txt  requirements-ci.txt  Dockerfile  docker-compose*.yml  Makefile
```

---

## Setup (WSL2 + venv)

PyTorch is installed separately because the correct wheel is hardware-specific.

```bash
make setup            # creates .venv, installs CPU torch + requirements
# or manually:
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Run the tests (need neither torch nor data):

```bash
make test             # 57 tests
```

> **Pins are load-bearing.** `numpy` stays `<2` because torch 2.2.2 requires it,
> and `scikit-learn==1.4.2` is required because `classifier.py` passes
> `multi_class=` to `LogisticRegression`, which newer versions removed. On
> scikit-learn 1.9 the suite fails with `TypeError: ... unexpected keyword
> argument 'multi_class'`. Fix the source before moving that pin.

---

## Usage

### Image pipeline

```bash
# 1. Acquire raw data into data/raw/ (idempotent; re-runs are no-ops)
python scripts/collect.py --source /path/to/rakuten/data      # local folder or .zip
python scripts/collect.py --from-ens                          # authenticated ENS download
dvc pull                                                      # or: the exact versioned data

# 2. Extract & cache MobileNetV2 features (the long step; resumable)
python scripts/process.py
python scripts/process.py --limit 540      # quick smoke run

# 3. Train the head on cached features (fast)
python scripts/train.py
RAKUTEN_CLASSIFIER=logreg python scripts/train.py

# 4. Evaluate -> reports/ (+ attaches test metrics to the SAME MLflow run)
python scripts/evaluate.py

# 5. Predict on a new image
python scripts/predict.py --image path/to/product.jpg --top-k 5
```

### Text pipeline

```bash
PYTHONPATH=src python -m rakuten_text.train                   # ~3 min on 2 cores
PYTHONPATH=src python -m rakuten_text.evaluate                # scores the TEST split, ~20s
PYTHONPATH=src python -m rakuten_text.evaluate --split val    # re-score val
```

`--max-features` is the only training flag. `--test-size` and `--full` were
removed deliberately: a flag that cannot change the shared split is a lie.

### Fusion

```bash
python scripts/tune_fusion_weight.py        # sweeps the weight on val, ~19s
```

All of the above are also `make` targets — run `make help`.

---

## Serving: three services behind one proxy

Nginx on **:80** is the only public entrypoint. **No service publishes a host
port.** The prefix is stripped by the trailing slash on both the `location` and
the `proxy_pass` target.

| route | goes to | notes |
|---|---|---|
| `/` `/predict` `/train` `/evaluate` | image service | unchanged legacy paths; the Airflow DAG drives these |
| `/text/…` | text service | prefix stripped |
| `/fusion/…` | gateway | prefix stripped; fans out to both services |

`/text` and `/fusion` without the trailing slash 301-redirect to the slashed
form. Basic auth guards `/train`, `/train/status`, `/evaluate`,
`/evaluate/status`. Rate limits are per-IP: 10r/s general (burst 20), 1r/s on
train (burst 5), `limit_req_status 429`. Body cap 10 MB.

```bash
curl -s -F "file=@product.jpg" "http://localhost/predict?top_k=3"
curl -s -X POST -H "Content-Type: application/json" \
     -d '{"designation":"Zzz","description":"..."}' "http://localhost/text/predict"
curl -s -F "file=@product.jpg" -F "designation=Zzz" "http://localhost/fusion/predict"
```

The gateway owns no model. It calls both services over HTTP, checks their
declared class order, and fuses at 0.85. If one upstream fails it returns the
other with `degraded=true` plus an `errors` block; if both fail, 502.

### Model loading order

**Both** serving services follow the same rule, via the shared
`rakuten_common/registry.py`:

1. **MLflow Model Registry, alias first** — the version carrying the
   `production` alias.
2. **Newest registered version**, labelled `(unpromoted)`, if no alias resolves.
3. **Local artifacts** if the registry is unreachable or tracking is unset —
   `models/image_classifier.joblib` for image, `models/text/*.pkl` for text.

`GET /health` reports the outcome as `model_source`, e.g.
`registry:rakuten-text-classifier@production/v1` for a deliberate promotion,
`registry:…/v6 (unpromoted)` for a fallback to the newest version, or
`local:…` when the registry was not reachable. That field — not `model_path` —
is the answer to "which model is serving?".

The two services differ in *when* they resolve: the text service loads eagerly at
startup (its artifacts are ~2.7 MB), the image service lazily on first `/predict`
(torch is expensive to import). Neither consults the registry per request.

Serving never hard-fails on a network problem. **Training never promotes** — the
`production` alias moves only via `scripts/promote.py`. Because the resolved
model is cached for the process lifetime, `docker compose restart api` (or
`text-api`) is what picks up a newly promoted version.

A serving process caps MLflow's HTTP retries (2 retries, 10s timeout) rather than
using its shipped defaults. Measured: against an unreachable tracking server the
defaults blocked for **over 170 seconds**, which — with compose's healthcheck and
the gateway's `depends_on: service_healthy` — would take the whole stack down
during a DagsHub outage instead of degrading to the local model. With the cap the
same startup took 10.7s. Raise `MLFLOW_HTTP_REQUEST_MAX_RETRIES` /
`MLFLOW_HTTP_REQUEST_TIMEOUT` to override.

---

## Docker

Four containers: three FastAPI services (all from the **same image**, with
per-service `command:` overrides) plus nginx.

```bash
# one-time: create the basic-auth file (gitignored)
printf "admin:$(openssl passwd -apr1)\n" > nginx/.htpasswd

docker compose build
docker compose up -d
docker compose ps           # all four should read (healthy)
curl http://localhost/health
curl http://localhost/text/health
curl http://localhost/fusion/health
```

Nginx waits for all three services to be *healthy*, not merely started. Memory
limits, measured with `docker stats` during a live fusion request:

| container | measured | limit |
|---|---|---|
| rakuten-image-api | 120 MiB idle / 389 MiB serving / ~1.5 GiB retraining | 2.5g |
| rakuten-text-api | 157 MiB | 1g |
| rakuten-gateway | 50 MiB | 512m |
| rakuten-nginx | 5.3 MiB | 128m |

All eight containers including the Airflow stack total ~1.04 GiB against a
4.807 GiB WSL2 ceiling. The text and gateway limits are generous rather than
tight; re-measure before lowering. Too tight a limit shows up as a retrain
OOM-killed midway, not an obvious error.

Secrets are injected at **runtime** via `env_file: .env`; `.env` and `.dvc/` are
excluded by `.dockerignore`, so credentials are never baked into image layers.

> ⚠️ **Nginx does not reload a bind-mounted config.** After editing
> `nginx/default.conf` you **must** run `docker compose restart nginx`. The tell
> that you forgot: `/text/health` returns FastAPI's `{"detail":"Not Found"}` —
> a 404 *body* means you reached a FastAPI app, just the wrong one.

---

## Experiment tracking & data versioning (DagsHub)

One account token drives three things:

- **MLflow tracking** — `train.py` logs params and train/val metrics;
  `evaluate.py` reopens the **same run** (the run_id is stored in the saved model
  payload) and attaches test metrics, the classification report and the confusion
  matrix. One run tells the full story. Both modalities do this.
- **Model Registry** — two registered models: `rakuten-image-classifier`
  (`production` → v2, six versions) and `rakuten-text-classifier`
  (`production` → v1). Both are served alias-first; promoting the text model did
  not move image serving at all.
- **DVC** — `data/raw` and `models` are tracked and pushed to the DagsHub remote
  over HTTP. Pointer files are committed; the data never is.

Everything is **best-effort by design**: if `MLFLOW_TRACKING_URI` is unset or the
server is unreachable, training and evaluation run normally and the model is
still saved locally. Tracking never blocks the pipeline.

**Teammate setup:** create a DagsHub account, grab your token, put it in `.env`
(as `MLFLOW_TRACKING_PASSWORD`) *and* `.dvc/config.local` (as the DVC password).
Both files are gitignored. Each person uses their own token.

### Configuration (env vars)

| Variable | Default | Meaning |
|---|---|---|
| `RAKUTEN_RAW_SOURCE` | – | default `--source` for collect.py |
| `RAKUTEN_DATA_DIR` | `./data` | data root |
| `RAKUTEN_MODELS_DIR` | `./models` | where classifiers are saved |
| `RAKUTEN_REPORTS_DIR` | `./reports` | where evaluate.py writes results |
| `RAKUTEN_CLASSIFIER` | `mlp` | `mlp` or `logreg` (image head) |
| `RAKUTEN_FEATURE_BATCH` | `64` | backbone batch size |
| `RAKUTEN_REGISTERED_MODEL` | `rakuten-image-classifier` | registry name (image) |
| `RAKUTEN_PRODUCTION_ALIAS` | `production` | alias consulted first when serving |
| `MLFLOW_TRACKING_URI` | – | DagsHub MLflow endpoint; unset = tracking off |
| `MLFLOW_TRACKING_USERNAME` / `_PASSWORD` | – | DagsHub username / token |
| `MLFLOW_EXPERIMENT_NAME` | `rakuten-image` | experiment for the image runs |
| `RAKUTEN_TEXT_REGISTERED_MODEL` | `rakuten-text-classifier` | registry name (text) |
| `RAKUTEN_TEXT_EXPERIMENT` | `rakuten-text` | experiment for the text runs |
| `MLFLOW_HTTP_REQUEST_MAX_RETRIES` | `2` when serving | capped so a dead registry cannot hang startup |
| `MLFLOW_HTTP_REQUEST_TIMEOUT` | `10` when serving | same reason |
| `ENS_USERNAME` / `ENS_PASSWORD` | – | credentials for `collect.py --from-ens` |
| `ENS_BASE` | `https://challengedata.ens.fr` | ENS site base URL |
| `ENS_CHALLENGE_ID` | `35` | ENS challenge number |

Text artifacts go to `models/text/` and `reports/text/`; the text experiment is
`rakuten-text`. A real exported variable takes precedence over `.env`.

---

## Orchestration (Airflow)

Airflow runs as a **separate compose project** (`rakuten-airflow`, Airflow 3.3.0,
LocalExecutor) and talks to the API **over HTTP through nginx** with basic auth —
it does not import the pipeline code.

```bash
# .env.airflow must set AIRFLOW_UID to your host uid, or the containers
# write root-owned files into airflow/logs/ that you then cannot delete
echo "AIRFLOW_UID=$(id -u)" >> airflow/.env.airflow

docker compose -f docker-compose.yml up -d                    # the services
docker compose -f docker-compose.airflow.yml up -d            # the scheduler stack
```

DAG `rakuten_image_pipeline`: `collect_guard >> process_guard >> train >> evaluate`,
`schedule=None`, `retries=0`, `max_active_runs=1`. A full run takes **~1015s**.
It registers a new model version and **never promotes**.

> The dag-processor refresh interval is the default **300s**, so a new or edited
> DAG can take up to five minutes to appear.

> ⚠️ **Airflow covers the image modality only.** Text is not orchestrated: the
> text service has no `/train` endpoint, so retraining text is a host-side
> command. Extending the DAG means first deciding whether orchestration drives an
> endpoint (which must be built) or a container command.

---

## CI

`.github/workflows/tests.yml` runs the tests on every push and pull request to
`main`, on Python 3.12. CI installs `requirements-ci.txt`, which omits mlflow, so
the tests that need a real registry **skip** there rather than fail: 45 pass and
12 skip in CI, while all 57 run locally.

CI installs `requirements-ci.txt` — a deliberate **subset** of
`requirements.txt` that omits mlflow, dagshub, dvc and the API stack, none of
which the tests import (every such import in `src/` is lazy). Measured: 128s and
995 MB for the full file versus 65s and 467 MB for the subset.

`scripts/check_ci_pins.py` runs **before** the install and fails if the two files
ever disagree on a version — a subset is only trustworthy if it cannot drift.

---

## Contributing

1. **Branch off `main`**: `git checkout -b your-feature`.
2. **Run `make test` before opening a PR.** CI runs the same tests, minus the
   ones needing mlflow.
3. **Open a pull request into `main`** and wait for the check to go green.
4. **Never commit data or secrets.** `data/`, `models/`, `reports/`, `.env`,
   `.dvc/config.local`, `nginx/.htpasswd`, `airflow/.env.airflow` and
   `airflow/logs/` are gitignored on purpose. Don't force-add them. Stage files
   explicitly rather than with `git add -A`.

---

## Notes & limitations

- **Resampling** balances each train class to ~4000 samples at the *feature*
  level via row indices + a memory-mapped read, so it never holds multiple full
  copies in RAM. `X_train.npy` is therefore the **resampled** set (108000 =
  27 × 4000) and train row identity is unrecoverable by design — use `train_raw`
  for the 67932 pre-resampling rows.
- **Crash-safe & resumable:** `process.py` writes each split's raw features
  before resampling, so the expensive backbone pass is never lost.
- **Processed images are not written to disk** — straight from raw image to
  feature vector, saving ~2.2 GB and an I/O pass.
- **The MLP's internal "Validation score" during training is optimistic** — it is
  measured on a slice of the *resampled* train set. The number that counts is
  weighted F1 on the untouched val/test splits.
- **`/health` reports `model_loaded` from the local joblib's existence**, which is
  unrelated to what is actually served. Fixing it would change the API response
  contract.
- **`model_source` reads `not-loaded` until the first `/predict`** (lazy torch
  import), so a freshly restarted image container looks unloaded but is healthy.
- **nginx keeps a stale backend IP after `docker compose up --build`.** Recreating
  a service gives it a new IP; nginx resolved the old one at startup and returns
  **502** until `docker compose restart nginx`. `docker compose restart <service>`
  reuses the container and its IP, so it does *not* need this. Same family as the
  bind-mounted-config wart above, different trigger.
- **Train/eval status is in-memory.** An API restart resets it to `idle`, which
  the DAG's poller treats as an error.
- **`models.dvc` tracks all of `models/` as one output**, so a text retrain
  invalidates the image artifacts' directory hash and vice versa. Splitting it is
  correct once the services get separate images.
- **The vectorizer's `stop_words_`** held 1.69M discarded terms (25 of 27 MB).
  It is introspection-only and is dropped before pickling; `models/text/` is
  ~2.7 MB.
- **Fixed-weight fusion is optimal on average, not per product.** Observed: a
  jacuzzi where image said `2583` at 0.9999 and text said `2583` at 0.309 fused
  to 0.413. Confidence-aware weighting might beat it, but it would have to be
  measured on val the same way — not assumed.
