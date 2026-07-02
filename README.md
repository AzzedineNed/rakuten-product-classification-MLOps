# Rakuten Product Classification  Image modality (MLOps)

This repository holds the **image** half of our group's *Rakuten France
Multimodal Product Data Classification* project (ENS Challenge Data #35). It
predicts a product `prdtypecode` (27 classes) from a product image.

> **Where this fits in the bigger project.** The full project is *multimodal*:
> an **image** model (this repo), a **text** model (CamemBERT), and a **fusion**
> step that combines the two. This repo is the image part only. The text model
> and fusion are owned by the rest of the team  see
> [Text modality & fusion](#text-modality--fusion-planned) for how this part
> plugs into them.

This is a deliberately **lighter re-implementation** of the modelisation work
(which trained LeNet / VGG16 / EfficientNetB3 on a rented H100). Instead of
training a heavy CNN, it uses a **frozen MobileNetV2 backbone as a feature
extractor** and trains a small scikit-learn classifier on the cached feature
vectors. That choice is what makes the whole thing run on a modest laptop and
makes the `/train` endpoint return in seconds.

---

## Architecture

```
raw images ──► process.py ──► cached features (.npy) ──► train.py ──► image_classifier.joblib
                  │                                          │              │
        zoom + MobileNetV2                          small MLP / logreg      ├──► MLflow run (params + metrics)
        (frozen, no backprop)                       (fast, re-runnable)     └──► Model Registry (new version)

predict.py / API:  image ──► zoom ──► MobileNetV2 ──► classifier ──► probs (canonical order)
                                                          ▲
                                     MLflow Model Registry (newest version)
                                     └─ fallback: local models/image_classifier.joblib

serving:  client ──► nginx :80 (rate limit, body cap, basic auth on /train) ──► FastAPI :8000
```

A full diagram of the pipeline lives in `docs/rakuten-pipeline.excalidraw`
(open it at https://excalidraw.com).

Why frozen features instead of training a CNN:

- **Runs on a laptop.** The only heavy step is one forward pass over the
  ~84k images during `process.py`; after that everything is small-matrix math.
- **`/train` is actually usable.** Retraining the classifier head on cached
  features takes seconds, so a live `/train` endpoint is realistic (retraining
  a full CNN over HTTP would time out).
- **Reasonable accuracy.** Frozen MobileNetV2 features + an MLP head give a
  respectable weighted-F1  below the EfficientNetB3 result from modelisation,
  which is expected and fine: the deliverable is a working reproducible
  pipeline, not a new SOTA score.

---

## Project structure

```
rakuten-image-mlops/
├── data/raw/            # collect.py lands raw data here (gitignored, DVC-tracked)
├── data/processed/      # process.py caches feature .npy + meta here (gitignored)
├── models/              # trained classifier: image_classifier.joblib (DVC-tracked)
├── reports/             # evaluate.py writes metrics / report / confusion matrix
├── src/rakuten_img/     # the package  all logic lives here
│   ├── config.py        # paths, labels, CANONICAL class order, constants, run_params()
│   ├── images.py        # zoom / white-canvas preprocessing
│   ├── data.py          # load/merge, split, resample
│   ├── backbone.py      # frozen MobileNetV2 (only module importing torch)
│   ├── classifier.py    # build / save / load the head, registry publish/pull, reorder
│   ├── ens_download.py  # authenticated download from ENS (used by --from-ens)
│   └── fusion.py        # late-fusion helper (documents the team contract; inert)
├── scripts/             # thin CLI entrypoints over the package
│   ├── _bootstrap.py    # makes src/ importable so scripts "just run"
│   ├── collect.py  process.py  train.py  evaluate.py  predict.py
├── api/main.py          # FastAPI: /predict, /train, /train/status, /health
├── nginx/default.conf   # reverse proxy: rate limits, body cap, auth on /train
├── tests/               # torch-free smoke tests
├── .dvc/                # DVC config (remote = DagsHub); token in config.local (gitignored)
├── data/raw.dvc  models.dvc   # DVC pointers (committed; the data itself is not)
├── .env.example         # copy to .env for ENS + DagsHub/MLflow credentials (gitignored)
├── requirements.txt  Dockerfile  docker-compose.yml  Makefile
```

The five scripts are thin wrappers; the importable logic lives in
`src/rakuten_img/`. That's what makes everything reusable and lets the API and
the CLI share identical preprocessing and class order.

> **Repository layout note (TBD, team).** The image code currently lives at the
> repository root. If/when the text model is added to this same repo, the team
> may move to a multi-part layout (e.g. `image/`, `text/`, `fusion/`) with a
> top-level project README. That restructure is not done yet  this README
> documents the repo as it stands today.

---

## Setup (WSL2 + venv)

PyTorch is installed separately because the correct wheel is hardware-specific.

```bash
# from the project root, inside WSL2
make setup            # creates .venv, installs CPU torch + requirements
# or manually:
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Run the tests (need neither torch nor data):

```bash
make test
```

> On the GPU: a 2 GB laptop GPU (e.g. GeForce 940MX) brings little benefit here
> and CUDA setup is often more trouble than it's worth. CPU is the recommended
> default. If `torch.cuda.is_available()` is true the backbone uses it
> automatically  no code change needed.

---

## Usage

The scripts run in order. Each has a clear, reusable role.

```bash
# 1. Acquire raw data into data/raw/ (idempotent; re-runs are no-ops)
#  (a) from a local folder or zip you already have:
python scripts/collect.py --source /path/to/rakuten/data
python scripts/collect.py --source /path/to/rakuten.zip
#  (b) OR download straight from challengedata.ens.fr (authenticated):
cp .env.example .env        # then set ENS_USERNAME / ENS_PASSWORD in .env
python scripts/collect.py --from-ens
#  (c) OR, for teammates: pull the exact versioned data from DagsHub via DVC:
dvc pull                    # needs your DagsHub token in .dvc/config.local
#    Credentials are read from .env (gitignored), never logged or committed.
#    Source-download lands ~2.2 GB in data/raw/ (reproducibility, not disk saving).

# 2. Extract & cache MobileNetV2 features for train/val/test (the long step)
python scripts/process.py
#    quick smoke run on a subset:
python scripts/process.py --limit 540
#    Raw features are saved to disk as soon as each split is extracted, so the
#    expensive pass is never lost. If interrupted, just rerun  it resumes by
#    skipping any split whose raw features are already cached.

# 3. Train the classifier head on cached features (fast)
python scripts/train.py
#    switch to logistic regression:
RAKUTEN_CLASSIFIER=logreg python scripts/train.py
#    With MLflow configured (see below) this logs the run to DagsHub AND
#    registers the model as a new version in the MLflow Model Registry.

# 4. Evaluate on the test set -> reports/  (+ attaches test metrics and the
#    confusion matrix to the SAME MLflow run as the training)
python scripts/evaluate.py

# 5. Predict on a new image (pulls the model from the registry; see Serving)
python scripts/predict.py --image path/to/product.jpg --top-k 5
```

All of the above are also available as `make` targets  run `make help`.

### Configuration (env vars)

| Variable | Default | Meaning |
|---|---|---|
| `RAKUTEN_RAW_SOURCE` | – | default `--source` for collect.py |
| `RAKUTEN_DATA_DIR` | `./data` | data root |
| `RAKUTEN_MODELS_DIR` | `./models` | where the classifier is saved |
| `RAKUTEN_REPORTS_DIR` | `./reports` | where evaluate.py writes results |
| `RAKUTEN_CLASSIFIER` | `mlp` | `mlp` or `logreg` |
| `RAKUTEN_FEATURE_BATCH` | `64` | backbone batch size |
| `RAKUTEN_REGISTERED_MODEL` | `rakuten-image-classifier` | name in the MLflow Model Registry |
| `MLFLOW_TRACKING_URI` | – | DagsHub MLflow endpoint; unset = tracking off |
| `MLFLOW_TRACKING_USERNAME` | – | your DagsHub username |
| `MLFLOW_TRACKING_PASSWORD` | – | your DagsHub token |
| `MLFLOW_EXPERIMENT_NAME` | `rakuten-image` | MLflow experiment to log under |
| `ENS_USERNAME` / `ENS_PASSWORD` | – | credentials for `collect.py --from-ens` (put in `.env`) |
| `ENS_BASE` | `https://challengedata.ens.fr` | ENS site base URL |
| `ENS_CHALLENGE_ID` | `35` | ENS challenge number |

These can go in a local `.env` file (auto-loaded) or be exported in the shell.
A real exported variable takes precedence over the same key in `.env`.

---

## Experiment tracking & data versioning (DagsHub)

The repo is connected to DagsHub, which hosts three things off one account
token:

- **MLflow tracking.** `train.py` logs each run's parameters and train/val
  metrics; `evaluate.py` reopens the **same run** (the run_id is stored inside
  the saved model payload) and attaches test metrics, the classification
  report and the confusion matrix  so one run tells the full story.
- **MLflow Model Registry.** `train.py` registers every trained model as a new
  version of `rakuten-image-classifier`. This is what serving pulls from (see
  next section).
- **DVC.** `data/raw` and `models` are DVC-tracked and pushed to the DagsHub
  DVC remote over HTTP. The `.dvc` pointer files are committed; the data
  itself never is.

Everything is **best-effort by design**: if `MLFLOW_TRACKING_URI` is unset or
the server is unreachable, training/evaluation run normally and the model is
still saved locally  tracking never blocks the pipeline.

**Teammate setup:** create a DagsHub account, grab your token, then put it in
two places  `.env` (as `MLFLOW_TRACKING_PASSWORD`) and `.dvc/config.local`
(as the DVC password; this file is gitignored). Each person uses their own
token.

---

## Model serving: registry first, local fallback

`predict.py` (and therefore the API) loads the model in this order:

1. **MLflow Model Registry**  the newest version of
   `rakuten-image-classifier` is downloaded once and cached in memory for the
   process lifetime.
2. **Local fallback**  if the registry is unreachable, empty, or
   `MLFLOW_TRACKING_URI` is unset, it falls back to
   `models/image_classifier.joblib` on disk (with automatic reload if the file
   changes, e.g. after a `/train`). A dead server is tried once per process,
   not once per request.

Serving therefore **never hard-fails on a network problem**. `GET /health`
reports which source is in use (`model_source`). Note: because the registry
copy is cached for the process lifetime, a restart
(`docker compose restart api`) is what picks up a newly registered version.

---

## API

In Docker the API sits **behind nginx** and is reached on port **80**; the
API container itself is not published to the host. For local development
without Docker, `make serve` still runs uvicorn directly on `:8000`.

```bash
make serve     # dev only: uvicorn api.main:app on http://localhost:8000
```

`POST /predict`  multipart image upload (open, rate-limited):

```bash
curl -s -F "file=@product.jpg" "http://localhost/predict?top_k=3"
```

Returns the top-k classes, the single best prediction, **and** the full
probability vector together with `canonical_classes` so a fusion layer can
consume it directly.

`POST /train`  retrain the classifier head (background). **Behind basic auth**
in the Docker setup:

```bash
curl -s -X POST -u admin:<password> "http://localhost/train"                  # classifier only (fast)
curl -s -X POST -u admin:<password> "http://localhost/train?reprocess=true"   # re-extract features first (slow)
curl -s -u admin:<password> "http://localhost/train/status"                   # poll progress + metrics
```

`GET /health`  liveness, whether a trained model is present, and where the
served model came from (`model_source`: registry vs local).

---

## Fusion contract

Fusion lives at the team level, not inside this pipeline, but this repo honors
the contract that makes it trivial:

> Every probability vector produced here is ordered by
> `config.CANONICAL_CLASSES`  the 27 `prdtypecode`s sorted **numerically**.

If the text model (CamemBERT) emits its probabilities in the same order, fusion
is a weighted average (`src/rakuten_img/fusion.py`). The `/predict` response
already exposes both the ordered class list and the full probability vector for
this purpose.

---

## Text modality & fusion (planned)

> **Status: not in this repo yet. Owned by the team. Details TBD.**

The full project combines this image model with a **text** model (CamemBERT) via
**late fusion** (a weighted average of the two probability vectors). What this
repo guarantees today, so the text side can plug in cleanly later:

- The image model emits a length-27 probability vector ordered by
  `CANONICAL_CLASSES` (numerically sorted `prdtypecode`s). The text model must
  use the **same order** for fusion to be correct.
- `src/rakuten_img/fusion.py` already implements the weighted-average combine and
  validates vector length; it is currently unused (no second input yet).
- The `/predict` endpoint returns the full ordered vector + `canonical_classes`,
  which is exactly what a fusion step needs to consume.

The planned service layout (agreed, not yet built) is: an **image service**
(this model + its API), a **text service** (teammates'), and a **fusion/gateway
service** that calls both and combines their outputs. Anything beyond that
(where the text code lives, how it's trained) is **to be decided with the
group**.

---

## Contributing

This is a shared group repository. Suggested flow:

1. **Branch off `main`** for any change: `git checkout -b your-feature`.
2. **Run the tests before opening a PR:** `make test` (they need neither torch
   nor the dataset, so they're fast and run anywhere).
3. **Open a pull request** into `main` and request a review from a teammate.
4. **Never commit data or secrets.** `data/`, `models/`, `reports/`, `.env`,
   `.dvc/config.local` and `nginx/.htpasswd` are gitignored on purpose  the
   dataset is ~2.2 GB and the rest hold credentials. Don't force-add them.
5. Keep the **image / text scopes separate** while the layout is being decided
   (see the repository layout note above) so the two parts don't entangle before
   the team agrees on structure.

---

## Docker

Two services: the API and an **nginx reverse proxy** in front of it. Nginx is
the only public entrypoint (port **80**); it adds per-IP rate limiting, a
10 MB upload cap, and HTTP basic auth on the `/train` endpoints.

```bash
# one-time: create the basic-auth file for /train (gitignored)
printf "admin:$(openssl passwd -apr1)\n" > nginx/.htpasswd

docker compose build
docker compose up -d        # nginx on :80 -> API (internal :8000)
curl http://localhost/health
```

Secrets are injected at **runtime** via `env_file: .env` in
`docker-compose.yml`  `.env` and `.dvc/` are excluded from the image by
`.dockerignore`, so credentials never end up baked into image layers.

The image is CPU-based (the pipeline doesn't need a GPU). A commented GPU block
is in `docker-compose.yml` if you want to expose a card via
nvidia-container-toolkit; note the Dockerfile installs the **CPU** torch wheel,
so the image is CPU-only until that wheel is changed.

---

## Notes & limitations

- **Resampling** balances each train class to ~4000 samples, reproducing the
  modelisation methodology. It is done at the *feature* level via row indices +
  a memory-mapped read of the cached features, so it never holds multiple full
  copies in RAM. `class_weight="balanced"` is a lighter alternative.
- **Crash-safe & resumable:** `process.py` saves each split's raw features
  (`X_<split>_raw.npy`) to disk immediately after extraction, before resampling.
  The expensive backbone pass is therefore never lost; rerunning resumes by
  skipping splits whose raw features already exist.
- **Processed images are not written to disk**  `process.py` goes straight from
  raw image to feature vector, saving ~2.2 GB and an I/O pass.
- **Memory:** the resampled train features are ~27×4000×1280×4 bytes ≈ 0.55 GB.
  Lower `RESAMPLE_TARGET` in `config.py` if your machine is tight on RAM.
- **First run downloads** the MobileNetV2 ImageNet weights (~14 MB), cached
  afterward.
- **CPU by default:** torch is installed as the CPU build; the backbone uses a
  GPU automatically only if a CUDA-enabled torch sees one.
- **The MLP's internal "Validation score" during training is optimistic**  it
  is measured on a slice of the *resampled* train set (oversampled duplicates
  leak between fit and internal validation). The number that counts is the
  weighted F1 on the untouched val/test splits, printed by train.py and
  evaluate.py.