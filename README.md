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
                  │                                          │
        zoom + MobileNetV2                          small MLP / logreg
        (frozen, no backprop)                       (fast, re-runnable)

predict.py / API:  image ──► zoom ──► MobileNetV2 ──► classifier ──► probs (canonical order)
```

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
├── data/raw/            # collect.py lands raw data here (gitignored)
├── data/processed/      # process.py caches feature .npy + meta here (gitignored)
├── models/              # trained classifier: image_classifier.joblib
├── reports/             # evaluate.py writes metrics / report / confusion matrix
├── src/rakuten_img/     # the package  all logic lives here
│   ├── config.py        # paths, labels, CANONICAL class order, constants, run_params()
│   ├── images.py        # zoom / white-canvas preprocessing
│   ├── data.py          # load/merge, split, resample
│   ├── backbone.py      # frozen MobileNetV2 (only module importing torch)
│   ├── classifier.py    # build / save / load the head + class-order reorder
│   ├── ens_download.py  # authenticated download from ENS (used by --from-ens)
│   └── fusion.py        # late-fusion helper (documents the team contract; inert)
├── scripts/             # thin CLI entrypoints over the package
│   ├── _bootstrap.py    # makes src/ importable so scripts "just run"
│   ├── collect.py  process.py  train.py  evaluate.py  predict.py
├── api/main.py          # FastAPI: /predict, /train, /train/status, /health
├── tests/               # torch-free smoke tests
├── .env.example         # copy to .env for ENS credentials (gitignored)
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

# 4. Evaluate on the test set -> reports/
python scripts/evaluate.py

# 5. Predict on a new image
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
| `ENS_USERNAME` / `ENS_PASSWORD` | – | credentials for `collect.py --from-ens` (put in `.env`) |
| `ENS_BASE` | `https://challengedata.ens.fr` | ENS site base URL |
| `ENS_CHALLENGE_ID` | `35` | ENS challenge number |

These can go in a local `.env` file (auto-loaded) or be exported in the shell.
A real exported variable takes precedence over the same key in `.env`.

---

## API

```bash
make serve     # uvicorn api.main:app on http://localhost:8000
```

`POST /predict`  multipart image upload:

```bash
curl -s -F "file=@product.jpg" "http://localhost:8000/predict?top_k=3"
```

Returns the top-k classes, the single best prediction, **and** the full
probability vector together with `canonical_classes` so a fusion layer can
consume it directly.

`POST /train`  retrain the classifier head from cached features (background):

```bash
curl -s -X POST "http://localhost:8000/train"             # classifier only (fast)
curl -s -X POST "http://localhost:8000/train?reprocess=true"  # re-extract features first (slow)
curl -s "http://localhost:8000/train/status"              # poll progress + metrics
```

`GET /health`  liveness + whether a trained model is present.

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

Anything beyond the contract above (where the text code lives, how it's trained,
how fusion is orchestrated) is **to be decided with the group** and is not
documented here yet to avoid stating things that aren't settled.

---

## Contributing

This is a shared group repository. Suggested flow:

1. **Branch off `main`** for any change: `git checkout -b your-feature`.
2. **Run the tests before opening a PR:** `make test` (they need neither torch
   nor the dataset, so they're fast and run anywhere).
3. **Open a pull request** into `main` and request a review from a teammate.
4. **Never commit data or secrets.** `data/`, `models/`, `reports/`, and `.env`
   are gitignored on purpose  the dataset is ~2.2 GB and `.env` holds ENS
   credentials. Don't force-add them.
5. Keep the **image / text scopes separate** while the layout is being decided
   (see the repository layout note above) so the two parts don't entangle before
   the team agrees on structure.

---

## Docker

```bash
docker compose build
docker compose up -d        # API on :8000, data/models mounted as volumes
```

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