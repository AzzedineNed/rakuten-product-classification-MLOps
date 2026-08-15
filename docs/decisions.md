# Design decisions

The README says what to run. This file says why the repository is shaped the
way it is. Everything here was moved verbatim out of the README, which had
grown into a mix of instructions and argument; nothing was rewritten in the
move, and the three section headings promoted from `###` to `##` are the only
lines that changed at all.

Back to the [README](../README.md).

---

## The fusion weight was measured, not guessed

`scripts/tune_fusion_weight.py` sweeps the weight on the **val** split:

| text_weight | f1_weighted | |
|---|---|---|
| 0.00 | 0.5468 | endpoint check, must equal image alone |
| 0.50 | 0.6621 | the old default: **worse than text alone**, silently |
| **0.85** | **0.7945** | chosen |
| 1.00 | 0.7818 | endpoint check, must equal text alone |

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
16984 rows). Text was being scored on rows the image pipeline held out, so the
two numbers were never comparable and fusion could not be evaluated honestly.

Guarantees now enforced in code:

- **No leakage.** The TF-IDF vectorizer is fit on the 67932 training rows only.
- **Guarded merge.** Modalities are merged on the ID column with a row-count
  guard and a duplicate-id check, never by row position.
- **Fingerprints.** `models/text/split.json` stores a SHA-256 of all three
  splits; `scripts/evaluate_text.py` verifies **all three** before scoring, so a
  change confined to training rows cannot pass unnoticed.
- **Proof, not assertion.** `scripts/check_split.py` demonstrates on the real
  data that the shared module reproduces `rakuten_img.data.split_dataframe`
  exactly: identical index, order and membership, 27 classes in each split.

### Canonical class order (the fusion contract)

> Every probability vector produced anywhere in this repo is ordered by
> `config.CANONICAL_CLASSES`, the 27 `prdtypecode`s sorted **numerically**.

Verified at train time (training aborts if `classes_` disagrees) and at predict
time. The fusion gateway **refuses to fuse** if an upstream reports a different
order. `tests/test_contract.py` covers this.

---

## Layout convention

The two modalities were shaped differently for a while. Text was ported from a
standalone repo and kept its own habits. They now follow one rule set, written
down here because a convention nobody wrote down is how the drift happened:

1. **Packages under `src/` are libraries.** They export functions and classes
   and contain no `if __name__ == "__main__"` block.
2. **`scripts/` holds every entrypoint, for every modality.** Anything you run
   is `python scripts/<something>.py`, never `python -m <package>.<module>`.
3. **One `tracking.py` per modality** for all MLflow code, with one stated
   exception: `scripts/promote.py`. It is the only file outside those two
   modules that imports MLflow, and it stays that way deliberately. Promotion
   acts on the registry itself rather than on either model, and `--list`
   reports both registered models, so no per-modality module can honestly own
   it: putting the import in `rakuten_img/tracking.py` would make an image
   module the home of an operation that also moves the text model's alias.
   Naming the exception here is more honest than hiding it behind a module
   that does not really own it.

   The import sits inside `_client()` rather than at module top level, and
   that is load-bearing, not style. The CI subset does not install MLflow, and
   `tests/test_promote.py` imports the script at collection time, so hoisting
   the import would skip that whole file in CI rather than just its
   MLflow-dependent tests. Leave it where it is.

   `promote.py` also imports `rakuten_text.config`, for the text model's name
   in `--list`. That is rule 2 working as intended and not a second exception:
   `scripts/` holds every entrypoint for every modality.
4. **One Makefile target per task per modality.**
5. **Anything both modalities need lives in `rakuten_common/`**, which is why
   `split_fingerprint` is there and not in a text module.

Image entrypoints keep their bare names (`train.py`) and text ones take a
`_text` suffix (`train_text.py`). Renaming the image scripts purely for
symmetry would churn the Makefile, the README and the DAG's error messages
without making anything work better.

| task | image | text |
|---|---|---|
| collect | `make collect` / `scripts/collect.py` | shares the same raw data |
| process | `make process` / `scripts/process.py` | n/a (no feature cache) |
| train | `make train` / `scripts/train.py` | `make train-text` / `scripts/train_text.py` |
| evaluate | `make evaluate` / `scripts/evaluate.py` | `make evaluate-text` / `scripts/evaluate_text.py` |
| predict | `make predict IMG=...` / `scripts/predict.py` | via the API or `TfidfPredictor` |
| serve | `make serve` / `api/image_main.py` | `api/text_main.py` |

---

## Model loading order

**Both** serving services follow the same rule, via the shared
`rakuten_common/registry.py`:

1. **MLflow Model Registry, alias first**: the version carrying the
   `production` alias.
2. **Newest registered version**, labelled `(unpromoted)`, if no alias resolves.
3. **Local artifacts** if the registry is unreachable or tracking is unset:
   `models/image_classifier.joblib` for image, `models/text/*.pkl` for text.

`GET /health` reports the outcome as `model_source`, e.g.
`registry:rakuten-text-classifier@production/v1` for a deliberate promotion,
`registry:.../v6 (unpromoted)` for a fallback to the newest version because no
alias is set, `registry:.../v6 (unpromoted, alias lookup failed)` when an alias
may well exist but the registry would not say, or `local:...` when the registry
was not reachable at all. That field, not `model_path`, is the answer to "which
model is serving?".

The two degraded strings are kept apart on purpose. The first means there is
nothing to honour; the second means a promotion may be being ignored and the
service should be restarted once the registry answers. The dashboard's
**Services on a degraded model** panel counts both, plus `not-loaded`.

**Both** services resolve **eagerly at startup**, in a FastAPI lifespan, and
never per request. A registry outage therefore cannot slow or break traffic
already being served, and `model_loaded` in `/health` means what it says from
boot rather than "somebody has sent a request".

Serving never hard-fails on a network problem. **Training never promotes**: the
`production` alias moves only via `scripts/promote.py`. Because the resolved
model is cached for the process lifetime, `docker compose restart api` (or
`text-api`) is what picks up a newly promoted version.

A serving process caps MLflow's HTTP retries (2 retries, 10s timeout) rather than
using its shipped defaults. Measured: against an unreachable tracking server the
defaults blocked for **over 170 seconds**, which, with compose's healthcheck and
the gateway's `depends_on: service_healthy`, would take the whole stack down
during a DagsHub outage instead of degrading to the local model. With the cap the
same startup took 10.7s. Raise `MLFLOW_HTTP_REQUEST_MAX_RETRIES` /
`MLFLOW_HTTP_REQUEST_TIMEOUT` to override.

Both modalities pay it at the same moment. Each resolves the registry inside its
lifespan, so an outage delays **startup** by at most the capped budget and never
delays a request. The image service used to resolve lazily on the first
`/predict`, moving the same delay onto one request instead; that is no longer
true of either service.

---

## Instrumentation design

`rakuten_common/observability.py` is **pure ASGI** and imports no fastapi, so its
tests run in both CI jobs with only `prometheus-client` added to
`requirements-ci.txt`. A test that imported `api/` would skip in *both* jobs and
vanish from CI silently. It uses `prometheus-client` alone rather than
`prometheus-fastapi-instrumentator`, whose current release requires
`starlette>=1.0.0` and breaks `fastapi==0.111.0`; `prometheus-client` has zero
dependencies and cannot disturb a pin.

---

## Notes & limitations

- **Resampling** balances each train class to ~4000 samples at the *feature*
  level via row indices + a memory-mapped read, so it never holds multiple full
  copies in RAM. `X_train.npy` is therefore the **resampled** set (108000 =
  27 x 4000) and train row identity is unrecoverable by design, so use `train_raw`
  for the 67932 pre-resampling rows.
- **Crash-safe & resumable:** `process.py` writes each split's raw features
  before resampling, so the expensive backbone pass is never lost.
- **Processed images are not written to disk**: straight from raw image to
  feature vector, saving ~2.2 GB and an I/O pass.
- **The MLP's internal "Validation score" during training is optimistic**: it is
  measured on a slice of the *resampled* train set. The number that counts is
  weighted F1 on the untouched val/test splits.
- **`/health`'s `model_loaded` means a model is resolved and serving**, the same
  thing it means on the text service. It used to report the local joblib's mere
  existence, which was unrelated to what was served; that value is still
  available as `local_model_present`, which is what you want when a registry
  fetch has failed and the fallback is what is running.
- **nginx follows a backend that changed IP, within 10 seconds.** It used to
  resolve each name once at startup, so recreating a service returned **502**
  until `docker compose restart nginx`. The config now targets a variable and
  uses Docker's embedded DNS at runtime, with `valid=10s` capping how long an
  address is reused. Editing the config file itself is unchanged: that is still
  a bind mount, and still needs the restart described above.
- **Train/eval status is in-memory.** An API restart resets it to `idle`, which
  the DAG's poller treats as an error.
- **Prometheus does not reload a bind-mounted config either.** Same trap as
  nginx: after editing `prometheus/prometheus.yml` you must
  `docker compose restart prometheus`. The tell is that
  `/api/v1/status/config` still shows the old file.
- **Never set a target label that the application also exports.** The scrape
  config originally attached `modality:` to each job, colliding with the
  `modality` label on `rakuten_model_info`; Prometheus keeps the target's and
  silently renames the application's to `exported_modality`. The dashboard grew
  a duplicate column and any `by (modality)` grouping would have been grouping
  by the wrong label. `job` already distinguishes the targets. Note that old
  series linger for their retention, so the duplicate persists for ~5 minutes
  after the fix.
- **The metrics counters are in-process and reset when a container is
  recreated.** `docker compose up -d` after a config change zeroes
  `rakuten_predictions_total` and the fusion counters, and a counter with no
  increments emits no series at all, so freshly-recreated services legitimately
  show "No data" until traffic arrives. Only `rakuten_model_info` is set at
  startup and therefore survives immediately.
- **`models.dvc` tracks all of `models/` as one output**, so a text retrain
  invalidates the image artifacts' directory hash and vice versa. Splitting it is
  correct once the services get separate images.
- **The vectorizer's `stop_words_`** held 1.69M discarded terms (25 of 27 MB).
  It is introspection-only and is dropped before pickling; `models/text/` is
  ~2.7 MB.
- **Fixed-weight fusion is optimal on average, not per product.** Observed: a
  jacuzzi where image said `2583` at 0.9999 and text said `2583` at 0.309 fused
  to 0.413. Confidence-aware weighting might beat it, but it would have to be
  measured on val the same way, not assumed.
