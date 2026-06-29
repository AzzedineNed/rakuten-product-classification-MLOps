# Convenience targets. Run `make help` for the list.
# Assumes a venv at .venv; `make setup` creates it.

PY := .venv/bin/python
PIP := .venv/bin/pip
SOURCE ?=

.PHONY: help setup install test collect process train evaluate predict serve \
        docker-build docker-up docker-down clean

help:
	@echo "Targets:"
	@echo "  setup        create .venv and install deps (CPU torch)"
	@echo "  test         run unit tests (no torch/data needed)"
	@echo "  collect      acquire raw data       (SOURCE=/path/or/url)"
	@echo "  process      extract & cache features"
	@echo "  train        train the classifier head"
	@echo "  evaluate     evaluate on the test set"
	@echo "  predict      predict one image       (IMG=path)"
	@echo "  serve        run the FastAPI app on :8000"
	@echo "  docker-build / docker-up / docker-down"

setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
	$(PIP) install -r requirements.txt

install:
	$(PIP) install -r requirements.txt

test:
	$(PY) -m pytest -q

collect:
	$(PY) scripts/collect.py --source "$(SOURCE)"

collect-ens:
	$(PY) scripts/collect.py --from-ens

process:
	$(PY) scripts/process.py

train:
	$(PY) scripts/train.py

evaluate:
	$(PY) scripts/evaluate.py

predict:
	$(PY) scripts/predict.py --image "$(IMG)"

serve:
	$(PY) -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

clean:
	rm -rf data/processed/*.npy data/processed/*.json reports/* models/*.joblib
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
