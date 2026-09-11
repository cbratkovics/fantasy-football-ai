.PHONY: help install test lint format train tiers evaluate score weekly api build frontend

PY ?= .venv/bin/python

help:
	@echo "install   - create .venv and install training + dev dependencies"
	@echo "test      - run the test suite (offline fixture; FFAI_TEST_DATA=full for the real pull)"
	@echo "lint      - ruff + black --check"
	@echo "format    - black"
	@echo "train     - train champion/challenger candidates (writes artifacts/models/<version>)"
	@echo "tiers     - build preseason GMM tiers (SEASON=2024)"
	@echo "evaluate  - frozen-test + rolling-origin evaluation, manifest, model card"
	@echo "score     - score one week: SEASON=2025 WEEK=1 [THROUGH=2024]"
	@echo "weekly    - dry-run the autonomous weekly job"
	@echo "api       - run the API locally on :7860"
	@echo "build     - build the API Docker image"
	@echo "frontend  - run the Next.js dev server on :3000"

install:
	uv venv --python 3.11 .venv
	uv pip install --python $(PY) -r requirements-train.txt
	uv pip install --python $(PY) -e .

test:
	$(PY) -m pytest tests

lint:
	$(PY) -m ruff check ffai tests scripts
	$(PY) -m black --check ffai tests scripts

format:
	$(PY) -m black ffai tests scripts

train:
	$(PY) scripts/train.py

SEASON ?= 2024
tiers:
	$(PY) scripts/tiers.py --season $(SEASON)

evaluate:
	$(PY) scripts/evaluate.py

WEEK ?= 1
THROUGH ?= $(SEASON)
score:
	$(PY) scripts/score_week.py --season $(SEASON) --week $(WEEK) --through-season $(THROUGH)

weekly:
	$(PY) scripts/run_weekly.py --dry-run

api:
	$(PY) -m uvicorn ffai.serve.app:app --host 127.0.0.1 --port 7860

build:
	docker build -t ffai-api .

frontend:
	cd frontend-next && npm run dev
