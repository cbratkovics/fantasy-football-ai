.PHONY: help install test lint publication-check format train tiers evaluate score weekly api build frontend dbt-deps dbt-dev dbt-state dbt-slim dbt-prod dbt-export dbt-docs dbt-lint

PY ?= .venv/bin/python

help:
	@echo "install   - create .venv and install training + dev dependencies"
	@echo "test      - run the test suite (offline fixture; FFAI_TEST_DATA=full for the real pull)"
	@echo "lint      - ruff + black --check"
	@echo "publication-check - scan tracked public text for coaching material"
	@echo "format    - black"
	@echo "train     - train champion/challenger candidates (writes artifacts/models/<version>)"
	@echo "tiers     - build preseason GMM tiers (SEASON=2024)"
	@echo "evaluate  - frozen-test + rolling-origin evaluation, manifest, model card"
	@echo "calibrate-drift - backtest the drift rule over past seasons, bucket vs hybrid reference (ADR-0031)"
	@echo "score     - score one week: SEASON=2025 WEEK=1 [THROUGH=2024]"
	@echo "weekly    - dry-run the autonomous weekly job"
	@echo "api       - run the API locally on :7860"
	@echo "build     - build the API Docker image"
	@echo "frontend  - run the Next.js dev server on :3000"
	@echo "dbt-dev   - dbt deps + build the medallion warehouse locally (.duckdb/ffai_dev.duckdb)"
	@echo "dbt-state - save the last dev build (manifest + DuckDB file) as slim-build state in .dbt-state/"
	@echo "dbt-slim  - build only state:modified+ against .dbt-state, deferring the rest (ADR-0026)"
	@echo "dbt-prod  - dbt build against MotherDuck (needs MOTHERDUCK_TOKEN in the environment)"
	@echo "dbt-export - export gold marts to artifacts/marts/*.parquet (DBT_TARGET=dev|prod)"
	@echo "dbt-docs  - generate the static dbt docs site into dbt/target"
	@echo "dbt-lint  - sqlfluff over the dbt project"

install:
	uv venv --python 3.11 .venv
	uv pip install --python $(PY) -c constraints.txt -r requirements-train.txt
	uv pip install --python $(PY) -e .

test:
	$(PY) -m pytest tests

lint:
	$(PY) -m ruff check ffai tests scripts
	$(PY) -m black --check ffai tests scripts

publication-check:
	$(PY) scripts/check_publication.py

format:
	$(PY) -m black ffai tests scripts

train:
	$(PY) scripts/train.py

SEASON ?= 2024
tiers:
	$(PY) scripts/tiers.py --season $(SEASON)

ARGS ?=
calibrate-drift:
	$(PY) scripts/calibrate_drift.py

evaluate:
	$(PY) scripts/evaluate.py $(ARGS)

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

# --- dbt (analytics warehouse; run from the repo root so relative file paths resolve) ---
DBT ?= .venv/bin/dbt
DBT_FLAGS = --project-dir dbt --profiles-dir dbt
DBT_TARGET ?= dev

dbt-deps:
	$(DBT) deps $(DBT_FLAGS)
	@# macOS occasionally leaves empty '<pkg> 2' copies next to the installed packages, which makes
	@# dbt refuse to run ("expects 3 package(s) ... found 6"); empty directories are safe to drop.
	@find dbt/dbt_packages -maxdepth 1 -type d -empty -delete

dbt-dev: dbt-deps
	mkdir -p .duckdb
	$(DBT) build $(DBT_FLAGS) --target dev

dbt-state:
	mkdir -p .dbt-state
	cp dbt/target/manifest.json .dbt-state/manifest.json
	cp .duckdb/ffai_dev.duckdb .dbt-state/ffai_dev.duckdb
	git rev-parse HEAD > .dbt-state/commit

dbt-slim: dbt-deps
	@test -f .dbt-state/manifest.json || (echo "no .dbt-state; run make dbt-dev && make dbt-state first" && exit 1)
	$(DBT) build $(DBT_FLAGS) --target dev --select state:modified+ --defer --state $(CURDIR)/.dbt-state

dbt-prod: dbt-deps
	$(DBT) build $(DBT_FLAGS) --target prod

dbt-export:
	GITHUB_SHA=$${GITHUB_SHA:-$$(git rev-parse HEAD)} $(DBT) run-operation export_gold $(DBT_FLAGS) --target $(DBT_TARGET)

dbt-docs: dbt-deps
	$(DBT) docs generate $(DBT_FLAGS) --target dev --static

dbt-lint:
	.venv/bin/sqlfluff lint dbt/models dbt/tests dbt/macros
