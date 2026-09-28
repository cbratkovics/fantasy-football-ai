.PHONY: help install test lint publication-check format train tiers evaluate score weekly api build frontend dbt-deps dbt-dev dbt-state dbt-slim dbt-prod dbt-export dbt-docs dbt-lint lab-build lab-check lab-verify lab-test lab-replay lab-golden lab-e2e lab-notes

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
	@echo "lab-build - export the Decision Lab bundle to artifacts/decision_lab and frontend-next/public/decision-lab"
	@echo "lab-check - rebuild the bundle in a temp dir and compare with the committed one; check the golden fixtures"
	@echo "lab-verify - verify the committed bundle's digests, schemas, and the public copy"
	@echo "lab-test  - Decision Lab pytest (-k decision_lab) + frontend vitest"
	@echo "lab-replay - re-validate and recompute a receipt: RECEIPT=receipt.json [ARGS=...]"
	@echo "lab-golden - rewrite tests/fixtures/decision_lab/golden.json from the Python reference"
	@echo "lab-e2e   - Playwright browser flow (installs chromium)"
	@echo "lab-notes - write private review notes OUTSIDE the repo: CASE=<case id> [ARGS=...]"

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

# --- Decision Lab (ADR-0032/0033/0034): an offline bundle over the committed marts + artifacts ---
LAB_OUT = artifacts/decision_lab
LAB_PUBLIC = frontend-next/public/decision-lab
RECEIPT ?= receipt.json

lab-build:
	$(PY) scripts/export_decision_lab.py --out $(LAB_OUT) --public-copy $(LAB_PUBLIC)

lab-check:
	@# Builds to a temp dir and compares; never rewrites the committed evidence.
	$(PY) scripts/export_decision_lab.py --check --out $(LAB_OUT) --public-copy $(LAB_PUBLIC)
	$(PY) -m ffai.decision_lab.golden --check

lab-verify:
	$(PY) scripts/export_decision_lab.py --verify --out $(LAB_OUT) --public-copy $(LAB_PUBLIC)

lab-test:
	$(PY) -m pytest tests -q -k decision_lab
	cd frontend-next && npm test

lab-replay:
	$(PY) -m ffai.decision_lab.replay $(RECEIPT) --bundle $(LAB_OUT) $(ARGS)

lab-golden:
	$(PY) -m ffai.decision_lab.golden --write

lab-e2e:
	cd frontend-next && npx playwright install chromium && npm run test:e2e

lab-notes:
	@# Writes outside the repository (the script refuses a directory inside it or any worktree).
	$(PY) scripts/lab_review_notes.py --case-id $(CASE) $(ARGS)
