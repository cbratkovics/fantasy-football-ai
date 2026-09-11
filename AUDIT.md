# Read-only deep audit — `fantasy-football-ai` (main @ `57c32cb`, 2026-09-10)

> This file is the verbatim record of the read-only audit that motivated the slim rebuild.
> Paths and line numbers refer to the tree at commit `57c32cb` (main) and, where noted, to
> `origin/codex/polish-ui-components-including-header-and-footer-mlj0ni`. Many of the files
> cited here were deleted or replaced by the rebuild; see `docs/REBUILD_REPORT.md`.

Method: read-only shell commands, `git` history reads, import checks with the repo owner's `agentic_ai_env` Python 3.10 interpreter, `pytest --collect-only`, `ruff`/`black` in check mode, and HTTP GETs against the owner's deployed site. No repo file was created, modified, or reformatted (`git status --porcelain` empty at end). Four parallel read-only sub-audits (ML, API/DB, frontend, docs/data) fed this report; every claim below was either verified directly or is attributed with `path:line`. Where the evidence is ambiguous the word **unclear** is used and the lines are quoted.

---

## 0. Repository snapshot

### 0.1 Git

```
57c32cb refactor: Update site content and simplify route structure
1a85490 docs: update README badges to larger for-the-badge style
bc6db05 fix: add type annotation to setResults callback parameter
23bd9b0 add: API debug page to troubleshoot backend connectivity
2b533ba update: correct Railway API URL to production endpoint
a0fd0a2 fix: add TypeScript type annotations to mock data generator
501859c docs: update environment setup status after Vercel configuration
6a39ec5 fix: add fallback mock data for player tiers when API unavailable
81d8eea chore: bump version to trigger new deployment
80ebf76 fix: replace non-existent Heroicons with correct ones
f3b0437 feat: add comprehensive frontend pages and fix build errors
3fd9fdc fix: add 'use client' directive to not-found page
8a018f1 fix: properly handle optional price property in signup page
efa3b98 fix: resolve TypeScript error in signup page
0cabb63 feat: integrate real-time API data and ML predictions into frontend
dd7bad3 feat: major frontend improvements for WinMyLeague.ai
c764ff9 fix: add explicit Railway start command configuration
57fe9c8 fix: resolve Railway PORT variable expansion issue
89c62e1 fix: add comprehensive Railway diagnostics with multiple fallback servers
261e25b fix: resolve Railway health check failures with comprehensive startup diagnostics
525bec7 fix: resolve Heroicons import errors for Vercel deployment
e12035c fix: resolve Railway deployment health check failures
9bb67db feat: implement interactive fantasy football AI frontend
4de61b4 fix: resolve Railway deployment health check failures
00da23b refactor: reorganize repository structure for better maintainability
23a1df7 fix: resolve Redis dependency conflict for Railway deployment
9bbf95f fix: resolve Railway deployment issues with MySQL dependencies and package conflicts
caad141 feat: implement production ML models with validated real NFL data
95266b7 feat: add LLM integration and enhanced services
8585963 feat: enhance database configuration and setup automation
```

| Item | Value |
|---|---|
| Newest commit date | 2026-09-07 (`57c32cb`) |
| Oldest commit date | 2025-07-28 (`76a7a7f Initial commit`) |
| Total commits on `main` | 61 |
| Local branches | `main` only |
| Remote branches | `origin/main`, `origin/codex/analyze-codebase-for-data-science-portfolio`, `…-k82eac`, `…-vna05y`, `origin/codex/polish-ui-components-including-header-and-footer`, `…-mlj0ni` (all dated 2026-09-08, 1–9 commits ahead of `main`, **unmerged**) |
| Tags | none |
| Tracked files | 240 |
| Remote | `https://github.com/cbratkovics/fantasy-football-ai.git` |

Note: all 60 commits before `57c32cb` are dated 2025-07-28 → 2025-07-31; a 13-month gap follows, then one commit on 2026-09-07 and five unmerged `codex/*` branches on 2026-09-08. The unmerged branches contain `backend/evaluation/decision_evaluator.py`, `analytics/sql/risk_strategy.sql`, `tests/test_decision_evaluator.py`, `docs/PORTFOLIO_CASE_STUDY.md`, and a rewritten `README.md`/`PerformanceDashboard.tsx`. **The live site at www.winmyleague.ai is built from that branch line, not from `main`** (see §6).

### 0.2 Tree (depth ≤ 3, excluding node_modules/.git/__pycache__/.next). Directories marked ✱ are git-ignored or empty.

```
.
├── .claude/settings.local.json ✱          ├── backend/
├── .env ✱  .env.example                   │   ├── __init__.py
├── .github/ISSUE_TEMPLATE/ (empty) ✱      │   ├── .claude/ ✱  .dockerignore  .env ✱  .env.example  .env.local ✱  .gitignore  .python-version
├── .github/workflows/ (empty) ✱           │   ├── alembic.ini  alembic/{env.py, README, script.py.mako, versions/817e005bc9f5_initial_migration.py}
├── .gitignore                             │   ├── api/{__init__, auth, llm_endpoints, payments, players, predictions, predictions_v2, subscriptions, tiers, websocket_routes}.py
├── .ipynb_checkpoints/ ✱ (5 files)        │   ├── backend/data/sources/ (empty, untracked) ✱
├── .pytest_cache/ ✱  .vscode/ ✱  .DS_Store│   ├── celery_app.py  check_imports.py
├── LICENSE  Makefile  README.md           │   ├── core/{cache, rate_limiter, websocket}.py
├── deploy-to-railway.sh                   │   ├── data/{__init__, data_pipeline, enhanced_data_collector, espn_client, fetch_players(0 bytes), scoring, sleeper_client, synthetic_data_generator}.py
├── docker-compose.yml                     │   ├── data/sources/{data_aggregator, espn_public_client, nfl_data_py_client, weather_client}.py
├── docs/{CLEANUP_SUMMARY_2024, DEPLOYMENT_ROADMAP, DEPLOYMENT, IMPROVEMENTS_SUMMARY, ML_ENHANCEMENTS_SUMMARY, PROJECT_STRUCTURE, QUICKSTART}.md
├── frontend-next/                         │   ├── Dockerfile  Dockerfile.minimal  Procfile  railway.json  healthcheck.sh  start.sh  test_railway_locally.sh
│   ├── .claude/ ✱ .dockerignore .env.example .env.local ✱ Dockerfile next.config.js next-env.d.ts ✱ package.json package-lock.json postcss.config.js tailwind.config.ts tsconfig.json tsconfig.tsbuildinfo ✱ VERCEL_ENVIRONMENT_SETUP.md
│   └── src/                               │   ├── docs/{CLEANUP_SUMMARY, COMMERCIAL_USE_COMPLIANCE, DEPLOYMENT_GUIDE, ML_DOCUMENTATION}.md
│       ├── app/{about, auth/signin, auth/signup, contact, dashboard, draft, features, help, how-it-works, learn, performance, player/[id], predictions, pricing, privacy, start-sit, terms, tiers}/page.tsx, globals.css, layout.tsx, not-found.tsx, page.tsx
│       ├── components/{dashboard/{DashboardLayout,PerformanceDashboard}, draft/DraftSimulator, landing/{Accuracy,Features,Hero,HowItWorks,Pricing}, layout/{Breadcrumb,Footer,Navigation}, player/PlayerProfile, predictions/{PlayerSearch,PredictionCard,PredictionsList,WeekSelector}, providers, start-sit/StartSitEngine, tiers/{TierChart,TierVisualization,TierVisualizationAPI}}.tsx
│       ├── data/{predictions_2024, tiers_2024}.json      │   ├── docs/archive/ ✱ (9 .md + dockerfiles/{Dockerfile.complex,.dev,.multistage,.prod})
│       ├── hooks/ (empty)  types/ (empty)                │   ├── emergency_server.py  health_check.py  main.py  main_optimized.py  main_simple.py
│       └── lib/{api/{client,players,tiers}.ts, constants.ts, utils.ts}   │   ├── ml/ (24 modules: advanced_models, draft_tier_storage, efficiency_ratio, enhanced_features, enhanced_training, ensemble_predictions, feature_engineering, feature_selection, features, gmm_clustering, hyperparameter_tuning, injury_impact, model_versioning, momentum_detection, neural_network, prediction_engine, predictions_simple, predictions, ranking_algorithm, scoring_engine, trade_analyzer, train, trend_analysis, ultra_accurate_model, weather_projections)
├── infrastructure/{.DS_Store, Dockerfile (0 bytes), docker-compose.yml (782 lines), nginx.conf, terraform/main.tf}
├── models/                                │   ├── models/{__init__, database, player_profile, schemas}.py
│   ├── .gitkeep  feature_importance.json  model_metadata.json             │   ├── models/archive/ ✱ (17 files)  models/production/ (9 tracked files)  models/production/archive/ ✱ (18 files)
│   ├── features_{QB,RB,TE,WR}.pkl ✱  nn_scaler_{QB,RB,TE,WR}.pkl ✱        │   ├── railway_debug.py  railway_direct_test.py  railway_direct_uvicorn.py  railway_simple_start.py  RAILWAY_DEPLOYMENT_GUIDE.md  RAILWAY_HEALTHCHECK_FIX.md  README.md
│   ├── nn_model_{QB,RB,TE,WR}/{metadata.json, model_X.keras} ✱             │   ├── requirements.txt  requirements-dev.txt  requirements-minimal.txt  requirements-prod.txt
│   └── nn_model_QB.h5/ ✱ (EMPTY DIRECTORY, not a file)                     │   ├── scripts/ ✱ (untracked: data_collection/ 6, tests/ 15, training/ 11, inspect_features.py, validate_requirements.py)
├── scripts/ (42 tracked files: 20 test_*/train_*/demonstrate_* .py, deploy*.sh, fetch_*.py, init_database.py, run/, setup/, README.md, …)   │   ├── services/{explainer, llm_service, predictor, stripe_service, subscription_service, vector_store}.py
├── ssl/ (empty) ✱                         │   ├── start_app.py  start_railway.py  start_simple.py  start.py  test_startup.py
├── tests/ (empty) ✱                       │   ├── tasks/{__init__, train_models, update_data}.py
└── vercel.json                            │   └── vector_db/chroma.sqlite3 (tracked, 160 KB)
```

### 0.3 Line counts (tracked files)

| Top-level | Files | Lines |
|---|---|---|
| `backend/` | 118 | 50,917 |
| `frontend-next/` | 59 | 16,773 (8,081 of which is `package-lock.json`) |
| `scripts/` | 42 | 7,031 |
| `docs/` | 7 | 1,214 |
| `infrastructure/` | 4 | 917 |
| `README.md` | 1 | 520 |
| `docker-compose.yml` | 1 | 136 |
| `deploy-to-railway.sh` | 1 | 91 |
| `Makefile` | 1 | 65 |
| `models/` | 3 | 52 |
| `LICENSE`, `vercel.json`, `.gitignore` | 3 | 137 |

| Language (ext) | Files | Lines |
|---|---|---|
| Python `.py` | 108 | 31,052 |
| TSX `.tsx` | 42 | 7,800 |
| TS `.ts` | 6 | 369 |
| Markdown `.md` | 17 | 2,562 |
| Shell `.sh` | 19 | 1,233 |
| JSON `.json` | 10 | 8,522 |
| YAML `.yml` | 2 | 918 |
| Other (`.txt` 169, `.conf` 107, `.css` 55, `.tf` 28, `.js` 13, `.sql` 9, `.ini`, `.mako`, `.pkl` ×8, `.sqlite3`) | | |

Untracked-but-present Python: 45 files (35 under `backend/scripts/**`, 10 `.ipynb_checkpoints` copies).

### 0.4 Files larger than 1 MB

| Path | Bytes | Tracked |
|---|---|---|
| `backend/models/production/prod_WR_model_20250731_171728.pkl` | 2,958,577 | yes |
| `backend/models/production/prod_RB_model_20250731_171728.pkl` | 2,457,025 | yes |
| `backend/models/production/prod_TE_model_20250731_171728.pkl` | 1,866,033 | yes |
| `backend/models/production/prod_QB_model_20250731_171728.pkl` | 1,323,585 | yes |
| `backend/models/archive/simple_random_forest_20250731_154126.pkl` | 8,609,153 | no |
| `backend/models/production/archive/prod_{WR,RB,TE,QB}_model_20250731_{171642,171705}.pkl` (8 files) | same sizes as tracked set | no |

No notebooks exist (`find -name '*.ipynb'` → none). `.git` is 7.7 MB.

### 0.5 `.gitignore` entries that matter for deployment

Root `.gitignore`: `.env`, `.env.local`, `.env.*` (:43-45); `*.db`, `*.sqlite` (:47-48); `models/saved/`, `models/trained/`, `data/raw/`, `data/processed/` (:49-52); `ssl/`, `*.pem`, `*.crt`, `*.key` (:59-62); `*.pkl`, `*.h5`, `*.joblib`, `*.pt`, `*.pth`, `*.keras`, `models/nn_model_*/`, `models/enhanced/`, `models/ultra_accurate/`, `*.png`, `*.parquet` (:75-86); `.next/`, `out/`, `build/`, `node_modules/` (:101-106).

`backend/.gitignore`: `.env`, `.env.*`, `!.env.example` (:47-49); `*.db`, `*.sqlite3`, `*.sqlite`, `vector_db/` (:52-55) — **but `backend/vector_db/chroma.sqlite3` is tracked anyway** (added before the rule); `*.pkl`, `*.joblib`, `*.h5`, `*.pt`, `*.pth`, `!models/production/*_20250731_171728.*` (:58-63); `*.csv`, `*.parquet`, `data_cache/`, `cache/` (:66-69); `archive/`, `scripts/` (:91-92) — this last pair silently excludes **all 35 files under `backend/scripts/`**, including the only trainer of the production models.

`backend/.dockerignore` excludes `*.json`, `*.pkl`, `*.h5`, `*.parquet`, `models/*.pkl`, `*.md`, `docs/`, `tests/`, `test_*.py` (lines ~60-85). Consequence: **a Docker image built from `backend/Dockerfile` contains no model artifacts at all** (`backend/models/production/*.pkl` and `models/*.json` are stripped at build).

---

## 1. Canonical paths

### 1.1 FastAPI app objects and launch points

| App object | File:line | Imports | Import result (from repo root, `agentic_ai_env`) |
|---|---|---|---|
| `app = FastAPI(title="Fantasy Football AI API", version="1.0.0", … lifespan=lifespan)` | `backend/main.py:85` | `from api import auth, players, predictions, subscriptions, tiers` (:21), `from models.database import engine, Base` (:22), `from api import llm_endpoints` / `services.llm_service` / `services.vector_store` (:34-36), `from api import predictions_v2, payments` (:127) | `import backend.main` → **OK but degraded**: `ERROR:backend.main:Database models not available: No module named 'api'` → 6 routes only |
| `app = FastAPI(title="Fantasy Football AI API - Optimized", version="2.0.0", …)` | `backend/main_optimized.py:68` | `prometheus_client`, `psutil` (:17-18, **not in requirements.txt**), `from backend.core.cache import cache, CacheWarmer` (:21), `backend.api.*`, `backend.models.database` (:22-24) | `import backend.main_optimized` → OK (Redis connect fails at import, logged). From `backend/` cwd: `ModuleNotFoundError: No module named 'backend.core'` |
| `app = FastAPI(title="Fantasy Football AI API", version="1.0.0", …)` | `backend/main_simple.py:17` | fastapi only | OK from both cwds; routes `/`, `/health`, `/ready` |
| `app = FastAPI()` | `backend/health_check.py:13` | fastapi | standalone diagnostic |
| `app = FastAPI(title="Fantasy Football AI - Minimal Fallback")` | `backend/start_railway.py:144` | fallback inside `start_minimal_server()` | diagnostic launcher |
| `app = FastAPI()` | `backend/railway_debug.py:84` | fallback | diagnostic launcher |
| `app = FastAPI()` | `backend/test_startup.py:113` | inside `test_minimal_server()` | diagnostic |

Launch points:

| Launcher | Command | cwd / PYTHONPATH implied | Referenced by |
|---|---|---|---|
| `backend/railway.json:7` | `"startCommand": "python start.py"`, `"builder": "DOCKERFILE"`, `healthcheckPath: /health` | `/app` (= `backend/`), `PYTHONPATH=/app` (`backend/Dockerfile:16,29`) | Railway |
| `backend/start.py:16` | `[sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", port]` via `subprocess.run` | `backend/` | `railway.json`, `Procfile` (`web: python start.py`) |
| `backend/Dockerfile:47` | `CMD ["python", "-u", "railway_direct_uvicorn.py"]` → `uvicorn.run("main:app", …)` (`railway_direct_uvicorn.py:36`; **exits if `PORT` unset**, :16-19) | `backend/` | Docker default (overridden by `railway.json` startCommand) |
| `backend/start.sh:21` | `exec python -m uvicorn main:app --host 0.0.0.0 --port $PORT` | `backend/` | nothing |
| `backend/start_app.py:38`, `start_simple.py:41` (`main_simple:app`), `railway_simple_start.py:13`, `start_railway.py:166,176`, `railway_debug.py:102,126`, `health_check.py:34`, `main_simple.py:73`, `main.py:155`, `main_optimized.py:321` (`workers=4, loop="uvloop"`; `uvloop` not pinned) | uvicorn.run variants | `backend/` | Dockerfile comments `:49-53` list them as "alternative CMD options" |
| `docker-compose.yml:65` | `uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload`, `PYTHONPATH: /app:/app/backend` (:51) | repo root | `make up` |
| `Makefile:41` | `cd backend && uvicorn main:app --reload …` | `backend/` | `make dev-backend` |
| `infrastructure/docker-compose.yml:66` | `uvicorn main:app …` with build context `./backend` | `backend/` | nothing (legacy; see 1.4) |
| `scripts/run/run_backend.sh:27`, `scripts/run/start_backend.sh:11` (absolute conda path) | `python -m uvicorn main:app --reload` | `backend/` | docs only |

**Empirical route registration test** (script in scratchpad, `DATABASE_URL` unset, Redis down):

| Configuration | `DATABASE_AVAILABLE` | Routes registered | Cause |
|---|---|---|---|
| cwd=`backend`, `PYTHONPATH=backend` (mirrors `backend/Dockerfile`/Railway) | False | 6: `/openapi.json /docs /docs/oauth2-redirect /redoc / /health` | `ERROR:main:Database models not available: No module named 'backend.models'` — `backend/api/auth.py:13` etc. use `from backend.models.database import …`, unresolvable when only `backend/` is on the path. (Locally the empty untracked `backend/backend/` dir makes the error read `backend.models` rather than `backend`; in the Railway image that dir is absent and the outcome is the same.) |
| cwd=`backend`, `PYTHONPATH=root:backend` (mirrors `docker-compose.yml`) | False | 6 | In this env: `ImportError: cannot import name 'is_data_content_block' from 'langchain_core.messages'` via `api/tiers.py:12 → services/llm_service.py:15` (installed langchain 0.2.16 ≠ pinned 0.1.0). With pinned versions the import may pass, but then `backend/api/tiers.py:20 llm_service = LLMService()` raises `TypeError` (`LLMService.__init__(self, openai_api_key: str, anthropic_api_key: str)` has no defaults, `backend/services/llm_service.py:117`), which is **not** an `ImportError` and is not caught by `main.py:25` → process dies. **Unclear** which of the two happens in the deployed image; neither yields a working API. Also `models.database` and `backend.models.database` load as two distinct module objects (two `Base`, two `engine`). |
| cwd=root, `import backend.main` (mirrors `uvicorn backend.main:app`) | False | 6 | `No module named 'api'` (`backend/main.py:21` bare import) |

`backend/RAILWAY_HEALTHCHECK_FIX.md:13-21` documents deliberately stripping the `backend.` prefix from `main.py` only, which created this split.

Live check (2026-09-10, outbound connectivity verified against api.github.com and api.sleeper.app): `https://fantasy-football-ai-production-4441.up.railway.app/health` → **timeout, HTTP 000 after 15 s**. The backend named in `frontend-next/.env.example:2`, `VERCEL_ENVIRONMENT_SETUP.md:9`, and `deploy-to-railway.sh:40` is unreachable; whether it is down or removed is **unclear**.

### 1.2 Dependency manifests

| File | Lines | Notable pins |
|---|---|---|
| `backend/requirements.txt` | 75 | `fastapi==0.104.1`, `uvicorn[standard]==0.24.0`, `pydantic==2.5.0`, `sqlalchemy==2.0.23`, `asyncpg`, `psycopg2-binary`, `alembic`, `redis==4.6.0`, `aioredis==2.0.1`, `fastapi-limiter`, `python-jose`, `passlib`, `PyJWT`, `stripe==7.8.0`, `celery==5.3.4`, `nfl_data_py==0.3.1`, `beautifulsoup4==4.12.3`, **`tensorflow-cpu==2.16.1`** (:41), `scikit-learn==1.3.2`, `pandas==2.1.3`, `numpy==1.24.3`, `joblib==1.3.2`, `scipy==1.11.4`, **`xgboost==2.0.3`** (:47) despite comment at :74 "xgboost and lightgbm removed", `langchain==0.1.0`, `langchain-openai`, `langchain-anthropic==0.1.1`, `openai==1.10.0`, `anthropic==0.17.0`, `weaviate-client==3.25.3` (never imported), `sentence-transformers==2.2.2` (pulls torch), `chromadb==0.4.22`, `websockets`, `sse-starlette`, `tiktoken`, `tenacity`, `asyncio-throttle` |
| `backend/requirements-prod.txt` | 47 | Same as above minus `nfl_data_py`, `beautifulsoup4`, `xgboost`, all LLM/vector packages; keeps `tensorflow-cpu==2.16.1` (:39). **Not referenced by any Dockerfile.** |
| `backend/requirements-minimal.txt` | 13 | fastapi, uvicorn, pydantic, python-dotenv — used by `Dockerfile.minimal` |
| `backend/requirements-dev.txt` | 37 | `tensorflow>=2.16.1` (unpinned upper), `matplotlib`, `seaborn`, `plotly`, `optuna==4.4.0`, `beautifulsoup4==4.13.4` (≠ prod 4.12.3), **`xgboost==3.0.3`** (≠ prod 2.0.3), `lightgbm==4.6.0` |
| `infrastructure/docker-compose.yml:226-262` (embedded text, see 1.4) | | `tensorflow==2.15.0`, `redis==5.0.1`, `requests==2.31.0` — a third TF pin |
| `frontend-next/package.json` | 45 | `next 14.2.25`, `react 18.2.0`, `@clerk/nextjs ^4.27.7`, `@stripe/stripe-js` (unused), `@tanstack/react-query`, `zustand` (unused), `recharts` (unused), `d3`, `axios`, `framer-motion`, `typescript 5.3.3`; eslint 8.57 + `eslint-config-next` but **no `.eslintrc`** |
| `pyproject.toml`, `setup.py`, `setup.cfg`, `Pipfile`, `environment.yml`, `pytest.ini`, `tox.ini` | | **none exist** |

Conflicting pins: TensorFlow `2.16.1` (prod/req) vs `>=2.16.1` (dev) vs `2.15.0` (infra text); xgboost `2.0.3` vs `3.0.3` (the archived XGB pickle was written by 3.0.3); redis `4.6.0` vs `5.0.1`; beautifulsoup4 `4.12.3` vs `4.13.4`. `matplotlib`/`seaborn` are imported at module level by `backend/ml/neural_network.py:19-20` (on the served import chain) but exist only in `requirements-dev.txt:26-27` — in an image from `requirements.txt` the `predictions_v2` router import would fail with `ModuleNotFoundError: matplotlib` and be swallowed at `main.py:137`.

**Is TensorFlow imported in runtime code?** Yes, transitively at import time on the served path: `backend/main.py:127 from api import predictions_v2` → `backend/api/predictions_v2.py:17 from backend.services.predictor import EnhancedPredictor` → `backend/services/predictor.py:18 from backend.ml.predictions import PredictionEngine` → `backend/ml/predictions.py:21 from backend.ml.neural_network import FantasyNeuralNetwork` → `backend/ml/neural_network.py:13-15 import tensorflow as tf / from tensorflow import keras / from tensorflow.keras import …`. No API/service module imports TF directly (grep over `backend/main.py backend/api backend/services backend/core backend/models/*.py` → 0 hits). TF is not functionally used for any served prediction (§5).

### 1.3 Training scripts / modules

| File | Trains | Inputs | Artifacts written | Imports cleanly? | Referenced by |
|---|---|---|---|---|---|
| `backend/ml/train.py` | GMM draft tiers (`GMMDraftOptimizer(n_components=16, n_pca_components=7)` :48) + per-position Keras NN (:288-300) | Postgres `players`/`player_stats` JSONB | `./models/gmm_scaler.pkl` :135, `gmm_draft_tiers.pkl` :186, `nn_scaler_{pos}.pkl` :284, `nn_model_{pos}.h5` **as a directory** (:314 → `save_model` does `os.makedirs`), `training_results.json` :355 | OK | `tasks/train_models.py:7`, `scripts/train_neural_network.py:20`, `Makefile:62`, `README.md:380` (`--evaluate` flag does not exist; `:397-401` has no argparse) |
| `backend/ml/enhanced_training.py` | XGB + RF + GB mean ensemble (:167-192, :296) | `DataAggregator.build_training_dataset` | `models/model_{ver}_{pos}_{name}.pkl`, `scaler_…`, `metadata_{ver}.json` (:333-358) | OK | nothing |
| `backend/ml/ultra_accurate_model.py` | XGB, LGBM, RF, ExtraTrees, 5× Keras NN, GB, Ridge stacking (:45-115, :268-370) | caller DataFrames | `{path}.pkl` + `{path}_nn_ensemble.keras` (:468-487) | OK (but `early_stopping_rounds` in `.fit` :295 is rejected by xgboost ≥2.0 → cannot train as written) | `scripts/train_ultra_accurate_models.py:20`, `scripts/test_ultra_accurate.py:16` |
| `backend/ml/neural_network.py` | `FantasyNeuralNetwork` Keras MLP, Huber loss, MC-dropout predict (:96-131, :299-315) | numpy | `{base}/model_{pos}.keras`, `scaler_{pos}.pkl`, `metadata.json` (:470-504); loader `:527` reads `model_{pos}` **without** `.keras` → round-trip broken | OK | `ml/train.py`, `ml/predictions.py:21`, `ml/ensemble_predictions.py:21`, scripts |
| `backend/ml/gmm_clustering.py` | StandardScaler → PCA → `GaussianMixture(covariance_type='full', n_init=5)` (:148-176), BIC search (:90-124) | numpy | `joblib.dump({'gmm','scaler','pca',…})` (:417-432) | OK | `ml/train.py:23`, `draft_tier_storage.py:16`, `ranking_algorithm.py:20`, scripts |
| `backend/ml/hyperparameter_tuning.py` | Optuna TPE over `advanced_models` (:56-107, :241-268) | numpy | `hyperparameter_results_{type}_{ts}.json` | **FAIL**: `optuna-integration[tfkeras]` missing (:7) | `scripts/train_enhanced_models.py:32` |
| `backend/ml/feature_selection.py` | F-test/MI/LassoCV/RF/RFECV/SHAP selection (:67-75) | DataFrame | none | OK | **nothing** |
| `backend/ml/advanced_models.py` | Transformer/LSTM/CNN class defs only | — | — | OK | `hyperparameter_tuning.py:18`, 2 scripts |
| `backend/ml/enhanced_features.py` | feature factory + SelectKBest (:491-518) | DataFrame | none | OK | 2 scripts + tests |
| `backend/ml/trend_analysis.py` | linear trend/streak analytics (no model) | Postgres `player_stats` (no week bound) | none | OK | `ml/predictions.py:20` (served import chain) |
| `backend/data/data_pipeline.py` | GMM + NN on `np.random.randn()` features (:394-396), target `points_per_game` (:425) | synthetic (`:189-293`) | `models/gmm_model.pkl`, `models/neural_network/` | **FAIL**: `No module named 'schedule'` (:15); also 33 undefined names (ruff F821, imports commented out at :20-26) | `docs/PROJECT_STRUCTURE.md:16` only |
| `backend/tasks/train_models.py` | Celery wrappers around `ModelTrainer` | DB | via `ModelTrainer` | OK | `celery_app.py:17,45,52` (weekly beat) |
| `scripts/train_and_save_models.py` | Keras NN per position | **synthetic** `generate_sample_data(2000)` (:32-106, :244); target is a linear function of the features (:61-67) | `models/nn_model_{pos}/`, `nn_scaler_{pos}.pkl`, `features_{pos}.pkl`, `feature_importance.json` (hardcoded dict :271-308, "simulated" :270), `model_metadata.json` with `accuracy = 0.892  # Simulated accuracy` (:268) | OK (static) | `scripts/deploy-all.sh:54` |
| `scripts/train_enhanced_models.py` | Optuna MLP | `EnhancedDataCollector` or `data/enhanced_nfl_dataset.parquet` | `models/enhanced/…` | **FAIL**: imports nonexistent `backend.ml.fantasy_predictor`, `backend.ml.draft_optimizer` (:33-34) | `docs/IMPROVEMENTS_SUMMARY.md:96,105` |
| `scripts/train_models_simple.py` | RF per position incl. K (:108-114) | Postgres JSONB same-week stats | `./models/rf_model_{pos}.pkl`, `rf_scaler_`, `rf_features_` (:124-130) — **not on disk** | OK | nothing (its outputs are what `ml/predictions_simple.py:51` and `ml/ensemble_predictions.py:72` look for) |
| `scripts/train_neural_network.py` | Keras via `ModelTrainer` | Postgres | `nn_model_{pos}.h5` dir, `nn_scaler_{pos}.pkl` | OK | docs |
| `scripts/train_ultra_accurate_models.py` | ultra ensemble | **synthetic** `SyntheticDataGenerator(years=10, players_per_position=100)` (:46-47) | `models/ultra_accurate/…` | OK; note `os.system("pip install xgboost lightgbm")` at :473 | docs |
| `scripts/demonstrate_92_accuracy.py`, `demonstrate_accuracy_simple.py`, `final_accuracy_demo.py` | RF/GB/NN blends | **synthetic**, target generated from the same features | PNG only | OK | docs |
| `backend/scripts/training/train_production_ml_models.py` (**untracked**) | RF vs XGB per position, keep lower val-MAE (:243-285) → RF won ×4 | `nfl_data_py.import_weekly_data([2019..2024])`, REG only (:30-40) | `models/production/prod_{pos}_model_{ts}.pkl`, `prod_{pos}_scaler_{ts}.pkl`, `prod_models_metadata_{ts}.json` (:369-417) → **this is the producer of the tracked production artifacts** (feature names on the scalers match :74-126; `backend/.gitignore:63` whitelists exactly this timestamp) | OK (static); **no `__main__` guard** — runs on import | nothing |
| Other untracked `backend/scripts/training/*` (10 files) | XGB/RF/GB ensembles, audits, rebuilds | `nfl_data_py` 2015–2024 or Sleeper → Supabase (**hard-coded DB password** in `quick_ml_test.py:18`, `run_ml_pipeline.py:20`, `train_real_data.py:24`) | `models/{simple,final,comprehensive,proper,fantasy}_*` → produced the untracked `backend/models/archive/*` | `run_ml_pipeline.py` fails (imports `EnsemblePredictor`/`MLPredictor`/`FeatureEngineer` under wrong names :25-27) | nothing |

### 1.4 Deployment files

| File | Launches | Ports | Env vars required | References to missing things |
|---|---|---|---|---|
| `backend/Dockerfile` (52 lines) | `python:3.10-slim`, apt `default-libmysqlclient-dev build-essential gcc g++ libpq-dev` (:4-14, MySQL client for a Postgres app), `COPY requirements.txt` + `pip install` (:22-23), `COPY . .` (:26), `ENV PYTHONPATH=/app` (:29), `CMD ["python","-u","railway_direct_uvicorn.py"]` (:47) | `$PORT` (exits if unset) | `PORT`; everything else optional | `.dockerignore` strips `*.pkl`/`*.json`/`*.md` → no models in image; import split (§1.1) |
| `backend/Dockerfile.minimal` | `main_simple:app` via `start_simple.py` | `$PORT`/8000 | — | health-check stub only |
| `backend/railway.json` | Dockerfile build, `startCommand: python start.py`, healthcheck `/health` 60 s, startup 120 s, restart ON_FAILURE ×3 | — | `PORT` | overrides Dockerfile CMD |
| `backend/Procfile` | `web: python start.py` | | | |
| `docker-compose.yml` (root, 136 lines) | postgres:15, redis:7, `backend` (`uvicorn backend.main:app --reload`, `PYTHONPATH=/app:/app/backend`, mounts `./backend`, `./scripts`, `./models`), `frontend-next` (`npm run dev`, :3000), `celery_worker` (`celery -A backend.celery_app worker --queues=ml_training,data_updates`), `celery_beat` | 5432, 6379, 8000, 3000 | `JWT_SECRET_KEY` (default), `STRIPE_*`, `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY`, `CLERK_SECRET_KEY`, `NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY` | `frontend-next/Dockerfile` copies `.next/standalone` (:34) but `next.config.js` has no `output: 'standalone'` and **no `module.exports`** (file ends at line 9 without export) |
| `infrastructure/docker-compose.yml` (782 lines) | Lines 1-153 are a compose file (postgres, redis, backend `uvicorn main:app`, **Streamlit `frontend` on 8501**, `celery -A tasks …` (no `tasks` top-level module), nginx 80/443). **Lines 154-782 are not YAML**: they are concatenated text of a `backend/Dockerfile` (python:3.11), a `frontend/Dockerfile` (Streamlit), a `requirements.txt` with `tensorflow==2.15.0`, an `.env` template (:385-402, includes an SMTP example with a Gmail address), a nginx conf, and a Makefile. `docker compose -f` would fail to parse it. | 8000, 8501, 80, 443 | `DB_USER/DB_PASSWORD/DB_NAME` | `./frontend` build context (missing), `nginx/ssl` |
| `infrastructure/Dockerfile` | **0 bytes** | | | |
| `infrastructure/nginx.conf` (107 lines) | upstreams `backend:8000`, `frontend:8501`; `/api/` → backend, `/` → frontend, `/_stcore/stream` (Streamlit) ; SSL certs at `/etc/nginx/ssl/{cert,key}.pem` | 80→301, 443 | | Streamlit frontend does not exist; `ssl/` empty |
| `infrastructure/terraform/main.tf` (28 lines) | provider `aws ~>5.0`, 3 variables, comment `# (Full terraform config is in the original artifact)` — **zero resources** | | `db_password` | `README.md:330 cd terraform` (root dir missing) |
| `Makefile` (65 lines) | `build/up/down/logs/clean/test/migrate/shell` via docker-compose; `dev-backend`; `dev-frontend: cd frontend && streamlit run app.py` (:44, dir missing); `deploy-prod/staging → ./scripts/deploy.sh`; `db-backup/restore → ./scripts/{backup,restore}-db.sh` (**missing**); `train-models → python -m ml.train`; `update-predictions → python -m data.update_predictions` (**no such module**) | | | |
| `vercel.json` | `regions: ["iad1"]`, `functions: {"frontend/src/app/api/create-checkout-session/route.ts": {maxDuration: 10}}` — **path does not exist** (no `route.ts` anywhere; no `frontend/`) | | | |
| `deploy-to-railway.sh` (91 lines) | Railway CLI: deploys `frontend-next` and `backend` as Railway services, sets `NEXT_PUBLIC_API_URL=https://fantasy-football-ai-production-4441.up.railway.app` (:40), placeholder secrets (:62-66) | | Railway CLI, `jq` | |
| `scripts/deploy.sh` | AWS ECR push + ECS or SSH `docker-compose up` on EC2; builds `./frontend` (:27, missing); sources `.env.production` (missing) | | AWS creds, `~/.ssh/fantasy-key.pem` | |
| `scripts/deploy-all.sh`, `deploy-backend.sh`, `deploy-frontend.sh` | Railway (`railway up`) + Vercel (`vercel --prod`); trains models via `scripts/train_and_save_models.py` if `models/*.h5` absent (:52-54) | | railway, vercel, docker CLIs | `frontend-next/.env.production` written by script |
| `frontend-next/Dockerfile` | node:18-alpine, `npm ci`, `npm run build`, copies `.next/standalone` → `node server.js` on 3000 | 3000 | | `standalone` output not configured |
| `backend/healthcheck.sh`, `test_railway_locally.sh` | curl `/health` | | | |
| `.github/workflows/` | **empty directory** (untracked) — no CI | | | |

### 1.5 `scripts/` one-liners

All 41 `scripts/**` files are tracked; only 8 are documented in `scripts/README.md`. "Ref" = referenced by another file (excluding `scripts/README.md`).

| File | Purpose | Ref |
|---|---|---|
| `demonstrate_92_accuracy.py` | synthetic-data RF+GB+NN "92% accuracy" demo, PNG output | `docs/IMPROVEMENTS_SUMMARY.md:98` |
| `demonstrate_accuracy_simple.py` | same, RF+GB | none |
| `final_accuracy_demo.py` | same, grid-searched blend | none |
| `deploy-all.sh` / `deploy-backend.sh` / `deploy-frontend.sh` / `deploy.sh` | Railway/Vercel/AWS deploy | `Makefile:48,51`; each other |
| `fetch_historical_stats.py` | Sleeper weekly stats **2021–2023** (`:46 self.seasons = ['2021','2022','2023']`) → `player_stats` | none |
| `fetch_sleeper_data.py` | Sleeper players → `players` table; stats fetching disabled (`:218-227`, `:262`) | `docs/QUICKSTART.md`, `run_data_setup.sh:40` |
| `init_database.py` | `Base.metadata.create_all` + extra indexes (duplicates alembic) | `run_data_setup.sh:30`, `setup_production.sh:126`, docs |
| `run_data_setup.sh` | init DB + fetch players | none |
| `run_with_env.sh` | exports **hard-coded Supabase `DATABASE_URL` with password** (:5) | none |
| `run/quick_start.sh`, `run_backend.sh`, `run_docker.sh`, `run_with_conda.sh` | local launchers with absolute `/Users/christopherbratkovics/...` paths | `docs/CLEANUP_SUMMARY_2024.md` |
| `run/start_backend.sh` | exports **the same hard-coded Supabase password** (:7), launches uvicorn via absolute conda path | docs |
| `setup_production.sh` | writes `frontend/.streamlit/secrets.toml`, curls :8501 | `docs/QUICKSTART.md` |
| `setup/setup_local.sh`, `setup_with_python310.sh`, `setup_postgres.sql` | local env; SQL creates role `fantasy_user`/`fantasy_pass` | docs, `run_with_conda.sh:32` |
| `test_all_improvements.py`, `test_apis_simple.py`, `test_data_quality.py` (imports nonexistent `backend.core.config`), `test_db_connection.py`, `test_efficiency_ratio.py`, `test_enhanced_pipeline.py`, `test_enhanced_training.py`, `test_ensemble_predictions.py`, `test_gmm_clustering.py`, `test_ml_complete.py`, `test_ml_system.py`, `test_mvp.sh`, `test_tier_integration.py`, `test_ultra_accurate.py` | ad-hoc smoke scripts, not pytest suites (several have 0 `def test_` functions and run at import) | mostly none; a few in `docs/ML_ENHANCEMENTS_SUMMARY.md` |
| `train_and_save_models.py`, `train_enhanced_models.py`, `train_models_simple.py`, `train_neural_network.py`, `train_ultra_accurate_models.py` | see §1.3 | `deploy-all.sh:54`, docs |
| `verify_api_methods.py` | checks Sleeper client method names | `run_data_setup.sh:26` |

Referenced but missing: `scripts/backup-db.sh`, `scripts/restore-db.sh` (`Makefile:55,58`), `scripts/verify_setup.sh` (`docs/QUICKSTART.md:98`), `test_llm_endpoints_full.py` (`docs/DEPLOYMENT_ROADMAP.md:173`).

### 1.6 Verdict

**The canonical runtime path is `backend/railway.json` → `python start.py` → `uvicorn main:app` (cwd `backend/`, `PYTHONPATH=/app`) → `backend/main.py` → [intended] `api/{auth,players,predictions,tiers,subscriptions,predictions_v2,payments,llm_endpoints}` → `services/predictor.py` → `ml/predictions.py`.** In that configuration the router import fails structurally (`No module named 'backend.models'`), so the deployed app exposes only `/` and `/health`. Even when both import roots are present (`docker-compose.yml`), `api/tiers.py:20` instantiates `LLMService()` with no arguments (TypeError) and, if that were fixed, the served prediction endpoints return constants or hash-seeded random numbers and load **no model** (§5, §7). The only real-data models (`backend/models/production/prod_*`) are loaded by nothing.

Redundant / legacy files (**52 files**): `backend/main_optimized.py`, `backend/main_simple.py`, `backend/health_check.py`, `backend/emergency_server.py`, `backend/railway_debug.py`, `backend/railway_direct_test.py`, `backend/railway_simple_start.py`, `backend/start_app.py`, `backend/start_railway.py`, `backend/start_simple.py`, `backend/start.sh`, `backend/test_startup.py`, `backend/check_imports.py`, `backend/healthcheck.sh`, `backend/test_railway_locally.sh`, `backend/Dockerfile.minimal`, `backend/requirements-prod.txt`, `backend/requirements-minimal.txt`, `backend/api/websocket_routes.py` (imports nonexistent `backend.core.auth`), `backend/core/websocket.py`, `backend/ml/{injury_impact, weather_projections, model_versioning, trade_analyzer, feature_selection, advanced_models, hyperparameter_tuning, enhanced_training, ultra_accurate_model, enhanced_features, predictions_simple, ensemble_predictions, ranking_algorithm, efficiency_ratio, momentum_detection, scoring_engine, feature_engineering, draft_tier_storage}.py` (none reachable from `main.py`), `backend/data/{espn_client, data_pipeline, fetch_players (0 bytes), enhanced_data_collector, synthetic_data_generator}.py`, `infrastructure/{docker-compose.yml, Dockerfile, nginx.conf, terraform/main.tf}`, `vercel.json`, `scripts/deploy.sh`, `scripts/demonstrate_*.py`, `scripts/final_accuracy_demo.py`, `scripts/train_ultra_accurate_models.py`, `scripts/train_enhanced_models.py`, `.ipynb_checkpoints/` (all).

---

## 2. Data sources and licensing

### 2.1 External sources

| Source | File:function | Endpoint (verbatim) | Auth | Rate limit / cache | Canonical path? |
|---|---|---|---|---|---|
| Sleeper | `backend/data/sleeper_client.py:63 SleeperAPIClient`; `:182 get_all_players`, `:208 get_nfl_state`, `:322 get_stats`, `:353 get_projections`, `:380 get_week_stats` | `:75 BASE_URL = "https://api.sleeper.app/v1"`; `players/{sport}`, `state/nfl`, `stats/{sport}/{season_type}/{season}/{week}`, `projections/...` | none; **requires reachable Redis at construction** (`:91-95`) | `:76 RATE_LIMIT = 900`; `:108-114 @sleep_and_retry @limits(calls=900, period=60) @backoff.on_exception(backoff.expo, …, max_tries=3)`; Redis TTL cache `:146-180` | No. Celery `tasks/update_data.py:7`; tracked `scripts/fetch_sleeper_data.py`, `fetch_historical_stats.py` |
| Sleeper raw `requests.get` | untracked `backend/scripts/data_collection/{collect_complete_nfl_data.py:72,170, continue_data_collection.py:41, quick_data_test.py:101}` | `https://api.sleeper.app/v1/stats/nfl/regular/{season}/{week}` | none (hard-coded Supabase DB password in same files) | `time.sleep(0.5)` | No |
| nflverse via `nfl_data_py` | `backend/data/sources/nfl_data_py_client.py:19`; `:126 nfl.import_weekly_data(years)`, `:86 import_pbp_data`, `:166 import_rosters`, `:201 import_schedules`, `:238 import_ngs_data`, `:273 import_combine_data`, `:310 import_qbr`, `:345 import_win_totals` | library (GitHub-hosted parquet) | none; requires Redis (`:29`) | Redis DataFrame cache `:42-58` | No. Only `data_aggregator.py:19` → `ml/enhanced_training.py:24` (unimported); untracked scripts; `nfl_data_py==0.3.1` pinned in prod requirements. `nflreadpy` not used. |
| ESPN Fantasy (authenticated) | `backend/data/espn_client.py:14 ESPNClient` | `:21 "https://fantasy.espn.com/apis/v3/games/ffl"`, `/seasons/{year}/segments/0/leagues/{id}`, `/players` | `ESPN_S2`, `ESPN_SWID` cookies (`:25-36`) | none | No importer anywhere |
| ESPN public | `backend/data/sources/espn_public_client.py:20` | `:34 "https://site.api.espn.com/apis/"` | none; `ESPN_RATE_LIMIT` (:48) | `:67-74 asyncio.sleep`; Redis cache `:86-106` | No (aggregator only) |
| sportsdata.io (paid) | `backend/data/enhanced_data_collector.py:134` | `:61 'https://api.sportsdata.io/v3/nfl/stats/json/'`, `:138 PlayerGameStatsByWeek/{year}/{type}/{week}`, header `Ocp-Apim-Subscription-Key` | `SPORTSDATA_API_KEY` (:70; in no `.env.example`) | none | No (3 tracked scripts) |
| collegefootballdata.com | `enhanced_data_collector.py:188` | `:190 stats/player/season`, Bearer | `CFBD_API_KEY` (:72) | none | No |
| OpenWeatherMap | `enhanced_data_collector.py:284` | `:294 onecall/timemachine` | `OPENWEATHER_API_KEY` (:71) | none | No |
| Open-Meteo | `backend/data/sources/weather_client.py:67`; `:100 get_game_weather` | `:73 "https://api.open-meteo.com/v1/forecast"`, `:74 archive-api…/v1/archive` | none; `WEATHER_CACHE_TTL` | Redis cache `:131-146`; no backoff | No (aggregator only) |
| Weather (mock) | `backend/ml/weather_projections.py:339` | none — `:344 "(Mock implementation…)"`, `np.random` | | | No importer |
| Synthetic | `backend/data/synthetic_data_generator.py:14`; `backend/data/data_pipeline.py:248` | none | | | scripts only |
| **Served "predictions"** | `backend/ml/prediction_engine.py:30` | none — `:21-22 "# In production, this would load actual trained models / # For now, we'll use a simulation"`; `:39 np.random.seed(hash(player_id) % 2**32)`; `:60,64,79-82` `np.random` | | | **Yes** via `api/tiers.py:13,21` → `/tiers/positions/{pos}` |
| Live scores / odds / injuries | `espn_public_client.py:239 get_week_games_with_odds`; `enhanced_data_collector.py:329 _get_injury_status → np.random.choice`, `:322 _get_defensive_rank → np.random.randint` | | | | No |

Unused HTTP targets in `scripts/test_apis_simple.py:37,62,86` (sportsdata, openweathermap, collegefootballdata).

### 2.2 HTML scraping (terms-of-service risk)

`backend/data/enhanced_data_collector.py`:
```
19  from bs4 import BeautifulSoup
20  import requests
65              'injuries': 'https://www.pro-football-reference.com/'
226     async def collect_combine_data(self, year: int) -> pd.DataFrame:
228         url = f"https://www.pro-football-reference.com/draft/{year}-combine.htm"
231             response = requests.get(url)
232             soup = BeautifulSoup(response.content, 'html.parser')
235             table = soup.find('table', {'id': 'combine'})
239             df = pd.read_html(str(table))[0]
386         for year in range(2014, 2025):
387             combine_df = await self.collect_combine_data(year)
```
No `User-Agent`, no delay, no robots.txt handling, 11 requests in a burst. The repo's own `backend/docs/COMMERCIAL_USE_COMPLIANCE.md:42-49` flags PFR scraping as "Verify… research/development only", yet `beautifulsoup4` ships in production `requirements.txt:37`. `:339` comment: `"# This would scrape PFF or Football Outsiders O-line rankings"` (returns `np.random.shuffle`). Not on the canonical path (imported by `scripts/train_enhanced_models.py:23`, `test_enhanced_pipeline.py:17`, `test_all_improvements.py:355`).

### 2.3 Historical training data

- **No CSV/parquet data is checked in or present on disk** (`find -name '*.csv' -o -name '*.parquet'` → none; `data/` directory does not exist). Untracked scripts write to `data/*.csv` (`collect_all_nfl_data.py:227` etc.).
- **Database-only** by design: `player_stats` (`backend/models/database.py:70-105`) written by tracked `scripts/fetch_historical_stats.py:99-112` (Sleeper, seasons `['2021','2022','2023']` at `:46`), Celery stub `tasks/update_data.py:99-111 "Stats update not implemented yet"`, and untracked `backend/scripts/data_collection/*` (2019–2024, Sleeper or nfl_data_py, into a Supabase DB whose password is hard-coded).
- **Is the 2019–2024 dataset reproducible from a tracked script? No.** The tracked ingest covers 2021–2023 via Sleeper. The 2019–2024 / 31,000-record dataset described in `backend/README.md:7`, `backend/docs/ML_DOCUMENTATION.md:4,32`, and `prod_models_metadata_20250731_171728.json` is produced only by untracked `backend/scripts/training/train_production_ml_models.py:30-40`:
  ```python
  data = nfl.import_weekly_data([2019, 2020, 2021, 2022, 2023, 2024])
  data = data[data['position'].isin(['QB','RB','WR','TE'])]
  data = data[data['season_type'] == 'REG']
  ```
  (and `:47-48` asserts `len(data) > 30000` and six seasons). Reproducibility from a clean clone: **not possible**. "2020–2024" appears nowhere in the repo.

### 2.4 Fixtures (`.json/.csv/.parquet` under backend/, frontend-next/, models/, data/)

| Path | Bytes | Tracked | Rows / keys | Read by |
|---|---|---|---|---|
| `frontend-next/src/data/predictions_2024.json` | 2,015 | yes (added `9bb67db` 2025-07-31) | `metadata{season:2024, generated:"2024-07-31", algorithm:"Ensemble Neural Network", accuracy{overall:0.931, QB{mae:6.17,within_3_points:0.89}, RB{4.92,0.91}, WR{4.99,0.90}, TE{3.88,0.94}}}`; `weekly_predictions.week_1` = **3 players**; `season_projections` = 3 players | `PerformanceDashboard.tsx:15`, `DraftSimulator.tsx:16`, `StartSitEngine.tsx:17`, `PlayerProfile.tsx:15` |
| `frontend-next/src/data/tiers_2024.json` | 6,666 | yes | `metadata.algorithm:"Gaussian Mixture Model (16 components)"`; `tiers{QB:3 tiers/8 players, RB:2/…, WR:1/3, TE:2/…}` (≈ 8–17 players total); `tier_breaks` | 6 components (`TierChart.tsx:6` always reads it even in API mode) |
| `backend/models/production/prod_models_metadata_20250731_171728.json` | 1,325 | yes | full contents in §5.2 | **no code** (only `backend/docs/ML_DOCUMENTATION.md:41`) |
| `models/model_metadata.json` | 217 | yes | `accuracy {QB:0.892, RB:0.892, WR:0.892, TE:0.892}` (literal from `train_and_save_models.py:268`) | `scripts/train_enhanced_models.py:344` only |
| `models/feature_importance.json` | 735 | yes | hand-written weights (names like `opponent_rank`, `touches` the NNs never saw) | `backend/services/predictor.py:41` (relative `./models`; from `backend/` cwd → not found → hardcoded dict `:47-84`) |
| `models/nn_model_{pos}/metadata.json` | 256 ×4 | no | `model_version "nn_v1_20250730", input_dim 3/5/4/4, feature_names: null` | `neural_network.py:512` |
| `backend/models/archive/*.json` (3), `production/archive/*.json` (2, truncated 105 B / 405 B) | | no | see §5.1 | nothing |
| `backend/vector_db/chroma.sqlite3` | 163,840 | **yes** (despite `vector_db/` ignore rule) | ChromaDB | `services/vector_store.py:46 path="./vector_db"` |
| `backend/railway.json`, `vercel.json`, `frontend-next/{package,package-lock,tsconfig}.json` | | yes | config | |

---

## 3. Database and caching

### 3.1 Tables (`backend/models/database.py`; migration `backend/alembic/versions/817e005bc9f5_initial_migration.py` agrees column-for-column)

| Table (model) | PK | Natural grain | Uniqueness at grain | Notes |
|---|---|---|---|---|
| `players` (`Player` :35-67) | `player_id String` | player | yes (PK) | indexes on position/team, name |
| `player_stats` (`PlayerStats` :70-105) | `id Integer` | player × season × week | **yes** — `UniqueConstraint('player_id','season','week')` (:102) | `stats JSONB`, `fantasy_points_{std,ppr,half} DECIMAL(5,2)` |
| `predictions` (`Prediction` :108-139) | `id Integer` | player × season × week (× model_version) | **NO** — only non-unique `Index('idx_predictions_player_week', player_id, season, week)` (:136) | readers filter by `week` only (`llm_endpoints.py:525-528,571-574`) → last row wins; `model_version String NOT NULL`, `confidence_interval JSONB`, `prediction_std` exist but the only writer uses invalid kwargs (§3.4) |
| `users` (`User` :142-164) | `id String uuid` | user | yes (email, username, stripe_customer_id unique) | `subscription_tier String` default 'free' (enum `SubscriptionTier` :28-32 unused) |
| `subscriptions` (:167-185) | `id` | user | yes (`user_id` unique) | |
| `prediction_usage` (:188-205) | `id` | user × week_start | yes | |
| `draft_tiers` (`DraftTier` :208-240) | `id` | player × season × model_version | **weak** — `UniqueConstraint('player_id','season','model_version')` (:238) but `model_version` nullable and writer `draft_tier_storage.py:52-60` never sets it → NULLs are distinct in Postgres | writer deletes per season first (:46) |
| `user_leagues` (:245-278) | `id` | user × platform × league | yes | |
| `model_performance` (`ModelPerformance` :281-312) | `id` | model × season × week × position | **NO** — non-unique index only (:308) | no writer on runtime path |

### 3.2 Is Postgres / Redis / Celery required to start?

| Service | Required for `backend/main.py` to start? | Code path / fallback |
|---|---|---|
| **Postgres** | **No** to boot; **yes** for every data route | `backend/models/database.py:361-367`: `DATABASE_URL = os.getenv("DATABASE_URL"); if DATABASE_URL: engine = create_engine(DATABASE_URL) else: engine = None; print("Warning: DATABASE_URL not set - database functionality disabled")`. `get_db()` (:376-384) raises `RuntimeError("Database not configured - SessionLocal is None")` per request. `main.py:51-58` wraps `Base.metadata.create_all` in try/except ("Continue anyway"). If the URL is set but unreachable, `create_engine` is lazy; each request fails at connect. **No SQLite/in-memory fallback** for the app DB. ML modules build their own engines with a Docker-hostname default: `backend/ml/predictions.py:26,34`, `ensemble_predictions.py:27`, `draft_tier_storage.py:21`, `trend_analysis.py:23`, … (`postgresql://fantasy_user:fantasy_pass@postgres:5432/fantasy_football`); `PredictionEngine()` is constructed at import of `api/predictions_v2.py:25`, so these engines are created at import time. `SessionLocal` is called directly (not via `get_db`) by `core/cache.py:318-322`, `tasks/*.py`, `ml/injury_impact.py:12` → `TypeError: 'NoneType' object is not callable` when unset. |
| **Redis** | **No** | `backend/core/cache.py:183 cache = RedisCache()` connects at import with 2 s timeout (:35-54); on failure `redis_client=None` and all methods return `None/False/0`. Observed at import: `Redis connection failed: Error 61 connecting to localhost:6379`. `rate_limiter.py:66-68` → `enabled=False` → allow-all. **But** LLM routes are effectively Redis-dependent: `llm_service.py:178-202 _check_rate_limits` returns `False` on any exception → `generate_response` yields `{"error": "Rate limit exceeded"}` (:249-254). `SleeperAPIClient` and `NFLDataPyClient` constructors require Redis (`sleeper_client.py:91-95`, `nfl_data_py_client.py:29`). |
| **Celery** | **No** | `backend/celery_app.py` imported only by `tasks/train_models.py:6`, `tasks/update_data.py:6`; `main.py` never imports it. Broker from `REDIS_URL` (`celery_app.py:9`); beat schedule daily `update_player_data`, weekly `train_all_models` (:38-48). Run only by `docker-compose.yml:109,129`. |
| ChromaDB / sentence-transformers | No (optional import `main.py:33-41`) | `vector_store.py:13-15` module-level imports; `initialize()` loads `all-MiniLM-L6-v2` and indexes **every** `Player` row synchronously in the event loop (`main.py:67-72`, `vector_store.py:114-175`); the instance is a local variable and used by no route |
| OpenAI/Anthropic keys | No, except `api/tiers.py:20 LLMService()` crash (§1.1) | per-request failures otherwise |
| Stripe | No | `stripe_service.py:18 stripe.api_key = os.getenv("STRIPE_SECRET_KEY")` may be None |

### 3.3 Predictions from static files instead of DB

| Code | Path | On disk? | Reached from `main.py`? |
|---|---|---|---|
| `backend/services/predictor.py:41-44` | `./models/feature_importance.json` | root `models/` yes; `backend/models/` no | yes |
| `backend/ml/predictions.py:51-52,58,62` | `./models/nn_model_{pos}.h5` + `nn_scaler_{pos}.pkl` | `models/nn_model_QB.h5` is an **empty directory**; RB/WR/TE absent; real weights at `models/nn_model_{pos}/model_{pos}.keras` (untracked) | yes — loads nothing |
| `backend/ml/ensemble_predictions.py:72-97` | `./models/rf_model_{pos}.pkl`, `rf_scaler_`, `rf_features_`, `nn_model_{pos}.h5` | none | no |
| `backend/ml/predictions_simple.py:51-59` | `rf_*` | none | no |
| `backend/ml/draft_tier_storage.py:34-37` | `./models/gmm_draft_tiers.pkl` | no | no |
| Frontend | `src/data/{predictions,tiers}_2024.json` static imports | yes | n/a (§8) |

Nothing reads `backend/models/production/*` (grep `prod_|models/production` over `*.py` → 0 hits outside untracked scripts).

### 3.4 Writer bug keeping `predictions` empty

`backend/ml/predictions.py:418-427` constructs `Prediction(… confidence_score=…, prediction_data=…)` — neither column exists on the ORM model, and NOT NULL `model_version` is omitted; SQLAlchemy raises `TypeError`, swallowed at `:433-434`. Consequently `/api/v2/predictions/accuracy/report` always returns 404 (`predictions.py:449-450`, `predictions_v2.py:212`).

---

## 4. Feature engineering and leakage

### 4.1 Where feature code lives

Four disjoint feature vocabularies exist: (a) `backend/scripts/training/train_production_ml_models.py:56-126` (untracked; the only one behind a real-data model), (b) `backend/ml/features.py` `FeatureEngineer` (26-field dataclass `:121-186`; `get_position_features` `:505-543` returns Sleeper JSONB key lists), (c) inference-time `backend/ml/predictions.py:175-228`, (d) `backend/ml/enhanced_features.py` (~100 columns, script-only), plus `backend/ml/feature_engineering.py` (DB-backed, instantiated but never called), `ensemble_predictions.py:520-582`, `predictions_simple.py:164-204`, `ultra_accurate_model.py:175-266`, and each untracked trainer's own `engineer_features`.

### 4.2 Production-model features (`train_production_ml_models.py`)

Base stats (`:64-72`): `passing_yards, passing_tds, rushing_yards, rushing_tds, receiving_yards, receiving_tds, receptions, targets, fantasy_points_ppr, attempts, completions, carries, interceptions, passing_air_yards, receiving_air_yards`. Rows sorted `['player_id','season','week']` (`:61`).

| Feature | Formula | Window | Excludes current row? |
|---|---|---|---|
| `{stat}_L1` | `data.groupby('player_id')[stat].shift(1)` (:79) | previous game (crosses seasons) | yes |
| `{stat}_L3_avg` | `data.groupby('player_id')[stat].rolling(3, min_periods=1).mean().shift(1).values` (:83) | t-3..t-1 | yes within player, **but** the `.shift(1)` is applied to the flattened groupby result, so each player's first row inherits the previous player's last value; positional `.values` assignment depends on pre-sorting. The contaminated first row is removed by `dropna(subset=['fantasy_points_ppr_L1'])` (:160-162). |
| `{stat}_L5_avg` | same, `rolling(5)` (:87) | t-5..t-1 | same |
| `{stat}_season_avg` | `data.groupby(['player_id','season'])[stat].expanding().mean().shift(1).values` (:92) | season-to-date through t-1 | yes; week 1 of season N receives season N-1's full average (a past value, mislabeled) |
| `completion_pct_L3`, `yards_per_carry_L3`, `catch_rate_L3` | ratios of L3 averages (:98-121) | derived | yes |
| `week_of_season` = `week`; `games_played` = `groupby(['player_id','season']).cumcount()` (:124-125) | | | known pre-game |

Final counts QB 44 / RB 40 / WR 27 / TE 27 (match `n_features_in_` of the pickles). Scaler fit on train only (:237-240). The "FORBIDDEN CHECK" (:130-136) is a substring test on names, not on data. No opponent, home/away, injury, or weather features. No same-week stats, no season-end aggregates, no target encoding — **this script is the exception**.

### 4.3 Leakage flags elsewhere

**Shift-less rolling windows (current row included):** `backend/ml/train.py:246` `groupby('player_id')[col].rolling(3, min_periods=1).mean()` (no shift, feeds the NN target row); `backend/ml/predictions.py:209` (inference; rolls over the last completed game, acceptable only because that row is a past game); `scripts/demonstrate_accuracy_simple.py:117-119`, `final_accuracy_demo.py:173-175`; `scripts/train_ultra_accurate_models.py:71-73,100-102`; `backend/ml/enhanced_features.py:316,355,365-368` (boom/bust rates with thresholds from `quantile(0.75)` over the player's whole history); untracked `run_comprehensive_ml_training.py:215-218`.

**Same-week raw stats as features (target components):** `backend/ml/train.py:230-255` (JSONB `pass_yd`, `rec`, … of the target week); `scripts/train_models_simple.py:70-93`; `scripts/train_and_save_models.py:61-101,117-125` (target = linear function of features + noise); untracked `run_simple_ml_training.py:33-49` (→ archive XGB with MAE 0.33), `train_fantasy_ml_complete.py:138-159` (→ `best_mae: 0.14`), `run_comprehensive…:200-229`, `run_final…:177-216`, `train_real_data.py:131-139`, `quick_ml_test.py:84-98`.

**Season-end / whole-history aggregates for in-season rows:** `train_real_data.py:118-140` (`AVG(fantasy_points_ppr) … WHERE season = 2023 GROUP BY player_id` joined to every 2023 week); `run_final_ml_training.py:63-107` (NGS per-season means merged on same season); `backend/ml/features.py:194-199,245,264-276` (`points_per_game`, `season_total_points`, variance over all rows ending at the featurized row); `enhanced_features.py:234,254-255,294-297,314`; `data_pipeline.py:388-425` (`points_per_game` is both feature and target); `backend/ml/trend_analysis.py:56-63` queried with **no (season, week) bound** and used at inference for the trend multiplier (`ml/predictions.py:113,262-278`).

**Inference history cut-off:** `backend/ml/predictions.py:96-99`:
```python
historical_stats = db.query(PlayerStats).filter(
    PlayerStats.player_id == player_id,
    PlayerStats.season <= season
).order_by(PlayerStats.season, PlayerStats.week).all()
```
No `week < week` filter → in any backtest or re-run with later weeks loaded, future rows enter the features. By contrast `ml/predictions_simple.py:82-86` and `ml/ensemble_predictions.py:121-125` do filter `week < week` (not on the served path).

**Target encoding from same rows:** `enhanced_features.py:234` (`groupby('offensive_coordinator')['team_points'].transform('mean')`), `:254` (`groupby(['player_id','opponent'])['fantasy_points'].transform('mean')`). **Scaler fit before split:** `enhanced_training.py:162-164`, `feature_selection.py:150-155`. **Stacking leak:** `ultra_accurate_model.py:353,357` (meta-features are base-model predictions on the same training rows).

### 4.4 Training vs inference consistency

| Aspect | Production RF training | On-disk NN training (`train_and_save_models.py`) | Served inference (`ml/predictions.py`) |
|---|---|---|---|
| Feature names | `passing_yards_L1`… (nfl_data_py, 44/40/27/27) | `['pass_yards','pass_tds','rush_yards']` (QB), 5/4/4 synthetic names (`models/features_*.pkl`) | Sleeper JSONB keys (`pass_yd`, `rec_tgt`, …) + `season, week, age, years_exp` + `fantasy_points_ppr_lag1/lag2/rolling_avg`, `pts_ppr_*` (~18 for QB; `features.py:508-519`) |
| Lag semantics | strictly prior games | none (same-row) | last completed game's raw stats (`iloc[-1]`), `season/week` overwritten to target (`:211-214`) |
| Scaler | `prod_{pos}_scaler` | `nn_scaler_{pos}` (3/5/4/4) | expects `nn_scaler_{pos}` → shape mismatch (18 vs 3) even if a model loaded |

Verdict: **features are not computed identically anywhere**; no inference code exists for the production RF feature set (the "usage example" in `backend/docs/ML_DOCUMENTATION.md:47-60` calls an undefined `prepare_lagged_features`).

### 4.5 Scoring and target definition

`backend/data/scoring.py`: `ScoringSettings` with `standard()` (reception 0.0, `:87-89`), `ppr()` (1.0, `:92-94`), `half_ppr()` (0.5, `:97-99`), `from_dict` (`:102-104`); `FantasyScorer.calculate_points` for QB/RB/WR/TE (`:154-204`), K (`:206-230`), DEF/DST with points/yards-allowed brackets (`:232-280`); nfl-style keys. Used by `scripts/fetch_sleeper_data.py:21`, `fetch_historical_stats.py:23`. `backend/ml/scoring_engine.py` is a second, Sleeper-keyed implementation (`:29-101`, `calculate_all_formats` `:218-227`, Redis-cached) used by `efficiency_ratio.py`/`momentum_detection.py` (off-path).

**Target:** for the production models it is literally `y_train = pos_train['fantasy_points_ppr']` (`train_production_ml_models.py:227`) — nfl_data_py PPR points of the same player-week, REG season, given prior-game features only. This is **not documented in prose** anywhere (`ML_DOCUMENTATION.md` never names the target; `README.md:21` says "weekly point projections"). The served code derives standard/half from PPR by fixed multipliers rather than scoring rules: `std_prediction = ppr_prediction * 0.85; half_prediction = ppr_prediction * 0.925` (`ml/predictions.py:285-290`; same in `ensemble_predictions.py:361-366`, `predictions_simple.py:141-144`).

---

## 5. Models and artifacts

### 5.1 Artifact inventory

| Path | Bytes | Format / type | Tracked | Library version inside | n_features_in_ | Loader |
|---|---|---|---|---|---|---|
| `backend/models/production/prod_QB_model_20250731_171728.pkl` | 1,323,585 | joblib `RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=20, min_samples_leaf=10, random_state=42)` | yes | sklearn 1.3.2 | 44 | **none** |
| `prod_RB_model_…pkl` / `prod_WR_…` / `prod_TE_…` | 2,457,025 / 2,958,577 / 1,866,033 | same | yes | 1.3.2 | 40 / 27 / 27 | none |
| `prod_{QB,RB,WR,TE}_scaler_…pkl` | 3,095 / 2,903 / 2,255 / 2,255 | StandardScaler with `feature_names_in_` (e.g. QB: `passing_yards_L1, passing_yards_L3_avg, …, completion_pct_L3, yards_per_carry_L3, week_of_season, games_played`) | yes | 1.3.2 | 44/40/27/27 | none |
| `prod_models_metadata_20250731_171728.json` | 1,325 | JSON (§5.2) | yes | | | none |
| `backend/models/production/archive/prod_*_20250731_{171642,171705}.*` | identical sizes to tracked set | same models, two earlier runs a minute apart (deterministic `random_state=42`); both metadata JSONs truncated (105 B / 405 B — **unclear** why) | no | | | none |
| `backend/models/archive/simple_xgboost_20250731_154126.pkl` | 375,606 | `XGBRegressor` | no | booster `version [3,0,3]` (≠ prod pin 2.0.3) | 29 (same-week features → leaky, MAE 0.33) | none |
| `backend/models/archive/simple_random_forest_…pkl` | 8,609,153 | RF | no | 1.3.2 | 29 | none |
| `backend/models/archive/scaler_{QB,RB,WR,TE,all}_20250731_{162146,162334}.pkl` | 5,455 / 5,631 | StandardScaler | no | 1.3.2 | 99 / 103 | none |
| `backend/models/archive/{fantasy_metadata_…334, proper_models_metadata_…, training_summary_…}.json` | 2,768 / 408 / 1,016 | `best_mae: 0.14` (literal at `train_fantasy_ml_complete.py:589`); `positions: []` (all failed the leakage gate); `training_samples 11018, test 5537, R² 0.99` | no | | | none |
| `models/nn_model_{QB,RB,WR,TE}/model_{pos}.keras` | 195,549 / 626,412 / 623,340 / 97,114 | Keras v3 zip, `keras_version 3.4.1`, saved 2025-07-30; Input `[None,3/5/4/4]` | no | keras 3.4.1 | 3/5/4/4 | `neural_network.py:527` loads `model_{pos}` **without extension** → broken; `ensemble_predictions.py:93` (off-path) |
| `models/nn_model_{pos}/metadata.json` | 256 | `model_version "nn_v1_20250730", feature_names: null, training_history: {}` | no | | | `neural_network.py:512` |
| `models/nn_model_QB.h5` | — | **empty directory** (Aug 23 2025) | no | | | `ml/predictions.py:51` (`Path.exists()` → True → load attempted → `TypeError`) |
| `models/nn_scaler_{pos}.pkl` | ~1 KB | StandardScaler | no | 1.3.2 | 3/5/4/4 | `ml/predictions.py:52,62` |
| `models/features_{pos}.pkl` | 53–81 | joblib list of 3/5/4/4 synthetic names | no | | | none |
| `models/model_metadata.json` | 217 | `accuracy 0.892 ×4` | yes | | | `scripts/train_enhanced_models.py:344` |
| `models/feature_importance.json` | 735 | hand-written weights | yes | | | `services/predictor.py:41` |

### 5.2 `prod_models_metadata_20250731_171728.json` (full)

```json
{
  "timestamp": "20250731_171728",
  "data_stats": { "total_records": 31000, "seasons": [2019, 2020, 2021, 2022, 2023, 2024], "unique_players": 1173 },
  "feature_engineering": { "total_features": 65, "all_lagged": true, "no_leakage": true },
  "model_performance": {
    "QB": { "mae": 6.173675600034888, "baseline_mae": 7.541418552398682, "improvement": 18.136414825149295, "model_type": "rf", "train_samples": 2402, "test_samples": 653 },
    "RB": { "mae": 4.921580516627673, "baseline_mae": 6.485996246337891, "improvement": 24.11989878337526, "model_type": "rf", "train_samples": 5125, "test_samples": 1313 },
    "WR": { "mae": 4.993198780252138, "baseline_mae": 6.26336669921875, "improvement": 20.279315900904287, "model_type": "rf", "train_samples": 7997, "test_samples": 2082 },
    "TE": { "mae": 3.8836971590208225, "baseline_mae": 4.706700801849375, "improvement": 17.485786275285793, "model_type": "rf", "train_samples": 4003, "test_samples": 1066 }
  },
  "validation": { "all_mae_above_3": true, "improvements_reasonable": true, "temporal_split_correct": true }
}
```
`total_records: 31000` is a round number — **unclear** whether nfl_data_py returned exactly 31,000 rows or the value was edited (the script computes `int(len(data))` at `:388`). Baseline definition in the script: position mean of training `fantasy_points_ppr` (untracked, `:250-262`).

### 5.3 What is loaded at API startup vs only in docs

| Model | Loaded on canonical path? | Evidence |
|---|---|---|
| Production RF (`prod_*`) | **No** | zero references in code |
| Keras NN | Attempted, **fails**, 0 loaded | `ml/predictions.py:46-66` looks for `nn_model_{pos}.h5`; observed log `Failed to load model for QB: FantasyNeuralNetwork.__init__() missing 1 required positional argument: 'input_dim'`; also calls nonexistent `predict_with_uncertainty` (`:242`) |
| XGBoost | No | only `enhanced_training.py`, `ultra_accurate_model.py`, scripts |
| LightGBM | No | `ultra_accurate_model.py:21` only; not in prod requirements |
| GMM / PCA | No | `gmm_clustering.py` via `train.py`/Celery/scripts; served `/tiers` uses fixed bucket sizes `[3,3,3,3,6,6,8,8]` (`api/tiers.py:124-165`) over hash-seeded random points, **not** 16 GMM tiers |
| "Ensemble" | No | `ensemble_predictions.py` reachable only from `main_optimized.py:273`, `core/cache.py:295`, un-mounted `websocket_routes.py:64` |
| SHAP / Optuna | No | `feature_selection.py`, `hyperparameter_tuning.py` unimported |

Served outputs: `/tiers/positions/{pos}` → `prediction_engine.py:38-68` (`np.random.seed(hash(player_id) % 2**32)`; base points chosen by substring test on the id (`"qb" in player_id.lower()`), which never matches numeric Sleeper ids → every player gets `base = 14.0, variance = 6.0`; `hash()` of str is salted per process → tiers change on every restart). `/predictions/custom`, `/predictions/week/{w}` → constants (`api/predictions.py:33-61, 87-104`). `/players/rankings` → `base_points[pos] - age*0.1 - years_exp*0.2 - idx*0.1` (`api/players.py:55-66`). `/api/v2/predictions/*` → `{"error": "No model available…"}` → 404 (`ml/predictions.py:92-93`, `predictions_v2.py:70-71`), plus `rate_limiter.check_prediction_usage` `AttributeError` (§7).

### 5.4 Ensemble weighting (off-path)

`backend/ml/ensemble_predictions.py:60-64` defines `self.ensemble_weights = {'rf': 0.5, 'nn': 0.3, 'tier': 0.2}` (never read); `:322-338` uses `weights = {'random_forest': 0.5, 'neural_network': 0.4, 'tier_baseline': 0.1}`, `weight = weights.get(model, 0.2)`, `base_prediction = weighted_sum / total_weight`; `:340-351` multiplies by `trend_factor` (×1.05/0.95, ×1.03/0.97; `:388-411`) and `momentum_factor = clip(1 + 0.05·momentum, 0.9, 1.1)`. The production script is not an ensemble: it keeps one of `{rf, xgb}` by validation MAE (`train_production_ml_models.py:264-285`).

### 5.5 Versioning / registry / champion-challenger

`backend/ml/model_versioning.py` implements `ModelVersioningSystem` (SQLite registry `model_registry.db`, `register_model` :180-247 with SHA-256 of `training_data.to_string()`, `promote_to_production` :408-450 with `is_champion`/`traffic_percentage`, `start_ab_test` :452-492, `get_champion_models` :569). **Nothing imports it** (the only occurrence of `model_versioning` in the repo is the file itself); no `model_registry.db` exists. Emitted versions are literals: `"model_version": "2.0"` (`services/predictor.py:161`), `"2.1.0"` (`prediction_engine.py:84`, then discarded by `tiers.py:58-70`), `'ensemble_v1'` (`ensemble_predictions.py:603`). Verdict: **dead code; none in effect.**

### 5.6 Uncertainty / intervals

| Where | Method | Exposed? |
|---|---|---|
| `backend/ml/neural_network.py:299-315` | MC dropout: `pred = model(features_scaled, training=True)` ×n; mean/std/2.5–97.5 percentiles; `confidence = 1/(1+std/(mean+1e-6))` | Only via `ml/predictions.py:242 model.predict_with_uncertainty(...)` — **method does not exist**; unreachable anyway |
| `ml/predictions.py:299-308, 311-359` | fixed bands ×0.8/1.2, ×0.85/1.15; heuristic confidence 0.5 + history/consistency bonuses clipped [0.2, 0.95] | would land in `EnhancedPredictionResponse.prediction.scoring_formats.ppr.*` (`schemas.py:117-123`, all `Dict[str, Any]`) — unreachable |
| `ml/ensemble_predictions.py:354-382` | `pred_std = np.std(model_preds)` or `0.15×estimate`; bounds ±1.5σ; stored as `prediction_std=(upper-lower)/3` | off-path |
| `ml/prediction_engine.py:63-68` | `floor = max(0, pred − variance)`, `ceiling = pred + variance`, variance ∈ {8,7,6,5}; `confidence = 0.75 + rand×0.20` | **yes**: `/tiers/positions/{pos}` `floor`, `ceiling`, `tier_confidence` (`api/tiers.py:68-69,183-193`) |
| `api/predictions.py:53-55` | `floor = base − 5`, `ceiling = base + 5`, `confidence = 0.75` | **yes**: `/predictions/custom` |
| `api/players.py:78-81` | `confidence_interval = {low: max(3, pred−3), high: pred+3}` | **yes**: `/players/rankings` |

No quantile, conformal, or residual-based intervals exist.

---

## 6. Evaluation — tracing the numbers on the live demo

### 6.1 Literal-number grep (excluding node_modules/.next/lock)

| Literal | Hits | Kind |
|---|---|---|
| `4.99`, `6.17`, `4.92`, `3.88` | `backend/models/production/prod_models_metadata_20250731_171728.json:22,30,38,46` (as 6.1736…, 4.9215…, 4.9931…, 3.8836…) | **generated artifact** of the untracked trainer (real nfl_data_py data, 2024 test season) |
| same | `backend/README.md:67-70`, `backend/docs/ML_DOCUMENTATION.md:11-14` | docs (transcribed) |
| same | `frontend-next/src/data/predictions_2024.json:8-11` | **hand-written fixture** (rounded; `within_3_points` 0.89/0.91/0.90/0.94 and `overall: 0.931` have **no source anywhere in code**; `generated: "2024-07-31"` while the trainer ran 2025-07-31) |
| `91%` | `frontend-next/src/components/dashboard/PerformanceDashboard.tsx:335` ("91% tier accuracy confirms GMM clustering…") | hardcoded JSX prose |
| `93.1`, `2.31`, `0.847`, `silhouette 0.73` | **0 hits at HEAD**. Present in `README.md` from `4d01b49` (2025-07-30) / `a5abcd5` through `HEAD~1`; **removed by `57c32cb` (2026-09-07)**, whose `README.md:369` now reads: *"This repository does not publish model accuracy, latency, or throughput figures."* Residual: `predictions_2024.json:7 "overall": 0.931`. `0.73` at `README.md:413` is a `conversion_rate` in a monetization example; `injury_impact.py:127` is `return_perf` for Adam Thielen. |
| `3808` / `3,808` | **0 hits on `main`** (also 0 in `git log -S` over `main`). Present only on the `codex/*` branches via arithmetic (below). | |
| `544`, `1088`, `1632` | `PerformanceDashboard.tsx:37,44,51,58` on main as `32 * 17`, `64 * 17`, `96 * 17`, `32 * 17` ("32 QBs × 17 weeks") ; on the codex branch as literals `count: 544/1088/1632/544` | **hardcoded assumptions**, not data counts |

### 6.2 What the Next.js performance page reads

`frontend-next/src/app/performance/page.tsx:1 import { PerformanceDashboard } from '@/components/dashboard/PerformanceDashboard'`; no fetch. `PerformanceDashboard.tsx:15-16`:
```ts
import predictionsData from '@/data/predictions_2024.json'
import tiersData from '@/data/tiers_2024.json'        // imported, never used
```
Nothing on this page calls an API. **No UI label on `/performance` (main) marks the numbers as demo.**

**Which build is live?** `GET https://www.winmyleague.ai/performance` (HTTP 200) contains `3,808`, `player-game records`, `Within ±3 points`, `91%` ×3, `Decision Lab` ×9 and the string *"Historical sample metrics shown for project demonstration. They are not a guarantee of future performance."* — strings that exist **only on the `origin/codex/*` branches** (`PerformanceDashboard.tsx:41-44,80` there), not on `main`. The homepage contains "Demo values are labeled" (branch `PortfolioHome.tsx:178`). **The deployed site is built from the unmerged codex branch line (last commit `403d7ca`, 2026-09-08).** The live tiers bundle still embeds the fallback string `Using demo data - API temporarily unavailable.`; the embedded API base URL could not be located in the fetched chunks (**unclear** which backend the live build targets).

**Branch formulas producing the reported figures** (`origin/codex/polish-ui-components-including-header-and-footer-mlj0ni:frontend-next/src/components/dashboard/PerformanceDashboard.tsx`):
```ts
10 const metrics = [
11   { position: 'QB', mae: predictionsData.metadata.accuracy.QB.mae, accuracy: predictionsData.metadata.accuracy.QB.within_3_points, count: 544, tier: .89 },
12   { position: 'RB', …RB.mae, …RB.within_3_points, count: 1088, tier: .91 },
13   { position: 'WR', …WR.mae, …WR.within_3_points, count: 1632, tier: .90 },
14   { position: 'TE', …TE.mae, …TE.within_3_points, count: 544, tier: .94 },
15 ]
27 const summary = useMemo(() => ({
28   accuracy: filtered.reduce((sum, item) => sum + item.accuracy, 0) / filtered.length,
29   mae: filtered.reduce((sum, item) => sum + item.mae, 0) / filtered.length,
30   tier: filtered.reduce((sum, item) => sum + item.tier, 0) / filtered.length,
31   count: filtered.reduce((sum, item) => sum + item.count, 0),
32 }), [filtered])
41 …{Math.round(summary.accuracy * 100)}<small>%</small></strong><p>within ±3 points</p>
42 …{summary.mae.toFixed(2)}</strong><p>fantasy points</p>
43 …{summary.count.toLocaleString()}</strong><p>player-game records</p>
44 …{Math.round(summary.tier * 100)}<small>%</small></strong><p>GMM assignment</p>
```
So: **91% within ±3** = unweighted mean (0.89+0.91+0.90+0.94)/4 = 0.91; **MAE 4.99** = (6.17+4.92+4.99+3.88)/4 = 4.99 (unweighted; the test-sample-weighted mean of the metadata MAEs is 4.98); **3,808** = 544+1088+1632+544 (hardcoded literals); **tier agreement 91%** = mean of the four literal `tier` values, which are numerically identical to the `within_3_points` values. On `main`, the same component instead shows **93%** overall (`Math.round(0.931*100)`, `:147`), "31,247 player-game records" (`:175,397`), "↑ 2.4% vs last month" (`:148`), "↓ 0.3 vs baseline" (`:162`) — all hardcoded.

### 6.3 `backend/evaluation/decision_evaluator.py`

**Does not exist on `main`** (`find -iname '*evaluat*'` → none). It exists on the five `codex/*` remote branches (137 lines; shown from `…-mlj0ni`). Description:

- **CLI** (`:114-119`): positional `input` (CSV), `--test-start SEASON-WEEK` (required, e.g. `2024-10`), `--output` JSON path (required). No other flags; thresholds fixed at `(0.5, 0.7, 0.9)` (`:81`).
- **Input contract** (`:21-23`): `REQUIRED_COLUMNS = {"player_id","season","week","position","prediction","actual","decision_score"}`; optional `prediction_floor`. `decision_score` must be finite in [0,1] (`:90-92`); all values must be finite (`:26-30`).
- **Baseline** (`:40-61` `_add_causal_baseline`): rows sorted by (season, week, player_id); for each period in order, `trailing_mean_baseline = mean(player's prior actuals)` falling back to the position's prior actuals, else the row's own `prediction`; a period's actuals are appended **only after** every row in that period is scored (`:56-60`) — causal.
- **Hold-out** (`:94`): `test = [row for row in enriched if (season, week) >= test_start]` — a single forward time split (`"strategy": "forward_time_holdout"`, `:99`), not rolling-origin.
- **Threshold sweep** (`:64-77`): for each threshold, `selected = decision_score >= t`; reports `eligible`, `recommended`, `recommendation_rate`, `selected_mae`, `downside_rate = mean(actual < prediction_floor)` over selected rows with a floor.
- **Output schema** (`:124-130`): `{artifact_version: "1.0", generated_at_utc, input: {path, sha256}, code_commit (git rev-parse HEAD or null), split: {strategy, test_start_season, test_start_week}, model: {n, mae, median_absolute_error}, baseline: {name: "causal_trailing_mean", n, mae, median_absolute_error}, cohorts: {position: {...}}, policy_sweep: [...]}`; written with `json.dumps(indent=2)` to `--output` (`:131-132`).
- **Has it been run?** **No checked-in output artifact exists on any branch** (branch trees contain no `*eval*.json`, no `manifest.json`, no model card). The branch README (`:93`) shows the invocation with a `predictions.csv` that does not exist in the repo. The only test is `tests/test_decision_evaluator.py` (3 `unittest` cases on 4 synthetic rows). `analytics/sql/risk_strategy.sql` (62 lines, branch only) defines a `fct_player_decisions ⋈ fct_player_outcomes` mart with `recommendation_mae`, `mean_regret`, `hit_rate`, `downside_rate` — no such tables exist in the SQLAlchemy schema.

### 6.4 The MAE-vs-accuracy arithmetic

The two headline numbers do **not** come from the same sample, and the "within ±3" number has no computational source:

- MAE 6.17/4.92/4.99/3.88 originate from the production RF metadata (2024 test season, n = 653/1,313/2,082/1,066 = **5,114** rows). That script computes **no within-3 metric** (it gates on `test_mae < 3.0 → "CHECK FOR LEAKAGE"`, `train_production_ml_models.py:293-294,346-356`).
- `within_3_points` 0.89/0.91/0.90/0.94 and `overall: 0.931` appear only in the hand-written fixture; grep for their provenance finds nothing (the untracked archive `ML_AUDIT_REPORT_CRITICAL.md` describes the earlier leaky runs; `docs/IMPROVEMENTS_SUMMARY.md:12` cites "89.9% (demonstrated with synthetic data)").
- The cohort sizes 544/1,088/1,632/544 are `32×17`, `64×17`, `96×17`, `32×17`, not the 653/1,313/2,082/1,066 test rows.
- Arithmetic check: if 91% of errors are ≤ 3 (say averaging ~1.5), then MAE 4.99 requires the other 9% to average ≈ (4.99 − 0.91×1.5)/0.09 ≈ **40 points**, which is implausible for weekly fantasy scoring (a typical top-QB week is ~25–35 points). With a realistic fat tail (say the top 9% averaging 12 points), MAE would be ≈ 2.4. The pairing is internally inconsistent, consistent with the numbers having separate origins.
- Where "within 3" **is** computed in code it is `np.mean(np.abs(y_test - y_pred) <= 3)` (`neural_network.py:406-408`, `ml/predictions.py:487`, `ultra_accurate_model.py:303`, `scripts/*demo*.py`), on synthetic data or unreachable paths; `model_versioning.py:517` uses a different definition (within 20% relative).

### 6.5 Train/test split code

| File:line | Kind | Quote |
|---|---|---|
| untracked `train_production_ml_models.py:143-153` | **time-based (season)** | `train = data[data['season'].isin([2019, 2020, 2021, 2022])]; val = data[data['season'] == 2023]; test = data[data['season'] == 2024]; assert train['season'].max() < val['season'].min()` |
| untracked `rebuild_ml_models_properly.py:255-292`, `train_fantasy_ml_complete.py:340-342`, `run_*_training.py`, `run_ml_pipeline.py:245-246` | time-based (season) | e.g. `train_mask = X['_season'] < 2023; test_mask = X['_season'] == 2023` |
| `backend/ml/enhanced_training.py:128-137, 378-379` | season split + `TimeSeriesSplit(n_splits=5)` on rows sorted by (season, week) | — |
| `backend/ml/train.py:239, 273-276` | labelled "Time-based split" but rows sorted by `['player_id','season','week']` → first 80% of **players**; then `:290-300` refits on train+test and evaluates on test | `split_idx = int(len(X) * 0.8); X_train, X_test = X[:split_idx], X[split_idx:]` |
| `scripts/train_and_save_models.py:132-134`, `train_models_simple.py:98-100`, `train_neural_network.py:52-54` (then refits on all), `train_enhanced_models.py:163-170`, `demonstrate_*.py`, `final_accuracy_demo.py:194`, untracked `train_real_data.py:185-187`, `quick_ml_test.py:139` | **random** | `train_test_split(X, y, test_size=0.2, random_state=42)` |
| `scripts/train_ultra_accurate_models.py:277-284` | positional split on player-sorted data, then random val split | — |
| `backend/ml/hyperparameter_tuning.py:287-290` | positional 80/20 on caller order | — |

No rolling-origin/walk-forward code exists on `main` (grep `walk.forward|rolling.origin|merge_asof|as_of` → none). `TimeSeriesSplit` is imported but unused in `ml/train.py:9`, `ultra_accurate_model.py:18`.

### 6.6 Evaluation notebooks

None. No `.ipynb` files exist; `.ipynb_checkpoints/` directories contain only `.py`/`.md`/`.sh`/`.yml` checkpoint copies from JupyterLab editing, all untracked.

---

## 7. API surface

### 7.1 Routes mounted by `backend/main.py` (prefixes `:129-135,145`)

Legend: **S** static/hardcoded · **DB** needs Postgres session · **R** Redis · **A** auth (`Depends(get_current_user)`) · **X** references missing attribute / crashes.

| Method | Path | Handler (file:line) | Data source | Class |
|---|---|---|---|---|
| GET | `/` | `root` `main.py:102` | literal | S |
| GET | `/health` | `health_check` `main.py:112` | literal | S |
| POST | `/auth/register` | `auth.py:74` | `users` | DB |
| POST | `/auth/login` | `auth.py:108` | `users`, PyJWT HS256 (`:24 JWT_SECRET_KEY` default `"your-secret-key-change-in-production"`) | DB |
| GET | `/auth/me` | `auth.py:137` | | DB, A |
| POST | `/auth/logout` | `auth.py:150` | literal | S |
| GET | `/players/rankings` | `players.py:18` | `players` rows; points/tiers hardcoded (`:55-81,142-172`) | DB (values S) |
| GET | `/players/{player_id}` | `players.py:93` | `players`; `season_stats` zeros (`:134-138`) | DB (values S) |
| POST | `/predictions/custom` | `predictions.py:16` | `players`; `base_points = {'QB': 20.0, 'RB': 15.0, …}` (`:33-46`) | DB (values S) |
| GET | `/predictions/week/{week}` | `predictions.py:66` | same (`:87-104`) | DB (values S) |
| GET | `/tiers/positions/{position}` | `tiers.py:24` | `players` + `prediction_engine` hash-seeded RNG; `tier_configs` sizes `[3,3,3,3,6,6,8,8]` (`:124-165`); `consistency = 0.65 + rand×0.30` (`:248-254`); `adp = index+1` (`:194`) | DB (values random); **X at import** (`:20 LLMService()` TypeError) |
| GET | `/tiers/all` | `tiers.py:103` | calls above | same |
| GET | `/api/v2/predictions/player/{player_id}` | `predictions_v2.py:30` | Stripe status → `EnhancedPredictor` → no models → 404 | DB, A, **X** (`:48 rate_limiter.check_prediction_usage` — method exists only on `PredictionRateLimiter`, `core/rate_limiter.py:408-475`, not on `RateLimiter` instantiated at `:27` → `AttributeError`) |
| POST | `/api/v2/predictions/bulk` | `predictions_v2.py:84` | same | DB, A, **X** (`:109`) |
| GET | `/api/v2/predictions/rankings/{position}` | `predictions_v2.py:146` | `rankings: []` (every prediction errors); `model_accuracy` literal `0.892` (`predictor.py:366`) | DB, A |
| GET | `/api/v2/predictions/accuracy/report` | `predictions_v2.py:192` | `predictions` table (never written) → always 404 | DB |
| GET | `/api/v2/predictions/lineup-optimizer` | `predictions_v2.py:220` | placeholder dict (`:243-247`, `# TODO` `:241`) | DB, A, S |
| GET | `/api/v2/predictions/draft-assistant/recommendations` | `predictions_v2.py:250` | placeholder (`:275-280`, `# TODO` `:273`) | DB, A, S |
| POST | `/api/payments/checkout-session` | `payments.py:25` | Stripe + DB | DB, A, Stripe |
| GET | `/api/payments/subscription/status` | `payments.py:82` | DB | DB, A |
| POST | `/api/payments/customer-portal` | `payments.py:104` | Stripe | DB, A, Stripe |
| POST | `/api/payments/webhook` | `payments.py:137` | Stripe signature (`STRIPE_WEBHOOK_SECRET`) | DB, Stripe |
| GET | `/api/payments/pricing` | `payments.py:178` | literal | S |
| GET | `/subscriptions/plans` | `subscriptions.py:16` | literal | S |
| POST | `/subscriptions/upgrade` | `subscriptions.py:60` | literal; `"expires_at": "2025-01-29T00:00:00Z"` (`:71`) | S (needs DB dep) |
| POST | `/subscriptions/cancel` | `subscriptions.py:75` | literal | S |
| GET | `/api/llm/api/llm/health` (**double prefix**: `llm_endpoints.py:70 APIRouter(prefix="/api/llm")` + `main.py:145 prefix="/api/llm"`) | `llm_endpoints.py:73` | `LLMService` | OpenAI/Anthropic |
| POST | `/api/llm/api/llm/draft/assistant`, `…/stream`, `…/analysis/injury`, `…/trades/analyze`, `…/lineup/optimize` | `llm_endpoints.py:116,179,230,281,344` | DB `draft_tiers`/`predictions` + LLM | DB, A, R (effective), OpenAI |
| GET | `/api/llm/api/llm/subscription/usage` | `llm_endpoints.py:407` | `SubscriptionService` | DB, A |
| WS | `/api/llm/api/llm/draft/live` | `llm_endpoints.py:432` | LLM; `user_id` query param | OpenAI |

Not mounted anywhere: `backend/api/websocket_routes.py` (`/ws`, `/ws/predictions/{id}`, `/ws/live-scores`, `/ws/alerts`, `POST /api/v1/demo/trigger-update`; `:16` imports nonexistent `backend.core.auth`; token check is `user_id = "demo_user"` `:54-56`); `backend/main_optimized.py` (`/health/detailed`, `/metrics`, `/api/v1/*`, `/api/v1/predictions/batch`); `main_simple.py` (`/`, `/health`, `/ready`). Frontend expects `/players/search` (`lib/api/players.ts:106`) — no such route (captured by `/{player_id}` with id `"search"` → 404).

### 7.2 Auth / payments / subscription code

| File | Wired into `main.py`? | Removable without breaking predictions? |
|---|---|---|
| `backend/api/auth.py` (JWT + bcrypt; `OAuth2PasswordBearer(tokenUrl="auth/login")`) | yes `:129`; also imported by `payments.py:17`, `llm_endpoints.py:19`, `predictions_v2.py:19` | `/predictions`, `/players`, `/tiers` use only `Depends(get_db)` (`predictions.py:19,71`, `players.py:22,96`, `tiers.py:28,106`) → **yes**, after editing `main.py:21,127,129,134,135` and dropping `predictions_v2.py`/`llm_endpoints.py`, and removing `tiers.py:12,20` (unused `llm_service`). |
| `backend/api/payments.py`, `services/stripe_service.py` (`STRIPE_SECRET_KEY`, `STRIPE_PRICE_ID` default `"price_fantasy_season_20"`, `STRIPE_WEBHOOK_SECRET`) | yes `:127,134` | yes |
| `backend/api/subscriptions.py` (canned responses, no DB writes) | yes `:135` | yes |
| `backend/services/subscription_service.py` (tiers `scout/analyst/gm` `:19-41`) | via `llm_endpoints`, `llm_service` | yes. **Tier vocabulary mismatch**: Stripe writes `'pro'` (`stripe_service.py:155`), `SubscriptionService.check_usage_limits` does `SUBSCRIPTION_TIERS[tier]` (`:132`) → `KeyError` → `{"allowed": False}` → LLM endpoints 429 for paying users; `schemas.SubscriptionTier` has a third vocabulary `free/pro/premium` (`schemas.py:11-15`). |
| `users`, `subscriptions`, `prediction_usage` tables | | droppable with the above |
| Frontend Clerk (`layout.tsx:27 ClerkProvider`; `useAuth` in `dashboard/page.tsx:11`, `Navigation.tsx:17`, `DashboardLayout.tsx:29`) | frontend only; **backend has no Clerk code**; `lib/api/client.ts:19` sends `Bearer localStorage.auth_token`, which nothing sets | n/a |

### 7.3 Response schemas (`backend/models/schemas.py`)

| Endpoint | Schema | model version | feature version | freshness | intervals |
|---|---|---|---|---|---|
| `POST /predictions/custom` | `PredictionResponse` (`:88-97`: `player_id, player_name, week, predicted_points, floor, ceiling, confidence, factors`) | no | no | no | `floor/ceiling` = ±5, `confidence` 0.75 |
| `GET /predictions/week/{w}` | dict | no | no | no | `confidence: 0.75` |
| `GET /players/rankings` | `PlayerRanking` (`:52-64`) | no | no | no | `confidence_interval {low, high}` = ±3 |
| `GET /tiers/positions/{p}` | inline dict (`tiers.py:93-100, 185-196`) | **discarded** (`prediction_engine.py:84-85` emits `"model_version": "2.1.0"`, `prediction_timestamp`; `tiers.py:58-70` copies only confidence/points/floor/ceiling) | no | `updated_at` = request time | `floor`, `ceiling`, `tier_confidence` |
| `GET /api/v2/predictions/player/{id}` | `EnhancedPredictionResponse` (`:117-124`, every field `Dict[str, Any]`) | `metadata.model_version` literal `"2.0"` (`predictor.py:161`); `model_accuracy` literal `0.892` | no | `metadata.generated_at` | `confidence.{score,level,factors}`; floor/ceiling inside untyped dicts |
| `GET /api/v2/predictions/rankings/{pos}` | `WeeklyRankingsResponse` (`:151-158`) | no | no | `generated_at` | `model_accuracy` 0.892 literal |
| `AccuracyReport` (`:161-168`) | | | | | unreachable |

### 7.4 Startup cost on a 2-CPU / 16 GB host

Import-time work if routers import successfully: TensorFlow + Keras (`neural_network.py:13-15`), matplotlib + seaborn (`:19-20`), scipy + sklearn (`features.py:13-14`, `trend_analysis.py:10-11`), pandas/joblib, langchain_openai/langchain_anthropic/openai/tiktoken (`llm_service.py:14-21`), stripe, Redis TCP connect (`cache.py:183`), three SQLAlchemy engines (`predictions_v2.py:25` → `PredictionEngine`, `PlayerTrendAnalyzer`), Keras/joblib load attempts that fail. Lifespan additionally loads ChromaDB + `SentenceTransformer('all-MiniLM-L6-v2')` (torch; ~90 MB model download on first run) and encodes every `players` row synchronously (`vector_store.py:114-175`). Measured installed sizes in the owner's env (proxy): tensorflow 1.1 GB, torch 544 MB, scipy 155 MB, onnxruntime 127 MB, pyarrow 116 MB, numpy 106 MB, transformers 86 MB, pandas 60 MB, sklearn 42 MB, xgboost 8.7 MB, lightgbm 7.5 MB; site-packages total 3.6 GB. Model artifacts contribute ~0 bytes at runtime because none load. In the degraded state actually observed (routers dropped) only fastapi/openai/anthropic are in `sys.modules`. TF import alone is typically 1–3 s and several hundred MB RSS on CPU; combined with torch this would be ~1–1.5 GB RSS before serving a request. Estimates are inferred, not measured (the app cannot fully boot here).

---

## 8. Frontend (`frontend-next`)

Framework: Next.js **14.2.25** (App Router), React 18.2.0, TypeScript 5.3.3, Tailwind 3.3.6; `@clerk/nextjs ^4.27.7`, `@tanstack/react-query ^5.12.2`, `axios`, `d3`, `framer-motion`; declared-but-unimported: `zustand`, `recharts`, `@stripe/stripe-js`. `src/hooks/` and `src/types/` are empty. No `.eslintrc`, no `middleware.ts`.

### 8.1 Pages and data sources

| Route | Data source (quoted) |
|---|---|
| `/` (`page.tsx`) → `Hero`, `Features`, `Accuracy`, `Pricing`, `HowItWorks` | inline literals (`Hero.tsx:102-143` "P. Mahomes 24.8 pts"; `Features.tsx:113-152` "Example Prediction" Josh Allen 26.5 / 85%) and `lib/constants.ts` |
| `/performance` | static JSON (§6.2) + hardcoded JSX |
| `/tiers` → `TierVisualizationAPI.tsx:45 const data = await tiersApi.getPositionTiers(selectedPosition, scoringType)` → `lib/api/tiers.ts:59` `` `/tiers/positions/${position}` `` via `lib/api/client.ts:4 const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'` | env API; on error `:51 setError('Using demo data - API temporarily unavailable.')` and `generateMockTierData()` (`:433-527`, 20 literal names/position, `Math.random()` stats) — **but the render ternary shows the error branch when `error` is set (`:205-215`), so mock data is never displayed**; chart view `:220 <TierChart>` always reads `tiers_2024.json` (`TierChart.tsx:52,115`) even when the API succeeds |
| `/predictions` → `predictions/page.tsx:40 playersApi.getWeeklyPredictions(` → `players.ts:93` `` `/predictions/week/${week}` `` | env API; on error `:48 setError('Failed to load predictions. Please try again.')` + `generateMockPredictions()` (`:305-327`, `25 - idx*1.5 + Math.random()*5`) — same dead-fallback pattern (`:194-204`) |
| `/dashboard` → `PredictionsList.tsx:20 axios.get(\`/api/predictions?${params}\`)`, `PlayerSearch.tsx:19 axios.get(\`/api/players/search?q=${search}\`)` | **relative Next.js paths; `src/app/api/` does not exist → 404** ("Error loading predictions"). Gate `dashboard/page.tsx:19-22 if (!userId) { window.location.href = '/auth/signin' }` via Clerk `useAuth` |
| `/draft` → `DraftSimulator.tsx:15-16` | static `tiers_2024.json` (≈8 players) + `predictions_2024.json` (3 players); AI pick delay `2000 + Math.random()*3000` (`:257`) |
| `/start-sit` → `StartSitEngine.tsx:16-17` | static JSON |
| `/player/[id]` → `page.tsx:5`, `PlayerProfile.tsx:15-16` | static JSON |
| `/auth/signin`, `/auth/signup` | simulated: `signin/page.tsx:42 // TODO: Implement actual authentication via Clerk.` … `await new Promise(resolve => setTimeout(resolve, 1000))` … `router.push('/dashboard')`; `signup/page.tsx:101` same with 1500 ms; social login → `console.log`. Backend `/auth/login`, `/auth/register` are never called. Sign-in then bounces back from `/dashboard` unless Clerk has a real session. |
| `/pricing`, `/about`, `/contact` (`:47 // TODO: Implement actual form submission`), `/features`, `/help`, `/how-it-works`, `/learn`, `/privacy`, `/terms`, `not-found` | static |

Endpoint wrappers vs backend: `/players/rankings` ✔ (`players.py:18`), `/players/{id}` ✔, `/predictions/week/{w}` ✔ (`predictions.py:66`), `/tiers/positions/{p}` ✔ (`tiers.py:24`), `/tiers/all` ✔, `/players/search` ✘.

### 8.2 Env vars and `vercel.json`

`process.env` reads: only `NEXT_PUBLIC_API_URL` (`lib/api/client.ts:4`, `next.config.js:7`, default `http://localhost:8000`). Clerk reads its key internally. `frontend-next/.env.example` lists `NEXT_PUBLIC_API_URL=https://fantasy-football-ai-production-4441.up.railway.app`, `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY`, `CLERK_SECRET_KEY`, `NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY`, `NEXT_PUBLIC_APP_NAME`, `NEXT_PUBLIC_APP_URL` (last four unused). `frontend-next/.env.local` (ignored) contains only `NEXT_PUBLIC_API_URL`. `next.config.js` has no `module.exports` (file ends at line 9) — **unclear** whether Next applies it. Root `vercel.json` (§1.4) pins region `iad1` and a `maxDuration` for a route file that does not exist.

### 8.3 Demo vs measured labels (verbatim)

Labelled demo/beta (main): `TierVisualizationAPI.tsx:51 'Using demo data - API temporarily unavailable.'`; `pricing/page.tsx:10 'WinMyLeague.ai is free while in beta. No payment details required, no paid plans today.'`, `:61 Free while in beta`, `:72 Beta access`; `landing/Pricing.tsx:21,37,44`; `terms/page.tsx:62 '… They are estimates, not predictions of fact, and they carry no guarantee of accuracy. …'`, `:70 'This is an independent project in beta…'`; `signin/page.tsx:219 … free while in beta.`; `lib/constants.ts:3-5` (comment only) `// This is a portfolio demo, not a commercial service. No prediction-accuracy, user-count, or usage figures are published here…`. Branch/live only: `PerformanceDashboard.tsx:80 Historical sample metrics shown for project demonstration. They are not a guarantee of future performance.`; `performance/page.tsx:17 Model evaluation · historical sample`; `PortfolioHome.tsx:178 Demo values are labeled.`

Presented as measured (main, no label): `PerformanceDashboard.tsx:147` 93% Overall Accuracy, `:148 ↑ 2.4% vs last month`, `:161` 4.99 Avg MAE, `:162 ↓ 0.3 vs baseline`, `:175 31,247` Predictions Made, `:190` 91% Tier Accuracy, `:245 Percentage of predictions within 3 points of actual score`, `:268` "544 predictions" etc., `:325 Tight ends show highest accuracy (94% within 3 points)…`, `:335 91% tier accuracy confirms GMM clustering…`, `:367 … reduce prediction accuracy by ~7%`, `:397-400 • 31,247 player-game records • 2019-2024 NFL seasons • Pre-game features only • No data leakage safeguards`, `:415 • 16-component mixture model`; `Hero.tsx:26 Ensemble Projections • 2019-2024 NFL Data`; `Accuracy.tsx:21 Proven Accuracy You Can Trust`, `:24 … independently verified weekly`, `:78 Accuracy numbers validated weekly against actual NFL results` (no numbers shown); `features/page.tsx:57 '11 unique styles'`, `:68 '20+ integrated'`, `:69 '< 1 minute'`, `:79 '25+ tracked'`; `how-it-works/page.tsx:40 'Real-time data from 20+ sources'`; `predictions/page.tsx:285 Updated hourly with latest injury reports and weather data`, `:289 ML models analyze 100+ factors…`; `help/page.tsx:92 'Yes! Our Pro and League plans include dynasty-specific features…'` (contradicts `:97` "no paid plans"). Pricing contradictions: signup `$19/$49` (`signup/page.tsx:17-50`), `constants.ts:45,50` `$14.99/$29.99`, pricing page `$0`.

### 8.4 Streamlit leftover

`frontend/` **does not exist** (removed after `19008b7`). References remain in `Makefile:44`, `vercel.json:4`, `.gitignore:66-68`, `infrastructure/nginx.conf:27,97`, `infrastructure/docker-compose.yml:76,83,88,192-265,295,365,450`, `README.md:141,164-165`, `docs/QUICKSTART.md:13,39,46,78`, `docs/PROJECT_STRUCTURE.md:35-46,97`, `scripts/test_mvp.sh:139-140`, `scripts/run_data_setup.sh:75`, `scripts/setup_production.sh:83-206`.

---

## 9. Tests and CI

### 9.1 `pytest --collect-only`

- `tests/` is an **empty, untracked directory**. No `pytest.ini`/`pyproject.toml`/`conftest.py`.
- Unrestricted `pytest --collect-only -q` from the repo root **hangs**: collection imports `scripts/test_*.py` and untracked `backend/scripts/tests/*` which execute at import (DB connects, `asyncio.run(...)`, and a server bound to port 8000 — `lsof` showed the pytest process `LISTEN` on 8000; candidates `backend/scripts/tests/comprehensive_test.py`, `final_system_test.py`, `test_llm_endpoint.py`). Killed after >5 min.
- Restricted collection (`tests backend/test_startup.py scripts`, ignoring the four files that open DB/network at import: `scripts/test_db_connection.py`, `test_data_quality.py`, `test_apis_simple.py`, `test_enhanced_pipeline.py`):
  ```
  backend/test_startup.py::test_environment
  backend/test_startup.py::test_basic_imports
  backend/test_startup.py::test_optional_imports
  backend/test_startup.py::test_app_imports
  backend/test_startup.py::test_minimal_server
  scripts/test_efficiency_ratio.py::test_efficiency_ratio
  scripts/test_enhanced_training.py::test_complete_pipeline
  scripts/test_ensemble_predictions.py::test_ensemble_predictions
  scripts/test_gmm_clustering.py::test_gmm_clustering
  scripts/test_ml_complete.py::test_ml_system
  scripts/test_tier_integration.py::test_tier_integration
  11 tests collected in 13.08s
  ```
  (one `MovedIn20Warning` from `backend/models/database.py:25 declarative_base()`). `scripts/test_all_improvements.py`, `test_ml_system.py`, `test_ultra_accurate.py`, `test_data_quality.py` define **zero** `def test_` functions.

### 9.2 Coverage

The 11 collected "tests" are smoke scripts (they instantiate ML classes on synthetic data or check that the app imports). **No tests exist for leakage, grain uniqueness, evaluation, or API contracts** on `main`. On the codex branches only: `tests/test_decision_evaluator.py` (3 unittest cases: model-vs-causal-baseline MAE on 4 rows, missing-column rejection, empty-window rejection).

### 9.3 CI

`.github/workflows/` and `.github/ISSUE_TEMPLATE/` are empty untracked directories (`git ls-files .github` → 0). **No workflows, no cron, nothing commits artifacts.** `README.md:401,449` and `Makefile:30-31` describe test/coverage commands that have no target.

### 9.4 Linting / formatting

No `ruff.toml`, `pyproject.toml`, `setup.cfg`, `.flake8`, `mypy.ini`, `.pre-commit-config.yaml`, `.eslintrc*`, `.prettierrc*` anywhere. Dry runs:

```
ruff check backend scripts --statistics   → Found 464 errors
  283 F401 unused-import · 54 F541 · 36 E402 · 35 F841 · 33 F821 undefined-name · 9 E722 · 8 F811 · 6 E712
black --check backend scripts             → 135 files would be reformatted, 7 files would be left unchanged
```
All 33 `F821 undefined-name` are in `backend/data/data_pipeline.py` (`DatabaseManager`, `SleeperAPIClient`, `FeatureEngineer`, `GMMDraftOptimizer`, `FantasyScorer`, `ScoringSettings`, `DBPlayer`, `PlayerStats`, `Prediction`, `DraftTier`, `FantasyNeuralNetwork` — imports commented out at `:20-26`). `next lint` would prompt to create a config (none exists).

---

## 10. Configuration, secrets, and hygiene

### 10.1 `.env.example` vs what is read

Root `.env.example` (46 lines) documents `DATABASE_URL`, `REDIS_URL`, `JWT_SECRET_KEY`, `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET`, `STRIPE_PRICE_ID`, `NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY`, `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY`, `CLERK_SECRET_KEY`, `NEXT_PUBLIC_API_URL`, `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND`, `MODEL_VERSION`, `MODELS_PATH`, `SLEEPER_API_URL`, `EMAIL_*`, `SENTRY_DSN`, `ENVIRONMENT`, `DEBUG`. `backend/.env.example` (38 lines) documents `DATABASE_URL`, `REDIS_URL`, `SECRET_KEY`, `ALGORITHM`, `ACCESS_TOKEN_EXPIRE_MINUTES`, `ENVIRONMENT`, `DEBUG`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `STRIPE_API_KEY`, `STRIPE_WEBHOOK_SECRET`, `ESPN_S2`, `ESPN_SWID`, `SLEEPER_API_URL`, `NFL_DATA_PY_CACHE_DIR`, `RATE_LIMIT_REQUESTS`, `RATE_LIMIT_PERIOD`, `MODEL_UPDATE_FREQUENCY`, `PREDICTION_CONFIDENCE_THRESHOLD`, `LOG_LEVEL`, `LOG_FILE`.

No `BaseSettings` is used anywhere (`pydantic-settings` pinned but unused). Every variable actually read:

| Variable | Read at | Default | Required? |
|---|---|---|---|
| `DATABASE_URL` | `models/database.py:361`; `alembic/env.py:31`; 11 ML modules with default `postgresql://fantasy_user:fantasy_pass@postgres:5432/fantasy_football`; launchers | `None` / docker hostname | optional to boot; absence disables all DB routes |
| `REDIS_URL` | `core/cache.py:18`, `celery_app.py:9`, `ml/scoring_engine.py:26`, `data/sleeper_client.py:81`, `sources/*` | `redis://localhost:6379/0` | optional |
| `JWT_SECRET_KEY` | `api/auth.py:24` | `"your-secret-key-change-in-production"` | optional (insecure default) |
| `STRIPE_SECRET_KEY`, `STRIPE_PRICE_ID` (default `price_fantasy_season_20`), `STRIPE_WEBHOOK_SECRET` | `services/stripe_service.py:18-20` | None | optional |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` | `main.py:67`, `api/llm_endpoints.py:81-82,110-111` | None | optional |
| `PORT` | `main.py:49`, all launchers | 8000 (`railway_direct_uvicorn.py:16-19`, `start_railway.py:74-78` **exit if unset**) | required by those launchers |
| `ENVIRONMENT` | `main.py:48`, `main_simple.py:55`, … | inconsistent defaults | log-only |
| `ESPN_S2`, `ESPN_SWID` (`data/espn_client.py:25-26`, unimported), `ESPN_RATE_LIMIT` (`espn_public_client.py:48`), `WEATHER_CACHE_TTL` (`weather_client.py:83`), `SPORTSDATA_API_KEY`, `OPENWEATHER_API_KEY`, `CFBD_API_KEY` (`enhanced_data_collector.py:70-72`), `RAILWAY_*` (`start_railway.py:43-53`), `PYTHONPATH`, `SECRET_KEY` (`railway_debug.py:29,34` diagnostic only) | | | optional |

Documented but never read: `SECRET_KEY`, `ALGORITHM`, `ACCESS_TOKEN_EXPIRE_MINUTES` (hardcoded 30 at `auth.py:26`), `STRIPE_API_KEY` (wrong name), `SLEEPER_API_URL`, `NFL_DATA_PY_CACHE_DIR`, `RATE_LIMIT_*`, `MODEL_UPDATE_FREQUENCY`, `PREDICTION_CONFIDENCE_THRESHOLD`, `LOG_*`, `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND`, `MODEL_VERSION`, `MODELS_PATH`, `EMAIL_*`, `SENTRY_DSN`, `DEBUG`. Read but undocumented: `JWT_SECRET_KEY` (backend example), `STRIPE_SECRET_KEY`, `PORT`, `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` (root example), `SPORTSDATA_API_KEY`, `OPENWEATHER_API_KEY`, `CFBD_API_KEY`, `ESPN_RATE_LIMIT`, `WEATHER_CACHE_TTL`.

### 10.2 Committed secrets / personal data (locations only)

| Finding | Location | Tracked? |
|---|---|---|
| **Supabase Postgres connection string with a real-looking password** (`postgresql://postgres:<pw>@db.ypxqifnqokwxrvqqtsgc.supabase.co:5432/postgres`) | `scripts/run/start_backend.sh:7`, `scripts/run_with_env.sh:5` | **yes**, since `00da23b`/`95266b7` (2025-07-31) — in public GitHub history |
| same string | `backend/scripts/data_collection/{collect_complete_nfl_data.py:20, continue_data_collection.py:15, quick_data_test.py:14}`, `backend/scripts/training/{quick_ml_test.py:18, run_ml_pipeline.py:20, train_real_data.py:24}` | no (ignored), on disk |
| **OpenAI project key (`sk-proj-…`) and Anthropic key (`sk-ant-api03-…`)** | `backend/.env.local:35-36` | no; never committed (`git log --all -- backend/.env.local` empty) |
| Stripe/Clerk/other key names | `.env`, `backend/.env`, `backend/.env.local`, `frontend-next/.env.local` | no (values not inspected beyond prefixes) |
| Personal email `chris@fantasyfootballai.com` | `README.md:510` | yes |
| `support@winmyleague.ai` | 13 frontend page locations (contact/help/pricing/privacy/terms) | yes (public contact) |
| A Gmail address as `SMTP_USER` example | `infrastructure/docker-compose.yml:402` | yes |
| Git author email `cbratkovics@gmail.com` | commit metadata | n/a |
| Default DB password `fantasy_pass` | `docker-compose.yml:9,45`, `scripts/setup/setup_postgres.sql`, ML module defaults | yes (dev default) |
| JWT default secret | `api/auth.py:24` | yes |
| Absolute home paths `/Users/christopherbratkovics/...` | `scripts/run/*.sh`, `scripts/setup/setup_local.sh`, `backend/scripts/tests/test_database_fix.py`, `.claude/settings.local.json` (untracked) | partly tracked |

### 10.3 Dead code (import graph)

Zero importers anywhere (excluding scripts): `api/websocket_routes.py`, `core/websocket.py`, `ml/injury_impact.py`, `ml/weather_projections.py`, `ml/model_versioning.py`, `ml/trade_analyzer.py`, `ml/feature_selection.py`, `data/espn_client.py`, `data/data_pipeline.py`, `data/fetch_players.py` (0 bytes), `main_optimized.py`, `health_check.py`, `emergency_server.py`, all `railway_*.py`/`start_*.py`/`test_startup.py`/`check_imports.py` launchers. Not reachable from `backend/main.py` but imported by scripts/Celery: `ml/{ensemble_predictions, draft_tier_storage, gmm_clustering, momentum_detection, efficiency_ratio, scoring_engine, feature_engineering, ranking_algorithm, predictions_simple, enhanced_features, advanced_models, ultra_accurate_model, hyperparameter_tuning, train, enhanced_training}`, `data/{sleeper_client, scoring, synthetic_data_generator, enhanced_data_collector}`, `data/sources/*`, `celery_app`, `tasks/*`, `models/player_profile`. Reachable from `main.py`: `api/{auth,players,predictions,subscriptions,tiers,predictions_v2,payments,llm_endpoints}`, `core/{cache,rate_limiter}`, `models/{database,schemas}`, `services/{llm_service,vector_store,subscription_service,stripe_service,predictor,explainer}`, `ml/{prediction_engine,predictions,features,trend_analysis,neural_network}`. Missing modules referenced: `backend.core.auth` (`websocket_routes.py:16`), `backend.core.config` (`scripts/test_data_quality.py:18`), `backend.ml.fantasy_predictor`, `backend.ml.draft_optimizer` (`scripts/train_enhanced_models.py:33-34`), `data.update_predictions` (`Makefile:65`), `schedule` (`data_pipeline.py:15`).

### 10.4 TODO/FIXME/HACK (tracked)

`backend/api/predictions_v2.py:241 # TODO: Implement lineup optimization logic`; `:273 # TODO: Implement draft recommendation logic`; `frontend-next/src/app/auth/signin/page.tsx:42 // TODO: Implement actual authentication via Clerk.`; `:59 // TODO: Implement social login`; `signup/page.tsx:101 // TODO: Implement actual registration via Clerk.`; `:118 // TODO: Implement social signup`; `contact/page.tsx:47 // TODO: Implement actual form submission`.

### 10.5 Notebooks

None (`*.ipynb` count 0). Ten `.ipynb_checkpoints/*-checkpoint.{py,sh,md,yml}` files exist untracked (JupyterLab editor artifacts), plus `.pytest_cache/`.

---

## 11. README and docs vs. code reconciliation

Status key: **Implemented** (on canonical path or invoked by a tracked script) · **Partially** · **Scaffold-only** (module exists, unreachable) · **Absent** · **Contradicted**. "README now" = whether `README.md@57c32cb` still states it.

| Claim | Where stated | Evidence in code | Status |
|---|---|---|---|
| Ensemble of XGBoost + LightGBM + NN | `README.md:21,27,49,80,94` (still); `docs/IMPROVEMENTS_SUMMARY.md:46-51`; `Features.tsx:16`; `constants.ts:12-13` | `ml/ultra_accurate_model.py:49,62,117` (scripts only); `enhanced_training.py:168` (no LGBM, unimported); shipped prod models are `"model_type": "rf"` ×4; `requirements.txt:74` "xgboost and lightgbm removed"; served path uses RF-less random/constant values | **Scaffold-only**; served **Contradicted** |
| GMM tiers with PCA (16 tiers) | `README.md:28-31,180-202`; `ML_ENHANCEMENTS_SUMMARY.md:8-14`; `tiers_2024.json` metadata | `ml/gmm_clustering.py:154 PCA`, `:168 GaussianMixture(n_components=16)`; trained only via `ml/train.py`/Celery/scripts; served `/tiers` uses fixed 8 buckets (`api/tiers.py:124`) | **Partially** (training side); served **Contradicted** |
| 100+ features / 50+ attributes | `README.md:34,35,79,229,476` (still); `IMPROVEMENTS_SUMMARY.md:20,38` | prod metadata `total_features: 65`; `ML_ENHANCEMENTS_SUMMARY.md:58 "26+ engineered features"`; on-disk NNs have 3–5 inputs | **Contradicted** |
| Monte Carlo dropout | `README.md:23,222` | `neural_network.py:299-315` real; caller uses nonexistent method (`predictions.py:242`); no model loads | **Scaffold-only** |
| Weather features | `README.md:37,81,238,245`; `features/page.tsx:43,46,65` | Open-Meteo client unreachable; served `weather_impact = 0.9 + rand×0.1` (`prediction_engine.py:82`) | **Scaffold-only**; served **Contradicted** |
| Injury NLP / survival analysis | `README.md:38,55`; `how-it-works:42` | `ml/injury_impact.py` lookup tables + `np.random`, no importer; LLM prompt endpoint `llm_endpoints.py:230` | **Scaffold-only** / NLP **Absent** |
| Sentiment analysis | `README.md:57` | grep → 0 | **Absent** |
| ARIMA | `README.md:63` | 0 | **Absent** |
| Genetic algorithms | `README.md:68` | 0 | **Absent** |
| Reinforcement learning | `README.md:69` | 0 | **Absent** |
| Bayesian optimisation / Optuna | `README.md:70,97,477` | `ml/hyperparameter_tuning.py:6,278,284` (fails to import; script-only; dev requirement only) | **Scaffold-only** |
| SHAP | `README.md:43,96,501`; `Accuracy.tsx:12` | `ml/feature_selection.py:23` no importer; `shap` in no requirements | **Scaffold-only** |
| Redis caching | `README.md:79,89,483` | `core/cache.py`, `sleeper_client.py`, `llm_service.py` | **Implemented** (optional) |
| Celery tasks | `README.md:90,163` | `celery_app.py`, `tasks/*`, compose; stats task stub | **Partially** |
| Kubernetes | `README.md:101` | no manifests | **Absent** |
| Terraform | `README.md:103,168-171,328-334`; `docs/DEPLOYMENT.md:60-64` | `infrastructure/terraform/main.tf` 28 lines, zero resources | **Scaffold-only** |
| A/B testing / canary / champion-challenger | `README.md:42,388-398,466,468` | `ml/model_versioning.py` (dead); no traffic routing | **Scaffold-only** |
| Drift detection | `README.md:403`; `ML_DOCUMENTATION.md:64` | grep `drift` in `*.py` → 0 | **Absent** |
| Automated retraining | `README.md:52,404,484` | weekly Celery beat only (`celery_app.py:44-48`); no trigger | **Partially** |
| Feature store | `README.md:491` | 0 | **Absent** |
| 95% coverage / test suite | `README.md:354-365,401,449`; `backend/README.md:126-133` | `tests/` empty; no CI; 11 smoke tests | **Contradicted** |
| Sub-200 ms latency | `README.md:41,89,483` (still) vs `README.md:369` | no benchmark code | **Absent**; README self-**Contradicted** |
| Real-time ingestion <30 s | removed from README; `features/page.tsx:64-70 '< 1 minute'`, `how-it-works:89`, `learn:135`, `Accuracy.tsx:24,87` | daily Celery schedule only | **Contradicted** |
| Streamlit frontend | `README.md:141,164-167`; `PROJECT_STRUCTURE.md`; `QUICKSTART.md`; `Makefile:44` | `frontend/` absent; Next.js app | **Contradicted** |
| As-of features | not claimed by name; `ML_DOCUMENTATION.md:24 "All features are lagged"` | lagging only in untracked trainer; served inference has no week bound | **Partially (untracked)** |
| Rolling-origin validation | `ML_DOCUMENTATION.md:32` (temporal split) | single season holdout in untracked script; no rolling-origin code on main | **Scaffold-only** |
| Policy / threshold simulator | not on main (codex branch README) | none on main; branch `decision_evaluator.py` sweep + `PortfolioHome.tsx` console | **Absent on main** |
| Decision marts `analytics/sql/risk_strategy.sql` | not on main | exists only on codex branches; references nonexistent tables | **Absent on main** |
| 93.1% accuracy | removed in `57c32cb` (old `README.md:21,27,50,81,369`) | residual `predictions_2024.json:7 overall: 0.931` → "93%" on `/performance` (main) | **Removed / residual Contradicted** |
| RMSE 2.31, R² 0.847, silhouette 0.73 | removed in `57c32cb` (old `:368-375`) | none (`gmm_clustering.py` uses BIC, no silhouette) | **Absent** |
| MAE 4.99 (WR) and 6.17/4.92/3.88 | `backend/README.md:66-70`; `ML_DOCUMENTATION.md:9-14`; fixture | prod metadata (untracked trainer; models unloaded) | **Partially** (metadata only) |
| 3,808 records | codex branch UI only (computed) | 544+1088+1632+544 literals | **Absent on main; fabricated arithmetic on branch** |
| Trained on 2019–2024 (31,000+ records) | `backend/README.md:7`; `ML_DOCUMENTATION.md:4,32`; `Hero.tsx:26` | metadata + untracked scripts; tracked ingest is 2021–2023 | **Partially** |
| Sleeper API | `README.md:79,161` | `sleeper_client.py`; Celery + scripts | **Implemented** (ingest side) |
| nfl_data_py | `backend/README.md:9,95` | `nfl_data_py_client.py` unreachable from tracked runtime | **Scaffold-only** (tracked) |
| PostgreSQL + JSONB | `README.md:88,280-314` | `models/database.py`, alembic | **Implemented** (README's `INDEX CONCURRENTLY` inside `CREATE TABLE` `:290` is invalid SQL) |
| JWT auth (+ RBAC) | `backend/README.md:11,120` | `api/auth.py`; no roles | **Implemented** (RBAC **Absent**) |
| Stripe | `backend/README.md:12,122` | `api/payments.py`, `stripe_service.py`; site says free beta | **Implemented** (backend); copy **Contradicted** |
| Clerk | `docs/DEPLOYMENT.md:125-126` | `layout.tsx:27`; sign-in simulated; no backend integration | **Partially** |
| WebSocket real-time | `backend/README.md:8,90`; `DEPLOYMENT_ROADMAP.md` | `websocket_routes.py` unmounted + broken import; LLM WS only | **Scaffold-only** |
| LLM / RAG / vector store | `DEPLOYMENT_ROADMAP.md:48-56` | `llm_service.py`, `vector_store.py`, `llm_endpoints.py`, tracked `chroma.sqlite3` | **Implemented** (conditional on keys; double prefix bug) |
| Prometheus metrics | `README.md:44,485` | `main_optimized.py:17` only; not in requirements | **Scaffold-only** |
| Rate limiting | `backend/README.md:10` | `core/rate_limiter.py` used by `predictions_v2` (wrong class → AttributeError); middleware not added | **Partially** |
| Efficiency ratio | `README.md:231-232,244` | `ml/efficiency_ratio.py` script-only | **Scaffold-only** |
| Momentum detection | `README.md:61,81,235,246` | `ml/momentum_detection.py` off-path; some columns in `features.py` | **Partially** |
| Trade analyzer | `README.md:56`; `how-it-works:55` | `ml/trade_analyzer.py` no importer; LLM prompt endpoint; no page | **Scaffold-only** |
| Draft simulator ("11 unique styles") | `features/page.tsx:53-61` | client-side over 8 static players; no backend route (`README.md:275 POST /draft/recommendations` **Absent**) | **Partially** (static demo) |
| "Real-time data from 20+ sources" | `how-it-works:40`; `features:68` | ≤9 URLs in code, 1 wired | **Contradicted** |
| Import team from ESPN/Yahoo/Sleeper | `how-it-works:23,85` | no Yahoo client; ESPN client unimported; no league-import route | **Absent** |
| "Independently verified weekly" | `Accuracy.tsx:21-25,76-79` | no writer for `model_performance` | **Absent** |
| `python -m backend.ml.train --evaluate` | `README.md:380` | no argparse (`train.py:397-401`) | **Absent** |
| `tests/test_ml.py`, `pytest --cov=app` | `README.md:361,364` | missing | **Absent** |

**`docs/PROJECT_STRUCTURE.md` vs tree:** lists `frontend/` with `app.py`, `components/{auth,charts}.py`, `pages/{account,draft_assistant,rankings,weekly_picks}.py`, `requirements.txt`, `Dockerfile` (lines 35-46) — **all missing**; `backend/data/fetch_players.py` (:17) is 0 bytes; service diagram `Frontend (Streamlit:8501)` (:97). Omits: `frontend-next/`, `docs/`, `models/`, `tests/`, `Makefile`, `vercel.json`, `deploy-to-railway.sh`, `.github/`, `infrastructure/Dockerfile`; under `backend/`: `api/{tiers,predictions_v2,payments,llm_endpoints,websocket_routes}.py`, `core/`, `services/`, `alembic/`, `docs/`, `scripts/`, `models/production/`, `vector_db/`, `data/sources/`, 20 of 24 `ml/` modules, `celery_app.py`, `main_optimized.py`, `main_simple.py`, all `start_*`/`railway_*` files, `requirements-{dev,prod,minimal}.txt`, `Procfile`, `railway.json`. `scripts/` lists 4 of 41 files. `README.md:156-176` tree has the same Streamlit entries and `README.md:330 cd terraform`, `:352 nginx.conf`, `:429 ./backups/` point at nonexistent root paths.

---

## 12. Fitness for the planned architecture

Plan: weekly GitHub Actions job (nflverse ingest → data contracts → drift check → score champion → shadow-evaluate challenger → deterministic publish/hold/promote → commit `manifest.json` + model card) and a FastAPI server on a free Docker host loading versioned artifacts from the repo at startup, no Postgres/Redis.

### 12.1 What blocks end-to-end scoring without a database today

1. **No inference feature code for the only real models.** The 44/40/27/27 lagged features exist solely inside untracked `train_production_ml_models.py:56-126`; nothing in `backend/` can build them from an nflverse frame.
2. **No loader for `prod_*` pickles.** The served `PredictionEngine` looks only for `nn_model_{pos}.h5` (`ml/predictions.py:51`).
3. **Every prediction route is DB-bound**: `Depends(get_db)` on all data routes; `get_db` raises without `DATABASE_URL` (`database.py:379`); ML modules create engines at import (`predictions.py:34`, etc.); history is fetched from `player_stats` (`predictions.py:96-99`).
4. **Import structure**: `main.py` bare imports vs `backend.*` everywhere else; `api/tiers.py:20` crashes on `LLMService()`; `predictions_v2.py:27` wrong rate-limiter class; `neural_network.py:19-20` needs matplotlib/seaborn.
5. **Build hygiene**: `backend/.dockerignore` strips `*.pkl`/`*.json`; `backend/.gitignore` hides the trainer; `.github/workflows` empty; no manifest, model card, or evaluation artifact exists anywhere on `main`.
6. **Artifact provenance**: prod pickles are sklearn 1.3.2 `RandomForestRegressor` fit on scaled arrays with no `feature_names_in_` on the model (only on the scaler); metadata records no input hash, no commit, no feature list.

### 12.2 Can inference run without TensorFlow?

**Yes, for the tree models — and nothing else is usable.** The NN is on the served *import* chain (§1.2) but not in the served *prediction* path (0 models load; caller uses nonexistent methods; on-disk `.keras` files are synthetic-data models with 3–5 inputs whose names match no real data). The RF pickles need only `scikit-learn==1.3.2` (+ numpy/scipy/joblib). Removing TF requires deleting the `neural_network` import from `ml/predictions.py:21` (or not shipping `ml/predictions.py` at all).

### 12.3 Minimal runtime dependency set (API-only, RF champion)

`fastapi`, `uvicorn`, `pydantic` (v2), `numpy`, `pandas` (feature frame), `scikit-learn==1.3.2` (unpickle RF/StandardScaler), `scipy` (sklearn dep), `joblib`, `python-dotenv` (optional). ≈ 9 packages. Everything else in `requirements.txt` (sqlalchemy/asyncpg/psycopg2/alembic, redis/aioredis/fastapi-limiter, jose/passlib/bcrypt/PyJWT, stripe, celery, httpx/aiohttp/ratelimit/backoff, nfl_data_py, beautifulsoup4, tensorflow-cpu, xgboost, langchain*, openai, anthropic, weaviate-client, sentence-transformers, chromadb, websockets, sse-starlette, tiktoken, tenacity, asyncio-throttle) is unnecessary for scoring. **Training set** (GitHub Actions job): the above + `nfl_data_py` (or `nflreadpy`) + `pyarrow`, plus `xgboost`/`lightgbm` only if the challenger uses them; `optuna`/`shap` optional.

### 12.4 Estimated API-only Docker image size

`python:3.10-slim` (~130 MB) + fastapi/uvicorn/pydantic (~25 MB) + numpy (~30–40 MB Linux wheel) + pandas (~60 MB) + scipy (~90–110 MB) + scikit-learn (~40 MB) + joblib + 9 MB of artifacts ≈ **380–450 MB** (≈ 150 MB compressed). Without pandas (build the feature vector with numpy only) ≈ 320 MB. For comparison the current `requirements.txt` (tensorflow-cpu ≈ 500 MB Linux wheel, torch via sentence-transformers ≈ 700 MB+, chromadb/onnxruntime, langchain, pyarrow) yields an estimated **3–4 GB** image and a build that needs `build-essential g++` (`Dockerfile:5-14`). These are estimates from wheel sizes, not a measured build.

### 12.5 Keep / refactor / delete

**Keep (as reference inputs, not as-is):** `backend/models/production/prod_*_20250731_171728.{pkl,json}` (baseline champion + its MAE record); `backend/scripts/training/train_production_ml_models.py` (must be **added to git** and split into feature builder + trainer); `backend/data/scoring.py` (scoring formats); `backend/ml/gmm_clustering.py` (if tiers are wanted); `frontend-next/` pages `/performance`, `/tiers`, `/predictions` shells (rewire to manifest-backed JSON); the codex-branch `backend/evaluation/decision_evaluator.py` + `tests/test_decision_evaluator.py` (causal baseline, forward holdout, artifact with sha256/commit — the closest thing to the plan's evaluator); `docs/COMMERCIAL_USE_COMPLIANCE.md` (after correction).

**Refactor:** `backend/main.py` (single import root, no DB/LLM guards, load artifacts from a `manifest.json` at startup, expose `model_version`, `feature_version`, `trained_at`, `data_through`, intervals); `backend/ml/features.py` → one as-of feature builder shared by training and inference (port the lagged logic, fix the flattened-`shift` bug by shifting inside the groupby); `backend/models/schemas.py` (typed responses); `frontend-next/src/lib/api/*` (drop `/players/search`, `/api/*` relative calls); `PerformanceDashboard.tsx` (read the evaluation artifact, label sample/window/baseline); `backend/Dockerfile` (`requirements-api.txt`, stop ignoring artifacts, non-root, no `PORT`-exit); `.gitignore`/`.dockerignore` (un-ignore `backend/scripts/`, model JSON).

**Delete:** all §1.6 legacy launchers and diagnostic servers; `backend/api/{websocket_routes,payments,subscriptions,llm_endpoints,auth}.py`, `services/{llm_service,vector_store,stripe_service,subscription_service}.py`, `core/{cache,rate_limiter,websocket}.py`, `celery_app.py`, `tasks/`, `alembic/`, `vector_db/`, `models/database.py` (for the no-DB plan); `backend/ml/{prediction_engine (random), predictions_simple, ensemble_predictions, ultra_accurate_model, enhanced_training, enhanced_features, hyperparameter_tuning, advanced_models, feature_selection, model_versioning, injury_impact, weather_projections, trade_analyzer, momentum_detection, efficiency_ratio, scoring_engine, ranking_algorithm, draft_tier_storage, feature_engineering, trend_analysis, neural_network}.py`; `backend/data/{data_pipeline, espn_client, fetch_players, enhanced_data_collector, synthetic_data_generator}.py`; `models/` root synthetic NN artifacts and `feature_importance.json`/`model_metadata.json`; `backend/models/archive/**`, `production/archive/**`; `scripts/demonstrate_*`, `final_accuracy_demo.py`, `train_*` (except a rewritten trainer), all `scripts/test_*.py`, `scripts/run*`, `scripts/setup*`, `deploy.sh`, `deploy-to-railway.sh`; `infrastructure/`; `vercel.json`; `docker-compose.yml` services for postgres/redis/celery; `frontend-next/src/data/*_2024.json` (or relabel as fixtures); `docs/{PROJECT_STRUCTURE, QUICKSTART, DEPLOYMENT, DEPLOYMENT_ROADMAP, IMPROVEMENTS_SUMMARY, ML_ENHANCEMENTS_SUMMARY, CLEANUP_SUMMARY_2024}.md`, `backend/RAILWAY_*.md`, `backend/docs/CLEANUP_SUMMARY.md`; `.ipynb_checkpoints/`; `backend/backend/`; `ssl/`.

### 12.6 Top 10 risks (severity-ordered)

1. **Committed Supabase database password** in `scripts/run/start_backend.sh:7` and `scripts/run_with_env.sh:5` (public history since 2025-07-31) — rotate the credential and purge history (or accept exposure and delete the DB).
2. **Live API keys on disk** in `backend/.env.local:35-36` (OpenAI, Anthropic) — rotate; they are untracked but present on a machine that also runs untrusted scripts.
3. **The live demo's headline metrics are arithmetic over hand-typed fixture values and hardcoded cohort sizes** (§6), while MAE values come from an unrecorded, unreproducible run — publish only artifact-backed numbers with sample/window/baseline, or label as illustrative.
4. **The deployed site is built from an unmerged branch**; `main` and production have diverged — merge or tag the deployed commit and make the Vercel build source explicit.
5. **The only real models are unreachable and their trainer is git-ignored** — commit the trainer, add a loader, and put the feature builder in shared code before any retraining pipeline.
6. **Two-import-root split makes the API unbootable as deployed** (only `/health` served; Railway backend unreachable) — one package root, one launcher, an import smoke test in CI.
7. **Leakage in every tracked training path** (same-week stats, shift-less rolling, no week bound at inference) — adopt the untracked lagged scheme as the single feature module and add a leakage test (feature timestamp < target timestamp).
8. **Pickle-based artifacts pinned to sklearn 1.3.2 with no feature-name contract on the model** — store feature list + versions in the manifest; consider ONNX or a `feature_names_in_`-bearing pipeline.
9. **Third-party terms exposure**: PFR HTML scraping, authenticated ESPN client, paid-API keys in scripts; `beautifulsoup4` shipped in prod — delete the scraper and ESPN client; document nflverse's license in the model card.
10. **No tests, no CI, 464 lint errors, 135 unformatted files, hanging pytest collection** — start CI with import checks, the evaluator unit tests, and a ruff/black baseline before adding the weekly job.

---

## 13. Open questions for the repo owner

1. Was `frontend-next/src/data/predictions_2024.json` (`within_3_points` 0.89/0.91/0.90/0.94, `overall: 0.931`, `generated: "2024-07-31"`) hand-edited, or produced by a script that no longer exists? Where did the `within_3_points` values come from?
2. Is the production site (www.winmyleague.ai) intentionally deployed from `origin/codex/polish-ui-components-including-header-and-footer-mlj0ni` rather than `main`? Will those branches be merged?
3. Is the Railway backend `fantasy-football-ai-production-4441.up.railway.app` still supposed to exist? It timed out during this audit.
4. Has the Supabase database `db.ypxqifnqokwxrvqqtsgc.supabase.co` password been rotated since it was committed on 2025-07-31? Does that database still hold the 30,406 stat rows the archived guide mentions?
5. Are the OpenAI/Anthropic keys in `backend/.env.local` live? Have they been rotated?
6. Can `backend/scripts/training/train_production_ml_models.py` (and the rest of `backend/scripts/`) be added to git? Was `backend/.gitignore`'s `scripts/` line intentional?
7. In `prod_models_metadata_20250731_171728.json`, is `total_records: 31000` the exact `len(data)` output or an edited round number? Were the two archived runs (`171642`, `171705`) interrupted, and why are their metadata files truncated?
8. Is `models/nn_model_QB.h5` (an empty directory dated 2025-08-23) a leftover from a failed save, and can the root `models/` NN artifacts (trained on synthetic data) be deleted?
9. Should the target remain nflverse `fantasy_points_ppr` (regular season), and do you want standard/half-PPR computed from scoring rules rather than the 0.85/0.925 multipliers?
10. Which frontend routes are meant to survive (tiers, predictions, performance, draft, start-sit) and which are marketing-only? Should auth, pricing, and Stripe be removed entirely given the "free beta" copy?
11. `next.config.js` has no `module.exports`: is the Vercel project using a different config, and is `NEXT_PUBLIC_API_URL` set there (the live bundle's API base could not be located)?
12. Do you intend to keep Clerk? The sign-in pages are simulated and the backend has its own JWT scheme.
13. Is any scheduled process (Celery beat, cron) actually running anywhere today, or is all ingestion manual?
14. Do you have the original evaluation notebooks/scripts behind the removed README figures (93.1%, RMSE 2.31, R² 0.847, silhouette 0.73), or should they be treated as never measured?
15. Are the paid data sources (sportsdata.io, OpenWeatherMap, collegefootballdata) under any active subscription, or can the `enhanced_data_collector.py` paths be deleted?
16. Is `docs/PORTFOLIO_CASE_STUDY.md` (branch) the intended framing going forward, i.e. should the plan's manifest/model-card format align with the `decision_evaluator.py` artifact schema (`artifact_version`, `input.sha256`, `code_commit`, `split`, `model`, `baseline`, `cohorts`, `policy_sweep`)?
