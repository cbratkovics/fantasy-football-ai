"""ffai.config.PROJECT is the one source of the project vocabulary; every mirror must agree
(ADR-0028): dbt vars, the frontend config JSON, the workflow schedule, and the deployment names
the YAML cannot compute."""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from ffai.config import PROJECT, REPO_ROOT, dbt_vars, frontend_config

FRONTEND_CONFIG = REPO_ROOT / "frontend-next" / "src" / "lib" / "project.config.json"


def test_dbt_project_vars_mirror_the_config() -> None:
    project = yaml.safe_load((REPO_ROOT / "dbt" / "dbt_project.yml").read_text(encoding="utf-8"))
    declared = {k: project["vars"][k] for k in dbt_vars()}
    assert declared == dbt_vars()
    assert project["name"] == PROJECT.dbt_project_name


def test_frontend_config_json_is_regenerated() -> None:
    committed = json.loads(FRONTEND_CONFIG.read_text(encoding="utf-8"))
    assert (
        committed == frontend_config()
    ), "run: python -m ffai.config --frontend > frontend-next/src/lib/project.config.json"


def test_weekly_schedule_mirrors_the_config() -> None:
    weekly = (REPO_ROOT / ".github" / "workflows" / "weekly.yml").read_text(encoding="utf-8")
    assert f'cron: "{PROJECT.schedule_cron}"' in weekly


def test_workflows_read_deployment_names_from_the_config() -> None:
    for name in ("weekly.yml", "ci.yml"):
        text = (REPO_ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")
        assert 'hf upload "$(python -m ffai.config hf_space)"' in text
        assert "hf upload cbratkovics/" not in text


def test_profiles_and_exposures_name_the_configured_deployments() -> None:
    profiles = (REPO_ROOT / "dbt" / "profiles.yml").read_text(encoding="utf-8")
    assert f"'{PROJECT.motherduck_database}'" in profiles
    exposures = (REPO_ROOT / "dbt" / "models" / "gold" / "_exposures.yml").read_text(
        encoding="utf-8"
    )
    assert PROJECT.api_url in exposures and PROJECT.site_url in exposures


def test_legacy_aliases_are_the_project_values() -> None:
    from ffai import config

    assert config.POSITIONS == PROJECT.cohorts
    assert config.TARGET == PROJECT.target_column
    assert config.TRAIN_SEASONS == PROJECT.train_seasons
    assert re.fullmatch(r"[a-z0-9-]+", PROJECT.slug)
    assert Path(PROJECT.package_name).name == "ffai"
