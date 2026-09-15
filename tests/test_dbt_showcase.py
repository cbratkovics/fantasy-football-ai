import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHOWCASE = ROOT / "frontend-next/src/data/dbt-showcase.json"


def test_showcase_is_current_and_inventory_is_version_aware() -> None:
    subprocess.run(
        [sys.executable, "scripts/generate_dbt_showcase.py", "--check"], cwd=ROOT, check=True
    )
    data = json.loads(SHOWCASE.read_text())
    assert data["inventory"] == {
        "executableModelNodes": 26,
        "logicalModelNames": 25,
        "snapshotNodes": 1,
        "layers": {"bronze": 9, "silver": 7, "gold": 10},
    }
    policy = {n["version"]: n for n in data["nodes"] if n["name"] == "fct_decision_policy"}
    assert policy[1]["alias"] == "fct_decision_policy"
    assert policy[2]["alias"] == "fct_decision_policy_v2"
    assert policy[1]["uniqueId"] != policy[2]["uniqueId"]


def test_showcase_allowlist_does_not_publish_private_or_runtime_fields() -> None:
    text = SHOWCASE.read_text().lower()
    for forbidden in (
        "motherduck_token",
        "owner_email",
        "invocation_id",
        "compiled_code",
        "root_path",
    ):
        assert forbidden not in text
    data = json.loads(text)
    for node in data["nodes"]:
        assert (ROOT / node["sqlpath"]).is_file()
        assert (ROOT / node["yamlpath"]).is_file()
