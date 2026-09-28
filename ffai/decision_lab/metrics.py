"""Outcome metrics for a saved decision (mirrored by ``metrics.ts``).

Observed outcomes never enter :mod:`ffai.decision_lab.policy`; they are joined here, after a
decision exists, to answer "what can we honestly conclude?". Every metric has its own eligibility
rule and reports why it is null instead of inventing a number:

* ``chosen_actual_points`` — the explicitly chosen player's observed PPR points, or null with a
  reason (``no_action_recorded``, ``declined``, ``outcome_missing``).
* ``choice_set_regret`` — max observed actual over the *comparable* choice set minus the chosen
  player's actual; only when every member of the choice set has an observed outcome. A missing
  alternative leaves the regret unavailable — it is never imputed to zero.
* ``points_vs_baseline_choice`` — chosen actual minus the baseline-preferred player's actual;
  needs a baseline preference and both outcomes.
* ``model_policy_vs_baseline_choice`` — the model policy's recommended player's actual minus the
  baseline-preferred player's actual. A hypothetical comparison between two comparators; it is
  not attributed to the user's action and exists even when no action was recorded.

Zero and negative actuals are valid observations. A recommendation is not a recorded action; a
recorded action is not proof it was carried out in a league; a hypothetical replay outcome is not
measured product impact — ``action_state`` and ``action_kind`` travel with every metric block.
"""

from __future__ import annotations

from typing import Any

from ffai.decision_lab import SCHEMA_VERSIONS
from ffai.decision_lab.canonical import q

METRICS_SCHEMA = SCHEMA_VERSIONS["outcome_metrics"]


def outcome_index(outcome_snapshot: dict[str, Any]) -> dict[str, float]:
    """``{player_id: observed actual}`` for the rows with an observed outcome."""
    idx: dict[str, float] = {}
    for row in outcome_snapshot.get("rows", []):
        actual = row.get("actual")
        if isinstance(actual, int | float) and not isinstance(actual, bool):
            idx[str(row["player_id"])] = q(float(actual))
    return idx


def compute_metrics(
    result: dict[str, Any],
    action: dict[str, Any],
    outcome_snapshot: dict[str, Any],
) -> dict[str, Any]:
    """Join a policy result, the (possibly absent) human action, and an outcome snapshot."""
    actuals = outcome_index(outcome_snapshot)
    choice_set = [r["player_id"] for r in result.get("ranking", []) if r.get("comparable")]
    nominated = [r["player_id"] for r in result.get("ranking", [])]
    observed = [pid for pid in choice_set if pid in actuals]
    missing = [pid for pid in choice_set if pid not in actuals]
    state = action.get("state", "not_recorded")
    chosen = action.get("chosen_player_id") if state == "recorded" else None

    per_alt = [
        {"player_id": pid, "actual": actuals.get(pid), "observed": pid in actuals}
        for pid in nominated
    ]

    # chosen actual
    chosen_actual = None
    chosen_reason: str | None
    if state == "not_recorded":
        chosen_reason = "no_action_recorded"
    elif state == "declined":
        chosen_reason = "declined"
    elif chosen not in actuals:
        chosen_reason = "outcome_missing"
    else:
        chosen_actual = actuals[chosen]
        chosen_reason = None

    # best in the comparable choice set (only meaningful when complete)
    best = None
    regret = None
    regret_reason: str | None = None
    if not choice_set:
        regret_reason = "empty_choice_set"
    elif missing:
        regret_reason = "incomplete_outcomes"
    else:
        best_pid = sorted(choice_set, key=lambda p: (-actuals[p], p))[0]
        best = {"player_id": best_pid, "actual": actuals[best_pid]}
        if chosen_actual is None:
            regret_reason = chosen_reason
        elif chosen not in choice_set:
            regret_reason = "chosen_outside_choice_set"
        else:
            regret = q(actuals[best_pid] - chosen_actual)

    # baseline comparator
    base_pid = (result.get("baseline") or {}).get("preferred_player_id")
    base_actual = actuals.get(base_pid) if base_pid else None
    vs_baseline = None
    vs_baseline_reason: str | None = None
    if base_pid is None:
        vs_baseline_reason = "no_baseline_preference"
    elif chosen_actual is None:
        vs_baseline_reason = chosen_reason
    elif base_actual is None:
        vs_baseline_reason = "baseline_outcome_missing"
    else:
        vs_baseline = q(chosen_actual - base_actual)

    model_pid = result.get("recommended_player_id")
    model_actual = actuals.get(model_pid) if model_pid else None
    model_vs_baseline = None
    model_vs_baseline_reason: str | None = None
    if model_pid is None:
        model_vs_baseline_reason = "no_model_recommendation"
    elif base_pid is None:
        model_vs_baseline_reason = "no_baseline_preference"
    elif model_actual is None or base_actual is None:
        model_vs_baseline_reason = "outcome_missing"
    else:
        model_vs_baseline = q(model_actual - base_actual)

    return {
        "schema_version": METRICS_SCHEMA,
        "decision_id": result.get("decision_id"),
        "outcome_snapshot_id": outcome_snapshot.get("snapshot_id"),
        "outcome_id": outcome_snapshot.get("outcome_id"),
        "action_state": state,
        "action_kind": action.get("kind") if state == "recorded" else None,
        "chosen_player_id": chosen,
        "choice_set": choice_set,
        "coverage": {
            "nominated": len(nominated),
            "choice_set_size": len(choice_set),
            "observed": len(observed),
            "missing": missing,
        },
        "chosen_actual_points": chosen_actual,
        "chosen_actual_reason": chosen_reason,
        "best_in_choice_set": best,
        "choice_set_regret": regret,
        "choice_set_regret_reason": regret_reason,
        "baseline_preferred_player_id": base_pid,
        "baseline_preferred_actual": base_actual,
        "points_vs_baseline_choice": vs_baseline,
        "points_vs_baseline_choice_reason": vs_baseline_reason,
        "model_recommended_player_id": model_pid,
        "model_recommended_actual": model_actual,
        "model_policy_vs_baseline_choice": model_vs_baseline,
        "model_policy_vs_baseline_choice_reason": model_vs_baseline_reason,
        "per_alternative": per_alt,
        "notes": [
            "A recommendation is not a recorded action; a recorded action is not proof it was carried out in a league.",
            "Hypothetical replay outcomes are not measured product impact.",
        ],
    }


def aggregate(metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate several metric blocks with metric-specific denominators.

    Each mean uses only the rows where that metric is non-null; zero denominators give ``null``
    with the counts. Nothing is averaged across groups of unequal size here — callers aggregate
    the rows of one group at a time and never re-average group means.
    """

    def _mean(key: str) -> dict[str, Any]:
        vals = [float(m[key]) for m in metric_rows if isinstance(m.get(key), int | float)]
        return {
            "n": len(vals),
            "mean": q(sum(vals) / len(vals)) if vals else None,
        }

    return {
        "rows": len(metric_rows),
        "with_recorded_action": sum(1 for m in metric_rows if m.get("action_state") == "recorded"),
        "chosen_actual_points": _mean("chosen_actual_points"),
        "choice_set_regret": _mean("choice_set_regret"),
        "points_vs_baseline_choice": _mean("points_vs_baseline_choice"),
        "model_policy_vs_baseline_choice": _mean("model_policy_vs_baseline_choice"),
        "model_beats_baseline": {
            "n": sum(
                1
                for m in metric_rows
                if isinstance(m.get("model_policy_vs_baseline_choice"), int | float)
            ),
            "wins": sum(
                1
                for m in metric_rows
                if isinstance(m.get("model_policy_vs_baseline_choice"), int | float)
                and m["model_policy_vs_baseline_choice"] > 0
            ),
            "losses": sum(
                1
                for m in metric_rows
                if isinstance(m.get("model_policy_vs_baseline_choice"), int | float)
                and m["model_policy_vs_baseline_choice"] < 0
            ),
            "ties": sum(
                1
                for m in metric_rows
                if isinstance(m.get("model_policy_vs_baseline_choice"), int | float)
                and m["model_policy_vs_baseline_choice"] == 0
            ),
        },
    }
