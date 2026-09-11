"""Preseason draft tiers per position from prior-season aggregates (as-of preseason).

Inputs for tiers of season ``S`` are computed **only** from season ``S-1`` regular-season rows:

* ``ppg_prev``: mean PPR points per game
* ``ppg_std_prev``: standard deviation of PPR points per game
* ``games_prev``: games played (stat rows)
* ``opportunity_share_prev``: player's share of the team's targets (WR/TE/RB) or carries+targets
  (RB) or attempts (QB) across the prior season; computed from the stats frame alone
* ``age``: from rosters (birth date) when available, else omitted

Players with fewer than ``MIN_GAMES`` prior-season games are excluded (too little signal).

Per position: StandardScaler → PCA (smallest number of components explaining ≥ 90% variance)
→ GaussianMixture with ``n_components`` chosen by BIC in [4, 12] subject to
``n_components <= n_players // MIN_PLAYERS_PER_COMPONENT`` (default 8), ``n_init=5``, fixed random
state. Tiers are numbered 1..k in decreasing order of the component's mean prior PPR/game.

Honest evaluation (``evaluate_tiers``): for the tiers of season ``S`` compare tier rank with the
realised season-``S`` PPR/game of the same players: Spearman correlation, and the share of players
whose realised rank falls inside their tier's rank band (tier 1 with n1 players owns ranks 1..n1,
tier 2 owns n1+1..n1+n2, ...). Both numbers are recorded in ``metadata.json``.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ffai.config import ARTIFACTS_DIR, POSITIONS, RANDOM_STATE
from ffai.features.asof import FEATURE_VERSION

TIER_FEATURE_VERSION = "tiers_prevseason_v1"
MIN_GAMES = 4
COMPONENT_RANGE = range(4, 13)
PCA_VARIANCE = 0.90
MIN_PLAYERS_FOR_GMM = 25
# A component must have room for at least this many players: n_components <= n_players // 8.
MIN_PLAYERS_PER_COMPONENT = 8

_OPPORTUNITY = {
    "QB": ["attempts"],
    "RB": ["carries", "targets"],
    "WR": ["targets"],
    "TE": ["targets"],
}


def prior_season_inputs(
    stats: pd.DataFrame, season: int, rosters: pd.DataFrame | None = None
) -> pd.DataFrame:
    """One row per player with season ``season-1`` aggregates (as-of preseason of ``season``)."""
    prev = stats[stats["season"] == season - 1]
    if prev.empty:
        raise ValueError(f"no stat rows for season {season - 1}")
    team_totals = prev.groupby(["team"])[["attempts", "carries", "targets"]].sum()

    def _share(g: pd.DataFrame) -> float:
        pos = g["position"].iloc[0]
        cols = _OPPORTUNITY.get(pos, ["targets"])
        team = g["team"].mode().iloc[0]
        denom = float(team_totals.loc[team, cols].sum()) if team in team_totals.index else 0.0
        return float(g[cols].sum().sum() / denom) if denom > 0 else 0.0

    grp = prev.groupby("player_id", sort=False)
    out = pd.DataFrame(
        {
            "position": grp["position"].first(),
            "team": grp["team"].last(),
            "player_display_name": grp["player_display_name"].last(),
            "ppg_prev": grp["fantasy_points_ppr"].mean(),
            "ppg_std_prev": grp["fantasy_points_ppr"].std(ddof=0),
            "games_prev": grp.size(),
        }
    )
    out["opportunity_share_prev"] = grp.apply(_share, include_groups=False)
    out = out[out["games_prev"] >= MIN_GAMES]
    if rosters is not None and "birth_date" in rosters.columns:
        r = rosters[rosters["season"] == season][["gsis_id", "birth_date"]].dropna()
        r = r.drop_duplicates("gsis_id").set_index("gsis_id")
        ref = pd.Timestamp(year=season, month=9, day=1)
        age = (ref - pd.to_datetime(r["birth_date"], errors="coerce")).dt.days / 365.25
        out["age"] = out.index.map(age)
        out["age"] = out["age"].fillna(out["age"].median())
    out.index.name = "player_id"
    out["season"] = season
    return out.reset_index()


def _tier_features(inputs: pd.DataFrame) -> list[str]:
    cols = ["ppg_prev", "ppg_std_prev", "games_prev", "opportunity_share_prev"]
    if "age" in inputs.columns:
        cols.append("age")
    return cols


def fit_position_tiers(inputs: pd.DataFrame, position: str) -> dict[str, Any]:
    """Fit scaler → PCA → GMM for one position; returns the pipeline and per-player tiers."""
    pos = inputs[inputs["position"] == position].copy()
    feats = _tier_features(pos)
    if len(pos) < MIN_PLAYERS_FOR_GMM:
        raise ValueError(f"{position}: only {len(pos)} players, need {MIN_PLAYERS_FOR_GMM}")
    x = pos[feats].to_numpy(dtype="float64")
    scaler = StandardScaler().fit(x)
    xs = scaler.transform(x)
    pca_full = PCA(random_state=RANDOM_STATE).fit(xs)
    n_pca = int(np.searchsorted(np.cumsum(pca_full.explained_variance_ratio_), PCA_VARIANCE) + 1)
    pca = PCA(n_components=n_pca, random_state=RANDOM_STATE).fit(xs)
    xp = pca.transform(xs)
    best = None
    bics = {}
    max_k = max(COMPONENT_RANGE.start, len(pos) // MIN_PLAYERS_PER_COMPONENT)
    for k in COMPONENT_RANGE:
        if k >= len(pos) or k > max_k:
            break
        gmm = GaussianMixture(
            n_components=k, covariance_type="full", n_init=5, random_state=RANDOM_STATE
        ).fit(xp)
        bics[k] = float(gmm.bic(xp))
        if best is None or bics[k] < bics[best[0]]:
            best = (k, gmm)
    assert best is not None
    k, gmm = best
    comp = gmm.predict(xp)
    proba = gmm.predict_proba(xp)
    # Order components by mean prior PPG → tier 1 = highest.
    comp_ppg = (
        pd.Series(pos["ppg_prev"].to_numpy()).groupby(comp).mean().sort_values(ascending=False)
    )
    tier_of_component = {c: i + 1 for i, c in enumerate(comp_ppg.index)}
    pos["tier"] = [tier_of_component[c] for c in comp]
    pos["tier_probability"] = proba.max(axis=1).round(4)
    pipeline = Pipeline([("scaler", scaler), ("pca", pca), ("gmm", gmm)])
    return {
        "pipeline": pipeline,
        "tier_of_component": tier_of_component,
        "features": feats,
        "n_components": k,
        "max_components_allowed": max_k,
        "bic": bics,
        "pca_components": n_pca,
        "pca_explained_variance": [float(v) for v in pca.explained_variance_ratio_],
        "players": pos.sort_values(["tier", "ppg_prev"], ascending=[True, False]),
    }


def evaluate_tiers(players: pd.DataFrame, realised: pd.DataFrame) -> dict[str, Any]:
    """Spearman(tier rank, realised PPG) and share-within-tier-band for one position.

    ``players`` has ``player_id, tier``; ``realised`` has ``player_id, ppg`` for the evaluated
    season. Only players present in both are scored.
    """
    m = players[["player_id", "tier"]].merge(realised[["player_id", "ppg"]], on="player_id")
    if len(m) < 3:
        return {"n": int(len(m)), "spearman": None, "within_band_rate": None}
    rho = float(spearmanr(m["tier"], -m["ppg"]).statistic)  # higher tier number = lower ppg
    m = m.sort_values("ppg", ascending=False).reset_index(drop=True)
    m["realised_rank"] = np.arange(1, len(m) + 1)
    sizes = m.groupby("tier").size().sort_index()
    ends = sizes.cumsum()
    starts = ends - sizes + 1
    band_ok = [
        starts[t] <= r <= ends[t] for t, r in zip(m["tier"], m["realised_rank"], strict=True)
    ]
    return {
        "n": int(len(m)),
        "spearman": round(rho, 4),
        "within_band_rate": round(float(np.mean(band_ok)), 4),
        "tier_sizes": {int(k): int(v) for k, v in sizes.items()},
    }


def _load_previous(previous_tier_version: str, artifacts: Path) -> dict[str, Any]:
    d = artifacts / "tiers" / previous_tier_version
    return {
        "meta": json.loads((d / "metadata.json").read_text(encoding="utf-8")),
        "tiers": json.loads((d / "tiers.json").read_text(encoding="utf-8")),
        "pipelines": joblib.load(d / "model.pkl"),
    }


def _worse_on_both(before: dict[str, Any], after: dict[str, Any]) -> bool:
    b_s, a_s = before.get("spearman"), after.get("spearman")
    b_w, a_w = before.get("within_band_rate"), after.get("within_band_rate")
    if None in (b_s, a_s, b_w, a_w):
        return False
    return a_s < b_s and a_w < b_w


def build_tiers(
    stats: pd.DataFrame,
    season: int,
    *,
    rosters: pd.DataFrame | None = None,
    artifacts: Path = ARTIFACTS_DIR,
    evaluate_on_realised: bool = True,
    previous_tier_version: str | None = None,
    now: dt.datetime | None = None,
) -> dict[str, Any]:
    """Fit tiers for preseason ``season`` and persist ``artifacts/tiers/<tier_version>/``.

    With ``previous_tier_version`` the new fit is compared position by position against that
    artifact's evaluation; a position that gets worse on **both** Spearman and within-band keeps
    the previous tiers (players and pipeline are copied over) and is marked ``kept: "previous"``
    in ``metadata.json -> comparison_to_previous``.
    """
    now = now or dt.datetime.now(dt.UTC)
    inputs = prior_season_inputs(stats, season, rosters)
    tier_version = f"{now:%Y%m%d}-{TIER_FEATURE_VERSION}-{season}-mpc{MIN_PLAYERS_PER_COMPONENT}"
    if previous_tier_version == tier_version:
        raise ValueError("previous_tier_version must differ from the new version")
    previous = _load_previous(previous_tier_version, artifacts) if previous_tier_version else None
    out_dir = artifacts / "tiers" / tier_version
    out_dir.mkdir(parents=True, exist_ok=True)

    realised = None
    if evaluate_on_realised:
        cur = stats[stats["season"] == season]
        if not cur.empty:
            realised = (
                cur.groupby("player_id")["fantasy_points_ppr"].mean().rename("ppg").reset_index()
            )

    positions_meta: dict[str, Any] = {}
    tiers_json: dict[str, Any] = {}
    evaluation: dict[str, Any] = {"season": season, "positions": {}}
    pipelines: dict[str, Any] = {}
    for pos in POSITIONS:
        fitted = fit_position_tiers(inputs, pos)
        players = fitted["players"]
        pipelines[pos] = {
            "pipeline": fitted["pipeline"],
            "tier_of_component": fitted["tier_of_component"],
            "features": fitted["features"],
        }
        positions_meta[pos] = {
            "n_players": int(len(players)),
            "n_components": fitted["n_components"],
            "bic": fitted["bic"],
            "pca_components": fitted["pca_components"],
            "pca_explained_variance": fitted["pca_explained_variance"],
            "features": fitted["features"],
            "tier_sizes": {int(k): int(v) for k, v in players.groupby("tier").size().items()},
        }
        tiers_json[pos] = [
            {
                "tier": int(t),
                "players": [
                    {
                        "player_id": r.player_id,
                        "name": r.player_display_name,
                        "team": r.team,
                        "ppg_prev": round(float(r.ppg_prev), 2),
                        "games_prev": int(r.games_prev),
                        "tier_probability": float(r.tier_probability),
                    }
                    for r in g.itertuples(index=False)
                ],
            }
            for t, g in players.groupby("tier", sort=True)
        ]
        if realised is not None:
            evaluation["positions"][pos] = evaluate_tiers(players, realised)

    comparison = None
    if previous is not None and realised is not None:
        comparison = {"previous_tier_version": previous_tier_version, "positions": {}}
        prev_eval = (previous["meta"].get("evaluation") or {}).get("positions", {})
        for pos in POSITIONS:
            before = {
                **prev_eval.get(pos, {}),
                **{
                    k: previous["meta"]["positions"][pos][k]
                    for k in ("n_components", "pca_components", "n_players")
                },
            }
            after = {
                **evaluation["positions"][pos],
                **{
                    k: positions_meta[pos][k]
                    for k in ("n_components", "pca_components", "n_players")
                },
            }
            kept = "new"
            if _worse_on_both(before, after):
                kept = "previous"
                pipelines[pos] = previous["pipelines"][pos]
                tiers_json[pos] = previous["tiers"]["positions"][pos]
                positions_meta[pos] = {
                    **previous["meta"]["positions"][pos],
                    "kept_from": previous_tier_version,
                }
                evaluation["positions"][pos] = prev_eval[pos]
            comparison["positions"][pos] = {"before": before, "after": after, "kept": kept}

    joblib.dump(pipelines, out_dir / "model.pkl", compress=3)
    (out_dir / "tiers.json").write_text(
        json.dumps(
            {"tier_version": tier_version, "season": season, "positions": tiers_json},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    metadata = {
        "tier_version": tier_version,
        "tier_feature_version": TIER_FEATURE_VERSION,
        "feature_version": FEATURE_VERSION,
        "season": season,
        "inputs": "prior-season (season-1) aggregates only; min games "
        f"{MIN_GAMES}; age from rosters when available",
        "method": (
            "StandardScaler -> PCA (>=90% variance) -> GaussianMixture(full, n_init=5), "
            "n_components by BIC in [4, 12]; tiers ordered by component mean prior PPG"
        ),
        "generated_at_utc": now.isoformat(timespec="seconds"),
        "positions": positions_meta,
        "evaluation": evaluation if realised is not None else None,
        "min_players_per_component": MIN_PLAYERS_PER_COMPONENT,
        "comparison_to_previous": comparison,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata
