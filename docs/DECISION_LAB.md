# Decision Lab

> Given these players as my actual alternatives for one starting slot, what does the available
> evidence recommend, what assumptions could change that recommendation, what did I choose, and
> what can we honestly conclude after outcomes become available?

The Decision Lab is a one-slot start/sit comparison over the repository's committed evidence. It
runs entirely from static files: no hosted API, no MotherDuck, no nflverse download, no secrets,
no model loading in the browser. It is not a lineup optimizer, draft engine, waiver marketplace,
trading system, or opponent simulator, and it makes no promise about winning a league.

* Route: `/decision-lab` in the Next.js app (`make frontend`, then http://localhost:3000/decision-lab).
* Evidence bundle: `artifacts/decision_lab/` (canonical) mirrored to `frontend-next/public/decision-lab/`.
* Reference policy: `ffai/decision_lab/policy.py`, specification `ffai/decision_lab/policy_spec.json`.
* TypeScript mirror: `frontend-next/src/lib/decision-lab/`.
* Decision records: ADR-0032 (evidence adapter and policy), ADR-0033 (receipts and trust
  boundary), ADR-0034 (frontend test tooling).

## Setup

```bash
make install                 # Python 3.11 venv (uv), training + dev dependencies, editable ffai
make lab-build               # rebuild artifacts/decision_lab from committed evidence and mirror it into frontend-next/public
make lab-check               # build into a temp dir and compare with the committed bundle + public copy (never rewrites)
make lab-test                # Python lab tests + frontend unit/parity/component tests
make lab-e2e                 # Playwright browser flow (installs Chromium on first run)
cd frontend-next && npm ci && npm run dev    # http://localhost:3000/decision-lab
```

Dependency installation needs the network; running the lab does not. The page loads
`/decision-lab/manifest.json` and one snapshot file per case from its own origin and verifies
byte and content digests before using anything.

## Three explicitly labelled contexts

| Context | Source | What it is | What it is not |
|---|---|---|---|
| **Historical replay** | frozen 2024 test season, out-of-sample 2025 season, completed 2026 weeks with observed outcomes | recorded retrospective evaluation rows: projection, floor/ceiling, causal baseline, and (separately) the observed outcome | proof of what anyone knew before kickoff; the population is players who appear in realized stats |
| **Published weekly snapshot** | the newest `artifacts/predictions/<season>/week_<ww>.json` without outcomes | a dated projection artifact with its data cutoff, generation time, and the run that published it | current, live, or game-day verified; availability is user-assumed |
| **Synthetic sandbox** | `inputs/syn-*.json` | small, unmistakably synthetic inputs for boundary conditions and regression behaviour | evidence about any real player; never mixed into historical numbers |

## Architecture

```
committed evidence                     offline exporter                       browser
artifacts/marts/fct_player_week.parquet ─┐
artifacts/predictions/<season>/week_*.json ├─ scripts/export_decision_lab.py ─► artifacts/decision_lab/ ─copy─► frontend-next/public/decision-lab/
artifacts/eval/*.json, rolling_<s>.json  │     (ffai.decision_lab.exporter)       manifest.json, inputs/, outcomes/, cases.json
artifacts/models/<v>/*.csv, metadata.json┘                                              │
                                                                                          ▼
ffai/decision_lab/policy.py  ◄── golden fixtures ──►  src/lib/decision-lab/policy.ts  (same statuses, ids, reasons, numbers)
ffai/decision_lab/receipts.py                          src/lib/decision-lab/receipts.ts + storage.ts (localStorage, import/export)
python -m ffai.decision_lab.replay  ◄── receipt JSON ──►  Export / Import in the Saved Decisions panel
```

The exporter reuses the tested gold marts and the source artifacts; it does not train, does not
overwrite `artifacts/marts`, and does not need the stats cache. It writes into a temporary
sibling directory, verifies the result with the same fail-closed loader the tests use, and only
then swaps it into place. A failed export leaves the previous bundle untouched and records
`artifacts/decision_lab/_last_attempt.json`.

### Bundle files and identities

| File | Schema | Identity |
|---|---|---|
| `manifest.json` | `decision_lab_manifest/1.0` | lists every file with `file_sha256` (bytes) and `content_sha256` (canonical JSON), row counts, population, baseline provenance, reconciliation, lineage, the code revision that produced the export |
| `inputs/<snapshot_id>.json` | `inputs_snapshot/1.0` | one (source family, season, week); champion candidate per position at the source; **never** contains actuals, errors, ranks or regret (the contract rejects them) |
| `outcomes/<snapshot_id>.json` | `outcome_snapshot/1.0` | observed PPR points per input row (null when unobserved) and the `inputs_content_sha256` it pairs with; its own content digest is the `outcome_id` |
| `cases.json` | `decision_lab_cases/1.0` | curated cases with stable ids, expected policy behaviour, experiments, sealed selection pattern |
| `_build.json` | volatile | build time, tool versions, the manifest's content digest (kept outside the manifest: no self-reference) |

Canonical JSON (`ffai/decision_lab/canonical.py`, `canonical.ts`): sorted keys, no whitespace,
`JSON.stringify` string escaping, numbers rounded half-up at six decimals and printed without
trailing zeros. Every digest is `sha256(canonical_json(document))`; `digest_coverage` in the
manifest says which fields each digest covers. Adding outcomes later never changes an inputs
digest or a decision id, because outcomes live in their own file with their own identity.

### Baseline provenance

The baseline is the evaluator's causal trailing mean (`ffai.eval.evaluator.add_causal_baseline`,
reproduced in SQL by `fct_player_week`): the player's earlier realised points, falling back to
the position's, falling back to the row's own prediction. The committed parquet exported before
this change carries the value but not which branch produced it, so the exporter derives it:

1. If the mart has `baseline_source` (added to `fct_player_week` in this change; it appears in
   the parquet after the next production export) the column is used as is.
2. Otherwise the exporter reconciles the champion-candidate aggregates of the whole evaluation
   window (n, MAE, baseline MAE, overall and per position) to the evaluation artifact (frozen
   test, out-of-sample season) or to `artifacts/eval/rolling_<season>.json` (completed weekly
   weeks) at 1e-4. A match confirms the warehouse held the full history; since every mart row
   belongs to a player with at least one prior game, every row's baseline came from player
   history. Provenance is then `player_history` with basis `population_rule_and_reconciliation`.
3. A week without a reference yet (the published snapshot) is `population_rule_unreconciled`.
4. A mismatch (for example a fixture-built mart) yields `unknown` for every row, and the policy
   answers REVIEW for those cases unless model-only exploration is explicitly enabled.

Nothing fabricates a baseline from the fixture, from current totals, or from future rows.

## Policy semantics (`policy_spec.json`, version 1.0.0)

Objective: the highest PPR projection among the explicitly allowed alternatives, subject to
evidence validation, an ambiguity margin, and an optional model-floor guardrail. Output is
`recommend`, `review`, or `hold`; `recommended_player_id` is present and non-null only for
`recommend`. Reasons are ordered and carry their measured values; explanations are rendered
deterministically from those reasons in both languages.

Precedence:

1. **HOLD** on invalid or incompatible evidence: corrupt or blocked snapshot, non-PPR scoring,
   unsupported slot, alternatives from different periods or model versions/candidates, invalid
   parameters, more than eight alternatives, duplicates. Invalid input never becomes a weaker
   recommendation, and lowering a threshold cannot unblock it.
2. Exclude alternatives marked unavailable or whose position cannot fill the slot (visible
   reasons). **REVIEW** when a remaining alternative has no finite projection, when availability
   is unresolved, when fewer than two comparable alternatives remain, or when an independent
   baseline is required and missing (model-only exploration proceeds with a recorded limitation).
3. Rank comparable alternatives by projection; ties are ordered by player id for display only.
4. Compute the top-two gap before the floor gate; `|gap| <= 1e-6` is a tie and produces REVIEW
   even when the minimum gap is zero.
5. `gap < min_projection_gap - 1e-6` produces REVIEW; equality qualifies.
6. With `min_floor` set: a missing floor on the leading option, or a floor below the threshold,
   produces REVIEW; equality qualifies. The policy never nominates a lower-ranked player instead.
7. Otherwise RECOMMEND the leading option, conditional on the recorded eligibility assumptions.

The baseline preference (highest causal trailing mean among comparable alternatives) is reported
separately with agreement/disagreement/tie; it never replaces the model and is never claimed when
its provenance is the model's own prediction. Floors and ceilings are position-level residual
quantiles, not probabilities or guaranteed bounds. The pipeline's PUBLISH/HOLD/PROMOTE is a
different state machine from these statuses.

## Outcome metrics

Computed only after a decision exists, from a separate outcome snapshot:

| Metric | Eligibility | Null reason codes |
|---|---|---|
| `chosen_actual_points` | an action was recorded and the chosen player's outcome is observed | `no_action_recorded`, `declined`, `outcome_missing` |
| `choice_set_regret` | every comparable alternative has an observed outcome and a choice was recorded inside the set | `incomplete_outcomes`, `empty_choice_set`, `chosen_outside_choice_set` |
| `points_vs_baseline_choice` | a baseline preference exists and both outcomes are observed | `no_baseline_preference`, `baseline_outcome_missing` |
| `model_policy_vs_baseline_choice` | the policy recommended someone and both outcomes are observed (hypothetical; not the user's action) | `no_model_recommendation`, `outcome_missing` |

Zero and negative actuals are valid observations. A missing alternative is never imputed. A
recommendation is not a recorded action; a recorded action is not proof it was carried out in a
league; a hypothetical replay outcome is not measured product impact. Aggregates use each
metric's own numerator and denominator and return null with counts for empty denominators.

The existing `fct_player_decisions` / `fct_decision_policy` marts keep their meaning where they
are reused: pool-wide hindsight gap (best scorer in the whole position pool) and a rank-k
projection benchmark, not roster regret or free-agent availability.

## Receipts and replay (`decision_receipt/1.0`)

A receipt holds the immutable decision inputs, the computed result and its digest, an ordered
event list, and two projections of that list (`action`, `outcome`). Initial state: `action.state
= not_recorded` with null id, time, player and kind; the recommendation is never auto-filled as
the user's choice. Events: `action_recorded` (hypothetical replay or self-reported real choice),
`action_declined`, `outcome_attached`. At most one action event; outcome events must belong to
the decision's own snapshot (same id, same inputs content digest, same season/week/model).

* `decision_id = sha256(canonical(decision_inputs))` — schema and policy versions, snapshot
  reference including the inputs content digest, slot, alternatives with availability fields,
  parameters. Replaying the same semantic inputs reproduces it; events never change it.
* `event_id = sha256(canonical({decision_id, seq, event_type, at_utc, payload}))`.
* Changing an assumption or a parameter is a new decision with `parent_decision_id`.

Import/replay validates the schema, recomputes the decision id, re-evaluates the policy and
compares to the stored result (a tampered result field is rejected even when the JSON parses),
verifies event ids and the action/outcome projections, checks the chosen player is a nominated
alternative, verifies the snapshot digest against the bundle, and recomputes attached metrics.
Duplicate imports merge idempotently; conflicting records with the same id are rejected. Receipts
never carry URLs or filesystem paths to fetch; notes are rendered as text.

```bash
# Python replay of a receipt exported from the browser (or created by the CLI)
python -m ffai.decision_lab.replay --new --case syn-ambiguity-floor-relaxation --bundle artifacts/decision_lab --out /tmp/r0.json
python -m ffai.decision_lab.replay /tmp/r0.json --bundle artifacts/decision_lab --record-action SYN-A --kind hypothetical_replay --out /tmp/r1.json
python -m ffai.decision_lab.replay /tmp/r1.json --bundle artifacts/decision_lab --attach-outcomes --out /tmp/r2.json
python -m ffai.decision_lab.replay /tmp/r2.json --bundle artifacts/decision_lab      # every check, decision_id unchanged
```

Trust boundary: digests let a reader check a receipt against known evidence; they are not
signatures and prove nothing about who acted when. Browser local storage is editable and erasable;
append-only is application behaviour, not a tamper-proof audit service. The outcome reveal
sequence is educational: the outcome files are public static artifacts and can be inspected.

## Curated cases

`artifacts/decision_lab/cases.json` (see the Decision Lab page for the list). Synthetic cases
form the executable regression suite: floor relaxation cannot bypass ambiguity, exact ties,
threshold equality, missing/zero/negative outcomes, slot-invalid and excluded alternatives,
unknown weekly availability, non-independent baselines, blocked evidence, one-option sets, and a
missing floor. Real cases are discovered deterministically from the evidence (top-four
projections per position and week; model/baseline disagreement with each side winning where the
evidence contains it, an agreement case, an ambiguity case, and the published weekly snapshot).
Titles do not reveal outcomes; the selection pattern is sealed in the file and shown after the
reveal step. Curated examples chosen to illustrate wins and losses are not an unbiased
performance sample.

## Limitations

* Historical rows exist only for players with realized stats; the lab cannot reconstruct a
  pregame roster universe, injury status, or a user's league availability.
* Weekly snapshots score every eligible player; availability is a user assumption, labelled so.
* Floors/ceilings are position-level residual quantiles; within a position an additive floor
  cannot reverse the projection order, and clipping at zero can create ties.
* The recorded 2026 rolling weeks show baseline MAE below champion MAE (week 1: 5.0275 vs
  5.1046; week 2: 4.7397 vs 4.8282). These are artifact observations, not proof of a lasting
  reversal, and lower MAE does not by itself prove a better lineup choice.
* Threshold sweeps on viewed historical outcomes are exploratory policy analysis, not a held-out
  policy result.
* The bundle is public; nothing in the reveal flow is secret.

## Rollback

The lab is additive. To remove it: delete the `/decision-lab` route and navigation entry, the
`frontend-next/src/lib/decision-lab/` and `frontend-next/public/decision-lab/` directories,
`artifacts/decision_lab/`, `ffai/decision_lab/`, `scripts/export_decision_lab.py`,
`scripts/lab_review_notes.py`, the `lab-*` Makefile targets, the lab steps in `ci.yml` and
`weekly.yml`, and the `tests/test_decision_lab_*.py` / `tests/test_lab_review_notes.py` files.
The `baseline_source` column on `fct_player_week` can stay (additive, contracted, documented) or
be dropped together with its YAML entry and the `decision_lab` exposure. No existing API route,
artifact, or mart contract changed.
