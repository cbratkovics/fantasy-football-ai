# Product presentation and defaults

The five task pages now put **choose → understand → inspect** in that order. Primary text is
16px and secondary text 14px. Warm neutrals and deep green replace technical banners; identifiers,
exact UTC, all policy reasons, parameters and method details remain in named disclosures.

- **Start/Sit:** latest verified published week, PPR, visible FLEX, two empty selection cards.
  Search supports names/forecast teams, keyboard selection and clear empty results. One explicit
  “Compare assuming these players can play” click constructs the exact next inputs atomically.
  Only selected, slot-eligible unknowns become `user_assumed`; explicit exclusions survive.
  Selection/slot/evidence changes clear confirmations; no action is recorded by comparing.
- **Player projections:** actual published periods from read-only `GET /prediction-periods`,
  PPR/FLEX, descending projection. Query overrides validated view preferences; unavailable explicit
  periods do not become another week. Search retains rank within the selected position group.
  Desktop tables/mobile cards show 25 at a time with explicit counts. Range scales include zero
  and negative values. All-null result summaries distinguish absent outcomes from withheld evidence.
- **Decision Lab:** latest verified weekly snapshot, FLEX, no fabricated nominations/availability.
  Explicit case/snapshot links initialize one coherent mode. Historical and synthetic modes/cases
  are explicit; examples are ordered independently of outcomes. Compatible saved display state can
  restore selections/settings, never availability, corrupt-evidence switches or recorded actions.
  Advanced parameters/testing controls are collapsed; custom settings can be reset.
- **Model results:** newest supported complete historical evaluation by metadata, all positions,
  average absolute error. Valid explicit evaluation/cohort/metric choices override saved display
  preferences. Finding, comparison bars, within-three-point frequency and player-week count all use
  the same selected artifact/cohort. Rate differences are percentage points; cohort errors are never
  averaged to fabricate an overall value. Frozen and rolling-origin results stay separate.
- **Historical tiers/Draft:** use the supported artifact’s target/input/evaluation seasons, not its
  generation year. Historical groups/teams precede plots and method. Assignment strength is only
  explained in details, never presented as win probability or exact draft rank.

Selections transfer as canonical IDs plus exact snapshot/slot, without numerical URL projections.
The destination verifies its evidence. Historical periods open matching Lab evidence; non-PPR
browsing explicitly prepares a PPR comparison and keeps its browsing preference. API/bundle
release mismatches block transfer. A linked weekly period is never silently changed on focus.

“Model range” means the existing position-level validation-residual offsets; actual scores can
fall outside it. A comparison margin is a policy threshold, not significance or confidence.
“Historical baseline” means the causal trailing history rule, not an invented last-three average.
“Points left behind among these choices” retains complete-choice-set-regret semantics in details.

Canonical policy/specification, scoring resolver, receipt identities/events, original evidence,
readiness rules, goldens and historical artifacts are unchanged. The human wording is a separate
presentation mapping. Same-origin comparison works during API outage. Missing optional context is
a compact limitation; corrupt/incompatible context and known restrictions remain visible blockers.
Failed refresh retains original facts/timestamps. Rechecks never rewrite saved records, steal focus
or announce a clock every second. Current restrictions permit only explicit historical exploration.

Local review uses the changed artifact-backed API on `http://127.0.0.1:7863` and a frontend built
with that `NEXT_PUBLIC_API_URL`, served on `http://localhost:3500`. The new period index is additive;
an older deployed API without it shows an explicit unavailable state, not speculative periods.
No deployed backend or frontend was changed. Screenshots/manifests remain outside publication
paths under `/tmp/ffai-ui-review/{before,after}`; before uses merged `e6d8bf8`, after uses this
uncommitted UI branch. Synthetic comparison captures are labeled and never mixed with live facts.

For a separately authorized release, deploy the additive backend first and verify
`GET /prediction-periods` at the frontend's actual API target, then deploy the frontend. Existing
clients continue to work with the updated backend. With an older API, the redesigned projections
page shows an explicit index-unavailable state and a link to the same-origin comparison; it does
not invent periods or silently change sources. A frontend-only preview using the old hosted API
does not verify the new backend. No deployment or mapping/secret change is authorized by this note.
