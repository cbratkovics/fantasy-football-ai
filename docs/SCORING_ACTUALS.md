# Actual scoring and provenance

`ffai.scoring` defines `ffai-nflverse-v1`: passing yards /25, four points per passing TD,
minus two per interception; rushing/receiving yards /10, six per TD; two per conversion,
minus two per fumble lost and six per special-teams TD. Reception points are zero (standard),
one half (half-PPR), or one (PPR). No yardage bonuses apply. Custom PPR leagues may differ.

The model predicts PPR. Non-PPR forecasts remain estimates derived using estimated receptions;
scoring reconciliation establishes the rules, not the accuracy of those estimates. Actuals
never use `receptions_estimate`.

`GET /predictions/{season}/{week}?scoring=standard|half|ppr` returns actuals in the requested
format and declares `actual_scoring_format`, `actual_source`, and `actual_null_reason` per row.
Player history, evaluation, and existing mart-history responses remain PPR and declare that
format. Legacy stored `actual` still means PPR. Historical artifacts are not rewritten.
Weekly player-history observations use the same resolver as the predictions route, including
revision/conflict checks, and expose the same additive actual-source and null-reason metadata.
This prevents a withheld prediction actual from reappearing as an unqualified history value.

`ffai.serve.actuals.resolve_actual` uses this precedence, validating agreement before resolving:

1. Legacy PPR actual for PPR; explicitly declared rules and format values for other formats.
2. Complete realized stat components from the same named outcome revision, scored by the
   existing scoring function. Provider-defined null components mean no opportunity; absent
   rows or columns cannot become zero.
3. Realized receptions with the same outcome revision: standard = PPR minus receptions;
   half-PPR = PPR minus half the receptions.
4. The exported `fct_player_week` fact at the same player, season, week and source family.
   All model/candidate duplicate outcomes must agree before reduction. The mart's PPR value
   must agree with any stored PPR actual before its non-PPR value can be used.

Values agree to 0.011 points (the existing artifacts round to hundredths). Zero and negative
observations are valid; nonfinite values are rejected. Conflicts, custom rules, absent conversion
evidence and inconsistent revisions return null with an explicit reason. Optional missing marts
and missing format columns do not prevent legacy PPR serving. The API performs no network repair.

Future `attach_actuals` calls add format-specific values and a digest of the realized components
only when the complete component row reconciles with the stored PPR observation. Refresh replaces
one revision atomically, clearing stale conversion evidence when a corrected row is missing or
incomplete. This changes future output only; no published benchmark is regenerated.

Focused verification:

```bash
.venv/bin/python -m pytest tests/test_actual_resolution.py tests/test_scoring_reconciliation.py tests/test_weekly_policy.py tests/test_api.py
```

Browser component tests cover format labels and missing versus observed actuals. Recommendations
support only the existing PPR Decision Lab policy; browsing support for other formats does not
expand the decision policy's scoring support.
