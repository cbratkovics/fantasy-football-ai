{#- Rules-based fantasy points from raw nflverse stat columns — the SQL twin of ffai.scoring.
    Reproduces nflfastR's calculate_player_stats definitions:
      standard = passing_yards/25 + 4*passing_tds - 2*passing_interceptions
               + rushing_yards/10 + 6*rushing_tds + receiving_yards/10 + 6*receiving_tds
               + 2*(all 2pt conversions) - 2*(all fumbles lost) + 6*special_teams_tds
      half = standard + 0.5*receptions ; ppr = standard + 1.0*receptions
    Missing stats count as zero (nflverse leaves a stat null when a player had no opportunity).
    tests/silver/assert_scoring_rules_reconcile_to_nflverse.sql proves this against nflverse's own
    fantasy_points / fantasy_points_ppr columns (max abs diff <= 0.01); the unit test in
    models/silver/_silver.yml pins the rules on fixed rows. -#}
{% macro fantasy_points(format, relation_alias=none) -%}
    {%- set weights = {'standard': 0.0, 'half': 0.5, 'ppr': 1.0} -%}
    {%- if format not in weights -%}
        {{ exceptions.raise_compiler_error("fantasy_points: unknown format '" ~ format ~ "'; expected standard, half, or ppr") }}
    {%- endif -%}
    {%- set p = relation_alias ~ '.' if relation_alias else '' -%}
    (
        0.04 * coalesce({{ p }}passing_yards, 0)
        + 4.0 * coalesce({{ p }}passing_tds, 0)
        - 2.0 * coalesce({{ p }}passing_interceptions, 0)
        + 0.1 * coalesce({{ p }}rushing_yards, 0)
        + 6.0 * coalesce({{ p }}rushing_tds, 0)
        + 0.1 * coalesce({{ p }}receiving_yards, 0)
        + 6.0 * coalesce({{ p }}receiving_tds, 0)
        + {{ weights[format] }} * coalesce({{ p }}receptions, 0)
        + 2.0 * (
            coalesce({{ p }}passing_2pt_conversions, 0)
            + coalesce({{ p }}rushing_2pt_conversions, 0)
            + coalesce({{ p }}receiving_2pt_conversions, 0)
        )
        - 2.0 * (
            coalesce({{ p }}sack_fumbles_lost, 0)
            + coalesce({{ p }}rushing_fumbles_lost, 0)
            + coalesce({{ p }}receiving_fumbles_lost, 0)
        )
        + 6.0 * coalesce({{ p }}special_teams_tds, 0)
    )
{%- endmacro %}
