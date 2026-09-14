{#- The raw stat columns kept from nflverse, in the order of ffai.data.nflverse.STAT_COLUMNS.
    One list, used by bronze typing, silver contracts, and the scoring macro. -#}
{% macro stat_columns() -%}
{{ return([
    'completions', 'attempts', 'passing_yards', 'passing_tds', 'passing_interceptions',
    'passing_air_yards', 'passing_2pt_conversions', 'sack_fumbles_lost',
    'carries', 'rushing_yards', 'rushing_tds', 'rushing_fumbles_lost', 'rushing_2pt_conversions',
    'receptions', 'targets', 'receiving_yards', 'receiving_tds', 'receiving_air_yards',
    'receiving_fumbles_lost', 'receiving_2pt_conversions', 'special_teams_tds',
    'fantasy_points', 'fantasy_points_ppr',
]) }}
{%- endmacro %}

{#- Counting stats that can never be negative (ffai.data.contracts.NON_NEGATIVE). -#}
{% macro non_negative_stat_columns() -%}
{{ return([
    'completions', 'attempts', 'passing_tds', 'passing_interceptions', 'passing_2pt_conversions',
    'sack_fumbles_lost', 'carries', 'rushing_tds', 'rushing_fumbles_lost',
    'rushing_2pt_conversions', 'receptions', 'targets', 'receiving_tds',
    'receiving_fumbles_lost', 'receiving_2pt_conversions', 'special_teams_tds',
]) }}
{%- endmacro %}
