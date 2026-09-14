-- Grain: one row per model_version. Training provenance from metadata.json plus the manifest's
-- champion / challenger slots. Versions that only appear in prediction files (no metadata)
-- are kept with null provenance so fct_player_week always has a parent.
with versions as (
    select model_version from {{ ref('brz_model_metadata') }}
    union
    select distinct model_version from {{ ref('slv_predictions') }}
),

manifest as (
    select * from {{ ref('brz_manifest') }}
)

select
    cast(v.model_version as varchar) as model_version,
    cast(m.feature_version as varchar) as feature_version,
    cast(m.trained_at_utc as timestamp) as trained_at_utc,
    cast(m.data_through_season as integer) as data_through_season,
    cast(m.data_through_week as integer) as data_through_week,
    cast(array_to_string(m.train_seasons, ',') as varchar) as train_seasons,
    cast(m.val_season as integer) as val_season,
    cast(m.test_season as integer) as test_season,
    cast(m.target as varchar) as target,
    cast(m.input_sha256 as varchar) as input_sha256,
    cast(m.input_rows as integer) as input_rows,
    cast(m.code_commit as varchar) as code_commit,
    cast(m.sklearn_version as varchar) as sklearn_version,
    cast(m.xgboost_version as varchar) as xgboost_version,
    cast(m.data_library as varchar) as data_library,
    cast(m.interval_method as varchar) as interval_method,
    cast(m.champion_qb as varchar) as champion_qb,
    cast(m.champion_rb as varchar) as champion_rb,
    cast(m.champion_wr as varchar) as champion_wr,
    cast(m.champion_te as varchar) as champion_te,
    cast(m.n_features_qb as integer) as n_features_qb,
    cast(m.n_features_rb as integer) as n_features_rb,
    cast(m.n_features_wr as integer) as n_features_wr,
    cast(m.n_features_te as integer) as n_features_te,
    cast(coalesce(mf.champion_model_version = v.model_version, false) as boolean) as is_champion,
    cast(coalesce(mf.challenger_model_version = v.model_version, false) as boolean) as is_challenger
from versions as v
left join {{ ref('brz_model_metadata') }} as m on v.model_version = m.model_version
left join manifest as mf on true
