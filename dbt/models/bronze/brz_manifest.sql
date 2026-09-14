{{ config(materialized='table') }}

-- artifacts/manifest.json as one row (the registered evaluations list is kept as a struct list).
select
    filename as source_file,
    cast(manifest_version as varchar) as manifest_version,
    cast(feature_version as varchar) as feature_version,
    cast(champion.model_version as varchar) as champion_model_version,
    champion.candidate as champion_candidate,
    cast(challenger.model_version as varchar) as challenger_model_version,
    challenger.candidate as challenger_candidate,
    cast(tiers_version as varchar) as tiers_version,
    cast(eval_id as varchar) as eval_id,
    evaluations,
    cast(data_through.season as integer) as data_through_season,
    cast(data_through.week as integer) as data_through_week,
    cast(predictions.latest as varchar) as predictions_latest,
    cast(last_run.run_id as varchar) as last_run_id,
    cast(last_run.at_utc as timestamp) as last_run_at_utc,
    cast(last_run.action as varchar) as last_run_action,
    cast(last_run.season as integer) as last_run_season,
    cast(last_run.week as integer) as last_run_week,
    cast(last_run.stats_rows as integer) as last_run_stats_rows,
    cast(updated_at_utc as timestamp) as updated_at_utc
from {{ source('repo_files', 'manifest') }}
