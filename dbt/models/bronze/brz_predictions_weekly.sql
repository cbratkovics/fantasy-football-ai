{{ config(materialized='table') }}

-- One row per prediction record in every artifacts/predictions/<season>/week_<ww>.json.
with files as (
    select * from {{ source('repo_files', 'predictions_weekly') }}
)

select
    f.filename as source_file,
    cast(f.season as integer) as season,
    cast(f.week as integer) as week,
    cast(f.model_version as varchar) as model_version,
    cast(f.feature_version as varchar) as feature_version,
    cast(f.generated_at_utc as timestamp) as generated_at_utc,
    cast(f.actuals_attached_at_utc as timestamp) as actuals_attached_at_utc,
    cast(f.data_through.season as integer) as data_through_season,
    cast(f.data_through.week as integer) as data_through_week,
    cast(p.player_id as varchar) as player_id,
    cast(p.name as varchar) as player_display_name,
    cast(p.team as varchar) as team,
    cast(p.position as varchar) as position,
    cast(p.candidate as varchar) as candidate,
    cast(p.prediction as double) as prediction,
    cast(p.floor as double) as prediction_floor,
    cast(p.ceiling as double) as prediction_ceiling,
    cast(p.receptions_estimate as double) as receptions_estimate,
    cast(p.actual as double) as actual
from files as f, unnest(f.predictions) as u (p)
