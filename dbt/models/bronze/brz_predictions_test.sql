{{ config(materialized='table') }}

-- artifacts/models/<version>/test_predictions.csv, typed. model_version comes from the path.
select
    filename as source_file,
    regexp_extract(filename, 'models/([^/]+)/test_predictions\.csv$', 1) as model_version,
    cast(player_id as varchar) as player_id,
    cast(season as integer) as season,
    cast(week as integer) as week,
    cast(position as varchar) as position,
    cast(player_display_name as varchar) as player_display_name,
    cast(team as varchar) as team,
    cast(candidate as varchar) as candidate,
    cast(prediction as double) as prediction,
    cast(prediction_floor as double) as prediction_floor,
    cast(prediction_ceiling as double) as prediction_ceiling,
    cast(actual as double) as actual
from {{ source('repo_files', 'predictions_test') }}
