{{ config(materialized='table') }}

-- One row per evaluation artifact; nested blocks kept as structs (silver flattens them).
select
    filename as source_file,
    cast(artifact_version as varchar) as artifact_version,
    cast(eval_id as varchar) as eval_id,
    cast(kind as varchar) as kind,
    cast(season as integer) as season,
    cast(generated_at_utc as timestamp) as generated_at_utc,
    cast(code_commit as varchar) as code_commit,
    input,
    model,
    split,
    metric_definitions,
    metrics,
    baseline,
    cohorts,
    rolling_origin
from {{ source('repo_files', 'eval_artifacts') }}
