{#- Export every gold model to artifacts/marts/<model>.parquet plus _export_manifest.json
    (model, row count, exported_at_utc, target, dbt invocation id, git commit). Run after
    `dbt build`:  dbt run-operation export_gold --target prod
    The API reads these files in-process with duckdb; it never holds a MotherDuck token
    (ADR-0018). Paths are relative to the repo root, like every file source in this project. -#}
{% macro export_gold(out_dir='artifacts/marts') %}
    {% if execute %}
        {% set gold = [] %}
        {% for node in graph.nodes.values() if node.resource_type == 'model' and 'gold' in node.tags %}
            {% do gold.append(node) %}
        {% endfor %}
        {% set gold = gold | sort(attribute='name') %}
        {% set manifest_rows = [] %}
        {% for node in gold %}
            {% set relation = adapter.get_relation(database=node.database, schema=node.schema, identifier=node.alias) %}
            {% if relation is none %}
                {{ exceptions.raise_compiler_error("export_gold: " ~ node.schema ~ "." ~ node.alias ~ " does not exist; run dbt build first") }}
            {% endif %}
            {% set path = out_dir ~ '/' ~ node.name ~ '.parquet' %}
            {% do run_query("copy (select * from " ~ relation ~ ") to '" ~ path ~ "' (format parquet, compression zstd)") %}
            {% set n = run_query("select count(*) from " ~ relation).columns[0].values()[0] %}
            {% do manifest_rows.append("select '" ~ node.name ~ "' as model, " ~ n ~ " as row_count, '" ~ path ~ "' as path") %}
            {% do log("export_gold: " ~ path ~ " (" ~ n ~ " rows)", info=true) %}
        {% endfor %}
        {% set commit = env_var('GITHUB_SHA', '') %}
        {% do run_query(
            "copy (select model, row_count, path, '" ~ run_started_at.strftime('%Y-%m-%dT%H:%M:%S+00:00') ~ "' as exported_at_utc, '"
            ~ target.name ~ "' as target, '" ~ invocation_id ~ "' as invocation_id, '" ~ commit ~ "' as code_commit from ("
            ~ manifest_rows | join(' union all ') ~ ") order by model) to '" ~ out_dir ~ "/_export_manifest.json' (format json, array true)"
        ) %}
        {% do log("export_gold: wrote " ~ out_dir ~ "/_export_manifest.json", info=true) %}
    {% endif %}
{% endmacro %}
