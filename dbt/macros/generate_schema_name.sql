{#- Medallion layers are schemas named exactly bronze / silver / gold (no target-schema prefix).
    Models without a custom schema land in the target schema (main). -#}
{% macro generate_schema_name(custom_schema_name, node) -%}
    {%- if custom_schema_name is none -%}
        {{ target.schema }}
    {%- else -%}
        {{ custom_schema_name | trim }}
    {%- endif -%}
{%- endmacro %}
