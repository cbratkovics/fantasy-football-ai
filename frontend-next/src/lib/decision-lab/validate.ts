/**
 * Strict structural and semantic validation for every Decision Lab document — the mirror of
 * `ffai/decision_lab/contracts.py`.
 *
 * Python validates with `jsonschema` (Draft 2020-12, `additionalProperties: false` throughout).
 * This module ships a tiny hand-written schema interpreter with the same semantics for the subset
 * the contracts use (types, unions, enum, const, pattern, min/max length, minimum, min/max items,
 * properties/required/additionalProperties, items, oneOf) and transcribes every schema. Semantic
 * checks (grain uniqueness, finite numbers, supported units, cross-file compatibility) follow the
 * `check_*` functions. Every failure throws {@link ContractError} naming the document and field;
 * nothing is coerced or defaulted.
 *
 * Stricter than Python on purpose: every `number`/`integer` must be finite (JSON cannot carry NaN
 * or Infinity, so an in-memory value that does is a bug, not data).
 *
 * Required-but-nullable fields are deliberate: `recommended_player_id` must be present and `null`
 * for HOLD / REVIEW, and `action` must be present with `state: not_recorded` and null ids before
 * anything is recorded.
 */
import { contentId, deepEqual, sortedKeys } from './canonical';
import { utf8ByteLength } from './sha256';
import { POLICY_VERSION, SCHEMA_VERSIONS, SPEC } from './spec';
import type {
  CasesFile,
  DecisionInputs,
  DecisionResult,
  InputsSnapshot,
  LabManifest,
  OutcomeSnapshot,
  Receipt,
} from './types';

/** A document violates its schema or a semantic rule. Always fail closed. */
export class ContractError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ContractError';
  }
}

/** Largest receipt JSON text (UTF-8 bytes) accepted by {@link validateReceiptShape}. */
export const RECEIPT_MAX_BYTES = 512 * 1024;
/** Largest event list a receipt may carry (also enforced by the schema's `maxItems`). */
export const RECEIPT_MAX_EVENTS = 256;

// --- mini schema interpreter ----------------------------------------------------------------

type JsonType = 'string' | 'number' | 'integer' | 'boolean' | 'null' | 'object' | 'array';

interface Schema {
  type?: JsonType | JsonType[];
  enum?: unknown[];
  const?: unknown;
  pattern?: string;
  minLength?: number;
  maxLength?: number;
  minimum?: number;
  minItems?: number;
  maxItems?: number;
  properties?: Record<string, Schema>;
  required?: string[];
  additionalProperties?: boolean | Schema;
  items?: Schema;
  oneOf?: Schema[];
}

function typeOf(value: unknown): JsonType | 'other' {
  if (value === null) return 'null';
  if (Array.isArray(value)) return 'array';
  switch (typeof value) {
    case 'string':
      return 'string';
    case 'boolean':
      return 'boolean';
    case 'number':
      return 'number';
    case 'object':
      return 'object';
    default:
      return 'other';
  }
}

function typeMatches(value: unknown, t: JsonType): boolean {
  const actual = typeOf(value);
  if (t === 'integer') {
    return actual === 'number' && Number.isInteger(value as number);
  }
  if (t === 'number') {
    return actual === 'number' && Number.isFinite(value as number);
  }
  return actual === t;
}

function describe(value: unknown): string {
  if (typeof value === 'string') return JSON.stringify(value);
  if (value === undefined) return 'undefined';
  try {
    const s = JSON.stringify(value);
    return s === undefined ? String(value) : s.length > 60 ? s.slice(0, 57) + '...' : s;
  } catch {
    return String(value);
  }
}

/** First violation of `schema` at `value`, as "<path>: <message>", or null when valid. */
function firstError(schema: Schema, value: unknown, path: string): string | null {
  const where = path || '<root>';
  if (schema.const !== undefined && !deepEqual(value, schema.const)) {
    return `${where}: ${describe(schema.const)} was expected`;
  }
  if (schema.type !== undefined) {
    const types = Array.isArray(schema.type) ? schema.type : [schema.type];
    if (!types.some((t) => typeMatches(value, t))) {
      return `${where}: ${describe(value)} is not of type ${types.map((t) => `'${t}'`).join(', ')}`;
    }
  }
  if (schema.enum !== undefined && !schema.enum.some((e) => deepEqual(e, value))) {
    return `${where}: ${describe(value)} is not one of ${describe(schema.enum)}`;
  }
  if (typeof value === 'string') {
    if (schema.pattern !== undefined && !new RegExp(schema.pattern).test(value)) {
      return `${where}: ${describe(value)} does not match '${schema.pattern}'`;
    }
    if (schema.minLength !== undefined && value.length < schema.minLength) {
      return `${where}: ${describe(value)} is too short`;
    }
    if (schema.maxLength !== undefined && value.length > schema.maxLength) {
      return `${where}: ${describe(value)} is too long`;
    }
  }
  if (typeof value === 'number' && schema.minimum !== undefined && value < schema.minimum) {
    return `${where}: ${value} is less than the minimum of ${schema.minimum}`;
  }
  if (Array.isArray(value)) {
    if (schema.minItems !== undefined && value.length < schema.minItems) {
      return `${where}: ${describe(value)} is too short`;
    }
    if (schema.maxItems !== undefined && value.length > schema.maxItems) {
      return `${where}: array has ${value.length} items; the maximum is ${schema.maxItems}`;
    }
    if (schema.items) {
      for (let i = 0; i < value.length; i += 1) {
        const err = firstError(schema.items, value[i], `${path}${path ? '/' : ''}${i}`);
        if (err) return err;
      }
    }
  }
  if (typeOf(value) === 'object') {
    const obj = value as Record<string, unknown>;
    const keys = sortedKeys(Object.keys(obj).filter((k) => obj[k] !== undefined));
    if (schema.required) {
      for (let i = 0; i < schema.required.length; i += 1) {
        const k = schema.required[i];
        if (!Object.prototype.hasOwnProperty.call(obj, k) || obj[k] === undefined) {
          return `${where}: '${k}' is a required property`;
        }
      }
    }
    if (schema.additionalProperties === false && schema.properties) {
      const extra = keys.filter((k) => !Object.prototype.hasOwnProperty.call(schema.properties, k));
      if (extra.length) {
        return `${where}: additional properties are not allowed (${extra.map((k) => `'${k}'`).join(', ')} ${extra.length === 1 ? 'was' : 'were'} unexpected)`;
      }
    }
    for (let i = 0; i < keys.length; i += 1) {
      const k = keys[i];
      const childPath = `${path}${path ? '/' : ''}${k}`;
      if (schema.properties && Object.prototype.hasOwnProperty.call(schema.properties, k)) {
        const err = firstError(schema.properties[k], obj[k], childPath);
        if (err) return err;
      } else if (schema.additionalProperties && typeof schema.additionalProperties === 'object') {
        const err = firstError(schema.additionalProperties, obj[k], childPath);
        if (err) return err;
      }
    }
  }
  if (schema.oneOf) {
    let matches = 0;
    let lastError: string | null = null;
    for (let i = 0; i < schema.oneOf.length; i += 1) {
      const err = firstError(schema.oneOf[i], value, path);
      if (err === null) {
        matches += 1;
      } else {
        lastError = err;
      }
    }
    if (matches !== 1) {
      return matches === 0
        ? `${where}: ${describe(value)} is not valid under any of the given schemas (${lastError})`
        : `${where}: ${describe(value)} is valid under more than one of the given schemas`;
    }
  }
  return null;
}

// --- schema definitions (transcribed from contracts.py) -------------------------------------

const SHA256: Schema = { type: 'string', pattern: '^[0-9a-f]{64}$' };
const NULLABLE_STR: Schema = { type: ['string', 'null'] };
const NULLABLE_NUM: Schema = { type: ['number', 'null'] };
const NULLABLE_INT: Schema = { type: ['integer', 'null'] };
const PLAYER_ID: Schema = { type: 'string', minLength: 1, maxLength: 64 };
const MODE: Schema = { type: 'string', enum: SPEC.modes };
const SOURCE_FAMILY: Schema = { type: 'string', enum: SPEC.source_families };
const AVAILABILITY: Schema = { type: 'string', enum: SPEC.availability.values };
const AVAILABILITY_BASIS: Schema = { type: 'string', enum: SPEC.availability.bases };
const BASELINE_PROVENANCE: Schema = { type: 'string', enum: SPEC.baseline_provenance };
const STATUS: Schema = { type: 'string', enum: SPEC.statuses };
const UTC_PATTERN = '^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}';
const UTC: Schema = { type: ['string', 'null'], pattern: UTC_PATTERN };
const STR: Schema = { type: 'string' };
const INT: Schema = { type: 'integer' };
const BOOL: Schema = { type: 'boolean' };
const NULLABLE_BOOL: Schema = { type: ['boolean', 'null'] };
const ANY_OBJECT: Schema = { type: 'object' };
const NULLABLE_OBJECT: Schema = { type: ['object', 'null'] };
const STRING_MAP: Schema = { type: 'object', additionalProperties: STR };

function obj(props: Record<string, Schema>, required?: string[]): Schema {
  return {
    type: 'object',
    properties: props,
    required: sortedKeys(required ?? Object.keys(props)),
    additionalProperties: false,
  };
}

function nullable(schema: Schema): Schema {
  return { oneOf: [schema, { type: 'null' }] };
}

const PERIOD = obj({ season: INT, week: { type: 'integer', minimum: 1 } });

const PUBLICATION = obj({
  status: { type: 'string', enum: ['recorded', 'published', 'synthetic', 'blocked'] },
  run_id: NULLABLE_STR,
  action: NULLABLE_STR,
  note: NULLABLE_STR,
});

const SNAPSHOT_REF = obj({
  snapshot_id: { type: 'string', minLength: 1 },
  mode: MODE,
  source_family: SOURCE_FAMILY,
  season: INT,
  week: { type: 'integer', minimum: 1 },
  model_version: STR,
  feature_version: NULLABLE_STR,
  candidate_by_position: STRING_MAP,
  scoring_format: STR,
  inputs_sha256: SHA256,
  data_cutoff: nullable(PERIOD),
  generated_at_utc: UTC,
  publication: PUBLICATION,
  evidence_status: { type: 'string', enum: ['verified', 'unverified', 'corrupt'] },
  evidence_detail: NULLABLE_STR,
  baseline_reconciled: NULLABLE_BOOL,
});

const ALTERNATIVE = obj({
  player_id: PLAYER_ID,
  display_name: NULLABLE_STR,
  team: NULLABLE_STR,
  position: STR,
  model_version: STR,
  candidate: STR,
  season: INT,
  week: INT,
  projection: NULLABLE_NUM,
  floor: NULLABLE_NUM,
  ceiling: NULLABLE_NUM,
  baseline: NULLABLE_NUM,
  baseline_provenance: BASELINE_PROVENANCE,
  availability: AVAILABILITY,
  availability_basis: AVAILABILITY_BASIS,
  source_row_ref: NULLABLE_STR,
});

const PARAMETERS = obj({
  min_projection_gap: { type: 'number' },
  min_floor: NULLABLE_NUM,
  require_independent_baseline: BOOL,
  model_only_exploration: BOOL,
});

const DECISION_INPUTS = obj({
  schema_version: { const: SCHEMA_VERSIONS.decision_inputs },
  policy_version: { const: POLICY_VERSION },
  snapshot: SNAPSHOT_REF,
  slot: STR,
  alternatives: { type: 'array', items: ALTERNATIVE, maxItems: 64 },
  parameters: PARAMETERS,
});

const REASON = obj({
  code: { type: 'string', enum: sortedKeys(Object.keys(SPEC.reasons)) },
  severity: { type: 'string', enum: ['hold', 'review', 'exclusion', 'info'] },
  detail: ANY_OBJECT,
});

const RANKING_ROW = obj({
  player_id: PLAYER_ID,
  display_name: NULLABLE_STR,
  position: NULLABLE_STR,
  projection: NULLABLE_NUM,
  floor: NULLABLE_NUM,
  ceiling: NULLABLE_NUM,
  baseline: NULLABLE_NUM,
  baseline_provenance: BASELINE_PROVENANCE,
  availability: NULLABLE_STR,
  availability_basis: NULLABLE_STR,
  comparable: BOOL,
  exclusion: NULLABLE_STR,
  rank: NULLABLE_INT,
  tied_with_next: BOOL,
});

const DECISION_RESULT = obj({
  schema_version: { const: SCHEMA_VERSIONS.decision_result },
  policy_version: { const: POLICY_VERSION },
  decision_id: SHA256,
  status: STATUS,
  recommended_player_id: NULLABLE_STR,
  reasons: { type: 'array', items: REASON },
  ranking: { type: 'array', items: RANKING_ROW },
  leading: obj({ first: NULLABLE_STR, second: NULLABLE_STR, gap: NULLABLE_NUM }),
  gates: obj({
    min_projection_gap: NULLABLE_NUM,
    gap_ok: NULLABLE_BOOL,
    min_floor: NULLABLE_NUM,
    floor_checked: BOOL,
    leading_floor: NULLABLE_NUM,
    floor_ok: NULLABLE_BOOL,
  }),
  baseline: obj({
    status: { type: 'string', enum: ['available', 'unavailable', 'tie'] },
    preferred_player_id: NULLABLE_STR,
    preferred_value: NULLABLE_NUM,
    agrees_with_model_leader: NULLABLE_BOOL,
    missing_for: { type: 'array', items: STR },
    provenance: { type: 'object', additionalProperties: BASELINE_PROVENANCE },
  }),
  counts: obj({ nominated: INT, excluded: INT, comparable: INT }),
  limitations: {
    type: 'array',
    items: { type: 'string', enum: sortedKeys(Object.keys(SPEC.limitations)) },
  },
  explanation: { type: 'array', items: STR },
});

const SNAPSHOT_ROW = obj({
  player_id: PLAYER_ID,
  display_name: NULLABLE_STR,
  team: NULLABLE_STR,
  position: STR,
  model_version: STR,
  candidate: STR,
  projection: NULLABLE_NUM,
  floor: NULLABLE_NUM,
  ceiling: NULLABLE_NUM,
  baseline: NULLABLE_NUM,
  baseline_provenance: BASELINE_PROVENANCE,
  availability_default: AVAILABILITY,
  display_source: {
    type: ['string', 'null'],
    enum: ['prediction_source', 'dim_player_current', 'synthetic', null],
  },
  source_row_ref: NULLABLE_STR,
});

const SOURCE_FILE = obj(
  { role: STR, path: STR, sha256: NULLABLE_STR, rows: NULLABLE_INT },
  ['role', 'path', 'sha256'],
);

const POPULATION = obj({
  conditioning: {
    type: 'string',
    enum: ['realized_stats_rows', 'scored_eligible_players', 'synthetic'],
  },
  description: STR,
  n_rows: INT,
  exclusions: { type: 'array', items: STR },
});

const BASELINE_META = obj({
  name: STR,
  provenance_basis: {
    type: 'string',
    enum: [
      'mart_column',
      'population_rule_and_reconciliation',
      'population_rule_unreconciled',
      'synthetic',
      'unknown',
    ],
  },
  reconciled: NULLABLE_BOOL,
  reconciled_to: NULLABLE_STR,
  note: NULLABLE_STR,
});

const MART_EXPORT = obj({
  exported_at_utc: NULLABLE_STR,
  target: NULLABLE_STR,
  code_commit: NULLABLE_STR,
  invocation_id: NULLABLE_STR,
});

const INPUTS_SNAPSHOT = obj({
  schema_version: { const: SCHEMA_VERSIONS.inputs_snapshot },
  snapshot_id: { type: 'string', minLength: 1 },
  mode: MODE,
  source_family: SOURCE_FAMILY,
  season: INT,
  week: { type: 'integer', minimum: 1 },
  model_version: STR,
  feature_version: NULLABLE_STR,
  candidate_by_position: STRING_MAP,
  scoring_format: STR,
  data_cutoff: nullable(PERIOD),
  generated_at_utc: UTC,
  publication: PUBLICATION,
  population: POPULATION,
  baseline: BASELINE_META,
  champion_at_source: nullable(
    obj({
      model_version: STR,
      candidate_by_position: STRING_MAP,
      basis: { type: 'string', enum: ['eval_artifact', 'predictions_file', 'synthetic'] },
    }),
  ),
  source: obj({
    mart: NULLABLE_STR,
    files: { type: 'array', items: SOURCE_FILE },
    mart_export: nullable(MART_EXPORT),
  }),
  rows: { type: 'array', items: SNAPSHOT_ROW },
});

const OUTCOME_ROW = obj({
  player_id: PLAYER_ID,
  actual: NULLABLE_NUM,
  actual_source: { type: ['string', 'null'], enum: ['stats', 'artifact', 'synthetic', null] },
});

const OUTCOME_SNAPSHOT = obj({
  schema_version: { const: SCHEMA_VERSIONS.outcome_snapshot },
  snapshot_id: { type: 'string', minLength: 1 },
  inputs_content_sha256: SHA256,
  season: INT,
  week: { type: 'integer', minimum: 1 },
  model_version: STR,
  scoring_format: STR,
  observed_at_utc: UTC,
  source: obj({ mart: NULLABLE_STR, files: { type: 'array', items: SOURCE_FILE } }),
  coverage: obj({ scored: INT, observed: INT }),
  rows: { type: 'array', items: OUTCOME_ROW },
});

const EVENT = obj({
  event_id: SHA256,
  seq: { type: 'integer', minimum: 1 },
  event_type: { type: 'string', enum: SPEC.event_types },
  at_utc: { type: 'string', pattern: UTC_PATTERN },
  payload: ANY_OBJECT,
});

const ACTION = obj({
  state: { type: 'string', enum: ['not_recorded', 'recorded', 'declined'] },
  action_id: nullable(SHA256),
  at_utc: UTC,
  chosen_player_id: NULLABLE_STR,
  kind: { type: ['string', 'null'], enum: [...SPEC.action_kinds, null] },
  note: NULLABLE_STR,
});

const OUTCOME_STATE = obj({
  state: { type: 'string', enum: ['not_attached', 'attached'] },
  outcome_snapshot_id: NULLABLE_STR,
  outcome_id: nullable(SHA256),
  attached_at_utc: UTC,
  metrics: NULLABLE_OBJECT,
});

const RECEIPT = obj({
  schema_version: { const: SCHEMA_VERSIONS.receipt },
  decision_id: SHA256,
  parent_decision_id: nullable(SHA256),
  created_at_utc: { type: 'string', pattern: UTC_PATTERN },
  case_id: NULLABLE_STR,
  prediction_note: NULLABLE_STR,
  inputs: DECISION_INPUTS,
  result: DECISION_RESULT,
  result_sha256: SHA256,
  events: { type: 'array', items: EVENT, maxItems: RECEIPT_MAX_EVENTS },
  action: ACTION,
  outcome: OUTCOME_STATE,
  trust: obj({ note: STR }),
});

const EXPERIMENT = obj(
  {
    label: STR,
    parameters: ANY_OBJECT,
    overrides: ANY_OBJECT,
    expected: obj({ status: STATUS, recommended_player_id: NULLABLE_STR }),
  },
  ['label', 'expected'],
);

const CASE = obj({
  case_id: { type: 'string', pattern: '^[a-z0-9][a-z0-9-]{2,80}$' },
  title: STR,
  mode: MODE,
  snapshot_id: STR,
  slot: STR,
  alternatives: { type: 'array', items: PLAYER_ID, minItems: 1, maxItems: 12 },
  overrides: {
    type: 'object',
    additionalProperties: obj({ availability: AVAILABILITY, availability_basis: AVAILABILITY_BASIS }),
  },
  parameters: PARAMETERS,
  setup: STR,
  experiments: { type: 'array', items: EXPERIMENT },
  expected: obj({
    status: STATUS,
    recommended_player_id: NULLABLE_STR,
    reason_codes: { type: 'array', items: STR },
  }),
  outcomes_available: BOOL,
  outcome_snapshot_id: NULLABLE_STR,
  limitations: { type: 'array', items: STR },
  selection: nullable(
    obj({
      method: { type: 'string', enum: ['curated_synthetic', 'deterministic_discovery'] },
      criteria: STR,
      sealed_pattern: NULLABLE_STR,
    }),
  ),
  source_row_refs: { type: 'array', items: STR },
});

const CASES = obj({
  schema_version: { const: SCHEMA_VERSIONS.cases },
  policy_version: { const: POLICY_VERSION },
  cases: { type: 'array', items: CASE },
  notes: { type: 'array', items: STR },
});

const FILE_REF = obj({ path: STR, file_sha256: SHA256, content_sha256: SHA256, n_rows: INT });

const OUTCOME_FILE_REF = obj({
  path: STR,
  file_sha256: SHA256,
  content_sha256: SHA256,
  n_rows: INT,
  n_observed: INT,
});

const RECONCILIATION = obj({
  status: { type: 'string', enum: ['matched', 'mismatch', 'unavailable'] },
  reference: NULLABLE_STR,
  expected: NULLABLE_OBJECT,
  observed: NULLABLE_OBJECT,
  tolerance: NULLABLE_NUM,
});

const MANIFEST_SNAPSHOT = obj({
  snapshot_id: STR,
  mode: MODE,
  source_family: SOURCE_FAMILY,
  season: INT,
  week: INT,
  model_version: STR,
  candidate_by_position: STRING_MAP,
  inputs: FILE_REF,
  outcomes: nullable(OUTCOME_FILE_REF),
  population: POPULATION,
  baseline: BASELINE_META,
  reconciliation: nullable(RECONCILIATION),
});

const MANIFEST = obj({
  schema_version: { const: SCHEMA_VERSIONS.manifest },
  exporter_version: STR,
  policy_version: { const: POLICY_VERSION },
  schema_versions: STRING_MAP,
  code_revision: obj({ produced_at: NULLABLE_STR, note: STR }),
  lineage: ANY_OBJECT,
  sources: { type: 'array', items: SOURCE_FILE },
  snapshots: { type: 'array', items: MANIFEST_SNAPSHOT },
  cases: FILE_REF,
  latest_weekly_snapshot_id: NULLABLE_STR,
  policy_spec: obj({ path: STR, sha256: SHA256 }),
  digest_coverage: STRING_MAP,
});

/** Schema names accepted by {@link validateShape}. */
export type SchemaName =
  | 'decision_inputs'
  | 'decision_result'
  | 'inputs_snapshot'
  | 'outcome_snapshot'
  | 'receipt'
  | 'cases'
  | 'manifest';

const SCHEMAS: Record<SchemaName, Schema> = {
  decision_inputs: DECISION_INPUTS,
  decision_result: DECISION_RESULT,
  inputs_snapshot: INPUTS_SNAPSHOT,
  outcome_snapshot: OUTCOME_SNAPSHOT,
  receipt: RECEIPT,
  cases: CASES,
  manifest: MANIFEST,
};

/** Structural validation against one of the schemas; throws {@link ContractError}. */
export function validateShape(name: SchemaName, document: unknown): void {
  const err = firstError(SCHEMAS[name], document, '');
  if (err) {
    throw new ContractError(`${name}: ${err}`);
  }
}

// --- semantic checks ---------------------------------------------------------------------------

type Loose = Record<string, unknown>;

function finiteOrNull(value: unknown, where: string): void {
  if (value === null || value === undefined) {
    return;
  }
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new ContractError(`${where}: expected a finite number or null, got ${describe(value)}`);
  }
}

function unique(ids: string[], where: string): void {
  const seen: Record<string, true> = {};
  for (let i = 0; i < ids.length; i += 1) {
    if (seen[ids[i]]) {
      throw new ContractError(`${where}: duplicate player_id ${ids[i]}`);
    }
    seen[ids[i]] = true;
  }
}

const FORBIDDEN_OUTCOME_FIELDS = [
  'actual',
  'abs_error',
  'regret',
  'hit',
  'downside',
  'prediction_rank',
  'best_eligible_points',
];

/** Structural + semantic validation of an `inputs_snapshot/1.0`; returns the typed document. */
export function validateInputsSnapshotShape(value: unknown): InputsSnapshot {
  validateShape('inputs_snapshot', value);
  const doc = value as InputsSnapshot;
  const tag = `inputs_snapshot ${doc.snapshot_id}`;
  if (doc.scoring_format !== SPEC.scoring_format) {
    throw new ContractError(`${tag}: scoring_format must be ${SPEC.scoring_format}`);
  }
  unique(doc.rows.map((r) => r.player_id), tag);
  if (doc.population.n_rows !== doc.rows.length) {
    throw new ContractError(`${tag}: population.n_rows != len(rows)`);
  }
  for (let i = 0; i < doc.rows.length; i += 1) {
    const r = doc.rows[i];
    const where = `${tag} row ${r.player_id}`;
    finiteOrNull(r.projection, `${where}.projection`);
    finiteOrNull(r.floor, `${where}.floor`);
    finiteOrNull(r.ceiling, `${where}.ceiling`);
    finiteOrNull(r.baseline, `${where}.baseline`);
    if (r.model_version !== doc.model_version) {
      throw new ContractError(`${where}: model_version differs from the snapshot`);
    }
    const expectedCand = Object.prototype.hasOwnProperty.call(doc.candidate_by_position, r.position)
      ? doc.candidate_by_position[r.position]
      : undefined;
    if (expectedCand !== undefined && r.candidate !== expectedCand) {
      throw new ContractError(`${where}: candidate ${r.candidate} != ${expectedCand}`);
    }
    if (!Object.prototype.hasOwnProperty.call(SPEC.slots, r.position)) {
      throw new ContractError(`${where}: unsupported position ${r.position}`);
    }
    if (r.floor !== null && r.projection !== null && r.floor > r.projection + 1e-9) {
      throw new ContractError(`${where}: floor above projection`);
    }
    if (r.ceiling !== null && r.projection !== null && r.ceiling < r.projection - 1e-9) {
      throw new ContractError(`${where}: ceiling below projection`);
    }
  }
  for (let i = 0; i < doc.rows.length; i += 1) {
    const row = doc.rows[i] as unknown as Loose;
    const leak = sortedKeys(
      FORBIDDEN_OUTCOME_FIELDS.filter((k) => Object.prototype.hasOwnProperty.call(row, k)),
    );
    if (leak.length) {
      throw new ContractError(`${tag}: outcome field(s) in inputs: ${JSON.stringify(leak)}`);
    }
  }
  return doc;
}

/**
 * Structural + semantic validation of an `outcome_snapshot/1.0`; with `inputs`, also checks the
 * cross-file compatibility (same snapshot, matching content digest, same period and model, no
 * players outside the inputs population).
 */
export function validateOutcomeSnapshotShape(
  value: unknown,
  inputs?: InputsSnapshot | null,
): OutcomeSnapshot {
  validateShape('outcome_snapshot', value);
  const doc = value as OutcomeSnapshot;
  const tag = `outcome_snapshot ${doc.snapshot_id}`;
  if (doc.scoring_format !== SPEC.scoring_format) {
    throw new ContractError(`${tag}: scoring_format must be ${SPEC.scoring_format}`);
  }
  unique(doc.rows.map((r) => r.player_id), tag);
  let observed = 0;
  for (let i = 0; i < doc.rows.length; i += 1) {
    const r = doc.rows[i];
    finiteOrNull(r.actual, `${tag} row ${r.player_id}.actual`);
    if (r.actual !== null) {
      observed += 1;
      if (r.actual_source === null || r.actual_source === undefined) {
        throw new ContractError(`${tag} row ${r.player_id}: observed actual without a source`);
      }
    }
  }
  if (doc.coverage.observed !== observed || doc.coverage.scored !== doc.rows.length) {
    throw new ContractError(`${tag}: coverage counts disagree with rows`);
  }
  if (inputs) {
    if (doc.snapshot_id !== inputs.snapshot_id) {
      throw new ContractError('outcome_snapshot pairs with a different snapshot_id');
    }
    if (doc.inputs_content_sha256 !== contentId(inputs)) {
      throw new ContractError(`${tag}: inputs_content_sha256 does not match the inputs snapshot`);
    }
    if (
      doc.season !== inputs.season ||
      doc.week !== inputs.week ||
      doc.model_version !== inputs.model_version
    ) {
      throw new ContractError(`${tag}: period or model differs from inputs`);
    }
    const inputIds: Record<string, true> = {};
    for (let i = 0; i < inputs.rows.length; i += 1) {
      inputIds[inputs.rows[i].player_id] = true;
    }
    const extra = sortedKeys(
      doc.rows.map((r) => r.player_id).filter((pid, i, arr) => !inputIds[pid] && arr.indexOf(pid) === i),
    );
    if (extra.length) {
      throw new ContractError(
        `${tag}: players not in the inputs snapshot: ${JSON.stringify(extra.slice(0, 5))}`,
      );
    }
  }
  return doc;
}

/** Structural + finite-number validation of a `decision_inputs/1.0` document. */
export function validateDecisionInputsShape(value: unknown): DecisionInputs {
  validateShape('decision_inputs', value);
  const doc = value as DecisionInputs;
  for (let i = 0; i < doc.alternatives.length; i += 1) {
    const a = doc.alternatives[i];
    const where = `decision_inputs alternative ${a.player_id}`;
    finiteOrNull(a.projection, `${where}.projection`);
    finiteOrNull(a.floor, `${where}.floor`);
    finiteOrNull(a.ceiling, `${where}.ceiling`);
    finiteOrNull(a.baseline, `${where}.baseline`);
  }
  finiteOrNull(doc.parameters.min_projection_gap, 'decision_inputs parameters.min_projection_gap');
  finiteOrNull(doc.parameters.min_floor, 'decision_inputs parameters.min_floor');
  return doc;
}

/** Structural + consistency validation of a `decision_result/1.0` document. */
export function validateDecisionResultShape(value: unknown): DecisionResult {
  validateShape('decision_result', value);
  const doc = value as DecisionResult;
  if (doc.status === 'recommend' && !doc.recommended_player_id) {
    throw new ContractError('decision_result: recommend without recommended_player_id');
  }
  if (doc.status !== 'recommend' && doc.recommended_player_id !== null) {
    throw new ContractError(`decision_result: ${doc.status} carries a recommended_player_id`);
  }
  const ids: Record<string, true> = {};
  for (let i = 0; i < doc.ranking.length; i += 1) {
    ids[doc.ranking[i].player_id] = true;
  }
  if (doc.recommended_player_id !== null && !ids[doc.recommended_player_id]) {
    throw new ContractError('decision_result: recommended_player_id not among the alternatives');
  }
  const leadingKeys: Array<'first' | 'second'> = ['first', 'second'];
  for (let i = 0; i < leadingKeys.length; i += 1) {
    const v = doc.leading[leadingKeys[i]];
    if (v !== null && !ids[v]) {
      throw new ContractError(`decision_result: leading.${leadingKeys[i]} not among the alternatives`);
    }
  }
  const bp = doc.baseline.preferred_player_id;
  if (bp !== null && !ids[bp]) {
    throw new ContractError('decision_result: baseline.preferred_player_id not among the alternatives');
  }
  return doc;
}

/**
 * Strict structural validation of a receipt (every level rejects unknown keys, unknown enum
 * values, non-finite numbers, wrong types and missing required-but-nullable fields), plus size
 * bounds: at most {@link RECEIPT_MAX_EVENTS} events and at most {@link RECEIPT_MAX_BYTES} of JSON
 * text. Pass `jsonText` when the receipt came from text so the byte bound is measured on the
 * original; otherwise it is measured on a compact re-serialization.
 *
 * Structural only: use `validateReceipt` from `receipts.ts` to replay the policy and check ids.
 */
export function validateReceiptShape(value: unknown, jsonText?: string): Receipt {
  const bytes = jsonText !== undefined ? utf8ByteLength(jsonText) : utf8ByteLength(safeStringify(value));
  if (bytes > RECEIPT_MAX_BYTES) {
    throw new ContractError(`receipt: ${bytes} bytes exceeds the ${RECEIPT_MAX_BYTES}-byte limit`);
  }
  validateShape('receipt', value);
  const doc = value as Receipt;
  if (doc.events.length > RECEIPT_MAX_EVENTS) {
    throw new ContractError(`receipt: ${doc.events.length} events exceeds the ${RECEIPT_MAX_EVENTS} limit`);
  }
  validateDecisionInputsShape(doc.inputs);
  validateDecisionResultShape(doc.result);
  return doc;
}

function safeStringify(value: unknown): string {
  try {
    const s = JSON.stringify(value);
    return s === undefined ? '' : s;
  } catch {
    return '';
  }
}

/** Structural + semantic validation of a `decision_lab_cases/1.0` file. */
export function validateCasesShape(value: unknown): CasesFile {
  validateShape('cases', value);
  const doc = value as CasesFile;
  const seen: Record<string, true> = {};
  for (let i = 0; i < doc.cases.length; i += 1) {
    const c = doc.cases[i];
    if (seen[c.case_id]) {
      throw new ContractError(`cases: duplicate case_id ${c.case_id}`);
    }
    seen[c.case_id] = true;
    unique(c.alternatives, `case ${c.case_id}`);
    const overridden = Object.keys(c.overrides);
    for (let j = 0; j < overridden.length; j += 1) {
      if (c.alternatives.indexOf(overridden[j]) < 0) {
        throw new ContractError(`case ${c.case_id}: override for a player not nominated: ${overridden[j]}`);
      }
    }
    finiteOrNull(c.parameters.min_projection_gap, `case ${c.case_id} parameters.min_projection_gap`);
    finiteOrNull(c.parameters.min_floor, `case ${c.case_id} parameters.min_floor`);
  }
  return doc;
}

/** Structural + semantic validation of a `decision_lab_manifest/1.0` file. */
export function validateManifestShape(value: unknown): LabManifest {
  validateShape('manifest', value);
  const doc = value as LabManifest;
  const ids = doc.snapshots.map((s) => s.snapshot_id);
  const seen: Record<string, true> = {};
  for (let i = 0; i < ids.length; i += 1) {
    if (seen[ids[i]]) {
      throw new ContractError('manifest: duplicate snapshot_id');
    }
    seen[ids[i]] = true;
  }
  if (doc.latest_weekly_snapshot_id !== null && !seen[doc.latest_weekly_snapshot_id]) {
    throw new ContractError('manifest: latest_weekly_snapshot_id is not a listed snapshot');
  }
  if (!deepEqual(doc.schema_versions, SCHEMA_VERSIONS)) {
    throw new ContractError("manifest: schema_versions differ from this code's policy_spec");
  }
  return doc;
}
