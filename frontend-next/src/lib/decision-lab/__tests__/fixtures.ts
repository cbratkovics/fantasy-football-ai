/**
 * A tiny synthetic bundle (manifest + inputs + outcomes + cases) with correct digests, for the
 * bundle and storage tests. Everything is deterministic so tamper tests can target one byte.
 */
import { contentId } from '../canonical';
import { sha256 } from '../sha256';
import { POLICY_VERSION, SCHEMA_VERSIONS } from '../spec';
import type {
  CasesFile,
  FileRef,
  InputsSnapshot,
  LabManifest,
  OutcomeFileRef,
  OutcomeSnapshot,
} from '../types';

export const SYN_MODEL = 'synthetic-model-v1';
export const SNAPSHOT_ID = 'syn-bundle';

export function syntheticInputsSnapshot(): InputsSnapshot {
  const row = (
    pid: string,
    projection: number,
    floor: number,
    baseline: number,
  ): InputsSnapshot['rows'][number] => ({
    player_id: pid,
    display_name: `Synthetic ${pid.slice(-1)}`,
    team: 'SYN',
    position: 'RB',
    model_version: SYN_MODEL,
    candidate: 'rf',
    projection,
    floor,
    ceiling: projection + 8,
    baseline,
    baseline_provenance: 'player_history',
    availability_default: 'assumed_available',
    display_source: 'synthetic',
    source_row_ref: null,
  });
  return {
    schema_version: 'inputs_snapshot/1.0',
    snapshot_id: SNAPSHOT_ID,
    mode: 'synthetic',
    source_family: 'synthetic',
    season: 2099,
    week: 1,
    model_version: SYN_MODEL,
    feature_version: null,
    candidate_by_position: { QB: 'rf', RB: 'rf', WR: 'rf', TE: 'rf' },
    scoring_format: 'ppr',
    data_cutoff: { season: 2098, week: 18 },
    generated_at_utc: null,
    publication: { status: 'synthetic', run_id: null, action: null, note: null },
    population: { conditioning: 'synthetic', description: 'three synthetic backs', n_rows: 3, exclusions: [] },
    baseline: { name: 'trailing_mean', provenance_basis: 'synthetic', reconciled: null, reconciled_to: null, note: null },
    champion_at_source: null,
    source: { mart: null, files: [], mart_export: null },
    rows: [row('SYN-A', 14.0, 6.0, 12.0), row('SYN-B', 13.6, 7.0, 13.0), row('SYN-C', 9.5, 2.0, 8.0)],
  };
}

export function syntheticOutcomeSnapshot(inputs: InputsSnapshot): OutcomeSnapshot {
  return {
    schema_version: 'outcome_snapshot/1.0',
    snapshot_id: inputs.snapshot_id,
    inputs_content_sha256: contentId(inputs),
    season: inputs.season,
    week: inputs.week,
    model_version: inputs.model_version,
    scoring_format: 'ppr',
    observed_at_utc: null,
    source: { mart: null, files: [] },
    coverage: { scored: 3, observed: 2 },
    rows: [
      { player_id: 'SYN-A', actual: 8.0, actual_source: 'synthetic' },
      { player_id: 'SYN-B', actual: 16.0, actual_source: 'synthetic' },
      { player_id: 'SYN-C', actual: null, actual_source: null },
    ],
  };
}

export function syntheticCases(): CasesFile {
  return {
    schema_version: 'decision_lab_cases/1.0',
    policy_version: POLICY_VERSION,
    cases: [
      {
        case_id: 'syn-bundle-ambiguity',
        title: 'Synthetic ambiguity',
        mode: 'synthetic',
        snapshot_id: SNAPSHOT_ID,
        slot: 'RB',
        alternatives: ['SYN-A', 'SYN-B'],
        overrides: {},
        parameters: {
          min_projection_gap: 0.5,
          min_floor: null,
          require_independent_baseline: true,
          model_only_exploration: false,
        },
        setup: 'Two backs 0.4 apart with a 0.5 gap threshold.',
        experiments: [],
        expected: { status: 'review', recommended_player_id: null, reason_codes: ['gap_below_minimum'] },
        outcomes_available: true,
        outcome_snapshot_id: SNAPSHOT_ID,
        limitations: ['synthetic_evidence'],
        selection: null,
        source_row_refs: [],
      },
    ],
    notes: [],
  };
}

export function fileText(doc: unknown): string {
  return JSON.stringify(doc, null, 1) + '\n';
}

export function fileRef(path: string, text: string, doc: { rows?: unknown[]; cases?: unknown[] }): FileRef {
  const rows = doc.rows ?? doc.cases ?? [];
  return { path, file_sha256: sha256(text), content_sha256: contentId(doc), n_rows: rows.length };
}

export interface SyntheticBundle {
  manifest: LabManifest;
  inputs: InputsSnapshot;
  outcomes: OutcomeSnapshot;
  cases: CasesFile;
  /** bundle-relative path → file text */
  files: Record<string, string>;
}

/** Build a consistent bundle; `mutate` lets a test alter documents before digests are computed. */
export function syntheticBundle(mutate?: (b: { inputs: InputsSnapshot; outcomes: OutcomeSnapshot; cases: CasesFile }) => void): SyntheticBundle {
  const inputs = syntheticInputsSnapshot();
  const outcomes = syntheticOutcomeSnapshot(inputs);
  const cases = syntheticCases();
  if (mutate) {
    mutate({ inputs, outcomes, cases });
  }
  const inputsText = fileText(inputs);
  const outcomesText = fileText(outcomes);
  const casesText = fileText(cases);
  const inputsRef = fileRef(`inputs/${inputs.snapshot_id}.json`, inputsText, inputs);
  const outcomeRef: OutcomeFileRef = {
    ...fileRef(`outcomes/${inputs.snapshot_id}.json`, outcomesText, outcomes),
    n_observed: outcomes.coverage.observed,
  };
  const manifest: LabManifest = {
    schema_version: 'decision_lab_manifest/1.0',
    exporter_version: 'test',
    policy_version: POLICY_VERSION,
    schema_versions: { ...SCHEMA_VERSIONS },
    code_revision: { produced_at: null, note: 'synthetic test bundle' },
    lineage: {},
    sources: [],
    snapshots: [
      {
        snapshot_id: inputs.snapshot_id,
        mode: inputs.mode,
        source_family: inputs.source_family,
        season: inputs.season,
        week: inputs.week,
        model_version: inputs.model_version,
        candidate_by_position: { ...inputs.candidate_by_position },
        inputs: inputsRef,
        outcomes: outcomeRef,
        population: { ...inputs.population },
        baseline: { ...inputs.baseline },
        reconciliation: null,
      },
    ],
    cases: fileRef('cases.json', casesText, cases),
    latest_weekly_snapshot_id: null,
    policy_spec: { path: 'policy_spec.json', sha256: 'f'.repeat(64) },
    digest_coverage: { file_sha256: 'bytes of the file', content_sha256: 'canonical JSON of the document' },
  };
  const files: Record<string, string> = {
    'manifest.json': fileText(manifest),
    [inputsRef.path]: inputsText,
    [outcomeRef.path]: outcomesText,
    'cases.json': casesText,
  };
  return { manifest, inputs, outcomes, cases, files };
}
