/** Context is a companion to v1 policy, never an input to its identity or ranking. */
import { contentId } from '../decision-lab/canonical';
import type { InputsSnapshot } from '../decision-lab/types';

export interface ContextSource {
  state: string;
  reasons: string[];
  source_id: string | null;
  sha256: string | null;
  provider_updated_at_utc: string | null;
  first_observed_at_utc: string | null;
  fetched_at_utc: string | null;
  coverage_complete: boolean;
}
export interface ContextPlayer {
  player_id: string;
  provider_id: string | null;
  name: string | null;
  position: string | null;
  current_team: string | null;
  identity_state: string;
  game: { game_id: string | null; opponent: string | null; kickoff_utc: string | null; state: string };
  forecast: { state: string; forecast_team: string | null };
  availability: { state: string; designation: string | null; basis: string };
  reasons: string[];
}
export interface ContextSnapshot {
  schema_version: 'context_snapshot/1.0';
  context_snapshot_id: string;
  mode: 'fixture' | 'live';
  target: { season: number; season_type: string; week: number | null; state: string };
  requested_as_of_utc: string;
  observed_at_utc: string;
  forecast: { snapshot_id: string | null; sha256: string | null; season: number | null; week: number | null };
  sources: Record<string, ContextSource>;
  players: ContextPlayer[];
}
export interface Readiness {
  state: 'unavailable' | 'incompatible' | 'blocked' | 'conditional';
  evaluated_at_utc: string;
  reasons: string[];
  players: ContextPlayer[];
  /** Always false: league rules, ownership, and actual lineup remain unverified. */
  league_verified: false;
}

// Explicit operational assumptions, not measured provider SLAs.
export const FRESHNESS_HOURS = { schedules: 24, rosters: 6, injuries: 6, context: 6, forecast: 168 };

function instant(value: unknown): number | null {
  if (typeof value !== 'string') return null;
  const parts = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.\d+)?(Z|[+-]\d{2}:\d{2})$/.exec(value);
  if (!parts) return null;
  const [, year, month, day, hour, minute, second, zone] = parts;
  const y = Number(year), m = Number(month), d = Number(day);
  const daysInMonth = new Date(Date.UTC(y, m, 0)).getUTCDate();
  if (m < 1 || m > 12 || d < 1 || d > daysInMonth || Number(hour) > 23 || Number(minute) > 59 || Number(second) > 59 ||
      (zone !== 'Z' && (Number(zone.slice(1, 3)) > 23 || Number(zone.slice(4, 6)) > 59))) return null;
  const n = Date.parse(value);
  return Number.isFinite(n) ? n : null;
}
function record(x: unknown): x is Record<string, unknown> {
  return x !== null && typeof x === 'object' && !Array.isArray(x);
}

/** Validate shape and content identity before rendering any optional context. */
export function parseContext(value: unknown): ContextSnapshot {
  if (!record(value) || value.schema_version !== 'context_snapshot/1.0' ||
      !record(value.target) || !record(value.forecast) || !record(value.sources) ||
      !Array.isArray(value.players) || !['live', 'fixture'].includes(String(value.mode)) ||
      instant(value.requested_as_of_utc) === null || instant(value.observed_at_utc) === null) {
    throw new Error('Invalid context contract');
  }
  const { context_snapshot_id, ...body } = value;
  if (typeof context_snapshot_id !== 'string' || contentId(body) !== context_snapshot_id) {
    throw new Error('Context digest mismatch');
  }
  const target = value.target;
  if (!Number.isInteger(target.season) || target.season_type !== 'REG' ||
      !['target', 'no_target', 'target_unknown'].includes(String(target.state)) ||
      !(target.week === null || (Number.isInteger(target.week) && Number(target.week) >= 1 && Number(target.week) <= 18)) ||
      ((target.state === 'target') !== (target.week !== null))) {
    throw new Error('Invalid context target');
  }
  for (const key of ['snapshot_id', 'sha256']) {
    if (!(value.forecast[key] === null || typeof value.forecast[key] === 'string')) throw new Error('Invalid forecast reference');
  }
  for (const key of ['season', 'week']) {
    if (!(value.forecast[key] === null || Number.isInteger(value.forecast[key]))) throw new Error('Invalid forecast period');
  }
  const seen = new Set<string>();
  for (const p of value.players) {
    if (!record(p) || typeof p.player_id !== 'string' || !p.player_id || seen.has(p.player_id) ||
        !record(p.game) || !record(p.availability) || !record(p.forecast) || !Array.isArray(p.reasons) ||
        !p.reasons.every((r) => typeof r === 'string') ||
        !['resolved', 'unresolved'].includes(String(p.identity_state)) ||
        !['unknown', 'questionable', 'unavailable'].includes(String(p.availability.state)) ||
        !['present', 'missing_history', 'unresolved_reference'].includes(String(p.forecast.state)) ||
        !['name', 'provider_id', 'position', 'current_team'].every((k) => p[k] === null || typeof p[k] === 'string') ||
        !['game_id', 'opponent', 'kickoff_utc'].every((k) => (p.game as Record<string, unknown>)[k] === null || typeof (p.game as Record<string, unknown>)[k] === 'string') ||
        !['scheduled', 'bye', 'unknown', 'postponed', 'cancelled', 'ambiguous'].includes(String(p.game.state)) ||
        !(p.availability.designation === null || typeof p.availability.designation === 'string') ||
        typeof p.availability.basis !== 'string') {
      throw new Error('Invalid or duplicate context identity');
    }
    if (p.identity_state === 'resolved' && (!p.current_team || !['QB', 'RB', 'WR', 'TE'].includes(String(p.position)) || p.provider_id !== p.player_id)) {
      throw new Error('Invalid resolved roster identity');
    }
    seen.add(p.player_id);
  }
  if (Object.keys(value.sources).some((name) => !['schedules', 'rosters', 'injuries'].includes(name))) throw new Error('Unexpected context source');
  for (const name of ['schedules', 'rosters', 'injuries']) {
    const s = value.sources[name];
    if (!record(s) || !Array.isArray(s.reasons) || !s.reasons.every((r) => typeof r === 'string') ||
        typeof s.coverage_complete !== 'boolean' ||
        typeof s.source_id !== 'string' || typeof s.sha256 !== 'string' || !/^[0-9a-f]{64}$/.test(s.sha256) ||
        !['fixture', 'verified', 'partial', 'unavailable', 'unsupported', 'network-unverified'].includes(String(s.state))) {
      throw new Error('Invalid context source');
    }
    for (const key of ['provider_updated_at_utc', 'first_observed_at_utc', 'fetched_at_utc']) {
      if (s[key] !== null && instant(s[key]) === null) throw new Error('Invalid source time');
    }
  }
  return value as unknown as ContextSnapshot;
}

/** Explicit time; no clock, I/O, storage, policy edits, or alternate winner selection. */
export function assessReadiness(
  snapshot: InputsSnapshot, playerIds: string[], context: ContextSnapshot | null, evaluatedAt: string,
): Readiness {
  const result: Readiness = { state: 'unavailable', evaluated_at_utc: evaluatedAt, reasons: [], players: [], league_verified: false };
  const now = instant(evaluatedAt);
  if (now === null) return { ...result, state: 'blocked', reasons: ['evaluation_time_invalid'] };
  if (!context) return { ...result, reasons: ['context_unavailable', 'dated_conditional_comparison_only'] };
  try { parseContext(context); } catch { return { ...result, state: 'incompatible', reasons: ['context_invalid'] }; }
  if (context.target.state !== 'target' || context.target.season_type !== 'REG') {
    return { ...result, state: 'blocked', reasons: ['no_supported_target'] };
  }
  if (context.target.season !== snapshot.season || context.target.week !== snapshot.week ||
      context.forecast.season !== snapshot.season || context.forecast.week !== snapshot.week ||
      context.forecast.snapshot_id !== snapshot.snapshot_id || context.forecast.sha256 !== contentId(snapshot)) {
    return { ...result, state: 'incompatible', reasons: ['forecast_context_reference_mismatch'] };
  }
  result.state = 'conditional';
  const block = (code: string) => { result.state = 'blocked'; result.reasons.push(code); };
  const asOf = instant(context.requested_as_of_utc);
  if (asOf === null || asOf > now) block('context_from_future');
  else if (now - asOf > FRESHNESS_HOURS.context * 3600000) block('context_stale');
  if ((instant(context.observed_at_utc) ?? Infinity) > now) block('context_observed_in_future');
  if (context.mode !== 'live') block('fixture_not_live_verified');
  const generated = instant(snapshot.generated_at_utc);
  if (generated === null || generated > now) block('forecast_time_unverified');
  else if (now - generated > FRESHNESS_HOURS.forecast * 3600000) block('forecast_stale');
  for (const name of ['schedules', 'rosters'] as const) {
    const src = context.sources[name];
    if (src.state !== 'verified') block(`${name}_not_verified`);
    const first = instant(src.first_observed_at_utc);
    const fetched = instant(src.fetched_at_utc);
    if (first === null || fetched === null || first > (asOf ?? now) || fetched > (asOf ?? now) || first > fetched) block(`${name}_time_unverified`);
    else if (now - fetched > FRESHNESS_HOURS[name] * 3600000) block(`${name}_fetch_stale`);
    const updated = instant(src.provider_updated_at_utc);
    if (updated !== null && (updated > (asOf ?? now) || now - updated > FRESHNESS_HOURS[name] * 3600000)) block(`${name}_content_stale_or_future`);
    if (updated === null) result.reasons.push(`${name}_content_age_unknown`);
  }
  for (const id of playerIds) {
    const p = context.players.find((row) => row.player_id === id);
    if (!p) { block(`identity_missing:${id}`); continue; }
    result.players.push(p);
    if (p.identity_state !== 'resolved') block(`identity_unresolved:${id}`);
    if (p.forecast.state !== 'present') block(`forecast_missing:${id}`);
    if (p.game.state === 'bye') block(`known_bye:${id}`);
    else {
      const kickoff = instant(p.game.kickoff_utc);
      if (kickoff === null || !p.game.game_id || !p.game.opponent) block(`game_time_unknown:${id}`);
      else if (kickoff <= now) block(`kickoff_passed:${id}`);
      if (p.game.state !== 'scheduled') block(`game_not_confirmed:${id}`);
    }
    if (p.availability.state === 'unavailable') block(`known_unavailable:${id}`);
    else result.reasons.push(`${p.availability.state === 'questionable' ? 'questionable' : 'availability_unknown'}:${id}`);
  }
  result.reasons.push('league_and_lineup_unverified', 'kickoff_lock_is_conservative_assumption');
  return result;
}
