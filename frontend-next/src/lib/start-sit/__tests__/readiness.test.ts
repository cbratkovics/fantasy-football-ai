import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { contentId } from '../../decision-lab/canonical';
import type { InputsSnapshot } from '../../decision-lab/types';
import { assessReadiness, parseContext, type ContextSnapshot } from '../readiness';

const forecast = {
  snapshot_id: 'weekly-2026-w09-example', season: 2026, week: 9,
  generated_at_utc: '2026-10-30T10:00:00Z',
} as InputsSnapshot;
const now = '2026-11-01T15:00:00Z';
const source = {
  state: 'verified', reasons: [], source_id: 'synthetic-test-source', sha256: 'a'.repeat(64),
  provider_updated_at_utc: null, first_observed_at_utc: '2026-11-01T14:00:00Z',
  fetched_at_utc: '2026-11-01T14:00:00Z', coverage_complete: true,
};
function seal(c: Omit<ContextSnapshot, 'context_snapshot_id'> | ContextSnapshot): ContextSnapshot {
  const { context_snapshot_id: ignored, ...body } = c as ContextSnapshot;
  return { ...body, context_snapshot_id: contentId(body) };
}
function context(): ContextSnapshot {
  return seal({
    schema_version: 'context_snapshot/1.0', mode: 'live',
    target: { season: 2026, season_type: 'REG', week: 9, state: 'target' },
    requested_as_of_utc: '2026-11-01T14:30:00Z', observed_at_utc: '2026-11-01T14:00:00Z',
    forecast: { snapshot_id: forecast.snapshot_id, sha256: contentId(forecast), season: 2026, week: 9 },
    sources: { schedules: { ...source }, rosters: { ...source }, injuries: { ...source, state: 'unavailable' } },
    players: [{ player_id: 'id-a', provider_id: 'id-a', name: 'Same Name', position: 'RB', current_team: 'B', identity_state: 'resolved',
      game: { game_id: 'game-1', opponent: 'C', kickoff_utc: '2026-11-01T18:00:00Z', state: 'scheduled' },
      forecast: { state: 'present', forecast_team: 'A' }, availability: { state: 'unknown', designation: null, basis: 'not_verified' }, reasons: [] }],
  });
}
describe('current context is a separate conservative readiness companion', () => {
  it('verifies the Python-produced synthetic context identity without rehashing or changing it', () => {
    const fixture = JSON.parse(readFileSync(resolve(process.cwd(), '../tests/fixtures/context/normalized.json'), 'utf8'));
    expect(parseContext(fixture).context_snapshot_id).toBe(fixture.context_snapshot_id);
    expect(assessReadiness(forecast, ['rookie'], fixture, now).state).toBe('incompatible');
  });
  it('preserves dated comparison when context is absent, not corrupt forecast evidence', () => {
    expect(assessReadiness(forecast, ['id-a'], null, now).state).toBe('unavailable');
  });
  it('never verifies a fixture, league roster, or absent injury report', () => {
    const c = context();
    const r = assessReadiness(forecast, ['id-a'], c, now);
    expect(r.state).toBe('conditional');
    expect(r.league_verified).toBe(false);
    expect(r.reasons).toContain('availability_unknown:id-a');
    c.mode = 'fixture';
    expect(assessReadiness(forecast, ['id-a'], seal(c), now).reasons).toContain('fixture_not_live_verified');
  });
  it('rejects corrupt content, mixed releases, future and stale context', () => {
    const c = context(); c.players[0].current_team = 'tampered';
    expect(() => parseContext(c)).toThrow('digest');
    expect(assessReadiness(forecast, ['id-a'], c, now).state).toBe('incompatible');
    const mixed = context(); mixed.forecast.snapshot_id = 'other';
    expect(assessReadiness(forecast, ['id-a'], seal(mixed), now).reasons).toContain('forecast_context_reference_mismatch');
    expect(assessReadiness(forecast, ['id-a'], context(), '2026-11-01T12:00:00Z').reasons).toContain('context_from_future');
    expect(assessReadiness(forecast, ['id-a'], context(), '2026-11-02T14:30:01Z').reasons).toContain('context_stale');
  });
  it('rejects resealed malformed presentation fields rather than crashing the context view', () => {
    const c = context();
    expect(() => parseContext(seal({ ...c, target: { ...c.target, season: {} } } as unknown as ContextSnapshot))).toThrow('target');
    c.sources.rosters.fetched_at_utc = 'unknown';
    expect(() => parseContext(seal(c))).toThrow('source time');
  });
  it.each(['bye', 'postponed', 'cancelled', 'unknown'])('blocks %s and never chooses another player', (state) => {
    const c = context(); c.players[0].game.state = state;
    expect(assessReadiness(forecast, ['id-a'], seal(c), now).state).toBe('blocked');
  });
  it('distinguishes questionable, unavailable, unresolved identity and missing forecast', () => {
    const c = context(); c.players[0].availability = { state: 'questionable', designation: 'Questionable', basis: 'report' };
    expect(assessReadiness(forecast, ['id-a'], seal(c), now).reasons).toContain('questionable:id-a');
    c.players[0].availability.state = 'unavailable';
    expect(assessReadiness(forecast, ['id-a'], seal(c), now).reasons).toContain('known_unavailable:id-a');
    c.players[0].identity_state = 'unresolved'; c.players[0].forecast.state = 'missing_history';
    expect(assessReadiness(forecast, ['id-a'], seal(c), now).reasons).toEqual(expect.arrayContaining(['identity_unresolved:id-a', 'forecast_missing:id-a']));
    expect(assessReadiness(forecast, ['id-other'], seal(c), now).reasons).toContain('identity_missing:id-other');
  });
  it('compares aware kickoffs across DST and blocks exactly at deadline or unknown times', () => {
    const c = context(); c.players[0].game.kickoff_utc = '2026-11-01T09:30:00-05:00';
    expect(assessReadiness(forecast, ['id-a'], seal(c), now).reasons).toContain('kickoff_passed:id-a');
    c.players[0].game.kickoff_utc = now;
    expect(assessReadiness(forecast, ['id-a'], seal(c), now).reasons).toContain('kickoff_passed:id-a');
    c.players[0].game.kickoff_utc = '2026-11-01T01:30:00';
    expect(assessReadiness(forecast, ['id-a'], seal(c), now).reasons).toContain('game_time_unknown:id-a');
  });
  it('retains content age when a payload is freshly fetched and blocks no-target periods', () => {
    const c = context(); c.sources.rosters.provider_updated_at_utc = '2026-10-01T00:00:00Z';
    expect(assessReadiness(forecast, ['id-a'], seal(c), now).reasons).toContain('rosters_content_stale_or_future');
    c.target.state = 'no_target'; c.target.week = null;
    expect(assessReadiness(forecast, ['id-a'], seal(c), now).reasons).toContain('no_supported_target');
  });
  it('rejects impossible calendar instants and sources fetched after the declared as-of time', () => {
    const c = context();
    c.players[0].game.kickoff_utc = '2026-02-30T18:00:00Z';
    expect(assessReadiness(forecast, ['id-a'], seal(c), now).reasons).toContain('game_time_unknown:id-a');
    const futureFetch = context(); futureFetch.sources.rosters.fetched_at_utc = '2026-11-01T14:45:00Z';
    expect(assessReadiness(forecast, ['id-a'], seal(futureFetch), now).reasons).toContain('rosters_time_unverified');
    const futureObservation = context(); futureObservation.observed_at_utc = '2026-11-02T14:00:00Z';
    expect(assessReadiness(forecast, ['id-a'], seal(futureObservation), now).reasons).toContain('context_observed_in_future');
  });
  it('rejects unchecked source extensions before they can enter the presentation path', () => {
    const c = context(); c.sources.extra = { nonsense: {} } as never;
    expect(() => parseContext(seal(c))).toThrow('source');
  });
  it('never changes original forecast identity or persisted comparison inputs', () => {
    const before = contentId(forecast);
    assessReadiness(forecast, ['id-a'], context(), now);
    assessReadiness(forecast, ['id-a'], context(), '2026-11-01T19:00:00Z');
    expect(contentId(forecast)).toBe(before);
  });
});
