import { act, cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import ContextEvidence from '../ContextEvidence';
import { contentId } from '@/lib/decision-lab/canonical';
import type { InputsSnapshot } from '@/lib/decision-lab/types';

const snapshot = { snapshot_id: 'test-weekly', season: 2026, week: 9, generated_at_utc: '2026-10-31T12:00:00Z' } as InputsSnapshot;
function context() {
  const c = JSON.parse(readFileSync(resolve(process.cwd(), '../tests/fixtures/context/normalized.json'), 'utf8'));
  c.mode = 'live';
  // Synthetic lifecycle simulation: observation occurs at the declared test decision time.
  c.observed_at_utc = c.requested_as_of_utc;
  for (const s of Object.values(c.sources) as Array<{state: string}>) s.state = 'verified';
  c.forecast = { snapshot_id: snapshot.snapshot_id, sha256: contentId(snapshot), season: 2026, week: 9 };
  c.players.find((p: {player_id: string}) => p.player_id === 'rookie').forecast.state = 'present';
  const { context_snapshot_id: ignored, ...body } = c;
  return { ...body, context_snapshot_id: contentId(body) };
}
afterEach(() => { cleanup(); vi.useRealTimers(); vi.unstubAllGlobals(); });
describe('optional context lifecycle', () => {
  it('reassesses a passed kickoff without fetching providers and retains known facts on failed revisit', async () => {
    vi.useFakeTimers(); vi.setSystemTime(new Date('2026-10-31T13:00:00Z'));
    const fetcher = vi.fn().mockResolvedValue({ ok: true, json: async () => context() });
    vi.stubGlobal('fetch', fetcher);
    const onStateChange = vi.fn();
    await act(async () => { render(<ContextEvidence snapshot={snapshot} playerIds={['rookie']} onStateChange={onStateChange} />); });
    expect(screen.getByRole('heading')).toHaveTextContent('conditional');
    expect(onStateChange).toHaveBeenLastCalledWith('conditional');
    vi.setSystemTime(new Date('2026-11-01T19:00:00Z'));
    await act(async () => { vi.advanceTimersByTime(1000); });
    expect(screen.getByText('kickoff passed:rookie')).toBeInTheDocument();
    expect(screen.getByRole('heading')).toHaveTextContent('blocked');
    expect(fetcher).toHaveBeenCalledTimes(1);
    fetcher.mockResolvedValue({ ok: false });
    await act(async () => { window.dispatchEvent(new Event('focus')); });
    expect(screen.getByRole('heading')).toHaveTextContent('blocked');
    expect(screen.getByText('kickoff passed:rookie')).toBeInTheDocument();
    expect(fetcher).toHaveBeenCalledTimes(2);
  });
  it('keeps malformed optional evidence incompatible rather than treating it as absent', async () => {
    const fetcher = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ schema_version: 'context_snapshot/1.0' }) });
    vi.stubGlobal('fetch', fetcher);
    const onStateChange = vi.fn();
    await act(async () => { render(<ContextEvidence snapshot={snapshot} playerIds={['rookie']} onStateChange={onStateChange} />); });
    expect(screen.getByRole('heading')).toHaveTextContent('incompatible');
    expect(onStateChange).toHaveBeenLastCalledWith('incompatible');
    fetcher.mockResolvedValue({ ok: false, status: 404 });
    await act(async () => { window.dispatchEvent(new Event('focus')); });
    expect(screen.getByRole('heading')).toHaveTextContent('incompatible');
  });
  it('rechecks a changed forecast digest even when its snapshot ID is unchanged', async () => {
    const fetcher = vi.fn().mockResolvedValue({ ok: false });
    vi.stubGlobal('fetch', fetcher);
    const view = render(<ContextEvidence snapshot={snapshot} playerIds={[]} />);
    await act(async () => {});
    view.rerender(<ContextEvidence snapshot={{ ...snapshot, generated_at_utc: '2026-10-31T12:01:00Z' }} playerIds={[]} />);
    await act(async () => {});
    expect(fetcher).toHaveBeenCalledTimes(2);
  });
});
