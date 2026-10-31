'use client';

import { useEffect, useState } from 'react';
import { contentId } from '@/lib/decision-lab/canonical';
import type { InputsSnapshot } from '@/lib/decision-lab/types';
import { assessReadiness, parseContext, FRESHNESS_HOURS, type ContextSnapshot, type Readiness } from '@/lib/start-sit/readiness';

/** Optional artifact companion. Missing/corrupt context never repairs or replaces a forecast. */
export default function ContextEvidence({ snapshot, playerIds, onStateChange }: { snapshot: InputsSnapshot; playerIds: string[]; onStateChange?: (state: Readiness['state']) => void }) {
  const [context, setContext] = useState<ContextSnapshot | null>(null);
  const [problem, setProblem] = useState('Context has not been checked.');
  const [invalidCompanion, setInvalidCompanion] = useState(false);
  const [now, setNow] = useState(() => new Date().toISOString());
  const snapshotDigest = contentId(snapshot);
  useEffect(() => {
    let active = true;
    let requestSequence = 0;
    let controller: AbortController | null = null;
    async function refresh() {
      const request = ++requestSequence;
      setNow(new Date().toISOString());
      controller?.abort();
      controller = new AbortController();
      try {
        const response = await fetch('/context/snapshot.json', { cache: 'no-store', credentials: 'same-origin', signal: controller.signal });
        if (!active || request !== requestSequence) return;
        if (!response.ok) {
          setProblem('Context refresh is unavailable. Any previously checked companion is retained with its original timestamps; no new verification is claimed.');
          return;
        }
        try {
          const next = parseContext(await response.json());
          if (active && request === requestSequence) { setContext(next); setInvalidCompanion(false); setProblem(''); }
        } catch (error) {
          if (active && request === requestSequence) {
            setInvalidCompanion(true);
            setProblem(`Context evidence is invalid: ${error instanceof Error ? error.message : 'contract could not be verified'}. A missing or failed later refresh cannot make it valid.`);
          }
        }
      } catch (error) {
        if (active && request === requestSequence && !(error instanceof Error && error.name === 'AbortError')) {
          setProblem('Context refresh failed. Any previously checked companion is retained with its original timestamps; no new verification is claimed.');
        }
      }
    }
    const revisit = () => { if (document.visibilityState === 'visible') void refresh(); };
    void refresh();
    window.addEventListener('focus', revisit);
    document.addEventListener('visibilitychange', revisit);
    // Reassess action deadlines even while a tab remains open. No high-frequency provider calls.
    const tick = window.setInterval(() => setNow(new Date().toISOString()), 1000);
    return () => {
      active = false;
      controller?.abort();
      window.clearInterval(tick);
      window.removeEventListener('focus', revisit);
      document.removeEventListener('visibilitychange', revisit);
    };
  }, [snapshotDigest]);
  const priorAssessment = assessReadiness(snapshot, playerIds, context, now);
  const assessment: Readiness = invalidCompanion
    ? { ...priorAssessment, state: 'incompatible', reasons: ['context_invalid', ...priorAssessment.reasons] }
    : priorAssessment;
  useEffect(() => { onStateChange?.(assessment.state); }, [assessment.state, onStateChange]);
  return <section aria-label="Current context assessment" className="rounded-xl border border-white/10 p-4 text-sm" data-testid="context-assessment">
    <h2 className="font-semibold">Current actionability: {assessment.state}</h2>
    <p>{problem || 'Context is separate from the dated policy comparison below.'}</p>
    <p>League roster, actual lineup, and platform lock rules are unverified. A user assumption is never provider verification.</p>
    <ul className="mt-2 list-disc pl-5">
      {assessment.reasons.map((reason) => <li key={reason}>{reason.replace(/_/g, ' ')}</li>)}
    </ul>
    {['blocked', 'incompatible'].includes(assessment.state) && <p className="mt-2 font-semibold">Do not treat the dated policy suggestion as an actionable current lineup change. Assumptions cannot override invalid evidence, a known bye, unavailability, or passed deadline. Historical hypothetical exploration remains separate.</p>}
    {assessment.players.map((p) => <p key={p.player_id} className="mt-2">
      {p.name || p.player_id}: current team {p.current_team || 'unresolved'}; {p.game.state === 'bye' ? 'bye' : `opponent ${p.game.opponent || 'unknown'}`}; kickoff {p.game.kickoff_utc || 'unknown'}; status {p.availability.designation || p.availability.state}; forecast coverage {p.forecast.state.replace(/_/g, ' ')}.
      {p.forecast.forecast_team && p.forecast.forecast_team !== p.current_team ? ` Forecast-era team: ${p.forecast.forecast_team}.` : ''}
    </p>)}
    {context && <details className="mt-2"><summary>Roster universe and model coverage for season {context.target.season}, week {context.target.week ?? 'unknown'}</summary>
      <p>Roster identities are independent of published projections. Unscored identities below have no invented forecast and cannot enter a complete comparison.</p>
      <ul>{context.players.filter((p) => p.forecast.state !== 'present').map((p) => <li key={p.player_id}>{p.name || p.player_id} ({p.player_id}) · {p.current_team || 'team unresolved'} · {p.forecast.state.replace(/_/g, ' ')}</li>)}</ul>
    </details>}
    <details className="mt-2"><summary>Context references and freshness assumptions</summary>
      <p>Assessed at {now}. Context maximum age {FRESHNESS_HOURS.context} hours; schedule fetch {FRESHNESS_HOURS.schedules} hours; roster fetch {FRESHNESS_HOURS.rosters} hours; forecast {FRESHNESS_HOURS.forecast} hours. These are conservative assumptions, not provider guarantees. Kickoff is a conservative action deadline.</p>
      <p className="break-all">Context ID: {context?.context_snapshot_id || 'none'}; forecast ID: {snapshot.snapshot_id}.</p>
      {context && Object.entries(context.sources).map(([name, source]) => <p key={name}>{name}: {source.state}; content update {source.provider_updated_at_utc || 'unknown'}; first observed {source.first_observed_at_utc || 'unknown'}; fetched {source.fetched_at_utc || 'unknown'}; {source.reasons.join(', ')}.</p>)}
    </details>
  </section>;
}
