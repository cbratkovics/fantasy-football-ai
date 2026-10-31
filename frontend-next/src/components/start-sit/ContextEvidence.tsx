'use client';

import { useEffect, useState } from 'react';
import { contentId } from '@/lib/decision-lab/canonical';
import { formatLocalDate } from '@/lib/presentation/display';
import { contextReason } from '@/lib/presentation/decision';
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
  const blocked = ['blocked', 'incompatible'].includes(assessment.state);
  const principal = [...assessment.reasons].sort((a, b) => {
    const priority = (x: string) => ['kickoff_passed', 'known_bye', 'known_unavailable', 'context_invalid', 'forecast_context_reference_mismatch'].findIndex((prefix) => x.startsWith(prefix));
    return (priority(a) < 0 ? 99 : priority(a)) - (priority(b) < 0 ? 99 : priority(b));
  })[0];
  return <section aria-label="Availability and context" className={blocked ? 'product-notice product-blocker' : 'product-notice'} data-testid="context-assessment" data-state={assessment.state}>
    <h2 className="text-base font-semibold">{blocked ? 'Check before changing a lineup' : 'Availability is not verified'}</h2>
    <p className="mt-1">{blocked ? contextReason(principal ?? '') : 'Use this as a dated comparison. Confirm player status and your league’s lineup rules before acting.'}</p>
    {assessment.players.length > 0 && <ul className="mt-2 space-y-2">{assessment.players.map((p) => <li key={p.player_id}>{p.name || p.player_id}: {p.current_team || 'team unresolved'} · {p.game.state === 'bye' ? 'bye' : `vs ${p.game.opponent || 'unknown opponent'}`} · {p.game.kickoff_utc ? formatLocalDate(p.game.kickoff_utc) : 'kickoff unknown'} · {p.availability.designation || 'availability unknown'}</li>)}</ul>}
    <details className="mt-3"><summary className="cursor-pointer font-semibold underline">Availability and source details</summary>
      <p className="mt-2">{problem || 'Context is separate from the forecast comparison.'} User assumptions do not verify provider status, league membership, or lineup legality.</p>
      <ul className="mt-2 list-disc pl-5">{assessment.reasons.map((reason) => <li key={reason}>{reason.replace(/_/g, ' ')}</li>)}</ul>
      {context && <p className="mt-2">Source period: season {context.target.season}, week {context.target.week ?? 'unknown'}. Roster identities are independent of published projections. {context.players.filter((p) => p.forecast.state !== 'present').length} identities have missing model coverage; no forecasts are invented.</p>}
      <p className="mt-2">Assessed at {now}. Assumed maximum ages: context {FRESHNESS_HOURS.context}h, schedule fetch {FRESHNESS_HOURS.schedules}h, roster fetch {FRESHNESS_HOURS.rosters}h, forecast {FRESHNESS_HOURS.forecast}h. These are operational assumptions, not provider guarantees. Kickoff is a conservative deadline, not a verified league lock.</p>
      <p className="mt-2 break-all">Context ID: {context?.context_snapshot_id || 'none'}; forecast ID: {snapshot.snapshot_id}.</p>
      {context && Object.entries(context.sources).map(([name, source]) => <p className="mt-2" key={name}>{name}: {source.state}; content update {source.provider_updated_at_utc || 'unknown'}; first observed {source.first_observed_at_utc || 'unknown'}; fetched {source.fetched_at_utc || 'unknown'}; {source.reasons.join(', ')}.</p>)}
    </details>
  </section>;
}
