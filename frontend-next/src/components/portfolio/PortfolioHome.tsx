'use client'

import { useMemo, useState } from 'react'
import Link from 'next/link'
import { useQuery } from '@tanstack/react-query'
import {
  ArrowRightIcon,
  BeakerIcon,
  BoltIcon,
  ChartBarSquareIcon,
  CheckIcon,
  CircleStackIcon,
  CommandLineIcon,
  ShieldCheckIcon,
} from '@heroicons/react/24/outline'
import { getHealth, getLatestWeek, getPerformance, getPredictions } from '@/lib/api/players'
import { fmtNum, fmtPct } from '@/lib/format'
import { describeError } from '@/lib/api/client'
import { Navigation } from '@/components/layout/Navigation'
import { Footer } from '@/components/layout/Footer'
import { ROUTES, SITE } from '@/lib/constants'

const architecture = [
  ['01', 'Ingest', 'nflverse weekly player data, the only source'],
  ['02', 'Features', 'As-of signals from each player’s own prior weeks'],
  ['03', 'Model', 'RandomForest champion, XGBoost challenger, per position'],
  ['04', 'Tiers', 'Gaussian-mixture preseason tiers from prior-season aggregates'],
  ['05', 'Publish', 'Weekly GitHub Actions job: publish, hold, or promote'],
]

const skills = [
  ['Decision science', 'Floors, ceilings, and an explicit policy over them'],
  ['Modeling', 'Forward-time splits, rolling-origin folds, causal baselines'],
  ['Analytics engineering', 'Versioned artifacts, hashed inputs, metric definitions'],
  ['Production', 'FastAPI, containers, committed manifests'],
]

const CONSOLE_ROWS = 6

function FoldBars({ folds }: { folds: Array<{ mae: number }> }) {
  const max = Math.max(1, ...folds.map((f) => f.mae))
  return (
    <div className="mini-bars" aria-label="Rolling-origin fold MAE (lower is better)">
      {folds.map((f, index) => (
        <span key={index} style={{ height: `${Math.max(8, (f.mae / max) * 100)}%` }} title={`fold MAE ${fmtNum(f.mae, 2)}`} />
      ))}
    </div>
  )
}

export function PortfolioHome() {
  const health = useQuery({ queryKey: ['health'], queryFn: getHealth })
  const perf = useQuery({ queryKey: ['performance'], queryFn: getPerformance })
  const latest = useQuery({ queryKey: ['latest-week'], queryFn: getLatestWeek })
  const preds = useQuery({
    queryKey: ['predictions', latest.data?.season, latest.data?.week, 'ppr', 'ALL'],
    queryFn: () => getPredictions(latest.data!.season, latest.data!.week, 'ppr'),
    enabled: !!latest.data,
  })

  const rows = useMemo(() => (preds.data?.predictions ?? []).slice(0, CONSOLE_ROWS), [preds.data])
  const maxFloor = useMemo(() => Math.ceil(Math.max(0, ...rows.map((r) => r.floor))), [rows])
  const [threshold, setThreshold] = useState<number | null>(null)
  const activeThreshold = threshold ?? Math.floor(maxFloor / 2)
  const approved = rows.filter((r) => r.floor >= activeThreshold)
  const consoleError = health.error ?? perf.error ?? latest.error ?? preds.error

  return (
    <div className="portfolio-shell">
      <Navigation />
      <main>
        <section className="portfolio-hero">
          <div className="hero-copy">
            <div className="eyebrow">
              <span /> Applied data science · end-to-end case study
            </div>
            <h1>
              Better predictions are only useful when they produce <em>better decisions.</em>
            </h1>
            <p className="hero-lede">
              I built a weekly fantasy football projection system that turns a player&apos;s own history into a point estimate with a floor and a ceiling, then
              evaluates it against an honest baseline before anything is published.
            </p>
            <div className="hero-actions">
              <Link href={ROUTES.predictions} className="portfolio-button primary">
                Open the predictions <ArrowRightIcon />
              </Link>
              <Link href={ROUTES.performance} className="portfolio-button secondary">
                Review evaluation
              </Link>
            </div>
            <div className="hero-proof">
              <div>
                <b>Model</b>
                <span>{health.data ? health.data.model_version : health.isError ? 'API unavailable' : 'loading…'}</span>
              </div>
              <div>
                <b>Data through</b>
                <span>{health.data ? `season ${health.data.data_through.season} · week ${health.data.data_through.week}` : '—'}</span>
              </div>
              <div>
                <b>Standard</b>
                <span>No metric without provenance</span>
              </div>
            </div>
          </div>

          <div className="decision-console" aria-label="Live floor policy over the latest published predictions">
            <div className="console-top">
              <div>
                <span className="live-dot" /> FLOOR POLICY
              </div>
              <span>{preds.data ? `SEASON ${preds.data.season} · WEEK ${preds.data.week} · PPR` : 'FROM API'}</span>
            </div>
            <div className="console-metrics">
              <div>
                <small>HOLDOUT MAE · CHAMPION</small>
                <strong>{perf.data ? fmtNum(perf.data.metrics.mae, 2) : '—'}</strong>
                <span>{perf.data ? `baseline ${fmtNum(perf.data.baseline.mae, 2)} · n ${perf.data.metrics.n.toLocaleString()}` : 'reading evaluation artifact'}</span>
              </div>
              <div>
                <small>WITHIN ±3 POINTS</small>
                <strong>{perf.data ? fmtPct(perf.data.metrics.within_3_rate, 0) : '—'}</strong>
                <span>{perf.data ? `baseline ${fmtPct(perf.data.baseline.within_3_rate, 0)}` : ''}</span>
              </div>
            </div>
            {perf.data && <FoldBars folds={perf.data.rolling_origin.folds} />}
            <div className="threshold-control">
              <div>
                <label htmlFor="floor-threshold">Minimum floor to start</label>
                <output>{activeThreshold} pts</output>
              </div>
              <input id="floor-threshold" type="range" min="0" max={maxFloor} value={activeThreshold} disabled={rows.length === 0} onChange={(event) => setThreshold(Number(event.target.value))} />
              <div className="range-labels">
                <span>More candidates</span>
                <span>Safer floors</span>
              </div>
            </div>
            <div className="decision-list">
              {rows.map((row) => {
                const ok = row.floor >= activeThreshold
                return (
                  <div key={row.player_id} className={ok ? 'selected' : ''}>
                    <span className="decision-state">{ok ? <CheckIcon /> : '—'}</span>
                    <b>{row.name ?? row.player_id}</b>
                    <small>{row.position}</small>
                    <span>{fmtNum(row.prediction, 1)} pts</span>
                    <em>
                      {fmtNum(row.floor, 1)}–{fmtNum(row.ceiling, 1)}
                    </em>
                  </div>
                )
              })}
              {rows.length === 0 && !consoleError && <div><span className="decision-state">…</span><b>Loading the latest predictions</b><small /><span /><em /></div>}
              {consoleError && <div className="selected"><span className="decision-state">!</span><b>API unavailable</b><small /><span /><em>{describeError(consoleError)}</em></div>}
            </div>
            <p className="console-note">
              {rows.length > 0 ? `${approved.length} of ${rows.length} top projections clear the floor. ` : ''}
              Live values from the API; MAE and fold bars come from evaluation artifact {perf.data?.eval_id ?? '…'}.
            </p>
          </div>
        </section>

        <section className="portfolio-strip" aria-label="Core technologies">
          <span>PYTHON</span>
          <i /> <span>SCIKIT-LEARN</span>
          <i /> <span>XGBOOST</span>
          <i /> <span>FASTAPI</span>
          <i /> <span>NEXT.JS</span>
          <i /> <span>GITHUB ACTIONS</span>
        </section>

        <section className="case-section" id="case-study">
          <div className="section-heading">
            <div>
              <span className="section-number">01</span>
              <p>THE CASE STUDY</p>
            </div>
            <h2>From a ranking model to a measurable decision strategy.</h2>
            <p>
              Fantasy lineup selection is a safe analogue for real-world risk work: incomplete information, asymmetric errors, shifting populations, and a hard
              decision deadline every week.
            </p>
          </div>
          <div className="problem-grid">
            <article className="statement-card dark-card">
              <span>THE BUSINESS QUESTION</span>
              <h3>Who should we act on, and what evidence would change that decision?</h3>
              <p>
                A point estimate does not answer this. The useful product combines an expected value, an interval around it, and an auditable trail from the
                published number back to the data and code that produced it.
              </p>
              <div className="formula">decision = f(prediction, floor, ceiling) · every input versioned</div>
            </article>
            <article className="statement-card">
              <span>SUCCESS CRITERIA</span>
              <ul>
                <li>
                  <b>Beat the baseline</b>
                  <small>Lower error than a causal trailing mean of the player&apos;s own points</small>
                </li>
                <li>
                  <b>Leak nothing</b>
                  <small>Features use only weeks that were already scored</small>
                </li>
                <li>
                  <b>Publish on evidence</b>
                  <small>A weekly job publishes, holds, or promotes based on the evaluation</small>
                </li>
                <li>
                  <b>Trace every figure</b>
                  <small>Model version, feature version, and eval id on every response</small>
                </li>
              </ul>
            </article>
          </div>
        </section>

        <section className="system-section" id="system">
          <div className="section-heading compact">
            <div>
              <span className="section-number">02</span>
              <p>SYSTEM DESIGN</p>
            </div>
            <h2>An observable path from raw data to a published projection.</h2>
          </div>
          <div className="architecture-row">
            {architecture.map(([number, title, text]) => (
              <article key={number}>
                <span>{number}</span>
                <div className="architecture-icon">
                  <CircleStackIcon />
                </div>
                <h3>{title}</h3>
                <p>{text}</p>
              </article>
            ))}
          </div>
          <div className="principles-grid">
            <article>
              <BeakerIcon />
              <div>
                <b>Evaluation before optimization</b>
                <p>A forward-time holdout, rolling-origin folds, and a causal baseline are fixed before any model is compared.</p>
              </div>
            </article>
            <article>
              <ChartBarSquareIcon />
              <div>
                <b>Metrics with definitions attached</b>
                <p>MAE, median AE, RMSE, and within-band rates are stored next to their formulas in the evaluation artifact.</p>
              </div>
            </article>
            <article>
              <ShieldCheckIcon />
              <div>
                <b>Guardrails over magic</b>
                <p>Hashed inputs, versioned features, a manifest that names the champion, and a policy that can refuse to publish.</p>
              </div>
            </article>
            <article>
              <BoltIcon />
              <div>
                <b>Built for iteration</b>
                <p>A challenger is trained beside the champion every run and promoted only when the policy says so.</p>
              </div>
            </article>
          </div>
        </section>

        <section className="evidence-section">
          <div className="section-heading compact light-heading">
            <div>
              <span className="section-number">03</span>
              <p>WHAT THIS DEMONSTRATES</p>
            </div>
            <h2>Technical depth, translated into practical value.</h2>
          </div>
          <div className="skills-list">
            {skills.map(([title, text], index) => (
              <div key={title}>
                <span>0{index + 1}</span>
                <b>{title}</b>
                <p>{text}</p>
              </div>
            ))}
          </div>
          <div className="truth-panel">
            <CommandLineIcon />
            <div>
              <span>PORTFOLIO PRINCIPLE</span>
              <h3>Credibility compounds.</h3>
              <p>
                Every figure on this site is read from a committed evaluation artifact. Nothing is typed into the UI by hand, and the API names the model,
                feature set, and evaluation behind each response so a reader can reproduce it from the repository.
              </p>
            </div>
            <a href={`${SITE.repoUrl}/blob/main/docs/PORTFOLIO_CASE_STUDY.md`} target="_blank" rel="noreferrer">
              Read the technical brief <ArrowRightIcon />
            </a>
          </div>
        </section>
      </main>
      <Footer />
    </div>
  )
}
