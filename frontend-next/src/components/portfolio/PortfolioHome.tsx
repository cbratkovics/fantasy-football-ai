'use client'

import { useMemo, useState } from 'react'
import Link from 'next/link'
import {
  ArrowRightIcon,
  ArrowTopRightOnSquareIcon,
  BeakerIcon,
  BoltIcon,
  ChartBarSquareIcon,
  CheckIcon,
  CircleStackIcon,
  CommandLineIcon,
  ShieldCheckIcon,
} from '@heroicons/react/24/outline'

const decisions = [
  { name: 'C. McCaffrey', position: 'RB', projection: 22.5, confidence: 85, downside: 16.2, signal: 91 },
  { name: 'J. Jefferson', position: 'WR', projection: 19.2, confidence: 83, downside: 12.8, signal: 79 },
  { name: 'P. Mahomes', position: 'QB', projection: 24.8, confidence: 87, downside: 18.6, signal: 88 },
  { name: 'S. LaPorta', position: 'TE', projection: 13.4, confidence: 72, downside: 7.9, signal: 62 },
  { name: 'D. Achane', position: 'RB', projection: 16.8, confidence: 67, downside: 8.1, signal: 55 },
]

const architecture = [
  ['01', 'Ingest', 'NFL stats, availability, matchup context'],
  ['02', 'Validate', 'Contracts, leakage checks, missingness'],
  ['03', 'Model', 'Time-aware baselines + ensemble challenger'],
  ['04', 'Decide', 'Calibrated score into an explicit policy'],
  ['05', 'Monitor', 'Cohorts, drift, errors, decision outcomes'],
]

const skills = [
  ['Decision science', 'Thresholds, expected value, constraints'],
  ['Modeling', 'Time splits, calibration, uncertainty'],
  ['Analytics engineering', 'SQL marts, tests, metric definitions'],
  ['Production', 'FastAPI, containers, versioned artifacts'],
]

function MiniBars({ threshold }: { threshold: number }) {
  const values = [42, 54, 48, 67, 61, 74, 69, 81, 77, 86, 82, 91]
  return (
    <div className="mini-bars" aria-label="Weekly decision quality trend">
      {values.map((value, index) => (
        <span key={index} style={{ height: `${Math.max(16, value - threshold / 3)}%` }} />
      ))}
    </div>
  )
}

export function PortfolioHome() {
  const [threshold, setThreshold] = useState(70)
  const approved = useMemo(() => decisions.filter((row) => row.signal >= threshold), [threshold])
  const approvalRate = Math.round((approved.length / decisions.length) * 100)
  const averageFloor = approved.length
    ? approved.reduce((sum, row) => sum + row.downside, 0) / approved.length
    : 0

  return (
    <main className="portfolio-shell">
      <nav className="portfolio-nav" aria-label="Primary navigation">
        <Link href="/" className="portfolio-mark" aria-label="Win My League home">
          <span>WML</span><b>Decision Lab</b>
        </Link>
        <div className="portfolio-links">
          <a href="#case-study">Case study</a>
          <a href="#system">System</a>
          <Link href="/performance">Evaluation</Link>
          <a href="https://github.com/cbratkovics/fantasy-football-ai" target="_blank" rel="noreferrer">
            Source <ArrowTopRightOnSquareIcon />
          </a>
        </div>
      </nav>

      <section className="portfolio-hero">
        <div className="hero-copy">
          <div className="eyebrow"><span /> Applied data science · end-to-end case study</div>
          <h1>Better predictions are only useful when they produce <em>better decisions.</em></h1>
          <p className="hero-lede">
            I built a decision system for fantasy football that turns noisy, time-sensitive signals into
            explainable lineup recommendations—then measures the trade-off between opportunity and downside.
          </p>
          <div className="hero-actions">
            <a href="#case-study" className="portfolio-button primary">Explore the case study <ArrowRightIcon /></a>
            <Link href="/performance" className="portfolio-button secondary">Review evaluation</Link>
          </div>
          <div className="hero-proof">
            <div><b>Problem</b><span>Choose under uncertainty</span></div>
            <div><b>Output</b><span>Ranked, explainable actions</span></div>
            <div><b>Standard</b><span>No metric without provenance</span></div>
          </div>
        </div>

        <div className="decision-console" aria-label="Interactive decision policy demonstration">
          <div className="console-top">
            <div><span className="live-dot" /> POLICY SIMULATOR</div>
            <span>DEMO DATA</span>
          </div>
          <div className="console-metrics">
            <div><small>RECOMMEND RATE</small><strong>{approvalRate}%</strong><span>{approved.length} of {decisions.length} candidates</span></div>
            <div><small>AVG DOWNSIDE FLOOR</small><strong>{averageFloor.toFixed(1)}</strong><span>points among selections</span></div>
          </div>
          <MiniBars threshold={threshold} />
          <div className="threshold-control">
            <div><label htmlFor="risk-threshold">Decision threshold</label><output>{threshold}</output></div>
            <input id="risk-threshold" type="range" min="50" max="95" value={threshold} onChange={(event) => setThreshold(Number(event.target.value))} />
            <div className="range-labels"><span>More opportunity</span><span>Less downside</span></div>
          </div>
          <div className="decision-list">
            {decisions.map((row) => (
              <div key={row.name} className={row.signal >= threshold ? 'selected' : ''}>
                <span className="decision-state">{row.signal >= threshold ? <CheckIcon /> : '—'}</span>
                <b>{row.name}</b><small>{row.position}</small><span>{row.projection.toFixed(1)} pts</span><em>{row.confidence}% conf.</em>
              </div>
            ))}
          </div>
          <p className="console-note">Illustrative policy sandbox—not a claim of live or historical performance.</p>
        </div>
      </section>

      <section className="portfolio-strip" aria-label="Core capabilities">
        <span>PYTHON</span><i /> <span>SQL</span><i /> <span>SCIKIT-LEARN</span><i /> <span>FASTAPI</span><i /> <span>NEXT.JS</span><i /> <span>DOCKER</span>
      </section>

      <section className="case-section" id="case-study">
        <div className="section-heading">
          <div><span className="section-number">01</span><p>THE CASE STUDY</p></div>
          <h2>From a ranking model to a measurable decision strategy.</h2>
          <p>Fantasy lineup selection is a safe analogue for real-time risk work: incomplete information, asymmetric errors, shifting populations, and a decision deadline.</p>
        </div>
        <div className="problem-grid">
          <article className="statement-card dark-card">
            <span>THE BUSINESS QUESTION</span>
            <h3>Who should we act on—and what evidence would change that decision?</h3>
            <p>A point estimate does not answer this. The useful product combines expected performance, uncertainty, eligibility, opportunity cost, and an auditable policy.</p>
            <div className="formula">utility = upside − λ(downside) − opportunity cost</div>
          </article>
          <article className="statement-card">
            <span>SUCCESS CRITERIA</span>
            <ul>
              <li><b>Decision value</b><small>Beat a simple expert-ranking baseline</small></li>
              <li><b>Reliability</b><small>Calibrated intervals across positions and weeks</small></li>
              <li><b>Coverage</b><small>Track recommendation rate alongside quality</small></li>
              <li><b>Trust</b><small>Freshness, drivers, and model version on every result</small></li>
            </ul>
          </article>
        </div>
      </section>

      <section className="system-section" id="system">
        <div className="section-heading compact">
          <div><span className="section-number">02</span><p>SYSTEM DESIGN</p></div>
          <h2>An observable path from raw event to human action.</h2>
        </div>
        <div className="architecture-row">
          {architecture.map(([number, title, text]) => (
            <article key={number}><span>{number}</span><div className="architecture-icon"><CircleStackIcon /></div><h3>{title}</h3><p>{text}</p></article>
          ))}
        </div>
        <div className="principles-grid">
          <article><BeakerIcon /><div><b>Evaluation before optimization</b><p>Naive and expert baselines, rolling-origin validation, segment error, and interval coverage before model complexity.</p></div></article>
          <article><ChartBarSquareIcon /><div><b>Metrics with a decision attached</b><p>Recommendation rate and regret sit beside MAE. Every view exposes cohort, time window, and denominator.</p></div></article>
          <article><ShieldCheckIcon /><div><b>Guardrails over magic</b><p>Schema validation, freshness checks, model versions, fallbacks, and explicit abstention for low-confidence cases.</p></div></article>
          <article><BoltIcon /><div><b>Built for iteration</b><p>Modular feature computation and stable API contracts allow a baseline and challenger to coexist safely.</p></div></article>
        </div>
      </section>

      <section className="evidence-section">
        <div className="section-heading compact light-heading">
          <div><span className="section-number">03</span><p>WHAT THIS DEMONSTRATES</p></div>
          <h2>Technical depth, translated into practical value.</h2>
        </div>
        <div className="skills-list">
          {skills.map(([title, text], index) => <div key={title}><span>0{index + 1}</span><b>{title}</b><p>{text}</p></div>)}
        </div>
        <div className="truth-panel">
          <CommandLineIcon />
          <div><span>PORTFOLIO PRINCIPLE</span><h3>Credibility compounds.</h3><p>Demo values are labeled. Measured results require a reproducible dataset, split, baseline, and generated evaluation artifact. The repository roadmap makes unfinished work visible instead of disguising it as production performance.</p></div>
          <a href="https://github.com/cbratkovics/fantasy-football-ai/blob/main/docs/PORTFOLIO_CASE_STUDY.md" target="_blank" rel="noreferrer">Read the technical brief <ArrowRightIcon /></a>
        </div>
      </section>

      <footer className="portfolio-footer">
        <div><span className="footer-mark">WML</span><div><b>Win My League · Decision Lab</b><p>An applied data science portfolio project by Christopher Bratkovics.</p></div></div>
        <div><a href="mailto:chris@fantasyfootballai.com">Email</a><a href="https://github.com/cbratkovics">GitHub</a><a href="https://linkedin.com/in/cbratkovics">LinkedIn</a></div>
      </footer>
    </main>
  )
}
