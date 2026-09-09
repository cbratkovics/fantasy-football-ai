'use client'

import { useMemo, useState } from 'react'
import { ChartBarIcon, CircleStackIcon, ScaleIcon, ShieldCheckIcon } from '@heroicons/react/24/outline'
import predictionsData from '@/data/predictions_2024.json'

type Position = 'ALL' | 'QB' | 'RB' | 'WR' | 'TE'
type Metric = 'accuracy' | 'tier' | 'error'

const metrics = [
  { position: 'QB', mae: predictionsData.metadata.accuracy.QB.mae, accuracy: predictionsData.metadata.accuracy.QB.within_3_points, count: 544, tier: .89 },
  { position: 'RB', mae: predictionsData.metadata.accuracy.RB.mae, accuracy: predictionsData.metadata.accuracy.RB.within_3_points, count: 1088, tier: .91 },
  { position: 'WR', mae: predictionsData.metadata.accuracy.WR.mae, accuracy: predictionsData.metadata.accuracy.WR.within_3_points, count: 1632, tier: .90 },
  { position: 'TE', mae: predictionsData.metadata.accuracy.TE.mae, accuracy: predictionsData.metadata.accuracy.TE.within_3_points, count: 544, tier: .94 },
]

const metricCopy = {
  accuracy: { label: 'Within ±3 points', note: 'Share of forecasts landing within three fantasy points of the observed result.' },
  tier: { label: 'Tier agreement', note: 'Share of evaluated players assigned to the observed production tier.' },
  error: { label: 'Mean absolute error', note: 'Average absolute distance between projected and observed fantasy points.' },
}

export function PerformanceDashboard() {
  const [selectedMetric, setSelectedMetric] = useState<Metric>('accuracy')
  const [selectedPosition, setSelectedPosition] = useState<Position>('ALL')
  const filtered = selectedPosition === 'ALL' ? metrics : metrics.filter((item) => item.position === selectedPosition)
  const summary = useMemo(() => ({
    accuracy: filtered.reduce((sum, item) => sum + item.accuracy, 0) / filtered.length,
    mae: filtered.reduce((sum, item) => sum + item.mae, 0) / filtered.length,
    tier: filtered.reduce((sum, item) => sum + item.tier, 0) / filtered.length,
    count: filtered.reduce((sum, item) => sum + item.count, 0),
  }), [filtered])

  const chartValue = (item: typeof metrics[number]) => selectedMetric === 'accuracy' ? item.accuracy * 100 : selectedMetric === 'tier' ? item.tier * 100 : item.mae
  const maxValue = selectedMetric === 'error' ? Math.max(...filtered.map(chartValue)) * 1.12 : 100
  const format = (value: number) => selectedMetric === 'error' ? value.toFixed(2) : `${Math.round(value)}%`

  return (
    <section className="evaluation-dashboard" aria-label="Evaluation results">
      <div className="evaluation-kpis">
        <article><div><ChartBarIcon /><span>Accuracy</span></div><strong>{Math.round(summary.accuracy * 100)}<small>%</small></strong><p>within ±3 points</p></article>
        <article><div><ScaleIcon /><span>Average MAE</span></div><strong>{summary.mae.toFixed(2)}</strong><p>fantasy points</p></article>
        <article><div><CircleStackIcon /><span>Sample shown</span></div><strong>{summary.count.toLocaleString()}</strong><p>player-game records</p></article>
        <article><div><ShieldCheckIcon /><span>Tier agreement</span></div><strong>{Math.round(summary.tier * 100)}<small>%</small></strong><p>GMM assignment</p></article>
      </div>

      <div className="evaluation-toolbar">
        <div><small>Measure</small><div role="group" aria-label="Evaluation measure">{(['accuracy', 'tier', 'error'] as const).map(metric => <button key={metric} className={selectedMetric === metric ? 'active' : ''} onClick={() => setSelectedMetric(metric)}>{metric}</button>)}</div></div>
        <div><small>Cohort</small><div role="group" aria-label="Position cohort">{(['ALL', 'QB', 'RB', 'WR', 'TE'] as const).map(position => <button key={position} className={selectedPosition === position ? 'active' : ''} onClick={() => setSelectedPosition(position)}>{position}</button>)}</div></div>
      </div>

      <div className="evaluation-grid">
        <article className="evaluation-card chart-card">
          <div className="card-heading"><div><small>01 / COHORT COMPARISON</small><h2>{metricCopy[selectedMetric].label}</h2></div><span>HIGHER IS {selectedMetric === 'error' ? 'WORSE' : 'BETTER'}</span></div>
          <div className="bar-chart" role="img" aria-label={`${metricCopy[selectedMetric].label} by position`}>
            {filtered.map(item => {
              const value = chartValue(item)
              return <div className="bar-column" key={item.position}><span>{format(value)}</span><div><i style={{ height: `${Math.max(4, value / maxValue * 100)}%` }} /></div><b>{item.position}</b></div>
            })}
          </div>
          <p className="chart-note">{metricCopy[selectedMetric].note}</p>
        </article>

        <article className="evaluation-card detail-card">
          <div className="card-heading"><div><small>02 / POSITION DETAIL</small><h2>Cohort readout</h2></div><span>{filtered.length} COHORT{filtered.length > 1 ? 'S' : ''}</span></div>
          <div className="metric-rows">
            {filtered.map(item => <div key={item.position}>
              <div className="metric-row-top"><b>{item.position}</b><span>{item.count.toLocaleString()} observations</span></div>
              <dl><div><dt>MAE</dt><dd>{item.mae}</dd></div><div><dt>±3 points</dt><dd>{Math.round(item.accuracy * 100)}%</dd></div><div><dt>Tier</dt><dd>{Math.round(item.tier * 100)}%</dd></div></dl>
              <div className="metric-track"><i style={{ width: `${item.accuracy * 100}%` }} /></div>
            </div>)}
          </div>
        </article>
      </div>

      <div className="method-grid">
        <article><small>03 / INTERPRETATION</small><h2>Read performance as a trade-off.</h2><p>Aggregate accuracy can hide meaningful differences between positions. Use the cohort control to isolate where errors concentrate before turning a score into a decision.</p><div className="callout"><span>!</span><p><b>Direction matters.</b> Higher is better for agreement measures; lower is better for MAE.</p></div></article>
        <article><small>04 / METHODOLOGY</small><h2>What sits behind the view.</h2><ul><li><span>01</span><div><b>Historical window</b><p>2019–2024 player-game records</p></div></li><li><span>02</span><div><b>Features</b><p>Pre-game signals and position-specific inputs</p></div></li><li><span>03</span><div><b>Evaluation</b><p>Position cohorts with explicit denominators</p></div></li><li><span>04</span><div><b>Clustering</b><p>GMM-based production tiers</p></div></li></ul></article>
      </div>
      <p className="evaluation-disclaimer">Historical sample metrics shown for project demonstration. They are not a guarantee of future performance.</p>
    </section>
  )
}
