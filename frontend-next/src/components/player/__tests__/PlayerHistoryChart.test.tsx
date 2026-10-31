import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import type { PlayerWeekRow } from '@/lib/api/types'
import { PlayerHistoryChart } from '../PlayerHistoryChart'

afterEach(cleanup)
const row = (actual: number | null, week: number): PlayerWeekRow => ({
  season: 2026, week, model_version: 'model', candidate: 'rf', position: 'QB', source: 'weekly',
  eval_window: 'weekly', prediction: 5, prediction_floor: 0, prediction_ceiling: 10, actual,
  actual_source: actual === null ? null : 'stats', abs_error: actual === null ? null : Math.abs(5 - actual),
  within_3: null, within_5: null, interval_hit: null, baseline: 4, baseline_abs_error: null,
})

describe('PPR player history', () => {
  it('labels scoring and plots negative and zero observations while leaving missing actuals absent', () => {
    const { container } = render(<PlayerHistoryChart history={[row(-2, 1), row(0, 2), row(null, 3)]} />)
    expect(screen.getByRole('img', { name: /Weekly PPR prediction versus actual/ })).toBeInTheDocument()
    expect(screen.getByText(/PPR prediction vs actual/)).toBeInTheDocument()
    const actuals = container.querySelectorAll('circle[fill="#ff6b35"]')
    expect(actuals).toHaveLength(2)
    for (const point of Array.from(actuals)) {
      expect(Number(point.getAttribute('cy'))).toBeGreaterThanOrEqual(16)
      expect(Number(point.getAttribute('cy'))).toBeLessThanOrEqual(240)
    }
  })
})
