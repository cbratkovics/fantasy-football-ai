import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, expect, it, vi } from 'vitest'
import { createMemoryFetch } from '@/lib/decision-lab/bundle'
import { loadReceipts } from '@/lib/decision-lab/storage'
import { weeklyBundle } from '@/lib/start-sit/__tests__/fixtures'
import type { Readiness } from '@/lib/start-sit/readiness'
import { StartSitEngine } from '../StartSitEngine'

vi.mock('../ContextEvidence', () => ({ default: ({ onStateChange }: { onStateChange: (state: Readiness['state']) => void }) => (
  <button type="button" onClick={() => onStateChange('blocked')}>Simulate a passed action deadline</button>
) }))
beforeEach(() => { localStorage.clear(); sessionStorage.clear(); window.history.replaceState({}, '', '/start-sit') })

it('keeps the original dated policy inspectable but gates blocked context behind explicit hypothetical exploration', async () => {
  const user = userEvent.setup()
  const b = weeklyBundle((s) => { s.rows[1].projection = 13 })
  render(<StartSitEngine fetchImpl={createMemoryFetch(b.files)} />)
  await screen.findByLabelText('Starting slot')
  await user.selectOptions(screen.getByLabelText('Starting slot'), 'RB')
  for (const name of ['Synthetic A', 'Synthetic B']) {
    await user.type(screen.getByLabelText('Search players to compare'), name)
    await user.click(screen.getByRole('button', { name: new RegExp(`Add ${name}`) }))
    const d = screen.getByText('Adjust player assumptions').closest('details')!
    if (!d.open) await user.click(screen.getByText('Adjust player assumptions'))
    await user.selectOptions(screen.getByLabelText(`Availability assumption for ${name}`), 'assumed_available')
  }
  await user.click(screen.getByRole('button', { name: 'Compare assuming these players can play' }))
  expect(within(screen.getByTestId('quick-result')).getByRole('heading')).toHaveTextContent('Synthetic A has the higher projection')
  await user.click(screen.getByRole('button', { name: 'Simulate a passed action deadline' }))
  expect(within(screen.getByTestId('quick-result')).getByRole('heading')).toHaveTextContent('This comparison cannot support a current lineup change')
  expect(screen.getByText('Original dated policy result: Synthetic A has the higher projection')).toBeInTheDocument()
  expect(screen.queryByTestId('action-panel')).not.toBeInTheDocument()
  expect(Object.keys(loadReceipts().receipts)).toHaveLength(0)
  await user.click(screen.getByRole('button', { name: 'Explore as historical hypothetical' }))
  await user.click(screen.getByText('Record my choice or save a hypothetical comparison'))
  expect(screen.getByTestId('action-panel')).toBeInTheDocument()
  expect(screen.queryByTestId('kind-self_reported_real')).not.toBeInTheDocument()
  expect(screen.getByTestId('kind-hypothetical_replay')).toBeChecked()
  expect(Object.keys(loadReceipts().receipts)).toHaveLength(0)
})
