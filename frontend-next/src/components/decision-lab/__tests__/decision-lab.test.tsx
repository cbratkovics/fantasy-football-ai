import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it } from 'vitest'

import { createMemoryFetch } from '@/lib/decision-lab/bundle'
import { STORAGE_KEY, exportAll, loadReceipts } from '@/lib/decision-lab/storage'
import { syntheticBundle, type SyntheticBundle } from '@/lib/decision-lab/__tests__/fixtures'
import { DecisionLabApp } from '../DecisionLabApp'

const CASE_ID = 'syn-bundle-ambiguity'
const NOW = () => '2026-01-01T00:00:00Z'

function mount(bundle: SyntheticBundle = syntheticBundle(), initialCaseId: string | null = null) {
  return render(<DecisionLabApp fetchImpl={createMemoryFetch(bundle.files)} now={NOW} initialCaseId={initialCaseId} />)
}

async function selectCaseAndWait(user: ReturnType<typeof userEvent.setup>) {
  await user.click(await screen.findByTestId(`case-${CASE_ID}`))
  await waitFor(() => expect(screen.getByTestId('compute')).toBeEnabled())
}

beforeEach(() => {
  window.localStorage.clear()
})

describe('DecisionLab container', () => {
  it('loads the bundle, lists the case library and warns about the spec digest without blocking', async () => {
    mount()
    expect(await screen.findByTestId(`case-${CASE_ID}`)).toBeInTheDocument()
    expect(screen.getByTestId('spec-warning')).toBeInTheDocument()
    expect(screen.getByTestId('decision-empty')).toBeInTheDocument()
    expect(screen.getByTestId('compute')).toBeDisabled()
  })

  it('fails closed on a corrupt manifest and explains that nothing was substituted', async () => {
    const b = syntheticBundle()
    const files = { ...b.files, 'cases.json': b.files['cases.json'].replace('\n', '\n\n') }
    render(<DecisionLabApp fetchImpl={createMemoryFetch(files)} now={NOW} />)
    const alert = await screen.findByTestId('bundle-error')
    expect(alert).toHaveTextContent(/byte digest mismatch/)
    expect(alert).toHaveTextContent(/Nothing was substituted/)
  })

  it('deep links to a case and marks it current', async () => {
    mount(syntheticBundle(), CASE_ID)
    const button = await screen.findByTestId(`case-${CASE_ID}`)
    await waitFor(() => expect(button).toHaveAttribute('aria-current', 'true'))
    await waitFor(() => expect(screen.getByTestId('compute')).toBeEnabled())
  })
})

describe('DecisionCard statuses', () => {
  it('renders REVIEW with a text label and no recommended player', async () => {
    const user = userEvent.setup()
    mount()
    await selectCaseAndWait(user)
    await user.click(screen.getByTestId('compute'))
    const pill = within(screen.getByTestId('decision-card')).getByTestId('status-pill')
    expect(pill).toHaveTextContent(/Review/)
    expect(pill).toHaveAttribute('data-status', 'review')
    expect(screen.getByTestId('recommended-player')).toHaveTextContent('No recommendation')
    expect(screen.getByTestId('gap-line')).toHaveTextContent('0.4 vs threshold 0.5')
    expect(screen.getByTestId('saved-badge')).toHaveTextContent('Unsaved exploration')
  })

  it('renders RECOMMEND after lowering the gap and shows what changed with a parent link', async () => {
    const user = userEvent.setup()
    mount()
    await selectCaseAndWait(user)
    await user.click(screen.getByTestId('compute'))
    const gap = screen.getByLabelText('Minimum projection gap')
    await user.clear(gap)
    await user.type(gap, '0.3')
    expect(screen.getByTestId('stale-notice')).toBeInTheDocument()
    await user.click(screen.getByTestId('compute'))
    expect(within(screen.getByTestId('decision-card')).getByTestId('status-pill')).toHaveAttribute('data-status', 'recommend')
    expect(screen.getByTestId('recommended-player')).toHaveTextContent('Synthetic A (SYN-A)')
    expect(screen.getByTestId('baseline-line')).toHaveTextContent('Baseline prefers Synthetic B (SYN-B) (13)')
    expect(screen.getByTestId('changed-controls')).toHaveTextContent('min_projection_gap')
    expect(screen.getByTestId('diff-row-status')).toHaveTextContent('review')
    expect(screen.getByTestId('parent-id')).not.toHaveTextContent('—')
  })

  it('renders HOLD when corrupt evidence is simulated and no parameter unblocks it', async () => {
    const user = userEvent.setup()
    mount()
    await selectCaseAndWait(user)
    await user.click(screen.getByLabelText('Simulate corrupt evidence'))
    const gap = screen.getByLabelText('Minimum projection gap')
    await user.clear(gap)
    await user.type(gap, '0')
    await user.click(screen.getByTestId('compute'))
    const pill = within(screen.getByTestId('decision-card')).getByTestId('status-pill')
    expect(pill).toHaveAttribute('data-status', 'hold')
    expect(pill).toHaveTextContent(/Hold/)
    expect(screen.getByTestId('recommended-player')).toHaveTextContent('No recommendation')
  })

  it('invalid numbers never reach the state: the field reports an error and Compute is disabled', async () => {
    const user = userEvent.setup()
    mount()
    await selectCaseAndWait(user)
    const gap = screen.getByLabelText('Minimum projection gap')
    await user.clear(gap)
    expect(screen.getByTestId('compute')).toBeDisabled()
    expect(gap).toHaveAttribute('aria-invalid', 'true')
    await user.type(gap, '99')
    expect(screen.getByTestId('compute')).toBeDisabled()
    await user.clear(gap)
    await user.type(gap, '1')
    expect(screen.getByTestId('compute')).toBeEnabled()
  })
})

describe('stale results', () => {
  it('changing a parameter after computing shows the stale state and hides the recommended player', async () => {
    const user = userEvent.setup()
    mount()
    await selectCaseAndWait(user)
    await user.click(screen.getByTestId('compute'))
    expect(screen.getByTestId('recommended-player')).toBeInTheDocument()
    await user.click(screen.getByLabelText('Model-only exploration'))
    expect(screen.getByTestId('decision-stale')).toHaveTextContent(/stale for the inputs above/)
    expect(screen.queryByTestId('recommended-player')).toBeNull()
    expect(screen.queryByTestId('action-panel')).toBeNull()
    expect(screen.getByTestId('stale-badge')).toBeInTheDocument()
  })

  it('changing the slot also invalidates the result', async () => {
    const user = userEvent.setup()
    mount()
    await selectCaseAndWait(user)
    await user.click(screen.getByTestId('compute'))
    await user.click(screen.getByTestId('slot-WR'))
    expect(screen.getByTestId('decision-stale')).toBeInTheDocument()
  })
})

describe('actions, receipts and outcomes', () => {
  it('records an action, saves the receipt, freezes controls and enables the reveal', async () => {
    const user = userEvent.setup()
    mount()
    await selectCaseAndWait(user)
    await user.click(screen.getByTestId('compute'))

    await user.click(screen.getByRole('button', { name: /^Outcomes$/ }))
    expect(screen.getByTestId('reveal-outcomes')).toBeDisabled()

    // The recommendation is never preselected.
    expect(screen.getByTestId('choose-SYN-A')).not.toBeChecked()
    expect(screen.getByTestId('choose-SYN-B')).not.toBeChecked()
    expect(screen.getByTestId('record-choice')).toBeDisabled()

    await user.click(screen.getByTestId('choose-SYN-B'))
    await user.click(screen.getByTestId('record-choice'))
    expect(screen.getByTestId('action-state')).toHaveTextContent('Action recorded')
    expect(screen.getByTestId('action-chosen')).toHaveTextContent('Synthetic B (SYN-B)')
    expect(screen.getByTestId('saved-badge')).toHaveTextContent('Recorded decision')
    expect(screen.getByLabelText('Minimum projection gap')).toBeDisabled()
    expect(screen.getByTestId('frozen-note')).toBeInTheDocument()

    const stored = loadReceipts(window.localStorage)
    expect(Object.keys(stored.receipts)).toHaveLength(1)
    const receipt = Object.values(stored.receipts)[0]
    expect(receipt.action).toMatchObject({ state: 'recorded', chosen_player_id: 'SYN-B', kind: 'hypothetical_replay' })
    expect(receipt.case_id).toBe(CASE_ID)

    expect(screen.getByTestId('reveal-outcomes')).toBeEnabled()
    await user.click(screen.getByTestId('reveal-outcomes'))
    const metrics = await screen.findByTestId('outcome-metrics')
    expect(within(metrics).getByTestId('metric-chosen')).toHaveTextContent('16')
    expect(within(metrics).getByTestId('metric-regret')).toHaveTextContent('0')
    expect(within(metrics).getByTestId('metric-vs-baseline')).toHaveTextContent('0')
    expect(within(metrics).getByTestId('metric-model-vs-baseline')).toHaveTextContent(/null — no model recommendation/)
    const after = loadReceipts(window.localStorage).receipts[receipt.decision_id]
    expect(after.outcome.state).toBe('attached')
    expect(after.decision_id).toBe(receipt.decision_id)
  })

  it('a new child decision starts after unlocking the frozen controls', async () => {
    const user = userEvent.setup()
    mount()
    await selectCaseAndWait(user)
    await user.click(screen.getByTestId('compute'))
    await user.click(screen.getByTestId('choose-SYN-A'))
    await user.click(screen.getByTestId('decline-choice'))
    expect(screen.getByTestId('action-state')).toHaveTextContent('Declined to choose')
    const parentId = Object.keys(loadReceipts(window.localStorage).receipts)[0]
    await user.click(screen.getByTestId('unlock'))
    await user.click(screen.getByLabelText('Require an independent baseline'))
    await user.click(screen.getByTestId('compute'))
    const parent = screen.getByTestId('parent-id').querySelector('code')
    expect(parent).toHaveAttribute('title', parentId)
    expect(screen.getByTestId('saved-badge')).toHaveTextContent('Unsaved exploration')
  })
})

describe('already-saved semantic decisions', () => {
  it('opens the saved record instead of creating a competing receipt with new metadata', async () => {
    const user = userEvent.setup()
    const first = mount()
    await selectCaseAndWait(user)
    await user.type(screen.getByTestId('prediction-note'), 'first note')
    await user.click(screen.getByTestId('compute'))
    await user.click(screen.getByTestId('choose-SYN-B'))
    await user.click(screen.getByTestId('record-choice'))
    const stored = loadReceipts(window.localStorage)
    const [savedId] = Object.keys(stored.receipts)
    const savedBefore = JSON.stringify(stored.receipts[savedId])
    expect(stored.receipts[savedId].prediction_note).toBe('first note')
    first.unmount()

    // A later session with a different clock and a different note reaches the same inputs.
    render(<DecisionLabApp fetchImpl={createMemoryFetch(syntheticBundle().files)} now={() => '2026-02-02T00:00:00Z'} />)
    await selectCaseAndWait(user)
    await user.type(screen.getByTestId('prediction-note'), 'second note')
    await user.click(screen.getByTestId('compute'))
    expect(screen.getByTestId('compute-message')).toHaveTextContent(/already saved/)
    expect(screen.getByTestId('saved-badge')).toHaveTextContent('Recorded decision')
    expect(screen.getByTestId('action-chosen')).toHaveTextContent('Synthetic B (SYN-B)')
    expect(screen.getByTestId('frozen-note')).toBeInTheDocument()
    const after = loadReceipts(window.localStorage)
    expect(Object.keys(after.receipts)).toEqual([savedId])
    expect(JSON.stringify(after.receipts[savedId])).toBe(savedBefore)
  })
})

describe('missing numbers', () => {
  it("renders '—' for a missing floor, never 0", async () => {
    const user = userEvent.setup()
    const b = syntheticBundle(({ inputs, cases }) => {
      inputs.rows[2].floor = null
      cases.cases[0].alternatives = ['SYN-A', 'SYN-B', 'SYN-C']
    })
    mount(b)
    await selectCaseAndWait(user)
    await user.click(screen.getByTestId('compute'))
    expect(screen.getByTestId('floor-SYN-C')).toHaveTextContent('—')
    expect(screen.getByTestId('floor-SYN-A')).toHaveTextContent('6')
    expect(screen.getByTestId('alternatives-scroll')).toHaveClass('overflow-x-auto')
  })
})

describe('import', () => {
  it('rejects a tampered receipt with a visible reason and keeps the good one', async () => {
    const user = userEvent.setup()
    mount()
    await selectCaseAndWait(user)
    await user.click(screen.getByTestId('compute'))
    await user.click(screen.getByTestId('choose-SYN-A'))
    await user.click(screen.getByTestId('record-choice'))
    const exported = JSON.parse(exportAll(NOW(), window.localStorage)) as { receipts: Array<{ result: { leading: { gap: number | null } } }> }
    const good = JSON.parse(JSON.stringify(exported))
    // A plausible value the shape check accepts; only the digest and replay checks can catch it.
    exported.receipts[0].result.leading.gap = 9
    const tampered = JSON.stringify({ ...exported, receipts: [...exported.receipts, ...good.receipts] })
    window.localStorage.removeItem(STORAGE_KEY)

    await user.click(screen.getByRole('button', { name: /Saved decisions/ }))
    const input = screen.getByTestId('import-file') as HTMLInputElement
    const file = new File([tampered], 'receipts.json', { type: 'application/json' })
    fireEvent.change(input, { target: { files: [file] } })
    const report = await screen.findByTestId('import-report')
    expect(report).toHaveTextContent('1 imported')
    expect(report).toHaveTextContent('1 rejected')
    expect(screen.getByTestId('import-rejections')).toHaveTextContent(/result_replay: stored result differs from a fresh evaluation \(tampered or stale\)/)
    expect(Object.keys(loadReceipts(window.localStorage).receipts)).toHaveLength(1)
  })
})
