import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it } from 'vitest'
import { createMemoryFetch } from '@/lib/decision-lab/bundle'
import { loadReceipts } from '@/lib/decision-lab/storage'
import { weeklyBundle } from '@/lib/start-sit/__tests__/fixtures'
import { StartSitEngine } from '../StartSitEngine'

beforeEach(() => { localStorage.clear(); sessionStorage.clear(); window.history.replaceState({}, '', '/start-sit') })
function mount(b = weeklyBundle()) { return render(<StartSitEngine fetchImpl={createMemoryFetch(b.files)} now={() => '2099-09-01T00:00:00Z'} />) }
async function nominate(user: ReturnType<typeof userEvent.setup>, name: string) {
  await user.type(screen.getByLabelText('Search players to compare'), name)
  await user.click(screen.getByRole('button', { name: new RegExp(`Add ${name}`) }))
  const details = screen.getByText('Adjust player assumptions').closest('details')!
  if (!details.open) await user.click(screen.getByText('Adjust player assumptions'))
}

describe('Start/Sit verified comparison', () => {
  it('defaults visibly to FLEX/PPR with two empty cards and collapsed evidence', async () => {
    mount()
    expect(await screen.findByLabelText('Starting slot')).toHaveValue('FLEX')
    expect(screen.getByText(/Latest published: Week/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Compare assuming these players can play' })).toBeDisabled()
    expect(screen.getByRole('button', { name: /Choose player 1/ })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Choose player 2/ })).toBeInTheDocument()
    expect(screen.getByText('Scoring rules, model range and source evidence').closest('details')).not.toHaveAttribute('open')
    expect(screen.queryByRole('button', { name: 'half' })).not.toBeInTheDocument()
  })
  it('requires intentional availability assumptions; computing never records the suggestion as action', async () => {
    const user = userEvent.setup()
    mount(weeklyBundle((s) => { s.rows[1].projection = 13 }))
    await screen.findByLabelText('Starting slot')
    await user.selectOptions(screen.getByLabelText('Starting slot'), 'RB')
    await nominate(user, 'Synthetic A'); await nominate(user, 'Synthetic B')
    await user.click(screen.getByRole('button', { name: 'Compare nominated alternatives' }))
    expect(within(screen.getByTestId('quick-result')).getByRole('heading')).toHaveTextContent('Check availability before choosing')
    expect(Object.keys(loadReceipts().receipts)).toHaveLength(0)
    await user.selectOptions(screen.getByLabelText('Availability assumption for Synthetic A'), 'assumed_available')
    await user.selectOptions(screen.getByLabelText('Availability assumption for Synthetic B'), 'assumed_available')
    await user.click(screen.getByRole('button', { name: 'Compare nominated alternatives' }))
    expect(within(screen.getByTestId('quick-result')).getByRole('heading')).toHaveTextContent('Synthetic A has the higher projection')
    expect(screen.getAllByRole('radio').filter((r) => (r as HTMLInputElement).name === 'chosen').every((r) => !(r as HTMLInputElement).checked)).toBe(true)
    expect(Object.keys(loadReceipts().receipts)).toHaveLength(0)
  })
  it('keeps a missing selected ID visible after navigation and withholds reduced-set ranking', async () => {
    window.history.replaceState({}, '', '/start-sit?snapshot=weekly-fixture&players=SYN-A,missing,SYN-B&slot=RB')
    mount()
    expect(await screen.findByText(/A selected projection is missing/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Remove missing' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Compare assuming these players can play' })).toBeDisabled()
  })
  it('refreshes evidence after tab focus and invalidates a computed comparison on snapshot change', async () => {
    const b = weeklyBundle()
    const user = userEvent.setup()
    let files = b.files
    const fetchImpl = (url: string) => createMemoryFetch(files)(url)
    render(<StartSitEngine fetchImpl={fetchImpl} />)
    await screen.findByLabelText('Starting slot')
    await user.selectOptions(screen.getByLabelText('Starting slot'), 'RB')
    await nominate(user, 'Synthetic A'); await nominate(user, 'Synthetic B')
    await user.click(screen.getByRole('button', { name: 'Compare nominated alternatives' }))
    expect(screen.getByTestId('quick-result')).toBeInTheDocument()
    files = weeklyBundle((s) => { s.rows[1].projection = 12 }).files
    fireEvent.focus(window)
    await waitFor(() => expect(screen.queryByTestId('quick-result')).not.toBeInTheDocument())
    expect(screen.getByText(/Recompute before using/)).toBeInTheDocument()
  })
  it('retains a labeled dated result during refresh, pauses actions and ignores an older request that finishes last', async () => {
    const old = weeklyBundle()
    const newer = weeklyBundle((s) => { s.snapshot_id = 'weekly-new'; s.rows[1].projection = 12 })
    const files = { ...old.files, ...newer.files }
    let queue = false
    const release: Array<(value: Awaited<ReturnType<ReturnType<typeof createMemoryFetch>>>) => void> = []
    const fetchImpl = (url: string) => {
      if (url.endsWith('/manifest.json')) {
        if (!queue) return createMemoryFetch(old.files)(url)
        return new Promise<Awaited<ReturnType<ReturnType<typeof createMemoryFetch>>>>((resolve) => { release.push(resolve) })
      }
      return createMemoryFetch(files)(url)
    }
    const user = userEvent.setup()
    render(<StartSitEngine fetchImpl={fetchImpl} />)
    await screen.findByLabelText('Starting slot')
    await user.selectOptions(screen.getByLabelText('Starting slot'), 'RB')
    await nominate(user, 'Synthetic A'); await nominate(user, 'Synthetic B')
    await user.click(screen.getByRole('button', { name: 'Compare nominated alternatives' }))
    expect(screen.getByTestId('quick-result')).toBeInTheDocument()
    queue = true
    fireEvent.focus(window)
    expect(screen.getByTestId('quick-result')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Compare nominated alternatives' })).toBeDisabled()
    fireEvent.focus(window)
    await act(async () => { release[1](await createMemoryFetch(newer.files)('/decision-lab/manifest.json')) })
    await screen.findByText(/Snapshot weekly-new ·/)
    await act(async () => { release[0](await createMemoryFetch(old.files)('/decision-lab/manifest.json')) })
    expect(screen.getByText(/Snapshot weekly-new ·/)).toBeInTheDocument()
    expect(screen.queryByText(/Snapshot weekly-fixture ·/)).not.toBeInTheDocument()
    expect(screen.queryByTestId('quick-result')).not.toBeInTheDocument()
  })
  it('requires fresh availability assumptions when verified evidence changes while preserving nominations', async () => {
    const user = userEvent.setup()
    let files = weeklyBundle((s) => { s.rows[1].projection = 13 }).files
    const fetchImpl = (url: string) => createMemoryFetch(files)(url)
    render(<StartSitEngine fetchImpl={fetchImpl} />)
    await screen.findByLabelText('Starting slot')
    await user.selectOptions(screen.getByLabelText('Starting slot'), 'RB')
    await nominate(user, 'Synthetic A'); await nominate(user, 'Synthetic B')
    for (const name of ['Synthetic A', 'Synthetic B']) {
      await user.selectOptions(screen.getByLabelText(`Availability assumption for ${name}`), 'assumed_available')
    }
    await user.click(screen.getByRole('button', { name: 'Compare nominated alternatives' }))
    expect(within(screen.getByTestId('quick-result')).getByRole('heading')).toHaveTextContent('Synthetic A has the higher projection')
    // An unchanged refresh does not revoke an intentional assumption about the same evidence.
    fireEvent.focus(window)
    await waitFor(() => expect(screen.getByRole('button', { name: 'Compare nominated alternatives' })).toBeEnabled())
    expect(screen.getByLabelText('Availability assumption for Synthetic A')).toHaveValue('assumed_available')

    files = weeklyBundle((s) => { s.snapshot_id = 'weekly-next-release'; s.rows[1].projection = 12 }).files
    fireEvent.focus(window)
    await screen.findByText(/Snapshot weekly-next-release ·/)
    expect(screen.getByLabelText('Availability assumption for Synthetic A')).toHaveValue('unknown')
    expect(screen.getByLabelText('Availability assumption for Synthetic B')).toHaveValue('unknown')
    expect(screen.getByRole('button', { name: 'Remove Synthetic A' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Remove Synthetic B' })).toBeInTheDocument()
    expect(screen.getByText(/Availability assumptions were cleared|assumptions were cleared/)).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Compare nominated alternatives' }))
    expect(within(screen.getByTestId('quick-result')).getByRole('heading')).toHaveTextContent('Check availability before choosing')
    expect(Object.keys(loadReceipts().receipts)).toHaveLength(0)
  })
  it('a broken same-origin bundle explains failure without an API fallback', async () => {
    const b = weeklyBundle(); b.files['manifest.json'] = '{}'
    mount(b)
    expect(await screen.findByRole('alert')).toHaveTextContent('Recommendation unavailable')
    expect(screen.getByText(/No other week, API forecast, or synthetic data was substituted/)).toBeInTheDocument()
  })
  it('atomically confirms only selected eligible unknowns on the explicit bulk click, preserving exclusions', async () => {
    const user = userEvent.setup()
    mount(weeklyBundle((s) => { s.rows[1].projection = 13 }))
    await screen.findByLabelText('Starting slot')
    await nominate(user, 'Synthetic A'); await nominate(user, 'Synthetic B')
    expect(screen.getByLabelText('Availability assumption for Synthetic A')).toHaveValue('unknown')
    expect(screen.queryByTestId('quick-result')).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Compare assuming these players can play' }))
    expect(screen.getByLabelText('Availability assumption for Synthetic A')).toHaveValue('assumed_available')
    expect(screen.getByLabelText('Availability assumption for Synthetic B')).toHaveValue('assumed_available')
    expect(within(screen.getByTestId('quick-result')).getByRole('heading')).toHaveTextContent('Synthetic A has the higher projection')
    expect(Object.keys(loadReceipts().receipts)).toHaveLength(0)
    await user.selectOptions(screen.getByLabelText('Availability assumption for Synthetic B'), 'unavailable')
    await user.click(screen.getByRole('button', { name: 'Compare assuming these players can play' }))
    expect(screen.getByLabelText('Availability assumption for Synthetic B')).toHaveValue('unavailable')
    expect(within(screen.getByTestId('quick-result')).getByRole('heading')).toHaveTextContent('Only one comparable player remains')
  })
  it('clears confirmations and result when another alternative is nominated', async () => {
    const user = userEvent.setup()
    mount()
    await screen.findByLabelText('Starting slot')
    await nominate(user, 'Synthetic A'); await nominate(user, 'Synthetic B')
    await user.click(screen.getByRole('button', { name: 'Compare assuming these players can play' }))
    expect(within(screen.getByTestId('quick-result')).getByRole('heading')).toHaveTextContent('No clear edge')
    await nominate(user, 'Synthetic C')
    expect(screen.queryByTestId('quick-result')).not.toBeInTheDocument()
    expect(screen.getByLabelText('Availability assumption for Synthetic A')).toHaveValue('unknown')
    expect(Object.keys(loadReceipts().receipts)).toHaveLength(0)
  })
  it('does not substitute latest evidence for a historical transfer', async () => {
    window.history.replaceState({}, '', '/start-sit?snapshot=older-period&players=SYN-A,SYN-B&slot=RB')
    mount()
    expect(await screen.findByRole('alert')).toHaveTextContent(/no week was substituted/i)
    expect(screen.queryByRole('button', { name: 'Compare assuming these players can play' })).not.toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Open matching evidence in Decision Lab' })).toHaveAttribute('href', expect.stringContaining('snapshot=older-period'))
    expect(screen.getByRole('link', { name: 'Open matching evidence in Decision Lab' })).toHaveAttribute('href', expect.stringContaining('slot=RB'))
  })
  it('supports keyboard search selection without asserting availability', async () => {
    const user = userEvent.setup()
    mount()
    const search = await screen.findByRole('combobox', { name: 'Search players to compare' })
    await user.type(search, 'Synthetic A'); await user.keyboard('{Enter}')
    expect(screen.getByRole('button', { name: 'Remove Synthetic A' })).toBeInTheDocument()
    expect(screen.queryByTestId('quick-result')).not.toBeInTheDocument()
    expect(Object.keys(loadReceipts().receipts)).toHaveLength(0)
  })
  it('never changes an explicitly linked period when a later manifest arrives on focus', async () => {
    window.history.replaceState({}, '', '/start-sit?snapshot=weekly-fixture&players=SYN-A,SYN-B&slot=FLEX')
    let files = weeklyBundle().files
    render(<StartSitEngine fetchImpl={(url) => createMemoryFetch(files)(url)} />)
    await screen.findByRole('button', { name: 'Remove Synthetic A' })
    files = weeklyBundle((s) => { s.snapshot_id = 'newer-period'; s.week += 1 }).files
    fireEvent.focus(window)
    expect(await screen.findByRole('alert')).toHaveTextContent('no week was substituted')
    expect(screen.getByRole('link', { name: 'Open matching evidence in Decision Lab' })).toHaveAttribute('href', expect.stringContaining('snapshot=weekly-fixture'))
    expect(screen.queryByRole('button', { name: 'Compare assuming these players can play' })).not.toBeInTheDocument()
  })
})
