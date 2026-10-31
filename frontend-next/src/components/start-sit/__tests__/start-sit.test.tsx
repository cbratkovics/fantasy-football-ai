import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it } from 'vitest'
import { createMemoryFetch } from '@/lib/decision-lab/bundle'
import { loadReceipts } from '@/lib/decision-lab/storage'
import { weeklyBundle } from '@/lib/start-sit/__tests__/fixtures'
import { StartSitEngine } from '../StartSitEngine'

beforeEach(() => { localStorage.clear(); sessionStorage.clear() })
function mount(b = weeklyBundle()) { return render(<StartSitEngine fetchImpl={createMemoryFetch(b.files)} now={() => '2099-09-01T00:00:00Z'} />) }
async function nominate(user: ReturnType<typeof userEvent.setup>, name: string) {
  await user.type(screen.getByLabelText('Search players to compare'), name)
  await user.click(screen.getByRole('button', { name: new RegExp(`Add ${name}`) }))
}

describe('Start/Sit verified comparison', () => {
  it('requires a starting slot and labels dated PPR evidence', async () => {
    mount()
    expect(await screen.findByLabelText('Starting slot')).toHaveValue('')
    expect(screen.getByRole('heading', { name: /Dated comparison/ })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Compare nominated alternatives' })).toBeDisabled()
    expect(screen.queryByRole('button', { name: 'half' })).not.toBeInTheDocument()
  })
  it('requires intentional availability assumptions; computing never records the suggestion as action', async () => {
    const user = userEvent.setup()
    mount(weeklyBundle((s) => { s.rows[1].projection = 13 }))
    await screen.findByLabelText('Starting slot')
    await user.selectOptions(screen.getByLabelText('Starting slot'), 'RB')
    await nominate(user, 'Synthetic A'); await nominate(user, 'Synthetic B')
    await user.click(screen.getByRole('button', { name: 'Compare nominated alternatives' }))
    expect(within(screen.getByTestId('quick-result')).getByRole('heading')).toHaveTextContent('Check player status')
    expect(Object.keys(loadReceipts().receipts)).toHaveLength(0)
    await user.selectOptions(screen.getByLabelText('Availability assumption for Synthetic A'), 'assumed_available')
    await user.selectOptions(screen.getByLabelText('Availability assumption for Synthetic B'), 'assumed_available')
    await user.click(screen.getByRole('button', { name: 'Compare nominated alternatives' }))
    expect(within(screen.getByTestId('quick-result')).getByRole('heading')).toHaveTextContent('Suggested starter')
    expect(screen.getAllByRole('radio').filter((r) => (r as HTMLInputElement).name === 'chosen').every((r) => !(r as HTMLInputElement).checked)).toBe(true)
    expect(Object.keys(loadReceipts().receipts)).toHaveLength(0)
  })
  it('keeps a missing selected ID visible after navigation and withholds reduced-set ranking', async () => {
    sessionStorage.setItem('ffai.start-sit/draft/1', JSON.stringify({ slot: 'RB', selected: ['SYN-A', 'missing', 'SYN-B'] }))
    mount()
    expect(await screen.findByText(/selected projections are missing/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Remove missing' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Compare nominated alternatives' })).toBeDisabled()
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
  it('suspends a result during refresh and ignores an older request that finishes last', async () => {
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
    expect(screen.queryByTestId('quick-result')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Compare nominated alternatives' })).toBeDisabled()
    fireEvent.focus(window)
    await act(async () => { release[1](await createMemoryFetch(newer.files)('/decision-lab/manifest.json')) })
    await screen.findByText(/weekly-new ·/)
    await act(async () => { release[0](await createMemoryFetch(old.files)('/decision-lab/manifest.json')) })
    expect(screen.getByText(/weekly-new ·/)).toBeInTheDocument()
    expect(screen.queryByText(/weekly-fixture ·/)).not.toBeInTheDocument()
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
    expect(within(screen.getByTestId('quick-result')).getByRole('heading')).toHaveTextContent('Suggested starter')
    // An unchanged refresh does not revoke an intentional assumption about the same evidence.
    fireEvent.focus(window)
    await waitFor(() => expect(screen.getByRole('button', { name: 'Compare nominated alternatives' })).toBeEnabled())
    expect(screen.getByLabelText('Availability assumption for Synthetic A')).toHaveValue('assumed_available')

    files = weeklyBundle((s) => { s.snapshot_id = 'weekly-next-release'; s.rows[1].projection = 12 }).files
    fireEvent.focus(window)
    await screen.findByText(/weekly-next-release ·/)
    expect(screen.getByLabelText('Availability assumption for Synthetic A')).toHaveValue('unknown')
    expect(screen.getByLabelText('Availability assumption for Synthetic B')).toHaveValue('unknown')
    expect(screen.getByRole('button', { name: 'Remove Synthetic A' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Remove Synthetic B' })).toBeInTheDocument()
    expect(screen.getByText(/Availability assumptions were cleared/)).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Compare nominated alternatives' }))
    expect(within(screen.getByTestId('quick-result')).getByRole('heading')).toHaveTextContent('Check player status')
    expect(Object.keys(loadReceipts().receipts)).toHaveLength(0)
  })
  it('a broken same-origin bundle explains failure without an API fallback', async () => {
    const b = weeklyBundle(); b.files['manifest.json'] = '{}'
    mount(b)
    expect(await screen.findByRole('alert')).toHaveTextContent('Recommendation unavailable')
    expect(screen.getByText(/No other week, API forecast, or synthetic data was substituted/)).toBeInTheDocument()
  })
})
