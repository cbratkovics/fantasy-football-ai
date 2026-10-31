import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, render, screen, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import userEvent from '@testing-library/user-event'
import { TierBoard } from '../TierBoard'
import { DraftSimulator } from '@/components/draft/DraftSimulator'
import { tierPeriod } from '@/lib/presentation/tiers'
import { getTiers } from '@/lib/api/tiers'
import type { TiersResponse } from '@/lib/api/types'

vi.mock('@/lib/api/tiers', () => ({ getTiers: vi.fn() }))
vi.mock('@/components/ui/PageShell', () => ({ PageShell: ({ title, lede, children }: { title: React.ReactNode; lede: React.ReactNode; children: React.ReactNode }) => <main><h1>{title}</h1><p>{lede}</p>{children}</main> }))
afterEach(cleanup)
beforeEach(() => { localStorage.clear(); window.history.replaceState({}, '', '/tiers') })

const artifact: TiersResponse = {
  position: 'RB', season: 2018, tier_version: 'stored-target-2018',
  method: 'StandardScaler → PCA → GaussianMixture',
  inputs: 'prior-season (season-1) aggregates only; min games 4',
  generated_at_utc: '2029-06-15T12:00:00Z',
  evaluation: { n: 2, spearman: 0.5, within_band_rate: 0.5 },
  tiers: [{ tier: 1, players: [{ player_id: 'one', name: 'Historical Receiver', team: 'OLD', ppg_prev: 12, games_prev: 10, tier_probability: 0.97 }] }],
}
function mount(child: React.ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(<QueryClientProvider client={client}>{child}</QueryClientProvider>)
}

describe('historical tier presentation', () => {
  it('distinguishes target, prior input and evaluated season from later generation', () => {
    expect(tierPeriod(artifact)).toEqual({ targetSeason: 2018, inputSeason: 2017, evaluationSeason: 2018, historical: true, title: 'Historical player groups: 2018' })
    expect(tierPeriod({ ...artifact, evaluation: null, inputs: 'unknown source' })).toEqual({ targetSeason: 2018, inputSeason: null, evaluationSeason: null, historical: false, title: 'Player groups: 2018' })
  })
  it('leads with historical player groups and keeps assignment strength inside collapsed method details', async () => {
    vi.mocked(getTiers).mockResolvedValue(artifact)
    mount(<TierBoard />)
    await screen.findByRole('heading', { name: 'Historical player groups: 2018' })
    expect(screen.getByText(/Based on 2017 production · evaluated on 2018 results/)).toBeInTheDocument()
    expect(screen.getByRole('link', { name: /latest published projections/ })).toHaveAttribute('href', '/predictions')
    const playerLink = screen.getByRole('link', { name: 'Historical Receiver' })
    expect(playerLink.closest('li')).toHaveTextContent('12.0 PPR/game')
    expect(playerLink.closest('li')).not.toHaveTextContent('97%')
    const method = screen.getByText('How these groups were built').closest('details')!
    expect(method.open).toBe(false)
    expect(within(method).getByText(/not their chance of winning/)).toBeInTheDocument()
    const evidence = screen.getByText('Historical evaluation and source evidence').closest('details')!
    expect(evidence.open).toBe(false)
    expect(evidence).toHaveTextContent('2029')
    expect(screen.getByRole('img')).toHaveAccessibleName('RB groups based on 2017 PPR points per game')
    const group = screen.getByRole('heading', { name: 'Group 1' })
    expect(group.compareDocumentPosition(screen.getByRole('img')) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
  })
  it('uses the same historical target in Draft and removes unlabeled assignment percentages from the board', async () => {
    vi.mocked(getTiers).mockImplementation(async (position) => ({ ...artifact, position, tiers: artifact.tiers.map((t) => ({ ...t, players: t.players.map((p) => ({ ...p, player_id: `${position}-${p.player_id}` })) })) }))
    mount(<DraftSimulator />)
    await screen.findByRole('heading', { name: '2018 historical exercise' })
    expect(screen.getByText(/uses 2017 production to group players for 2018/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Start historical exercise' })).toBeInTheDocument()
    expect(screen.queryByText('97%')).not.toBeInTheDocument()
    expect(screen.getByText(/not a current draft ranking/)).toBeInTheDocument()
  })
  it('respects an explicit supported position before saved view preferences', async () => {
    localStorage.setItem('ffai.tiers/view/1', JSON.stringify({ position: 'QB' }))
    window.history.replaceState({}, '', '/tiers?position=WR')
    vi.mocked(getTiers).mockImplementation(async (position) => ({ ...artifact, position }))
    mount(<TierBoard />)
    expect(await screen.findByRole('heading', { name: '2018 season · 1 WR groups' })).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'TE' }))
    expect(await screen.findByRole('heading', { name: '2018 season · 1 TE groups' })).toBeInTheDocument()
    expect(JSON.parse(localStorage.getItem('ffai.tiers/view/1')!)).toEqual({ position: 'TE' })
  })
  it('explains an invalid explicit position', async () => {
    window.history.replaceState({}, '', '/tiers?position=K')
    vi.mocked(getTiers).mockResolvedValue(artifact)
    mount(<TierBoard />)
    expect(await screen.findByText('That position is not supported. Choose QB, RB, WR, or TE below.')).toBeInTheDocument()
  })
  it('shows a bounded group preview with explicit counts and can find a player beyond it', async () => {
    const players = Array.from({ length: 8 }, (_, i) => ({ ...artifact.tiers[0].players[0], player_id: `player-${i}`, name: `Player ${i}` }))
    vi.mocked(getTiers).mockResolvedValue({ ...artifact, tiers: [{ tier: 1, players }] })
    mount(<TierBoard />)
    await screen.findByRole('heading', { name: 'Historical player groups: 2018' })
    expect(screen.getByText('6 of 8 players')).toBeInTheDocument()
    expect(screen.queryByRole('link', { name: 'Player 7' })).not.toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Show all 8 players in group 1' }))
    expect(screen.getByRole('link', { name: 'Player 7' })).toBeInTheDocument()
    await userEvent.type(screen.getByLabelText('Search historical players'), 'Player 7')
    expect(screen.getByText('1 matching of 8 players')).toBeInTheDocument()
    expect(screen.queryByRole('link', { name: 'Player 0' })).not.toBeInTheDocument()
  })
  it('clears an invalid saved position with a small notice rather than using it', async () => {
    localStorage.setItem('ffai.tiers/view/1', JSON.stringify({ position: 'K' }))
    vi.mocked(getTiers).mockResolvedValue(artifact)
    mount(<TierBoard />)
    expect(await screen.findByText('An outdated position preference was cleared.')).toBeInTheDocument()
    expect(JSON.parse(localStorage.getItem('ffai.tiers/view/1')!)).toBeNull()
  })
  it('refuses mixed target seasons instead of combining incompatible draft groups', async () => {
    vi.mocked(getTiers).mockImplementation(async (position) => ({ ...artifact, position, season: position === 'QB' ? 2017 : 2018 }))
    mount(<DraftSimulator />)
    expect(await screen.findByText(/do not belong to one compatible historical artifact/)).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Start historical exercise' })).not.toBeInTheDocument()
  })
})
