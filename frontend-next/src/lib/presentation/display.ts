/** View-only formatting/preferences. Never stores availability, actions, or policy evidence. */
export function formatLocalDate(value: string | null | undefined): string {
  if (!value || !Number.isFinite(Date.parse(value))) return 'Update time unknown'
  return new Intl.DateTimeFormat(undefined, { month: 'short', day: 'numeric', year: 'numeric', hour: 'numeric', minute: '2-digit', timeZoneName: 'short' }).format(new Date(value))
}
export function formatPeriod(season: number, week: number): string { return `Week ${week}, ${season}` }
export function readPreference(key: string): unknown {
  try { return JSON.parse(localStorage.getItem(key) ?? 'null') } catch { return null }
}
export function writePreference(key: string, value: unknown): void {
  try { localStorage.setItem(key, JSON.stringify(value)) } catch { /* Optional view preferences. */ }
}
export function selectionHref(path: '/start-sit' | '/decision-lab', snapshotId: string, playerIds: string[], slot = 'FLEX'): string {
  const query = new URLSearchParams({ snapshot: snapshotId, players: playerIds.join(','), slot })
  return `${path}?${query.toString()}`
}
