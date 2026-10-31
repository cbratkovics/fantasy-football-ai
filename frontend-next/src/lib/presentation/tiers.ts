import type { TiersResponse } from '@/lib/api/types'

/** Display metadata only; never changes tier order or interprets assignment as a forecast. */
export function tierPeriod(data: Pick<TiersResponse, 'season' | 'inputs' | 'evaluation'>) {
  const inputSeason = /prior.season|season\s*-\s*1/i.test(data.inputs) ? data.season - 1 : null
  const historical = data.evaluation !== null
  return {
    targetSeason: data.season,
    inputSeason,
    evaluationSeason: historical ? data.season : null,
    historical,
    title: `${historical ? 'Historical player groups' : 'Player groups'}: ${data.season}`,
  }
}
