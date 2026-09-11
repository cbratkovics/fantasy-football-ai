// Single entry point for every request to the ffai FastAPI service.
// The API serves committed artifacts only; there is no auth and no write path.

export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:7860'

export class ApiError extends Error {
  status: number
  detail: string

  constructor(status: number, detail: string) {
    super(`${status}: ${detail}`)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
  }
}

type Params = Record<string, string | number | undefined>

export async function apiGet<T>(path: string, params?: Params): Promise<T> {
  const url = new URL(path, API_BASE_URL)
  if (params) {
    for (const [key, value] of Object.entries(params)) {
      if (value !== undefined && value !== '') url.searchParams.set(key, String(value))
    }
  }
  let response: Response
  try {
    response = await fetch(url.toString(), { headers: { Accept: 'application/json' } })
  } catch (error) {
    throw new ApiError(0, `could not reach the API at ${API_BASE_URL}`)
  }
  if (!response.ok) {
    let detail = response.statusText
    try {
      const body = await response.json()
      if (body && typeof body.detail === 'string') detail = body.detail
    } catch {
      // non-JSON error body; keep the status text
    }
    throw new ApiError(response.status, detail)
  }
  return (await response.json()) as T
}

export function describeError(error: unknown): string {
  if (error instanceof ApiError) return error.detail
  if (error instanceof Error) return error.message
  return 'unknown error'
}
