import Link from 'next/link'
import { ROUTES } from '@/lib/constants'

export default function NotFound() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-cream px-6 text-ink">
      <main className="max-w-lg border border-[#c5c7c0] bg-[#f8f7f2] p-10">
        <p className="font-mono text-[10px] font-bold uppercase tracking-widest text-ember">Not found</p>
        <h1 className="mt-4 text-4xl font-semibold tracking-[-0.04em]">There is nothing at this address.</h1>
        <p className="mt-4 text-sm leading-6 text-[#52605d]">
          The route may have been removed in the rebuild, or a player or week you asked for has no published artifact.
        </p>
        <div className="mt-8 flex flex-wrap gap-3">
          <Link href={ROUTES.home} className="portfolio-button primary">Home</Link>
          <Link href={ROUTES.predictions} className="portfolio-button secondary">Predictions</Link>
        </div>
      </main>
    </div>
  )
}
