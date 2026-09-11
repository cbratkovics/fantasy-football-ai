import { Navigation } from '@/components/layout/Navigation'
import { Footer } from '@/components/layout/Footer'
import { Breadcrumb } from '@/components/layout/Breadcrumb'

interface PageShellProps {
  title: React.ReactNode
  eyebrow?: string
  lede?: React.ReactNode
  crumb: string
  wide?: boolean
  children: React.ReactNode
}

/** Shared chrome for every non-home route: cream background, mono eyebrow, oversized title. */
export function PageShell({ title, eyebrow, lede, crumb, wide = false, children }: PageShellProps) {
  return (
    <div className="min-h-screen bg-cream text-ink">
      <Navigation />
      <main className={`mx-auto px-6 pb-24 pt-10 lg:px-10 ${wide ? 'max-w-[1400px]' : 'max-w-5xl'}`}>
        <Breadcrumb items={[{ name: crumb, current: true }]} />
        <header className="mb-10 mt-8 border-b border-[#c5c7c0] pb-8">
          {eyebrow && (
            <div className="eyebrow">
              <span /> {eyebrow}
            </div>
          )}
          <h1 className="mt-5 text-4xl font-semibold leading-none tracking-[-0.05em] sm:text-6xl">{title}</h1>
          {lede && <div className="mt-5 max-w-3xl text-base leading-7 text-[#52605d]">{lede}</div>}
        </header>
        {children}
      </main>
      <Footer />
    </div>
  )
}
