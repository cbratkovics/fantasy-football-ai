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

/** Compact product chrome; the task is visible before engineering detail. */
export function PageShell({ title, eyebrow, lede, crumb, wide = false, children }: PageShellProps) {
  return (
    <div className="min-h-screen bg-cream text-ink">
      <Navigation />
      <main className={`product-main mx-auto px-4 pb-16 pt-5 sm:px-8 ${wide ? 'max-w-[1400px]' : 'max-w-6xl'}`}>
        <Breadcrumb items={[{ name: crumb, current: true }]} />
        <header className="product-page-heading mb-6 mt-4">
          {eyebrow && (
            <p className="text-sm text-[#52605d]">{eyebrow}</p>
          )}
          <h1 className="mt-1 text-[28px] font-semibold leading-tight tracking-tight sm:text-[32px]">{title}</h1>
          {lede && <div className="mt-2 max-w-3xl text-base leading-relaxed text-[#52605d]">{lede}</div>}
        </header>
        {children}
      </main>
      <Footer />
    </div>
  )
}
