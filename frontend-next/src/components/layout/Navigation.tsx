'use client'

import { useEffect, useRef } from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { ROUTES, SITE } from '@/lib/constants'

const primary = [
  { name: 'Start/Sit', href: ROUTES.startSit },
  { name: 'Player projections', href: ROUTES.predictions },
]
const explore = [
  { name: 'Decision Lab', href: ROUTES.decisionLab },
  { name: 'Model results', href: ROUTES.performance },
  { name: 'Historical tiers', href: ROUTES.tiers },
  { name: 'Historical draft practice', href: ROUTES.draft },
  { name: 'Data platform', href: ROUTES.dataPlatform },
  { name: 'How it works', href: ROUTES.howItWorks },
  { name: 'Learn', href: ROUTES.learn },
  { name: 'Help', href: ROUTES.help },
  { name: 'About', href: ROUTES.about },
]
export function Navigation() {
  const pathname = usePathname()
  const menu = useRef<HTMLDetailsElement>(null)
  useEffect(() => { if (menu.current) menu.current.open = false }, [pathname])
  return <header className="product-header">
    <nav aria-label="Primary navigation" className="product-nav">
      <Link href="/" className="product-brand" aria-label="Win My League home"><span>WML</span><b>Decision Lab</b></Link>
      <div className="product-nav-links">
        {primary.map((item) => <Link key={item.href} href={item.href} aria-current={pathname === item.href ? 'page' : undefined}>{item.name}</Link>)}
        <details ref={menu} className="product-explore" onKeyDown={(e) => { if (e.key === 'Escape' && menu.current) { menu.current.open = false; menu.current.querySelector('summary')?.focus() } }}>
          <summary>Explore</summary>
          <div className="product-explore-menu">
            {explore.map((item) => <Link key={item.href} href={item.href} aria-current={pathname === item.href ? 'page' : undefined}>{item.name}</Link>)}
            <a href={SITE.repoUrl} target="_blank" rel="noreferrer">Source ↗</a>
          </div>
        </details>
      </div>
    </nav>
  </header>
}
