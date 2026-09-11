'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { ArrowTopRightOnSquareIcon, Bars3Icon, XMarkIcon } from '@heroicons/react/24/outline'
import { AnimatePresence, motion } from 'framer-motion'
import { ROUTES, SITE } from '@/lib/constants'

const primary = [
  { name: 'Predictions', href: ROUTES.predictions },
  { name: 'Tiers', href: ROUTES.tiers },
  { name: 'Draft', href: ROUTES.draft },
  { name: 'Start/Sit', href: ROUTES.startSit },
  { name: 'Evaluation', href: ROUTES.performance },
]

const secondary = [
  { name: 'How it works', href: ROUTES.howItWorks },
  { name: 'Learn', href: ROUTES.learn },
  { name: 'Help', href: ROUTES.help },
  { name: 'About', href: ROUTES.about },
]

export function Navigation() {
  const pathname = usePathname()
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  useEffect(() => setMobileMenuOpen(false), [pathname])
  useEffect(() => {
    document.body.style.overflow = mobileMenuOpen ? 'hidden' : ''
    return () => {
      document.body.style.overflow = ''
    }
  }, [mobileMenuOpen])

  const all = [...primary, ...secondary]

  return (
    <header className="site-header">
      <nav className="site-nav" aria-label="Primary navigation">
        <Link href="/" className="portfolio-mark" aria-label="Win My League home">
          <span>{SITE.shortName}</span>
          <b>{SITE.tagline}</b>
        </Link>
        <div className="site-links">
          {primary.map((item) => (
            <Link key={item.name} href={item.href} aria-current={pathname === item.href ? 'page' : undefined}>
              {item.name}
            </Link>
          ))}
          <Link href={ROUTES.howItWorks} aria-current={pathname === ROUTES.howItWorks ? 'page' : undefined}>
            How it works
          </Link>
          <a href={SITE.repoUrl} target="_blank" rel="noreferrer">
            Source <ArrowTopRightOnSquareIcon />
          </a>
        </div>
        <button
          className="site-menu-button"
          type="button"
          onClick={() => setMobileMenuOpen(true)}
          aria-expanded={mobileMenuOpen}
          aria-controls="mobile-navigation"
        >
          <span className="sr-only">Open navigation</span>
          <Bars3Icon />
        </button>
      </nav>

      <AnimatePresence>
        {mobileMenuOpen && (
          <motion.div id="mobile-navigation" className="site-menu" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <button className="site-menu-scrim" aria-label="Close navigation" onClick={() => setMobileMenuOpen(false)} />
            <motion.div className="site-menu-panel" initial={{ x: '100%' }} animate={{ x: 0 }} exit={{ x: '100%' }} transition={{ duration: 0.22 }}>
              <div className="site-menu-top">
                <span>Navigate</span>
                <button onClick={() => setMobileMenuOpen(false)}>
                  <span className="sr-only">Close navigation</span>
                  <XMarkIcon />
                </button>
              </div>
              <div className="site-menu-links">
                {all.map((item, index) => (
                  <Link key={item.name} href={item.href}>
                    <small>{String(index + 1).padStart(2, '0')}</small>
                    {item.name}
                  </Link>
                ))}
                <a href={SITE.repoUrl} target="_blank" rel="noreferrer">
                  <small>{String(all.length + 1).padStart(2, '0')}</small>Source <ArrowTopRightOnSquareIcon />
                </a>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </header>
  )
}
