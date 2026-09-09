'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { useAuth } from '@clerk/nextjs'
import { ArrowTopRightOnSquareIcon, Bars3Icon, XMarkIcon } from '@heroicons/react/24/outline'
import { AnimatePresence, motion } from 'framer-motion'

const navigation = [
  { name: 'Case study', href: '/#case-study' },
  { name: 'System', href: '/#system' },
  { name: 'Evaluation', href: '/performance' },
]

export function Navigation() {
  const { isSignedIn } = useAuth()
  const pathname = usePathname()
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  useEffect(() => setMobileMenuOpen(false), [pathname])
  useEffect(() => {
    document.body.style.overflow = mobileMenuOpen ? 'hidden' : ''
    return () => { document.body.style.overflow = '' }
  }, [mobileMenuOpen])

  return (
    <header className="site-header">
      <nav className="site-nav" aria-label="Primary navigation">
        <Link href="/" className="portfolio-mark" aria-label="Win My League home">
          <span>WML</span><b>Decision Lab</b>
        </Link>
        <div className="site-links">
          {navigation.map((item) => (
            <Link key={item.name} href={item.href} aria-current={pathname === item.href ? 'page' : undefined}>{item.name}</Link>
          ))}
          <a href="https://github.com/cbratkovics/fantasy-football-ai" target="_blank" rel="noreferrer">
            Source <ArrowTopRightOnSquareIcon />
          </a>
          <Link className="site-nav-cta" href={isSignedIn ? '/dashboard' : '/auth/signin'}>
            {isSignedIn ? 'Dashboard' : 'Sign in'}
          </Link>
        </div>
        <button className="site-menu-button" type="button" onClick={() => setMobileMenuOpen(true)} aria-expanded={mobileMenuOpen} aria-controls="mobile-navigation">
          <span className="sr-only">Open navigation</span><Bars3Icon />
        </button>
      </nav>

      <AnimatePresence>
        {mobileMenuOpen && (
          <motion.div id="mobile-navigation" className="site-menu" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <button className="site-menu-scrim" aria-label="Close navigation" onClick={() => setMobileMenuOpen(false)} />
            <motion.div className="site-menu-panel" initial={{ x: '100%' }} animate={{ x: 0 }} exit={{ x: '100%' }} transition={{ duration: .22 }}>
              <div className="site-menu-top"><span>Navigate</span><button onClick={() => setMobileMenuOpen(false)}><span className="sr-only">Close navigation</span><XMarkIcon /></button></div>
              <div className="site-menu-links">
                {navigation.map((item, index) => <Link key={item.name} href={item.href}><small>0{index + 1}</small>{item.name}</Link>)}
                <a href="https://github.com/cbratkovics/fantasy-football-ai" target="_blank" rel="noreferrer"><small>04</small>Source <ArrowTopRightOnSquareIcon /></a>
              </div>
              <Link className="portfolio-button primary" href={isSignedIn ? '/dashboard' : '/auth/signin'}>{isSignedIn ? 'Open dashboard' : 'Sign in'}</Link>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </header>
  )
}
