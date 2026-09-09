import Link from 'next/link'
import { ArrowUpRightIcon } from '@heroicons/react/24/outline'

export function Footer() {
  return (
    <footer className="site-footer">
      <div className="site-footer-main">
        <div className="site-footer-brand">
          <span className="footer-mark">WML</span>
          <div><b>Win My League · Decision Lab</b><p>An applied data science portfolio project by Christopher Bratkovics.</p></div>
        </div>
        <nav aria-label="Footer navigation">
          <div><small>Explore</small><Link href="/#case-study">Case study</Link><Link href="/#system">System design</Link><Link href="/performance">Evaluation</Link></div>
          <div><small>Connect</small><a href="mailto:chris@fantasyfootballai.com">Email <ArrowUpRightIcon /></a><a href="https://github.com/cbratkovics" target="_blank" rel="noreferrer">GitHub <ArrowUpRightIcon /></a><a href="https://linkedin.com/in/cbratkovics" target="_blank" rel="noreferrer">LinkedIn <ArrowUpRightIcon /></a></div>
          <div><small>Project</small><Link href="/about">About</Link><Link href="/privacy">Privacy</Link><Link href="/terms">Terms</Link></div>
        </nav>
      </div>
      <div className="site-footer-meta"><span>© {new Date().getFullYear()} Win My League</span><span>Built with transparent assumptions.</span></div>
    </footer>
  )
}
