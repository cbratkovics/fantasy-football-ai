import Link from 'next/link'
import { ArrowUpRightIcon } from '@heroicons/react/24/outline'
import { ROUTES, SITE } from '@/lib/constants'

export function Footer() {
  return (
    <footer className="site-footer">
      <div className="site-footer-main">
        <div className="site-footer-brand">
          <span className="footer-mark">{SITE.shortName}</span>
          <div>
            <b>
              {SITE.name} · {SITE.tagline}
            </b>
            <p>An applied data science portfolio project by Christopher Bratkovics. Every figure on this site is read from a committed evaluation artifact.</p>
          </div>
        </div>
        <nav aria-label="Footer navigation">
          <div>
            <small>Tools</small>
            <Link href={ROUTES.predictions}>Predictions</Link>
            <Link href={ROUTES.tiers}>Tiers</Link>
            <Link href={ROUTES.draft}>Draft board</Link>
            <Link href={ROUTES.startSit}>Start/Sit</Link>
            <Link href={ROUTES.performance}>Evaluation</Link>
          </div>
          <div>
            <small>Read</small>
            <Link href={ROUTES.howItWorks}>How it works</Link>
            <Link href={ROUTES.learn}>Learn</Link>
            <Link href={ROUTES.help}>Help</Link>
            <Link href={ROUTES.about}>About</Link>
          </div>
          <div>
            <small>Project</small>
            <a href={`mailto:${SITE.supportEmail}`}>
              Email support <ArrowUpRightIcon />
            </a>
            <a href={SITE.repoUrl} target="_blank" rel="noreferrer">
              GitHub <ArrowUpRightIcon />
            </a>
            <a href={SITE.authorLinkedin} target="_blank" rel="noreferrer">
              LinkedIn <ArrowUpRightIcon />
            </a>
            <Link href={ROUTES.privacy}>Privacy</Link>
            <Link href={ROUTES.terms}>Terms</Link>
          </div>
        </nav>
      </div>
      <div className="site-footer-meta">
        <span>© {new Date().getFullYear()} {SITE.name}</span>
        <span>Built with transparent assumptions.</span>
      </div>
    </footer>
  )
}
