import { NavLink, Outlet } from 'react-router-dom'

import type { Role } from '../lib/api'
import { useAuth } from '../lib/auth'
import { useI18n, type Key } from '../lib/i18n'

const NAV: Record<Role, { to: string; label: Key }[]> = {
  admin: [{ to: '/admin/users', label: 'navUsers' }, { to: '/admin/audit', label: 'navAudit' }],
  doctor: [{ to: '/doctor', label: 'navPatients' }, { to: '/doctor/review', label: 'navReview' }],
  patient: [{ to: '/patient', label: 'navMyResults' }],
}
const ROLE_LABEL: Record<Role, Key> = { admin: 'roleAdmin', doctor: 'roleDoctor', patient: 'rolePatient' }

export function LangSwitch() {
  const { lang, setLang } = useI18n()
  return (
    <div className="lang" role="group" aria-label="Language">
      {(['vi', 'en'] as const).map((l) => (
        <button key={l} type="button" aria-pressed={lang === l} onClick={() => setLang(l)}>{l.toUpperCase()}</button>
      ))}
    </div>
  )
}

export default function AppShell() {
  const { user, logout } = useAuth()
  const { t } = useI18n()
  if (!user) return null
  return (
    <>
      <a className="skip-link" href="#main">Skip to content</a>
      <header className="topbar">
        <strong className="brand">{t('appName')}</strong>
        <nav aria-label="Main">
          {NAV[user.role].map((n) => (
            <NavLink key={n.to} to={n.to} end>{t(n.label)}</NavLink>
          ))}
        </nav>
        <div className="topbar-right">
          <span className="who">{user.display_name} · {t(ROLE_LABEL[user.role])}</span>
          <LangSwitch />
          <button type="button" className="btn-ghost" onClick={logout}>{t('signOut')}</button>
        </div>
      </header>
      <main id="main" className="page">
        <Outlet />
      </main>
    </>
  )
}
