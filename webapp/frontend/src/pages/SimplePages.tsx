import { Link } from 'react-router-dom'

import { useI18n, type Key } from '../lib/i18n'

// shortcut: doctor and patient homes are placeholders until milestones M2 (cases) and M3 (patient portal).
export function Placeholder({ title, milestone }: { title: Key; milestone: string }) {
  const { t } = useI18n()
  return (
    <section>
      <h1>{t(title)}</h1>
      <p className="card muted">{t('comingSoon', { m: milestone })}</p>
    </section>
  )
}

export function Message({ text }: { text: Key }) {
  const { t } = useI18n()
  return (
    <main className="page">
      <p>{t(text)}</p>
      <Link to="/">{t('back')}</Link>
    </main>
  )
}
