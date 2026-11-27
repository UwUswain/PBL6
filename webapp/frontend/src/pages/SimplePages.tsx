import { Link } from 'react-router-dom'

import { useI18n, type Key } from '../lib/i18n'

export function Message({ text }: { text: Key }) {
  const { t } = useI18n()
  return (
    <main className="page">
      <p>{t(text)}</p>
      <Link to="/">{t('back')}</Link>
    </main>
  )
}
