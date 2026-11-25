import { useState, type FormEvent } from 'react'
import { useNavigate } from 'react-router-dom'

import { LangSwitch } from '../components/AppShell'
import { ApiError } from '../lib/api'
import { useAuth } from '../lib/auth'
import { useI18n } from '../lib/i18n'
import { homeFor } from '../routes'

export default function ChangePasswordPage() {
  const { user, changePassword, logout } = useAuth()
  const { t } = useI18n()
  const navigate = useNavigate()
  const [current, setCurrent] = useState('')
  const [next, setNext] = useState('')
  const [confirm, setConfirm] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  async function submit(e: FormEvent) {
    e.preventDefault()
    if (next !== confirm) { setError(t('errMismatch')); return }
    setBusy(true)
    setError('')
    try {
      const me = await changePassword(current, next)
      navigate(homeFor(me), { replace: true })
    } catch (err) {
      setError(err instanceof ApiError ? err.message : t('errNetwork'))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="auth-screen">
      <header className="topbar"><strong className="brand">{t('appName')}</strong><LangSwitch /></header>
      <main className="auth-card">
        <h1>{t('changeTitle')}</h1>
        {user?.must_change_password && <p className="notice">{t('changeForced')}</p>}
        <form onSubmit={submit} noValidate>
          <label>{t('currentPassword')}
            <input type="password" value={current} onChange={(e) => setCurrent(e.target.value)} autoComplete="current-password" required autoFocus />
          </label>
          <label>{t('newPassword')}
            <input type="password" value={next} onChange={(e) => setNext(e.target.value)} autoComplete="new-password" minLength={10} required aria-describedby="pw-rule" />
          </label>
          <p id="pw-rule" className="muted small">{t('passwordRule')}</p>
          <label>{t('confirmPassword')}
            <input type="password" value={confirm} onChange={(e) => setConfirm(e.target.value)} autoComplete="new-password" required />
          </label>
          {error && <p className="form-error" role="alert">{error}</p>}
          <button className="btn-primary" disabled={busy || !current || !next || !confirm}>{busy ? t('saving') : t('save')}</button>
          <button type="button" className="btn-ghost dark" onClick={logout}>{t('signOut')}</button>
        </form>
      </main>
    </div>
  )
}
