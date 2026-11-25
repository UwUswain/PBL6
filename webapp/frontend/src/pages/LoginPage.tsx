import { useState, type FormEvent } from 'react'
import { Navigate } from 'react-router-dom'

import { LangSwitch } from '../components/AppShell'
import { ApiError } from '../lib/api'
import { useAuth } from '../lib/auth'
import { useI18n } from '../lib/i18n'
import { homeFor } from '../routes'

export default function LoginPage() {
  const { user, login } = useAuth()
  const { t } = useI18n()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  if (user) return <Navigate to={homeFor(user)} replace />

  async function submit(e: FormEvent) {
    e.preventDefault()
    setBusy(true)
    setError('')
    try {
      await login(username, password)
    } catch (err) {
      if (err instanceof ApiError && err.status === 429) {
        setError(t('errRateLimited', { min: Math.ceil((err.retryAfter ?? 60) / 60) }))
      } else if (err instanceof ApiError && err.status === 401) {
        setError(t('errLogin'))
      } else {
        setError(t('errNetwork'))
      }
      setPassword('')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="auth-screen">
      <header className="topbar"><strong className="brand">{t('appName')}</strong><LangSwitch /></header>
      <main className="auth-card">
        <h1>{t('loginTitle')}</h1>
        <p className="muted">{t('loginHint')}</p>
        <form onSubmit={submit} noValidate>
          <label>{t('username')}
            <input value={username} onChange={(e) => setUsername(e.target.value)} autoComplete="username" required autoFocus />
          </label>
          <label>{t('password')}
            <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} autoComplete="current-password" required />
          </label>
          {error && <p className="form-error" role="alert">{error}</p>}
          <button className="btn-primary" disabled={busy || !username || !password}>{busy ? t('signingIn') : t('signIn')}</button>
        </form>
      </main>
    </div>
  )
}
