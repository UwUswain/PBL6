import { useEffect, useRef, useState, type FormEvent } from 'react'

import { api, ApiError, type Role, type User } from '../lib/api'
import { useAuth } from '../lib/auth'
import { useI18n, type Key } from '../lib/i18n'

const ROLE_LABEL: Record<Role, Key> = { admin: 'roleAdmin', doctor: 'roleDoctor', patient: 'rolePatient' }

interface Created { user: User; temporary_password: string }

export default function AdminUsersPage() {
  const { user: me } = useAuth()
  const { t } = useI18n()
  const [users, setUsers] = useState<User[] | null>(null)
  const [error, setError] = useState('')
  const [form, setForm] = useState({ username: '', display_name: '', role: 'doctor' as Role })
  const [temp, setTemp] = useState<Created | null>(null)
  const dialog = useRef<HTMLDialogElement>(null)

  const load = () => api<User[]>('/admin/users').then(setUsers).catch((e) => setError(String(e.message)))
  useEffect(() => { load() }, [])
  useEffect(() => { if (temp) dialog.current?.showModal() }, [temp])

  async function run(action: () => Promise<unknown>) {
    setError('')
    try { await action(); await load() } catch (e) { setError(e instanceof ApiError ? e.message : String(e)) }
  }

  async function create(e: FormEvent) {
    e.preventDefault()
    await run(async () => {
      setTemp(await api<Created>('/admin/users', { method: 'POST', body: JSON.stringify(form) }))
      setForm({ username: '', display_name: '', role: 'doctor' })
    })
  }

  const toggle = (u: User) => run(() => api(`/admin/users/${u.id}`, { method: 'PATCH', body: JSON.stringify({ is_active: !u.is_active }) }))
  const reset = (u: User) => {
    if (!window.confirm(t('confirmReset', { user: u.username }))) return
    run(async () => setTemp(await api<Created>(`/admin/users/${u.id}/reset-password`, { method: 'POST' })))
  }

  return (
    <section>
      <h1>{t('usersTitle')}</h1>

      <form className="card inline-form" onSubmit={create}>
        <h2>{t('createUser')}</h2>
        <label>{t('username')}
          <input value={form.username} onChange={(e) => setForm({ ...form, username: e.target.value })} pattern="[A-Za-z0-9._\-]{3,64}" required />
        </label>
        <label>{t('displayName')}
          <input value={form.display_name} onChange={(e) => setForm({ ...form, display_name: e.target.value })} required />
        </label>
        <label>{t('role')}
          <select value={form.role} onChange={(e) => setForm({ ...form, role: e.target.value as Role })}>
            {(['doctor', 'patient', 'admin'] as Role[]).map((r) => <option key={r} value={r}>{t(ROLE_LABEL[r])}</option>)}
          </select>
        </label>
        <button className="btn-primary" disabled={!form.username || !form.display_name}>{t('createUser')}</button>
      </form>

      {error && <p className="form-error" role="alert">{error}</p>}

      <div className="card table-wrap">
        {!users ? <p className="muted">{t('loading')}</p> : (
          <table>
            <thead>
              <tr><th>{t('username')}</th><th>{t('displayName')}</th><th>{t('role')}</th><th>{t('status')}</th><th>{t('actions')}</th></tr>
            </thead>
            <tbody>
              {users.map((u) => (
                <tr key={u.id}>
                  <td>{u.username}{u.id === me?.id && <span className="muted"> ({t('you')})</span>}</td>
                  <td>{u.display_name}</td>
                  <td><span className={`tag role-${u.role}`}>{t(ROLE_LABEL[u.role])}</span></td>
                  <td>
                    <span className={`tag ${u.is_active ? 'ok' : 'off'}`}>{t(u.is_active ? 'active' : 'disabled')}</span>
                    {u.must_change_password && <span className="tag warn">{t('mustChange')}</span>}
                  </td>
                  <td className="row-actions">
                    {u.id !== me?.id && <button type="button" className="btn-small" onClick={() => toggle(u)}>{t(u.is_active ? 'disable' : 'enable')}</button>}
                    <button type="button" className="btn-small" onClick={() => reset(u)}>{t('resetPassword')}</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <dialog ref={dialog} onClose={() => setTemp(null)} aria-labelledby="temp-title">
        {temp && (
          <>
            <h2 id="temp-title">{t('tempPasswordFor', { user: temp.user.username })}</h2>
            <p className="temp-password">{temp.temporary_password}</p>
            <p className="muted small">{t('tempPasswordNote')}</p>
            <div className="row-actions">
              <button type="button" className="btn-small" onClick={() => navigator.clipboard?.writeText(temp.temporary_password)}>{t('copy')}</button>
              <button type="button" className="btn-primary" onClick={() => dialog.current?.close()}>{t('close')}</button>
            </div>
          </>
        )}
      </dialog>
    </section>
  )
}
