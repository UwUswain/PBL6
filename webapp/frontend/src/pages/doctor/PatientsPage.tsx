import { useEffect, useState, type FormEvent } from 'react'
import { Link } from 'react-router-dom'

import { api, ApiError, type Patient } from '../../lib/api'
import { useI18n, type Key } from '../../lib/i18n'
import { useFormat } from './common'

export default function PatientsPage() {
  const { t } = useI18n()
  const f = useFormat()
  const [q, setQ] = useState('')
  const [patients, setPatients] = useState<Patient[] | null>(null)
  const [form, setForm] = useState({ code: '', birth_year: '', sex: '' })
  const [error, setError] = useState('')

  const load = (query = q) => api<Patient[]>(`/patients?q=${encodeURIComponent(query)}`).then(setPatients)
  useEffect(() => { load('') }, [])  // eslint-disable-line react-hooks/exhaustive-deps

  async function create(e: FormEvent) {
    e.preventDefault()
    setError('')
    try {
      await api('/patients', {
        method: 'POST',
        body: JSON.stringify({ code: form.code.trim(), birth_year: form.birth_year ? Number(form.birth_year) : null, sex: form.sex || null }),
      })
      setForm({ code: '', birth_year: '', sex: '' })
      await load()
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err))
    }
  }

  return (
    <section>
      <h1>{t('patientsTitle')}</h1>
      <form className="card inline-form" onSubmit={create}>
        <h2>{t('addPatient')}</h2>
        <label>{t('patientCode')}
          <input value={form.code} onChange={(e) => setForm({ ...form, code: e.target.value })} pattern="[A-Za-z0-9._\-]+" required aria-describedby="code-hint" />
        </label>
        <label>{t('birthYear')}
          <input type="number" min={1900} max={2100} value={form.birth_year} onChange={(e) => setForm({ ...form, birth_year: e.target.value })} />
        </label>
        <label>{t('sex')}
          <select value={form.sex} onChange={(e) => setForm({ ...form, sex: e.target.value })}>
            <option value="">–</option>
            {(['M', 'F', 'O'] as const).map((s) => <option key={s} value={s}>{t(`sex${s}` as Key)}</option>)}
          </select>
        </label>
        <button className="btn-primary" disabled={!form.code.trim()}>{t('addPatient')}</button>
        <p id="code-hint" className="muted small span-all">{t('patientCodeHint')}</p>
      </form>
      {error && <p className="form-error" role="alert">{error}</p>}

      <div className="toolbar">
        <label className="grow">{t('search')}
          <input type="search" value={q} onChange={(e) => { setQ(e.target.value); load(e.target.value) }} />
        </label>
      </div>

      <div className="card table-wrap">
        {!patients ? <p className="muted pad">{t('loading')}</p> : patients.length === 0 ? <p className="muted pad">{t('noPatients')}</p> : (
          <table>
            <thead><tr><th>{t('patientCode')}</th><th>{t('birthYear')}</th><th>{t('sex')}</th><th>{t('cases')}</th><th>{t('updated')}</th><th /></tr></thead>
            <tbody>
              {patients.map((p) => (
                <tr key={p.id}>
                  <td><strong>{p.code}</strong></td>
                  <td>{p.birth_year ?? '–'}</td>
                  <td>{p.sex ? t(`sex${p.sex}` as Key) : '–'}</td>
                  <td>{p.case_count}</td>
                  <td>{f.date(p.created_at)}</td>
                  <td><Link className="btn-small" to={`/doctor/patients/${p.id}`}>{t('open')}</Link></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </section>
  )
}
