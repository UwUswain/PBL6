import { useEffect, useState, type FormEvent } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'

import { api, ApiError, type CaseDetail, type CaseSummary, type Patient } from '../../lib/api'
import { useI18n, type Key } from '../../lib/i18n'
import { StatusTag, useFormat } from './common'

export default function PatientPage() {
  const { id = '' } = useParams()
  const { t } = useI18n()
  const f = useFormat()
  const navigate = useNavigate()
  const [patient, setPatient] = useState<Patient | null>(null)
  const [cases, setCases] = useState<CaseSummary[] | null>(null)
  const [form, setForm] = useState({ study_date: '', notes: '' })
  const [error, setError] = useState('')

  useEffect(() => {
    api<Patient>(`/patients/${id}`).then(setPatient).catch((e) => setError(e.message))
    api<CaseSummary[]>(`/patients/${id}/cases`).then(setCases).catch((e) => setError(e.message))
  }, [id])

  async function create(e: FormEvent) {
    e.preventDefault()
    try {
      const c = await api<CaseDetail>(`/patients/${id}/cases`, {
        method: 'POST',
        body: JSON.stringify({ study_date: form.study_date || null, notes: form.notes || null }),
      })
      navigate(`/doctor/cases/${c.id}`)
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err))
    }
  }

  return (
    <section>
      <p><Link to="/doctor">← {t('navPatients')}</Link></p>
      <h1>{t('patientTitle', { code: patient?.code ?? '…' })}</h1>
      {patient && (
        <p className="muted">
          {t('birthYear')}: {patient.birth_year ?? '–'} · {t('sex')}: {patient.sex ? t(`sex${patient.sex}` as Key) : '–'}
        </p>
      )}
      <form className="card inline-form" onSubmit={create}>
        <h2>{t('newCase')}</h2>
        <label>{t('studyDate')}<input type="date" value={form.study_date} onChange={(e) => setForm({ ...form, study_date: e.target.value })} /></label>
        <label className="span-2">{t('notes')}<input value={form.notes} maxLength={2000} onChange={(e) => setForm({ ...form, notes: e.target.value })} /></label>
        <button className="btn-primary">{t('newCase')}</button>
      </form>
      {error && <p className="form-error" role="alert">{error}</p>}
      <div className="card table-wrap">
        {!cases ? <p className="muted pad">{t('loading')}</p> : cases.length === 0 ? <p className="muted pad">{t('noCases')}</p> : (
          <table>
            <thead><tr><th>{t('studyDate')}</th><th>{t('status')}</th><th>{t('notes')}</th><th>{t('updated')}</th><th /></tr></thead>
            <tbody>
              {cases.map((c) => (
                <tr key={c.id}>
                  <td>{f.date(c.study_date)}</td>
                  <td><StatusTag status={c.status} /></td>
                  <td className="muted">{c.notes}</td>
                  <td>{f.time(c.updated_at)}</td>
                  <td><Link className="btn-small" to={`/doctor/cases/${c.id}`}>{t('open')}</Link></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </section>
  )
}
