import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'

import { api, type CaseSummary } from '../../lib/api'
import { useI18n } from '../../lib/i18n'
import { StatusTag, useFormat } from './common'

export default function ReviewPage() {
  const { t } = useI18n()
  const f = useFormat()
  const [cases, setCases] = useState<CaseSummary[] | null>(null)
  useEffect(() => { api<CaseSummary[]>('/cases?status=awaiting_review').then(setCases) }, [])
  return (
    <section>
      <h1>{t('reviewTitle')}</h1>
      <div className="card table-wrap">
        {!cases ? <p className="muted pad">{t('loading')}</p> : cases.length === 0 ? <p className="muted pad">{t('noReview')}</p> : (
          <table>
            <thead><tr><th>{t('patient')}</th><th>{t('studyDate')}</th><th>{t('status')}</th><th>{t('updated')}</th><th /></tr></thead>
            <tbody>
              {cases.map((c) => (
                <tr key={c.id}>
                  <td><strong>{c.patient_code}</strong></td>
                  <td>{f.date(c.study_date)}</td>
                  <td><StatusTag status={c.status} /></td>
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
