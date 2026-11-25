import { useEffect, useState } from 'react'

import { api } from '../lib/api'
import { useI18n } from '../lib/i18n'

interface AuditEvent {
  id: string
  created_at: string
  actor_id: string | null
  action: string
  resource_type: string
  resource_id: string | null
  outcome: 'success' | 'failure'
  ip_address: string | null
  detail: Record<string, unknown> | null
}

export default function AdminAuditPage() {
  const { t, lang } = useI18n()
  const [events, setEvents] = useState<AuditEvent[] | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    api<AuditEvent[]>('/admin/audit-events?limit=200').then(setEvents).catch((e) => setError(String(e.message)))
  }, [])

  const fmt = (iso: string) => new Date(iso).toLocaleString(lang === 'vi' ? 'vi-VN' : 'en-GB')

  return (
    <section>
      <h1>{t('auditTitle')}</h1>
      {error && <p className="form-error" role="alert">{error}</p>}
      <div className="card table-wrap">
        {!events ? <p className="muted">{t('loading')}</p> : (
          <table>
            <thead><tr><th>{t('time')}</th><th>{t('action')}</th><th>{t('outcome')}</th><th>{t('ip')}</th><th>Detail</th></tr></thead>
            <tbody>
              {events.map((e) => (
                <tr key={e.id}>
                  <td className="nowrap">{fmt(e.created_at)}</td>
                  <td><code>{e.action}</code></td>
                  <td><span className={`tag ${e.outcome === 'success' ? 'ok' : 'off'}`}>{e.outcome}</span></td>
                  <td>{e.ip_address}</td>
                  <td className="muted small">{e.detail ? JSON.stringify(e.detail) : ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </section>
  )
}
