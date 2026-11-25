import { useCallback, useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'

import { api, ApiError, downloadFile, type CaseDetail, type MaskVersion, type ModelInfo } from '../../lib/api'
import { useI18n, type Key } from '../../lib/i18n'
import { StatusTag, useFormat } from './common'

const SEQUENCES = ['dwi', 'adc', 'flair'] as const
const ACTIVE = ['queued', 'running']

export default function CasePage() {
  const { id = '' } = useParams()
  const { t, lang } = useI18n()
  const f = useFormat()
  const [c, setCase] = useState<CaseDetail | null>(null)
  const [models, setModels] = useState<ModelInfo[]>([])
  const [model, setModel] = useState('')
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState('')
  const [approval, setApproval] = useState({ mask: '', note: '', show: true })

  const load = useCallback(() => api<CaseDetail>(`/cases/${id}`).then((d) => {
    setCase(d)
    setApproval((a) => ({ ...a, mask: a.mask || d.masks[0]?.id || '' }))
  }), [id])

  useEffect(() => {
    load().catch((e) => setError(e.message))
    api<ModelInfo[]>('/models').then((ms) => {
      setModels(ms)
      setModel(ms.find((m) => m.is_default && m.available)?.name ?? ms.find((m) => m.available)?.name ?? '')
    })
  }, [load])

  const activeJob = c?.jobs.find((j) => ACTIVE.includes(j.state))
  useEffect(() => {  // poll while the worker has the job
    if (!activeJob) return
    const timer = setInterval(() => load(), 2000)
    return () => clearInterval(timer)
  }, [activeJob, load])

  async function act(key: string, fn: () => Promise<unknown>) {
    setBusy(key)
    setError('')
    try { await fn(); await load() } catch (e) { setError(e instanceof ApiError ? e.message : String(e)) } finally { setBusy(null) }
  }

  const upload = (seq: string, file: File) => act(`up-${seq}`, () => {
    const body = new FormData()
    body.append('file', file, file.name)
    return api(`/cases/${id}/volumes/${seq}`, { method: 'PUT', body })
  })

  if (!c) return <p className="muted" role="status">{error || t('loading')}</p>
  const selectedModel = models.find((m) => m.name === model)
  const have = new Set(c.volumes.map((v) => v.modality))
  const missing = selectedModel?.required.filter((r) => !have.has(r as 'dwi')) ?? []
  const approvedMask = c.approval ? c.masks.find((m) => m.id === c.approval!.mask_version_id) : null

  return (
    <section className="case-page">
      <p><Link to={`/doctor/patients/${c.patient_id}`}>← {t('patientTitle', { code: c.patient_code })}</Link></p>
      <div className="case-head">
        <h1>{t('caseTitle', { code: c.patient_code })}</h1>
        <StatusTag status={c.status} />
      </div>
      <p className="muted">{t('studyDate')}: {f.date(c.study_date)}{c.notes ? ` · ${c.notes}` : ''}</p>
      {error && <p className="form-error" role="alert">{error}</p>}

      <div className="card">
        <h2>{t('stepImages')}</h2>
        {c.inputs_locked && <p className="notice">{t('locked')}</p>}
        <div className="seq-grid">
          {SEQUENCES.map((seq) => {
            const v = c.volumes.find((x) => x.modality === seq)
            return (
              <div key={seq} className={`seq ${v ? 'has' : ''}`}>
                <strong>{seq.toUpperCase()}</strong>{seq === 'flair' && <span className="muted small"> ({t('optional')})</span>}
                <p className="muted small">{v ? `${v.shape.join('×')} · ${v.spacing_mm.map((s) => s.toFixed(1)).join('×')} mm` : '–'}</p>
                {!c.inputs_locked && (
                  <label className="btn-small file-btn">
                    {busy === `up-${seq}` ? t('uploading') : v ? t('replace') : t('upload')}
                    <input type="file" accept=".nii,.nii.gz,.gz" disabled={!!busy}
                      onChange={(e) => { const file = e.target.files?.[0]; if (file) upload(seq, file); e.target.value = '' }} />
                  </label>
                )}
              </div>
            )
          })}
        </div>
      </div>

      <div className="card">
        <h2>{t('stepRun')}</h2>
        <div className="run-row">
          <label className="grow">{t('model')}
            <select value={model} onChange={(e) => setModel(e.target.value)} disabled={!!activeJob}>
              {models.map((m) => (
                <option key={m.name} value={m.name} disabled={!m.available}>{m.label}{m.available ? '' : ` — ${t('modelUnavailable')}`}</option>
              ))}
            </select>
          </label>
          <button className="btn-primary" disabled={!model || !!activeJob || missing.length > 0 || !!c.approval || busy === 'run'}
            onClick={() => act('run', () => api(`/cases/${id}/jobs`, { method: 'POST', body: JSON.stringify({ model_name: model }) }))}>
            {t('run')}
          </button>
        </div>
        {selectedModel?.info?.note?.[lang] && <p className="notice small">{selectedModel.info.note[lang]}</p>}
        {missing.length > 0 && <p className="muted small">{missing.map((m) => m.toUpperCase()).join(', ')} ?</p>}
        {c.jobs.slice(0, 3).map((j) => (
          <p key={j.id} className={`job job-${j.state}`} role={ACTIVE.includes(j.state) ? 'status' : undefined}>
            <strong>{t(`job_${j.state}` as Key)}</strong>
            {j.stage && ` · ${t(`stage_${j.stage}` as Key)}`} · {j.model_name} · {f.time(j.created_at)}
            {j.error && <span className="form-error"> · {j.error}</span>}
            {j.state === 'queued' && (
              <button className="btn-small" onClick={() => act('cancel', () => api(`/jobs/${j.id}/cancel`, { method: 'POST' }))}>{t('cancel')}</button>
            )}
          </p>
        ))}
      </div>

      <div className="card">
        <h2>{t('stepResults')}</h2>
        {c.masks.length === 0 ? <p className="muted">{t('noResults')}</p> : (
          <div className="table-wrap">
            <table>
              <thead><tr><th>{t('version')}</th><th>{t('source')}</th><th>{t('volume')}</th><th>{t('lesions')}</th><th>{t('hemisphere')}</th><th>{t('regionsTop')}</th><th /></tr></thead>
              <tbody>{c.masks.map((m) => <MaskRow key={m.id} m={m} />)}</tbody>
            </table>
          </div>
        )}
        <p className="muted small">{t('viewerSoon')}</p>
      </div>

      {c.masks.length > 0 && (
        <div className="card">
          <h2>{t('stepApprove')}</h2>
          {c.approval ? (
            <>
              <p className="approved">{t('approvedAt', { v: approvedMask?.version_no ?? '?', time: f.time(c.approval.approved_at) })}</p>
              {c.approval.patient_note && <p className="note">{c.approval.patient_note}</p>}
              <button className="btn-small" disabled={!!busy} onClick={() => act('unapprove', () => api(`/cases/${id}/unapprove`, { method: 'POST' }))}>{t('unapprove')}</button>
            </>
          ) : (
            <>
              <label>{t('approveVersion')}
                <select value={approval.mask} onChange={(e) => setApproval({ ...approval, mask: e.target.value })}>
                  {c.masks.map((m) => <option key={m.id} value={m.id}>v{m.version_no} · {m.model_name ?? t('srcDoctor')} · {f.num(m.stats.volume_ml)} mL</option>)}
                </select>
              </label>
              <label>{t('patientNote')}
                <textarea rows={3} maxLength={2000} value={approval.note} onChange={(e) => setApproval({ ...approval, note: e.target.value })} />
              </label>
              <label className="check">
                <input type="checkbox" checked={approval.show} onChange={(e) => setApproval({ ...approval, show: e.target.checked })} /> {t('showFunctions')}
              </label>
              <button className="btn-primary" disabled={!approval.mask || !!activeJob || !!busy}
                onClick={() => act('approve', () => api(`/cases/${id}/approve`, { method: 'POST', body: JSON.stringify({ mask_version_id: approval.mask, patient_note: approval.note || null, show_functions: approval.show }) }))}>
                {t('approve')}
              </button>
            </>
          )}
        </div>
      )}
    </section>
  )
}

function MaskRow({ m }: { m: MaskVersion }) {
  const { t } = useI18n()
  const f = useFormat()
  const top = (m.stats.regions ?? []).filter((r) => r.region && r.share >= 0.15).slice(0, 3)
  return (
    <tr>
      <td>v{m.version_no}<div className="muted small">{f.time(m.created_at)}</div></td>
      <td>{m.source === 'ai' ? `${t('srcAi')} · ${m.model_name}` : t('srcDoctor')}</td>
      <td><strong className="big-num">{f.num(m.stats.volume_ml)}</strong> mL</td>
      <td>{m.stats.lesion_count}</td>
      <td>{t(`hemi_${m.stats.hemisphere}` as Key)}</td>
      <td className="small">{top.map((r) => `${f.region(r)} ${Math.round(r.share * 100)}%`).join(', ') || '–'}</td>
      <td><button className="btn-small" onClick={() => downloadFile(`/masks/${m.id}/file`, `mask_v${m.version_no}.nii.gz`)}>{t('download')}</button></td>
    </tr>
  )
}
