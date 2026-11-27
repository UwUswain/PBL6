import { lazy, Suspense, useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'

import TEXT from '../../content/patient_text.json'
import { api, type Region } from '../../lib/api'
import { useI18n } from '../../lib/i18n'
import { useFormat } from '../doctor/common'

// NiiVue (~1 MB) only when a result is opened.
const PatientBrain = lazy(() => import('../../components/PatientBrain'))

interface Result {
  case_id: string
  study_date: string | null
  approved_at: string
  doctor_name: string
  patient_note: string | null
  show_functions: boolean
  edited_by_doctor: boolean
  stats: { volume_ml: number; lesion_count: number; hemisphere: string; regions?: Region[]; atlas_fit?: number }
}

type PatientText = (typeof TEXT)['vi']
const fill = (s: string, vars: Record<string, string | number>) => s.replace(/\{(\w+)\}/g, (_, k: string) => String(vars[k] ?? ''))

// Same safety rules as the legacy patient mode (WEB_PLAN 5.1): name a region only when it holds enough of
// the lesion; name its function only when that region alone is large enough; hide all of it on a poor atlas fit.
const REGION_MIN_SHARE = 0.15, FUNCTION_MIN_SHARE = 0.2, FUNCTION_MIN_ML = 0.5, ATLAS_MIN_FIT = 0.8

function summary(P: PatientText, r: Result, num: (x: number, d?: number) => string): string[] {
  const s = r.stats
  if (!s.lesion_count) return [P.none]
  return [
    fill(P.volume, { volume: num(s.volume_ml, 1) }),
    s.lesion_count === 1 ? P.count_one : fill(P.count_many, { count: s.lesion_count }),
    (P as Record<string, unknown>)[`side_${s.hemisphere}`] as string | undefined,
    r.edited_by_doctor ? P.edited : undefined,
  ].filter((x): x is string => !!x)
}

function regionLines(P: PatientText, r: Result): string[] {
  const rows = (r.stats.regions ?? []).filter((x) => x.region && x.share >= REGION_MIN_SHARE)
  if (!rows.length || !((r.stats.atlas_fit ?? 0) >= ATLAS_MIN_FIT)) return []
  const names = P.region_names as Record<string, string>
  const sides = P.side_names as Record<string, string>
  const label = (x: Region) => (x.side ? fill(P.region_with_side, { region: names[x.region!], side: sides[x.side] }) : names[x.region!])
  const lines = rows.map((x) => fill(P.region_share, { pct: Math.round(x.share * 100), region: label(x) }))
  if (r.show_functions) {
    const funcs = rows.filter((x) => x.ml >= FUNCTION_MIN_ML && x.share >= FUNCTION_MIN_SHARE)
    lines.push(...new Set(funcs.map((x) => (P.region_functions as Record<string, string>)[x.region!])))
    if (!funcs.length) lines.push(P.region_unclear)
  }
  lines.push(P.region_note)
  return lines
}

export function PatientHome() {
  const { t } = useI18n()
  const f = useFormat()
  const [results, setResults] = useState<Result[] | null>(null)
  useEffect(() => { api<Result[]>('/me/results').then(setResults) }, [])
  return (
    <section>
      <h1>{t('patientHome')}</h1>
      {!results ? <p className="muted">{t('loading')}</p> : results.length === 0 ? (
        <p className="card">{t('noApprovedResults')}</p>
      ) : (
        <div className="result-cards">
          {results.map((r) => (
            <Link key={r.case_id} className="result-card" to={`/patient/results/${r.case_id}`}>
              <strong>{t('studyOn', { date: f.date(r.study_date ?? r.approved_at) })}</strong>
              <span className="muted small">{t('reviewedBy', { name: r.doctor_name, time: f.time(r.approved_at) })}</span>
              <span className="go">{t('openResult')} →</span>
            </Link>
          ))}
        </div>
      )}
    </section>
  )
}

export function PatientResult() {
  const { id = '' } = useParams()
  const { t, lang } = useI18n()
  const f = useFormat()
  const [r, setResult] = useState<Result | null>(null)
  const [error, setError] = useState('')
  useEffect(() => { api<Result>(`/me/results/${id}`).then(setResult).catch((e) => setError(e.message)) }, [id])

  if (!r) return <p className="muted" role="status">{error || t('loading')}</p>
  const P = TEXT[lang] as PatientText
  const regions = regionLines(P, r)
  const list = (items: string[]) => <ul>{items.map((x) => <li key={x}>{x}</li>)}</ul>

  return (
    <section className="patient-result">
      <p><Link to="/patient">← {t('patientHome')}</Link></p>
      <div className="patient-grid">
        <Suspense fallback={<p className="muted">{t('loading')}</p>}>
          <PatientBrain caseId={r.case_id} side={r.stats.hemisphere} />
        </Suspense>
        <article className="card patient-text">
          <h1>{P.heading}</h1>
          <p className="approved">{fill(P.approved, { time: f.time(r.approved_at) })}</p>
          <h2>{P.what_title}</h2>
          <p>{P.what}</p>
          <h2>{P.summary_title}</h2>
          {list(summary(P, r, f.num))}
          {regions.length > 0 && (<><h2>{P.regions_title}</h2>{list(regions)}</>)}
          {r.patient_note && (<><h2>{P.note_title}</h2><p className="note">{r.patient_note}</p></>)}
          <h2>{P.limits_title}</h2>
          {list(P.limits)}
          <h2>{P.questions_title}</h2>
          {list(P.questions)}
        </article>
      </div>
    </section>
  )
}
