import type { CaseStatus, Region } from '../../lib/api'
import { useI18n, type Key } from '../../lib/i18n'

export function StatusTag({ status }: { status: CaseStatus }) {
  const { t } = useI18n()
  const tone = status === 'approved' ? 'ok' : status === 'failed' ? 'off' : status === 'awaiting_review' ? 'warn' : ''
  return <span className={`tag ${tone}`}>{t(`st_${status}` as Key)}</span>
}

export function useFormat() {
  const { lang, t } = useI18n()
  const locale = lang === 'vi' ? 'vi-VN' : 'en-GB'
  return {
    date: (iso: string | null) => (iso ? new Date(iso).toLocaleDateString(locale) : '–'),
    time: (iso: string) => new Date(iso).toLocaleString(locale),
    num: (x: number, digits = 2) => x.toLocaleString(locale, { maximumFractionDigits: digits }),
    region: (r: Region) => r.region
      ? `${t(`region_${r.region}` as Key)}${r.side ? ' ' + t(`side_${r.side}` as Key) : ''}`
      : '–',
  }
}
