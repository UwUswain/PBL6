// API client. The access token lives only in memory (never localStorage); the refresh token is an
// httpOnly cookie the browser sends to /api/v1/auth/* by itself.

export type Role = 'admin' | 'doctor' | 'patient'

export interface User {
  id: string
  username: string
  display_name: string
  role: Role
  is_active: boolean
  must_change_password: boolean
  created_at: string
}

interface TokenOut {
  access_token: string
  expires_in: number
  user: User
}

export class ApiError extends Error {
  status: number
  retryAfter?: number
  constructor(status: number, message: string, retryAfter?: number) {
    super(message)
    this.status = status
    this.retryAfter = retryAfter
  }
}

let accessToken: string | null = null
let refreshing: Promise<User | null> | null = null
let onSessionEnd: () => void = () => {}

export function setSessionEndHandler(fn: () => void) {
  onSessionEnd = fn
}

async function parse<T>(res: Response): Promise<T> {
  if (res.ok) return (res.status === 204 ? undefined : await res.json()) as T
  const body = await res.json().catch(() => ({}))
  const detail = typeof body.detail === 'string' ? body.detail : res.statusText
  throw new ApiError(res.status, detail, Number(res.headers.get('retry-after')) || undefined)
}

function take(t: TokenOut): User {
  accessToken = t.access_token
  return t.user
}

/** Uses the refresh cookie; one refresh at a time even if many requests hit 401 together. */
export function refreshSession(): Promise<User | null> {
  refreshing ??= fetch('/api/v1/auth/refresh', { method: 'POST' })
    .then((r) => (r.ok ? r.json().then(take) : null))
    .catch(() => null)
    .finally(() => { refreshing = null })
  return refreshing
}

export async function api<T>(path: string, init: RequestInit = {}, retried = false): Promise<T> {
  const headers = new Headers(init.headers)
  if (accessToken) headers.set('Authorization', `Bearer ${accessToken}`)
  if (init.body && !(init.body instanceof FormData)) headers.set('Content-Type', 'application/json')
  const res = await fetch(`/api/v1${path}`, { ...init, headers })
  if (res.status === 401 && !retried && !path.startsWith('/auth/')) {
    if (await refreshSession()) return api<T>(path, init, true)
    accessToken = null
    onSessionEnd()
  }
  return parse<T>(res)
}

export async function login(username: string, password: string): Promise<User> {
  const res = await fetch('/api/v1/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  })
  return take(await parse<TokenOut>(res))
}

export async function changePassword(current_password: string, new_password: string): Promise<User> {
  return take(await api<TokenOut>('/auth/change-password', {
    method: 'POST',
    body: JSON.stringify({ current_password, new_password }),
  }))
}

export async function logout(): Promise<void> {
  await fetch('/api/v1/auth/logout', { method: 'POST' }).catch(() => undefined)
  accessToken = null
}

// ---------- M2 clinical types ----------

export type CaseStatus = 'draft' | 'ready' | 'queued' | 'running' | 'awaiting_review' | 'approved' | 'failed'

export interface Patient {
  id: string
  code: string
  birth_year: number | null
  sex: 'M' | 'F' | 'O' | null
  created_at: string
  has_account: boolean
  case_count: number
}

export interface CaseSummary {
  id: string
  patient_id: string
  patient_code: string
  study_date: string | null
  notes: string | null
  status: CaseStatus
  created_at: string
  updated_at: string
}

export interface Region { region: string | null; side: 'left' | 'right' | null; ml: number; share: number }

export interface MaskStats {
  volume_ml: number
  lesion_count: number
  hemisphere: 'left' | 'right' | 'both' | 'none'
  left_ml: number
  right_ml: number
  regions?: Region[]
  atlas_fit?: number
}

export interface MaskVersion {
  id: string
  version_no: number
  source: 'ai' | 'doctor'
  model_name: string | null
  stats: MaskStats
  note: string | null
  created_at: string
}

export interface Job {
  id: string
  model_name: string
  state: 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled'
  stage: string | null
  error: string | null
  created_at: string
  started_at: string | null
  finished_at: string | null
}

export interface CaseDetail extends CaseSummary {
  volumes: { modality: 'dwi' | 'adc' | 'flair'; shape: number[]; spacing_mm: number[]; size_bytes: number; uploaded_at: string }[]
  jobs: Job[]
  masks: MaskVersion[]
  approval: { id: string; mask_version_id: string; approved_at: string; patient_note: string | null; show_functions: boolean } | null
  inputs_locked: boolean
}

export interface ModelInfo {
  name: string
  label: string
  modalities: string[]
  required: string[]
  available: boolean
  is_default: boolean
  info: { metrics?: { source?: string; values?: Record<string, number> } | null; note?: Record<string, string> | null } | null
}

/** GET with the bearer token, retrying once after a refresh (files cannot go through api(), which parses JSON). */
export async function fetchAuthed(path: string): Promise<Response> {
  const go = () => fetch(`/api/v1${path}`, { headers: accessToken ? { Authorization: `Bearer ${accessToken}` } : {} })
  let res = await go()
  if (res.status === 401 && (await refreshSession())) res = await go()
  if (!res.ok) throw new ApiError(res.status, res.statusText)
  return res
}

/** Authenticated file download (the token is a header, so a plain <a href> cannot be used). */
export async function downloadFile(path: string, filename: string): Promise<void> {
  const url = URL.createObjectURL(await (await fetchAuthed(path)).blob())
  const a = Object.assign(document.createElement('a'), { href: url, download: filename })
  a.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
