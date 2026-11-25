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
