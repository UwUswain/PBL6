import type { ReactNode } from 'react'
import { Navigate, useLocation } from 'react-router-dom'

import type { Role, User } from './lib/api'
import { useAuth } from './lib/auth'
import { useI18n } from './lib/i18n'

const HOME: Record<Role, string> = { admin: '/admin/users', doctor: '/doctor', patient: '/patient' }

export function homeFor(user: User): string {
  return user.must_change_password ? '/change-password' : HOME[user.role]
}

/** Signed in, password already changed, and (optionally) one of the given roles. */
export function Guard({ roles, children }: { roles?: Role[]; children: ReactNode }) {
  const { user, ready } = useAuth()
  const { t } = useI18n()
  const location = useLocation()
  if (!ready) return <p className="page muted" role="status">{t('loading')}</p>
  if (!user) return <Navigate to="/login" replace state={{ from: location.pathname }} />
  if (user.must_change_password) return <Navigate to="/change-password" replace />
  if (roles && !roles.includes(user.role)) return <Navigate to="/403" replace />
  return children
}

/** Only for signed-in users (also those who still must change the password). */
export function SignedIn({ children }: { children: ReactNode }) {
  const { user, ready } = useAuth()
  const { t } = useI18n()
  if (!ready) return <p className="page muted" role="status">{t('loading')}</p>
  return user ? children : <Navigate to="/login" replace />
}

export function HomeRedirect() {
  const { user, ready } = useAuth()
  if (!ready) return null
  return <Navigate to={user ? homeFor(user) : '/login'} replace />
}
