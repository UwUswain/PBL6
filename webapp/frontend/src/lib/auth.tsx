import { createContext, useContext, useEffect, useState, type ReactNode } from 'react'

import * as apiClient from './api'
import type { User } from './api'

interface Auth {
  user: User | null
  ready: boolean // false until the first refresh attempt finishes
  login: (username: string, password: string) => Promise<User>
  logout: () => Promise<void>
  changePassword: (current: string, next: string) => Promise<User>
}

const Ctx = createContext<Auth | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [ready, setReady] = useState(false)

  useEffect(() => {
    apiClient.setSessionEndHandler(() => setUser(null))
    // A reload loses the in-memory access token; the refresh cookie restores the session.
    apiClient.refreshSession().then((u) => { setUser(u); setReady(true) })
  }, [])

  const value: Auth = {
    user,
    ready,
    login: async (u, p) => { const me = await apiClient.login(u, p); setUser(me); return me },
    logout: async () => { await apiClient.logout(); setUser(null) },
    changePassword: async (c, n) => { const me = await apiClient.changePassword(c, n); setUser(me); return me },
  }
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useAuth(): Auth {
  const v = useContext(Ctx)
  if (!v) throw new Error('useAuth outside AuthProvider')
  return v
}
