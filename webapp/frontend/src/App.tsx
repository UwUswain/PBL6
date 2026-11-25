import { useEffect, useState } from 'react'

type Health = 'checking' | 'ready' | 'down'

// M0 page: proves browser -> nginx -> API -> PostgreSQL works end to end.
export default function App() {
  const [health, setHealth] = useState<Health>('checking')

  useEffect(() => {
    fetch('/api/v1/health/ready')
      .then((r) => setHealth(r.ok ? 'ready' : 'down'))
      .catch(() => setHealth('down'))
  }, [])

  return (
    <>
      <header className="topbar">
        <strong>PBL6 · Stroke Lesion Platform</strong>
      </header>
      <main className="center">
        <h1>Hệ thống đang khởi tạo</h1>
        <p className={`status ${health}`} role="status">
          API + cơ sở dữ liệu:{' '}
          {health === 'checking' ? 'đang kiểm tra…' : health === 'ready' ? 'sẵn sàng' : 'không kết nối được'}
        </p>
      </main>
    </>
  )
}
