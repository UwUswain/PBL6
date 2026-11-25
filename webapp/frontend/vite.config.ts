import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev server proxies the API so the browser sees one origin, same as nginx in Docker.
export default defineConfig({
  plugins: [react()],
  server: { proxy: { '/api': 'http://127.0.0.1:8001' } },
})
