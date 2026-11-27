import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

declare const process: { env: Record<string, string | undefined> }  // Node global; avoids @types/node for one line

// Dev server proxies the API so the browser sees one origin, same as nginx in Docker.
// API_TARGET=http://127.0.0.1:8080 points it at the Docker stack instead of a local uvicorn on :8001.
export default defineConfig({
  plugins: [react()],
  server: { proxy: { '/api': process.env.API_TARGET ?? 'http://127.0.0.1:8001' } },
})
