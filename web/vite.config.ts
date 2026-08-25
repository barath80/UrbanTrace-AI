import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The built bundle is served by FastAPI at the same origin as /api, so the
// dev server proxies /api to the backend and the production build emits into
// web-dist/ for the API to mount.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: '../web-dist',
    emptyOutDir: true,
  },
  server: {
    // Bind all interfaces so the dashboard can be opened from a phone on the
    // same network - the layout is built for that and needs testing there.
    host: true,
    port: 5173,
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true },
    },
  },
})
