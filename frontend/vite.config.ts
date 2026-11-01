import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // 开发时 API 代理到后端（避免跨域）
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  // build 产物放到 dist/，FastAPI 会 serve 这个目录
  build: {
    outDir: 'dist',
    sourcemap: false,
  },
})
