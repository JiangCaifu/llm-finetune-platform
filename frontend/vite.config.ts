import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  // 相对路径：支持部署在根路径，也支持 DSW/网关的 /proxy/8000/ 子路径
  base: './',
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
