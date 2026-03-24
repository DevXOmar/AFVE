import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";
import { componentTagger } from "lovable-tagger";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "VITE_");

  return {
  server: {
    host: "::",
    port: 8080,
    hmr: {
      overlay: false,
    },
    // Dev-only proxy to the FastAPI backend (avoids CORS in development).
    proxy: {
      "/api": {
        target: env.VITE_AFVE_API_TARGET || "http://127.0.0.1:8000",
        changeOrigin: true,
        secure: false,
        rewrite: (p) => p.replace(/^\/api/, ""),
      },
    },
  },
  plugins: [react(), mode === "development" && componentTagger()].filter(Boolean),
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  };
});
