import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const backendTarget = process.env.VITE_BACKEND_URL ?? "http://127.0.0.1:8001";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    proxy: {
      "/api": {
        target: backendTarget,
        changeOrigin: true,
      },
    },
  },
  preview: {
    host: "0.0.0.0",
    port: 4173,
  },
});
