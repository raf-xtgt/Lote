// Two entries (ARCHITECTURE.md > Project Folder Structure): the React tool UI and the Phaser player (no React).
// `npm run build` writes dist/tool/, dist/player/ and dist/assets/, which the Modal web function serves.
import { resolve } from "node:path";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Dev: point /api at the `modal serve` URL, e.g. LOTE_API_URL=https://...-dev.modal.run npm run dev
const api = process.env.LOTE_API_URL;

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "dist",
    emptyOutDir: true,
    // Phaser alone is ~1.4 MB minified (~360 kB gzip); the player entry is meant to be one bundle.
    chunkSizeWarningLimit: 1600,
    rolldownOptions: {
      input: {
        tool: resolve(import.meta.dirname, "tool/index.html"),
        player: resolve(import.meta.dirname, "player/index.html"),
      },
    },
  },
  server: {
    open: "/tool/",
    proxy: api ? { "/api": { target: api, changeOrigin: true } } : undefined,
  },
});
