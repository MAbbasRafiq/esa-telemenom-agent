/**
 * Vite config for the React frontend.
 *
 * - plugin-react: enables JSX / Fast Refresh
 * - @tailwindcss/vite: Tailwind CSS v4 (no separate postcss file needed)
 */
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    // Use 5180 so this UI can run beside another app on the default 5173.
    // FastAPI CORS in api/main.py allows this origin.
    port: 5180,
    strictPort: true,
  },
});
