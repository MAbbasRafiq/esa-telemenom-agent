/// <reference types="vite/client" />

/**
 * Type for Vite env variables we might add later.
 * Example: VITE_API_BASE=http://127.0.0.1:8100
 */
interface ImportMetaEnv {
  readonly VITE_API_BASE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
