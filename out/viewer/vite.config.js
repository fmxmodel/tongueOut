import { defineConfig } from 'vite';

// The rigged head (head_arkit_v2.glb) and the MediaPipe WASM/model are staged into
// public/ by the predev/prebuild scripts (copy-model.mjs, copy-mediapipe-assets.mjs,
// fetch-model.mjs), so Vite serves them locally in dev and copies them into dist/ on
// build — no runtime CDN, no dev-server middleware needed. Fully static output.
export default defineConfig({
  // Keep the MediaPipe wasm out of the bundler; it is loaded by URL at runtime.
  optimizeDeps: { exclude: ['@mediapipe/tasks-vision'] },
  build: {
    // The 11 MB GLB lives in public/ (copied verbatim, not bundled), so the only large
    // JS chunk is three.js; relax the warning limit to keep the build output clean.
    chunkSizeWarningLimit: 1500,
  },
});
