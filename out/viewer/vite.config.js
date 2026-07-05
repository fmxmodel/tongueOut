import { defineConfig } from 'vite';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { existsSync, statSync, createReadStream } from 'node:fs';

const here = dirname(fileURLToPath(import.meta.url));
// The pipeline bus. The viewer lives at out/viewer/, artifacts sit one level up at out/.
const OUT_DIR = resolve(here, '..');

const MIME = {
  '.glb': 'model/gltf-binary',
  '.gltf': 'model/gltf+json',
  '.json': 'application/json',
};

/**
 * Serve pipeline-bus artifacts straight from out/ at their runtime URLs, WITHOUT
 * copying them into the viewer. This is what lets the app load the head from
 * `out/head_arkit.glb` at runtime and degrade gracefully (clean 404) when the
 * GPU/real-artifact stage hasn't produced it yet.
 */
function serveBusArtifacts() {
  const mounts = {
    '/head_arkit.glb': resolve(OUT_DIR, 'head_arkit.glb'),
    // B1 shape contract (authored by arkit-rigger) — the coverage/unresolved panels read it.
    '/shapes/arkit_manifest.json': resolve(OUT_DIR, 'shapes', 'arkit_manifest.json'),
    // Legacy B2 51<->52 name map, kept mounted for back-compat / manual inspection.
    '/arkit_51_52_map.json': resolve(OUT_DIR, 'arkit_51_52_map.json'),
  };
  const handler = (req, res, next) => {
    const url = (req.url || '').split('?')[0];
    const file = mounts[url];
    if (!file) return next();
    if (!existsSync(file)) {
      res.statusCode = 404;
      res.setHeader('Content-Type', 'application/json');
      res.end(JSON.stringify({ error: `not produced yet: out${url}` }));
      return;
    }
    const ext = url.slice(url.lastIndexOf('.'));
    res.setHeader('Content-Type', MIME[ext] || 'application/octet-stream');
    res.setHeader('Content-Length', String(statSync(file).size));
    createReadStream(file).pipe(res);
  };
  return {
    name: 'serve-bus-artifacts',
    configureServer(server) {
      server.middlewares.use(handler);
    },
    configurePreviewServer(server) {
      server.middlewares.use(handler);
    },
  };
}

export default defineConfig({
  plugins: [serveBusArtifacts()],
  server: {
    // Allow the dev server to read the parent out/ dir (GLB + contract live there).
    fs: { allow: [here, OUT_DIR] },
  },
  // Keep the MediaPipe wasm out of the bundler; it is loaded by URL at runtime.
  optimizeDeps: { exclude: ['@mediapipe/tasks-vision'] },
});
