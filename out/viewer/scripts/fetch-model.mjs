// Download the MediaPipe FaceLandmarker model into public/models so the viewer
// runs fully offline. OPTIONAL: if you skip this, the app falls back to the Google
// CDN at runtime (see src/config.js modelAssetPathCdn).
//
// This downloads a model FILE; it does NOT run inference. In the no-network /
// contamination-guard session, skip it — the CDN fallback covers the later run.

import { mkdirSync, createWriteStream, existsSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Readable } from 'node:stream';

const URL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

const here = dirname(fileURLToPath(import.meta.url));
const dest = resolve(here, '..', 'public/models/face_landmarker.task');

if (existsSync(dest) && !process.argv.includes('--force')) {
  console.log(`[fetch-model] already present: ${dest} (use --force to re-download)`);
  process.exit(0);
}

mkdirSync(dirname(dest), { recursive: true });
console.log(`[fetch-model] downloading ${URL}`);
const res = await fetch(URL);
if (!res.ok) {
  console.error(`[fetch-model] HTTP ${res.status}. Skipping — CDN fallback still works at runtime.`);
  process.exit(1);
}
await new Promise((ok, err) => {
  const out = createWriteStream(dest);
  Readable.fromWeb(res.body).pipe(out).on('finish', ok).on('error', err);
});
console.log(`[fetch-model] saved -> ${dest}`);
