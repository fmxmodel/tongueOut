// Copy the MediaPipe FaceLandmarker WASM runtime out of node_modules into
// public/mediapipe/wasm so the viewer serves it locally (offline-first). Runs
// automatically on predev/prebuild. No extra dependency — plain Node fs.
//
// This is a file copy, NOT model inference: safe to run in the contamination-guard
// session. It does not download or execute the FaceLandmarker model.

import { cpSync, existsSync, mkdirSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..');
const src = resolve(root, 'node_modules/@mediapipe/tasks-vision/wasm');
const dest = resolve(root, 'public/mediapipe/wasm');

if (!existsSync(src)) {
  console.error(
    `[copy-assets] MediaPipe wasm not found at ${src}. Run \`npm install\` first.`,
  );
  process.exit(1);
}

mkdirSync(dest, { recursive: true });
cpSync(src, dest, { recursive: true });
console.log(`[copy-assets] MediaPipe wasm -> public/mediapipe/wasm`);
