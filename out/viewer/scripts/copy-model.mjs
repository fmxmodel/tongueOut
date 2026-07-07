// Stage the rigged head GLB from the pipeline bus (out/head_arkit_v2.glb) into the
// viewer's public/ dir so Vite serves it locally in dev AND emits it into dist/ on a
// production build. Runs automatically on predev/prebuild. Plain Node fs — no dep.
//
// This is a file copy, not model inference: safe in the contamination-guard session.
// public/head_arkit_v2.glb is a generated copy of a tracked artifact and is gitignored.

import { cpSync, existsSync, mkdirSync, statSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..');
const src = resolve(root, '..', 'head_arkit_v2.glb'); // out/head_arkit_v2.glb
const dest = resolve(root, 'public', 'head_arkit_v2.glb');

if (!existsSync(src)) {
  console.error(
    `[copy-model] pipeline artifact not found: ${src}\n` +
      `             The reconstruction/rigging stage must produce head_arkit_v2.glb first.`,
  );
  process.exit(1);
}

mkdirSync(dirname(dest), { recursive: true });
cpSync(src, dest);
console.log(`[copy-model] out/head_arkit_v2.glb (${statSync(dest).size} bytes) -> public/head_arkit_v2.glb`);
