// Preflight name-contract check for the new stack (head_arkit_v2.glb).
//
// This PARSES THE REAL GLB (stdlib GLB reader — no three.js, no webcam, no inference)
// and proves, by measurement (CLAUDE.md invariant #4):
//
//   1. The head mesh carries EXACTLY 52 morph targets.
//   2. Their names == the ARKit-52 contract (src/arkit52.js) — set-equal, 52 distinct,
//      no renames, no extras, no missing.  (This is invariant #2, the shared contract.)
//   3. Every primitive of the head mesh exposes all 52 targets (multi-material split).
//   4. The PURE driver (src/driver.js) resolves every MediaPipe-emitted category to a
//      morph target on EVERY primitive — 51/51, resolved by name against a dict whose
//      indices are DELIBERATELY REVERSED to prove no morph order is assumed.
//   5. Exactly ONE contract morph is never driven by the webcam: tongueOut (MediaPipe
//      does not emit it) — it is present/mappable and exercised via the manual slider.
//   6. `_neutral` is skipped; nothing is aliased onto a different-named morph.
//
// Wired as `prebuild` — a non-zero exit FAILS `npm run build`.

import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { applyBlendshapes, resolveNames } from '../src/driver.js';
import { ARKIT_52, MP_EMITTED, MP_NOT_EMITTED, MP_SKIP } from '../src/arkit52.js';

const here = dirname(fileURLToPath(import.meta.url));
const glbPath = resolve(here, '..', '..', 'head_arkit_v2.glb'); // out/head_arkit_v2.glb

// ---------------------------------------------------------------- stdlib GLB reader
// Reads the JSON chunk of a binary glTF and returns the parsed glTF JSON. No deps.
function readGlbJson(path) {
  const buf = readFileSync(path);
  const magic = buf.readUInt32LE(0);
  if (magic !== 0x46546c67) throw new Error(`not a GLB (magic=0x${magic.toString(16)})`);
  const version = buf.readUInt32LE(4);
  if (version !== 2) throw new Error(`unexpected glTF version ${version}`);
  let off = 12;
  const chunkLen = buf.readUInt32LE(off);
  const chunkType = buf.readUInt32LE(off + 4);
  if (chunkType !== 0x4e4f534a) throw new Error('first chunk is not JSON'); // 'JSON'
  const json = JSON.parse(buf.toString('utf8', off + 8, off + 8 + chunkLen));
  return json;
}

const problems = [];
const assert = (cond, msg) => { if (!cond) problems.push(msg); };

const gltf = readGlbJson(glbPath);
const meshes = gltf.meshes || [];

// The head mesh = the (single) mesh that carries morph targets.
const morphMeshes = meshes.filter((m) => (m.primitives || []).some((p) => (p.targets || []).length));
assert(morphMeshes.length >= 1, 'no mesh in the GLB carries morph targets');
const head = morphMeshes[0];
const targetNames = head?.extras?.targetNames || [];

// --- 1 & 2: exactly 52 targets, names set-equal to the ARKit-52 contract ---
assert(targetNames.length === 52, `head mesh has ${targetNames.length} targetNames, expected 52`);
assert(new Set(targetNames).size === targetNames.length, 'head mesh has duplicate morph-target names');
const contractSet = new Set(ARKIT_52);
const glbSet = new Set(targetNames);
const missing = ARKIT_52.filter((n) => !glbSet.has(n)); // contract names absent from GLB
const extra = targetNames.filter((n) => !contractSet.has(n)); // GLB names not in contract
assert(missing.length === 0, `contract names MISSING from GLB (naming drift): ${missing.join(', ')}`);
assert(extra.length === 0, `GLB morph names NOT in the ARKit-52 contract (renamed/extra): ${extra.join(', ')}`);

// --- 3: every primitive exposes all 52 targets ---
const primCounts = (head?.primitives || []).map((p) => (p.targets || []).length);
assert(
  primCounts.length >= 1 && primCounts.every((c) => c === 52),
  `not every head primitive has 52 targets: [${primCounts.join(', ')}]`,
);

// --- 4: drive a simulated MediaPipe frame through the PURE driver, per primitive ---
// Build a mock morphTargetDictionary the way three.js will (name -> index from
// targetNames order) but REVERSE the index assignment to prove the driver never
// assumes an order — it must still resolve every name correctly.
function mockMeshFrom(names) {
  const dict = {};
  [...names].reverse().forEach((name, i) => (dict[name] = i));
  return { morphTargetDictionary: dict, morphTargetInfluences: new Array(names.length).fill(0) };
}

// One real MediaPipe frame = `_neutral` + the 51 emitted ARKit categories.
const frame = [
  { categoryName: '_neutral', score: 0.9 },
  ...MP_EMITTED.map((n) => ({ categoryName: n, score: 0.5 })),
];

let drivenNames = null;
for (let pi = 0; pi < primCounts.length; pi++) {
  const mesh = mockMeshFrom(targetNames);
  const r = applyBlendshapes(mesh, frame, { smoothing: 0, skipNames: MP_SKIP });
  assert(
    r.driven.length === MP_EMITTED.length,
    `primitive ${pi}: driver drove ${r.driven.length}/${MP_EMITTED.length} MediaPipe categories`,
  );
  assert(
    r.unresolved.length === 0,
    `primitive ${pi}: MediaPipe categories left unresolved (naming drift): ${r.unresolved.join(', ')}`,
  );
  assert(r.skipped.includes('_neutral'), `primitive ${pi}: _neutral was not skipped`);
  // Value actually written by name (not order): jawOpen is carried; check its influence.
  assert(
    mesh.morphTargetInfluences[mesh.morphTargetDictionary['jawOpen']] === 0.5,
    `primitive ${pi}: jawOpen influence not written by name`,
  );
  drivenNames = new Set(r.driven);
}

// --- 5: exactly one contract morph is never driven by the webcam -> tongueOut ---
const notDriven = targetNames.filter((n) => !drivenNames.has(n));
assert(
  notDriven.length === MP_NOT_EMITTED.size && notDriven.every((n) => MP_NOT_EMITTED.has(n)),
  `webcam-undriven morphs != {${[...MP_NOT_EMITTED].join(', ')}}: got [${notDriven.join(', ')}]`,
);

// --- 6: no aliasing — tongueOut IS present (drivable manually) and resolves by name ---
assert(glbSet.has('tongueOut'), 'tongueOut morph missing from GLB (should be present, just not webcam-driven)');
const tongueRes = resolveNames({ ...mockMeshFrom(targetNames).morphTargetDictionary }, ['tongueOut']);
assert(tongueRes.resolved.length === 1, 'tongueOut must resolve by name for the manual slider');

// ---------------------------------------------------------------- report
console.log('── viewer/driver name-contract preflight — new stack (head_arkit_v2.glb) ──');
console.log(`glb:                 ${glbPath}`);
console.log(`materials:           ${(gltf.materials || []).map((m) => `${m.name}[${m.alphaMode || 'OPAQUE'}]`).join(', ')}`);
console.log(`head mesh:           ${head?.name || '(unnamed)'}  primitives=${primCounts.length}  targets/prim=[${primCounts.join(', ')}]`);
console.log(`morph target count:  ${targetNames.length}`);
console.log(`contract (ARKit-52): ${ARKIT_52.length}  → set-equal to GLB: ${missing.length === 0 && extra.length === 0}`);
console.log(`MediaPipe emits:     ${MP_EMITTED.length} ARKit + _neutral  (never emits: ${[...MP_NOT_EMITTED].join(', ')})`);
console.log(`DRIVEN from MP:      ${drivenNames ? drivenNames.size : 0} / ${MP_EMITTED.length}  (per primitive, resolved by name on reversed indices)`);
console.log(`webcam-undriven:     ${notDriven.join(', ') || 'none'}  (present & mappable; manual slider — expected, not drift)`);
console.log(`renamed/aliased:     none`);

if (problems.length) {
  console.error('\nFAIL:');
  for (const p of problems) console.error('  - ' + p);
  process.exit(1);
}

console.log(
  `\nPASS: GLB carries 52/52 ARKit morph targets across ${primCounts.length} primitives; ` +
    `all ${MP_EMITTED.length} MediaPipe categories resolve 1:1 by exact name; ` +
    `tongueOut present but webcam-undriven (manual); no renames, no extras, no missing.`,
);
