// STATIC verification of the driver's name-resolution for Track B1 (FLAME 2023 Open,
// CC-BY-4.0) — NO inference, NO GLB, NO webcam. Exercises the exact same PURE driver
// (src/driver.js) the browser uses, reconciled against the authoritative B1 shape
// manifest `out/shapes/arkit_manifest.json` (authored by arkit-rigger).
//
// Answers, by measurement (CLAUDE.md invariant #4):
//   - Which of Apple's 52 does the B1 rig CARRY (supported) vs declare unsupported?
//   - Do all carried names resolve 1:1 by exact, case-sensitive name? (must: yes)
//   - Do the unsupported names (a-priori: tongueOut, cheekPuff, cheekSquintLeft,
//     cheekSquintRight — plus ANY pod-demoted shapes) no-op cleanly
//     (dict[name] === undefined -> skipped + reported), and are NEVER aliased?
//   - Of the 51 ARKit categories MediaPipe actually emits, how many drive, and which
//     land in the honest "unresolved" log (the naming-drift channel for qa-verifier)?
//   - Are there any renamed / aliased shapes? (must: none)
//
// run_state handling (task item 2): while the manifest is `DEFERRED-pod-run` NOTHING
// in it is measured — the 52 delta meshes do not exist yet. We therefore verify the
// driver against the A-PRIORI classification (a shape is carried unless it is
// declared `supported: false` / `intended: "unsupported"`), and print loudly that the
// FINAL, authoritative check reruns against the pod-built GLB once the manifest flips
// to `run_state == "measured-on-pod"` (whereupon carried == `supported === true` and
// any pod-demoted weak shapes are folded into the unsupported set automatically).
//
// Exit code 0 = all assertions pass.

import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { applyBlendshapes, resolveNames } from '../src/driver.js';

const here = dirname(fileURLToPath(import.meta.url));
const manifestPath = resolve(here, '..', '..', 'shapes', 'arkit_manifest.json');
const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));

// MediaPipe FaceLandmarker v2 emits 52 categories = `_neutral` + 51 ARKit blendshapes.
// The ONE Apple-52 name it never emits is `tongueOut` (per the model card AND the
// manifest's own tongueOut.reason). Everything else in Apple-52 is emitted verbatim,
// so "what MediaPipe emits" is derived from the manifest itself, not a stale B2 list.
const MP_NOT_EMITTED = new Set(['tongueOut']);

// The B1 a-priori-unsupported set the task pins. These MUST be `supported: false` in
// the manifest regardless of run_state (a weak shape may be demoted TO unsupported on
// the pod, but these four can never be promoted).
const APRIORI_UNSUPPORTED = ['tongueOut', 'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight'];

const runState = manifest.run_state || 'unknown';
const measured = runState === 'measured-on-pod';
const shapes = manifest.shapes || {};
const allNames = Object.keys(shapes); // the canonical 52, in manifest order

// A shape is "carried by the (a-priori | measured) GLB" iff:
//   - measured:  its measured `supported === true`
//   - deferred:  it is NOT declared unsupported (supported !== false)
const isCarried = (name) => {
  const s = shapes[name] || {};
  return measured ? s.supported === true : s.supported !== false;
};

const carried = allNames.filter(isCarried);
const unsupported = allNames.filter((n) => !isCarried(n));
const mpEmitted = allNames.filter((n) => !MP_NOT_EMITTED.has(n)); // 51 ARKit names

const problems = [];
const assert = (cond, msg) => { if (!cond) problems.push(msg); };

// --- 0. Manifest sanity: exactly the canonical 52, and the a-priori-unsupported four
//        are declared unsupported (supported: false) no matter the run_state ---
assert(allNames.length === 52, `manifest lists ${allNames.length} shapes, expected 52`);
assert(
  (manifest.counts?.total ?? 52) === 52,
  `manifest.counts.total = ${manifest.counts?.total}, expected 52`,
);
for (const n of APRIORI_UNSUPPORTED) {
  assert(shapes[n] !== undefined, `a-priori-unsupported name '${n}' missing from manifest`);
  assert(
    shapes[n]?.supported === false,
    `a-priori-unsupported '${n}' must be supported:false in the manifest, got ${shapes[n]?.supported}`,
  );
}
// In the DEFERRED (a-priori) state the unsupported set must be EXACTLY those four.
// Once measured, it is those four PLUS any pod-demoted weak shapes (superset).
const aprioriSet = new Set(APRIORI_UNSUPPORTED);
if (!measured) {
  const extra = unsupported.filter((n) => !aprioriSet.has(n));
  const missing = APRIORI_UNSUPPORTED.filter((n) => !unsupported.includes(n));
  assert(extra.length === 0, `a-priori unsupported has unexpected extras: ${extra.join(', ')}`);
  assert(missing.length === 0, `a-priori unsupported is missing: ${missing.join(', ')}`);
} else {
  const missing = APRIORI_UNSUPPORTED.filter((n) => !unsupported.includes(n));
  assert(missing.length === 0, `measured unsupported must still include the a-priori four; missing: ${missing.join(', ')}`);
}

// Build a MOCK morphTargetDictionary exactly as the pod-built GLB will expose it: the
// CARRIED names only, in REVERSED order (indices assigned by insertion) to prove the
// driver never assumes a morph-target order (CLAUDE.md invariant, "resolve by name").
const dict = {};
[...carried].reverse().forEach((name, i) => (dict[name] = i));
const mesh = { morphTargetDictionary: dict, morphTargetInfluences: new Array(carried.length).fill(0) };

// --- 1. Every carried name resolves 1:1 against the dict ---
const carriedRes = resolveNames(dict, carried);
assert(carriedRes.resolved.length === carried.length, `expected ${carried.length} carried names to resolve, got ${carriedRes.resolved.length}`);
assert(carriedRes.unresolved.length === 0, `carried names failed to resolve: ${carriedRes.unresolved.join(', ')}`);

// --- 2. Canonical-52 vs GLB: exactly the unsupported names are absent from the GLB ---
const all = resolveNames(dict, allNames);
assert(all.resolved.length === carried.length, `expected ${carried.length}/52 canonical names drivable, got ${all.resolved.length}`);
assert(
  all.unresolved.length === unsupported.length &&
    all.unresolved.every((n) => !isCarried(n)),
  `canonical-52 unresolved set != unsupported set: [${all.unresolved.join(', ')}]`,
);

// --- 3. Simulate a real MediaPipe frame (_neutral + 51 ARKit) through the driver ---
const frame = [{ categoryName: '_neutral', score: 0.9 }, ...mpEmitted.map((n) => ({ categoryName: n, score: 0.5 }))];
const drivenMp = mpEmitted.filter(isCarried);
const unresolvedMp = mpEmitted.filter((n) => !isCarried(n)); // MediaPipe emits it, GLB doesn't carry it
const r1 = applyBlendshapes(mesh, frame, { smoothing: 0, skipNames: ['_neutral'] });
assert(r1.driven.length === drivenMp.length, `driver drove ${r1.driven.length}/${drivenMp.length} MediaPipe shapes`);
assert(
  r1.unresolved.length === unresolvedMp.length &&
    unresolvedMp.every((n) => r1.unresolved.includes(n)),
  `MediaPipe unresolved set mismatch: got [${r1.unresolved.join(', ')}], expected [${unresolvedMp.join(', ')}]`,
);
assert(r1.skipped.includes('_neutral'), `_neutral was not skipped`);
// influences actually written, by name (not order): jawOpen is always carried in B1.
assert(dict['jawOpen'] !== undefined, 'jawOpen expected carried by the B1 rig');
assert(mesh.morphTargetInfluences[dict['jawOpen']] === 0.5, 'jawOpen influence not written by name');

// --- 4. Every UNSUPPORTED name no-ops cleanly and is NEVER aliased onto another shape ---
for (const name of unsupported) {
  const m = { morphTargetDictionary: { ...dict }, morphTargetInfluences: new Array(carried.length).fill(0) };
  const r = applyBlendshapes(m, [{ categoryName: name, score: 1.0 }], { smoothing: 0 });
  assert(r.driven.length === 0, `unsupported '${name}' should not drive anything`);
  assert(r.unresolved.length === 1 && r.unresolved[0] === name, `unsupported '${name}' should be reported unresolved`);
  assert(!m.morphTargetInfluences.includes(1.0), `unsupported '${name}' must NOT have aliased onto another shape`);
}

// --- 5. Smoothing (EMA) behaves ---
const m2 = { morphTargetDictionary: { jawOpen: 0 }, morphTargetInfluences: [0] };
applyBlendshapes(m2, [{ categoryName: 'jawOpen', score: 1.0 }], { smoothing: 0.5 });
assert(Math.abs(m2.morphTargetInfluences[0] - 0.5) < 1e-9, `EMA smoothing wrong: ${m2.morphTargetInfluences[0]}`);

// --- 6. No renames/aliases: every MediaPipe ARKit name is ACCOUNTED FOR by the
//        manifest — it is either carried (drives 1:1) or unsupported (honest no-op).
//        None is silently remapped onto a differently-named morph target. ---
const accounted = new Set([...carried, ...unsupported]);
const unaccounted = mpEmitted.filter((n) => !accounted.has(n));
assert(unaccounted.length === 0, `MediaPipe names not present verbatim in the manifest (would need a rename): ${unaccounted.join(', ')}`);

// ---------------------------------------------------------------- report
console.log('── viewer/driver static name-resolution check — Track B1 (FLAME 2023 Open) ──');
console.log(`manifest:            ${manifestPath}`);
console.log(`run_state:           ${runState}${measured ? ' (MEASURED)' : ' (a-priori — nothing measured yet)'}`);
console.log(`canonical ARKit:     ${allNames.length}`);
console.log(`GLB carried (mock):  ${carried.length}   ${measured ? '(measured supported)' : '(a-priori supported)'}`);
console.log(`unsupported:         ${unsupported.length}  → ${unsupported.join(', ') || 'none'}`);
console.log(`MediaPipe emits:     ${mpEmitted.length} ARKit + _neutral  (never emits: ${[...MP_NOT_EMITTED].join(', ')})`);
console.log(`DRIVEN from MP:      ${drivenMp.length} / ${mpEmitted.length}`);
console.log(`unresolved from MP:  ${unresolvedMp.length}  → ${unresolvedMp.join(', ') || 'none'}  (honest no-ops; expected, not drift)`);
console.log(`renamed/aliased:     none`);

if (problems.length) {
  console.error('\nFAIL:');
  for (const p of problems) console.error('  - ' + p);
  process.exit(1);
}

if (!measured) {
  console.log(
    `\nPASS (A-PRIORI): ${carried.length}/52 carried resolve by exact name; ` +
      `${unsupported.length} unsupported no-op & report; ${drivenMp.length}/${mpEmitted.length} MediaPipe ` +
      `categories drive; no renames.`,
  );
  console.log(
    'NOTE: manifest.run_state == "' + runState + '" — NOTHING here is measured. The FINAL, ' +
      'authoritative check reruns against the POD-BUILT GLB once the manifest flips to ' +
      '"measured-on-pod"; the driver then treats that GLB\'s morphTargetDictionary as the ' +
      'source of truth and any pod-demoted weak shape (eyeSquint*/eyeWide*/noseSneer*) folds ' +
      'into the unsupported/no-op set automatically — no code change.',
  );
} else {
  console.log(
    `\nPASS (MEASURED): ${carried.length}/52 pod-built morph targets resolve by exact name; ` +
      `${unsupported.length} unsupported no-op & report; ${drivenMp.length}/${mpEmitted.length} MediaPipe ` +
      `categories drive; no renames.`,
  );
}
