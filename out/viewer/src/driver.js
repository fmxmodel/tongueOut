// The driver — the load-bearing 30 lines of Phase 5.
//
// This module is deliberately PURE: it imports nothing (no three.js, no DOM), so
// it can be exercised statically by scripts/verify-names.mjs under plain Node as
// well as in the browser. Its only contract with three.js is the duck-typed shape
// `{ morphTargetDictionary: {name->index}, morphTargetInfluences: number[] }`.
//
// INVARIANT (CLAUDE.md #2): morph targets are addressed by EXACT, case-sensitive
// name. There is no order assumption and no remap/alias table. A MediaPipe
// `categoryName` that does not resolve is reported as honestly-unsupported — never
// silently coerced onto a different shape to hide a naming-contract gap.

/**
 * Apply one frame of MediaPipe blendshape categories to a morph-target mesh by
 * exact-name lookup, with optional temporal smoothing (EMA).
 *
 * @param {{morphTargetDictionary: Record<string,number>, morphTargetInfluences: number[]}} mesh
 * @param {Array<{categoryName: string, score: number}>} categories
 * @param {{ smoothing?: number, skipNames?: string[] }} [opts]
 *   smoothing: inertia in [0,1). new = prev*s + target*(1-s). 0 = instant.
 *   skipNames: names to ignore entirely (e.g. MediaPipe's `_neutral`).
 * @returns {{ driven: string[], unresolved: string[], skipped: string[] }}
 */
export function applyBlendshapes(mesh, categories, opts = {}) {
  const smoothing = clamp01(opts.smoothing ?? 0);
  const skip = new Set(opts.skipNames ?? []);
  const dict = mesh.morphTargetDictionary || {};
  const influences = mesh.morphTargetInfluences || [];

  const driven = [];
  const unresolved = [];
  const skipped = [];

  for (const c of categories) {
    const name = c.categoryName;
    if (skip.has(name)) {
      skipped.push(name);
      continue;
    }
    const idx = dict[name];
    if (idx === undefined) {
      // Contract violation OR an expected 52nd shape (tongueOut) that this GLB
      // does not carry. Either way: no-op cleanly and REPORT it. Never alias.
      unresolved.push(name);
      continue;
    }
    const target = clamp01(c.score); // MediaPipe scores are 0..1; clamp defensively
    const prev = influences[idx] ?? 0;
    influences[idx] = smoothing > 0 ? prev * smoothing + target * (1 - smoothing) : target;
    driven.push(name);
  }

  return { driven, unresolved, skipped };
}

/**
 * Resolve a set of ARKit names against a morph-target dictionary WITHOUT touching
 * influences. Used by the static verifier and the on-screen coverage panel to
 * answer "which of the 52 would actually be driven?" ahead of any inference.
 *
 * @param {Record<string,number>} dict  morphTargetDictionary (name -> index)
 * @param {string[]} names              ARKit names to test (e.g. Apple's 52)
 * @returns {{ resolved: string[], unresolved: string[] }}
 */
export function resolveNames(dict, names) {
  const d = dict || {};
  const resolved = [];
  const unresolved = [];
  for (const name of names) {
    if (d[name] !== undefined) resolved.push(name);
    else unresolved.push(name);
  }
  return { resolved, unresolved };
}

function clamp01(x) {
  const n = Number(x);
  if (!Number.isFinite(n)) return 0;
  return n < 0 ? 0 : n > 1 ? 1 : n;
}
