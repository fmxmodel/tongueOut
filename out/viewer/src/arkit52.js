// The ARKit-52 name contract — the single source of truth shared by:
//   - the GLB morph-target names (mesh.extras.targetNames in head_arkit_v2.glb),
//   - MediaPipe FaceLandmarker `categoryName` values,
//   - the three.js driver keys (morphTargetDictionary lookups).
//
// These strings are LOAD-BEARING and case-sensitive. They are copied verbatim from
// the pipeline contract `newstack/pipe/arkit_names.py` (ARKIT_52). Never edit spelling
// to hide a mismatch — that is exactly the naming drift the verifier exists to catch.
//
// This module is PURE (no three.js, no DOM), so scripts/verify-names.mjs imports it
// under plain Node and the browser imports it too — one list, checked both ways.

export const ARKIT_52 = [
  'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight',
  'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight', 'eyeBlinkLeft', 'eyeBlinkRight',
  'eyeLookDownLeft', 'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft',
  'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft', 'eyeSquintRight',
  'eyeWideLeft', 'eyeWideRight', 'jawForward', 'jawLeft', 'jawOpen', 'jawRight',
  'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft', 'mouthFrownRight',
  'mouthFunnel', 'mouthLeft', 'mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthPressLeft',
  'mouthPressRight', 'mouthPucker', 'mouthRight', 'mouthRollLower', 'mouthRollUpper',
  'mouthShrugLower', 'mouthShrugUpper', 'mouthSmileLeft', 'mouthSmileRight', 'mouthStretchLeft',
  'mouthStretchRight', 'mouthUpperUpLeft', 'mouthUpperUpRight', 'noseSneerLeft', 'noseSneerRight',
  'tongueOut',
];

if (ARKIT_52.length !== 52) throw new Error(`ARKit contract must be 52, got ${ARKIT_52.length}`);
if (new Set(ARKIT_52).size !== 52) throw new Error('ARKit contract has duplicate names');

// MediaPipe FaceLandmarker emits 52 categories = `_neutral` + 51 ARKit blendshapes.
// The ONE ARKit-52 name it never emits is `tongueOut` (per the model card). So the
// avatar's tongueOut morph is present and mappable but simply won't be driven by the
// webcam — it is exercised manually via the UI slider. This is EXPECTED, not drift.
export const MP_NOT_EMITTED = new Set(['tongueOut']);

// MediaPipe emits `_neutral` alongside the 51 ARKit shapes; it is not a morph target
// and must be skipped (it would never resolve and would only clutter the drift log).
export const MP_SKIP = ['_neutral'];

// The 51 ARKit names MediaPipe actually emits (derived, not hand-maintained).
export const MP_EMITTED = ARKIT_52.filter((n) => !MP_NOT_EMITTED.has(n));
