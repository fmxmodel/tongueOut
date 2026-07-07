// Central, editable configuration for the ARKit avatar viewer/driver (new stack:
// ICT-FaceKit-based head_arkit_v2.glb, all 52 ARKit morph targets present).
//
// No ARKit shape name is hardcoded for DRIVING — names always resolve by lookup
// against the loaded GLB's `morphTargetDictionary` (see src/driver.js). The 52-name
// contract lives in src/arkit52.js and is used only to LABEL coverage, never to remap.
// These entries are just asset locations and tuning defaults.

const BASE = import.meta.env.BASE_URL || '/';

export const CONFIG = {
  // --- The rigged head. Copied from the pipeline bus `out/head_arkit_v2.glb` into
  //     public/ by `npm run copy-model` (runs automatically on predev/prebuild) so it
  //     is served locally in dev AND emitted into dist/ on a production build. 52 ARKit
  //     morph targets, three opaque materials (HeadMat + EyeMat + RestMat), real eyes,
  //     tongueOut. If the copy is missing the viewer degrades with a clear message. ---
  glbUrl: BASE + 'head_arkit_v2.glb',

  // --- MediaPipe FaceLandmarker WASM runtime. Copied out of
  //     node_modules/@mediapipe/tasks-vision/wasm into public/mediapipe/wasm by
  //     `npm run copy-assets` (runs automatically on predev/prebuild). Offline-first. ---
  wasmBase: BASE + 'mediapipe/wasm',

  // --- FaceLandmarker model (~3-4 MB). Not vendored in node_modules. `npm run
  //     fetch-model` downloads it into public/models. If the local copy is absent
  //     the loader transparently falls back to the Google CDN below. ---
  modelAssetPath: BASE + 'models/face_landmarker.task',
  modelAssetPathCdn:
    'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',

  // --- Driver tuning ---
  // Smoothing is temporal inertia in [0,1): 0 = raw (instant, jittery), 0.6 = balanced
  // default, ->1 = very smooth/laggy. Applied as an EMA per shape.
  smoothingDefault: 0.6,

  // Apply MediaPipe head pose (facialTransformationMatrixes rotation) to the head group.
  // Nice-to-have; toggleable in the UI. Off by default so the head stays framed.
  applyHeadPoseDefault: false,

  // MediaPipe emits `_neutral` alongside the 51 ARKit shapes; skip it (never a morph).
  skipNames: ['_neutral'],
};
