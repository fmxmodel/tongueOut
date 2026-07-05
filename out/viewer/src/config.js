// Central, editable configuration for the Track B1 (FLAME 2023 Open) viewer/driver.
//
// Everything an operator needs to point the viewer at real artifacts lives here.
// No ARKit shape name is hardcoded for driving — names always resolve by lookup
// against the loaded GLB's `morphTargetDictionary` (see src/driver.js). These are
// only *asset locations* and *tuning defaults*.

const BASE = import.meta.env.BASE_URL || '/';

export const CONFIG = {
  // --- The rigged head. Served at runtime from the pipeline bus `out/head_arkit.glb`
  //     via the dev-server middleware in vite.config.js (URL `/head_arkit.glb`).
  //     If the file does not exist yet (GPU/real-artifact stage deferred), the
  //     viewer degrades gracefully with a clear on-screen message. ---
  glbUrl: BASE + 'head_arkit.glb',

  // --- The B1 ARKit shape CONTRACT, authored by arkit-rigger:
  //     `out/shapes/arkit_manifest.json` (all 52 ARKit names + supported/unsupported +
  //     run_state). Loaded READ-ONLY for the on-screen coverage/unresolved panels so
  //     they can distinguish an EXPECTED honest no-op (a-priori/pod-demoted
  //     unsupported) from real naming drift. It is NEVER used to remap or rename — the
  //     loaded GLB's `morphTargetDictionary` is the sole source of truth for driving.
  //     (Loader also accepts the legacy `arkit_51_52_map.json` shape for back-compat.) ---
  contractUrl: BASE + 'shapes/arkit_manifest.json',

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
  // Smoothing is temporal inertia in [0,1): 0 = raw (instant, jittery),
  // 0.6 = balanced default, ->1 = very smooth/laggy. Applied as an EMA per shape.
  smoothingDefault: 0.6,

  // MediaPipe emits `_neutral` alongside the 51 ARKit shapes; it is not a morph
  // target and must be skipped (never resolves, would only clutter the log).
  skipNames: ['_neutral'],
};
