# ARKit Avatar Viewer / Driver (Track B1 — FLAME 2023 Open, Phase 5)

Loads the rigged head `out/head_arkit.glb` (ARKit-named morph targets — the
FLAME-supported subset of Apple's 52) in three.js and drives its
`morphTargetInfluences` from **MediaPipe FaceLandmarker** blendshapes — by **exact,
case-sensitive `categoryName` → `morphTargetDictionary` lookup**. No shape order is
assumed; nothing is renamed. The loaded GLB's `morphTargetDictionary` is the **sole
source of truth** for what can be driven: the driver drives all 52 MediaPipe ARKit
categories and **no-ops + logs** any name the GLB doesn't carry. For B1 that honestly
covers the a-priori-unsupported shapes (`tongueOut`, `cheekPuff`, `cheekSquintLeft`,
`cheekSquintRight`) plus **any shape the pod later demotes**. Supports **live webcam**
and a **supplied video file**, with influence **smoothing**.

Pinned: `three@0.170.0`, `@mediapipe/tasks-vision@0.10.14`, `vite@^6`.

## Install & run

```bash
cd out/viewer
npm install            # already done in the scaffold; re-run only if node_modules is gone
npm run dev            # http://localhost:5173  (predev copies MediaPipe wasm locally)
```

`npm run dev` serves the app AND (via `vite.config.js` middleware) serves the pipeline-bus
artifacts straight from `out/`:

- `GET /head_arkit.glb`   → streamed from `out/head_arkit.glb`; **clean 404** with a clear
  message if it doesn't exist yet (the GLB is produced later on the GPU pod).
- `GET /shapes/arkit_manifest.json` → the B1 ARKit shape contract (from `arkit-rigger`), read
  read-only so the coverage/unresolved panels can label an **expected** no-op vs real drift.
  (The legacy `GET /arkit_51_52_map.json` map is still mounted for back-compat.)

So the app is **turnkey**: drop the exported head at `out/head_arkit.glb`, reload, and it
loads. Until then it shows a graceful "not produced yet" message and keeps the UI live.

### Point it at the GLB

Nothing to configure — it loads from `out/head_arkit.glb` automatically. To use a different
path, edit `glbUrl` in `src/config.js`.

### Webcam vs. video file

In the UI: **Source** dropdown → `Webcam (live)` or `Video file` (then pick a file) → **Start**.
Or via URL params (handy for automation):

```
http://localhost:5173/?source=webcam&autostart=1
http://localhost:5173/?video=/clips/subject.mp4&autostart=1&smoothing=0.5
```

### Smoothing

Slider (0–0.95). Temporal EMA per shape: `new = prev*s + target*(1-s)`.
`0` = raw/instant (jittery), `0.6` = default, higher = smoother/laggier.

## The MediaPipe model

The ~3–4 MB `face_landmarker.task` is **not** vendored. Either:

```bash
npm run fetch-model    # downloads into public/models/ for fully-offline use
```

or skip it — the app **falls back to the Google CDN** at runtime automatically
(`modelAssetPathCdn` in `src/config.js`). The WASM runtime is always served locally
(`npm run copy-assets`, run automatically on predev/prebuild).

## Verify (no GPU / no inference needed)

```bash
npm run verify-names   # static: exercises src/driver.js against out/shapes/arkit_manifest.json
npm run build          # bundle check (vite build)
```

`verify-names` reconciles the pure driver against the B1 manifest: it proves every
**carried** (FLAME-supported) name resolves 1:1 by exact name, the **unsupported** names
(`tongueOut`, `cheekPuff`, `cheekSquintLeft`, `cheekSquintRight` a-priori, plus any
pod-demoted) no-op cleanly and are reported, and nothing is renamed/aliased. It also
handles `run_state`: while the manifest is `DEFERRED-pod-run` it checks the **a-priori**
classification and prints that the **final, authoritative check reruns against the
pod-built GLB** once the manifest flips to `measured-on-pod`.

## Headless / verification path (swap out the webcam)

The webcam is only one input. A screenshot/QA harness can drive the rig with **no camera
and no MediaPipe model** by injecting blendshape frames through the *same* driver:

```js
// in a Puppeteer/Playwright page, after window.__viewer.ready resolves:
await window.__viewer.ready;                       // GLB + contract loaded
window.__viewer.injectBlendshapes([                // one frame of {categoryName, score}
  { categoryName: 'jawOpen',        score: 0.8 },
  { categoryName: 'mouthSmileLeft', score: 0.6 },
]);
// then screenshot window.__viewer.renderer.domElement (preserveDrawingBuffer is on)
window.__viewer.state;                             // {glbLoaded, morphTargets, unresolved, …}
```

Other swaps:
- **Supplied video** instead of live webcam: `?video=<url>&autostart=1`, or
  `window.__viewer.loadVideoUrl(url)`.
- **Recorded blendshape trace**: feed frames from a JSON capture into
  `injectBlendshapes(frame)` on a timer — deterministic, no model, reproducible pixels.

`window.__viewer.state.unresolved` surfaces any MediaPipe `categoryName` that found no
morph target, and `window.__viewer.state.expectedUnsupported` lists the contract's declared
no-ops — so `qa-verifier` can tell an **expected** unresolved name (B1 unsupported /
pod-demoted) from real naming drift.

## Attribution / third-party licenses (commercial B1)

This product redistributes/builds on third-party components, so it ships their required
notices and surfaces them in-app:

- `public/THIRD-PARTY-NOTICES.md` (→ `dist/THIRD-PARTY-NOTICES.md`) — the bundled notices:
  **FLAME 2023 Open** (CC-BY-4.0 credit + FLAME-paper citation + "changes were made"),
  **MediaPipe `@mediapipe/tasks-vision`** (full Apache-2.0 text; sourced from the upstream
  `google-ai-edge/mediapipe` `LICENSE` because the npm tarball ships no LICENSE/NOTICE, and
  required because `dist/mediapipe/wasm/*` redistributes the binaries; upstream has **no**
  NOTICE file so §4(d) is not triggered), **three.js** (MIT), and a recorded note that
  pod-side build tools (PyTorch3D/OpenCV/Blender) are **not** shipped so their notices are
  not required by the web product.
- `public/licenses/*.txt` (→ `dist/licenses/`) — verbatim `mediapipe-Apache-2.0-LICENSE.txt`
  and `three.js-MIT-LICENSE.txt`.
- **In-app:** the **"Credits / Licenses"** button (bottom of the controls panel) opens a modal
  showing the FLAME CC-BY credit + cite and the MediaPipe/three.js attributions, with links to
  the full notices — satisfying CC-BY-4.0's "reasonably visible in the product" requirement.

Wiring attribution does **not** by itself clear the ship: FLAME 2023 Open asset provenance and
input-photo rights remain open per `out/compliance_report.md` (Track B1 is still
`SHIP-CLEARED: no`).

## Files

- `index.html` — UI (source toggle, start/stop, smoothing, coverage + unresolved panels,
  Credits / Licenses modal).
- `src/main.js` — three.js scene, GLB load (graceful degrade), MediaPipe VIDEO-mode setup,
  driver loop, headless hook.
- `src/driver.js` — **pure** name-resolution + smoothing (no three/DOM imports; reused by
  the static verifier).
- `src/config.js` — asset locations + tuning (the only place you edit paths).
- `vite.config.js` — serves `out/` artifacts; excludes wasm from bundling.
- `scripts/` — `copy-mediapipe-assets.mjs`, `fetch-model.mjs`, `verify-names.mjs`.
