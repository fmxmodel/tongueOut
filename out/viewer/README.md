# ARKit Avatar Viewer / Driver (new stack — head_arkit_v2, 52/52)

Loads the rigged head `out/head_arkit_v2.glb` in three.js and drives its
`morphTargetInfluences` live from **MediaPipe FaceLandmarker** blendshapes — by
**exact, case-sensitive `categoryName` → `morphTargetDictionary` lookup**. No shape
order is assumed; nothing is renamed. The GLB is a single mesh (`HeadARKit`) split into
**three opaque primitives** (`HeadMat` + `EyeMat` + `RestMat`), each carrying the **same
52 ARKit morph targets**; three.js loads it as three morph-bearing meshes and the driver
pushes every frame to **all of them**, resolving each target by name per-mesh (the
per-primitive index order can differ — that's why we never assume an index).

All **51** MediaPipe-emitted ARKit categories resolve **1:1** to a morph target. The one
ARKit name MediaPipe never emits is **`tongueOut`** — it is present and mappable in the
GLB, just not driven by the webcam, so it has a dedicated **manual slider**. Supports
**live webcam** and a **supplied video file**, with influence **smoothing** and an
optional **head-pose** rotation from `facialTransformationMatrixes`.

Pinned: `three@0.170.0`, `@mediapipe/tasks-vision@0.10.14`, `vite@^6`.

## Install & run

```bash
cd out/viewer
npm install
npm run dev            # http://localhost:5173
```

`predev`/`prebuild` stage the local assets automatically (no runtime CDN):
- `copy-model.mjs`   → copies `out/head_arkit_v2.glb` into `public/` (served at `/head_arkit_v2.glb`).
- `copy-mediapipe-assets.mjs` → copies the MediaPipe WASM runtime into `public/mediapipe/wasm`.

So the app is turnkey. To point at a different GLB, edit `glbUrl` in `src/config.js`.

### Webcam vs. video file

In the UI: **Source** → `Webcam (live)` or `Video file` (pick a file) → **Start**. Or via
URL params (handy for automation):

```
http://localhost:5173/?source=webcam&autostart=1
http://localhost:5173/?video=/clips/subject.mp4&autostart=1&smoothing=0.5&headpose=1
```

### UI

- **Start / Stop**, **Reset morphs**, live **FPS** readout + MediaPipe status.
- **mirror** — flips the preview (and, when on, the head-pose yaw so the avatar turns with you).
- **head pose** — applies MediaPipe head rotation to the head group (off by default; toggleable).
- **Smoothing** slider (0–0.95): temporal EMA per shape, `new = prev*s + target*(1-s)`.
  `0` = raw/instant (jittery), `0.6` = default, higher = smoother/laggier.
- **tongueOut** slider (always available — webcam never drives it).
- **Manual morph sliders (all 52)** — collapsible; exercise any morph without a camera.
  While the webcam runs the 51 tracked morphs are overwritten each frame; tongueOut holds.

## The MediaPipe model

The ~3–4 MB `face_landmarker.task` is **not** vendored. Either:

```bash
npm run fetch-model    # downloads into public/models/ for fully-offline use
```

or skip it — the app **falls back to the Google CDN** at runtime automatically
(`modelAssetPathCdn` in `src/config.js`). The WASM runtime is always served locally.

## Verify (no GPU / no inference / no webcam needed)

```bash
npm run verify-names   # PARSES out/head_arkit_v2.glb and exercises src/driver.js
npm run build          # production bundle (prebuild runs verify-names → FAILS on mismatch)
```

`verify-names` reads the **real GLB** (stdlib GLB reader) and proves, by measurement:
the head mesh carries **exactly 52** morph targets; their names are **set-equal to the
ARKit-52 contract** (`src/arkit52.js`) with no renames / extras / missing; **every
primitive** exposes all 52; the pure driver resolves **all 51** MediaPipe categories 1:1
by name (against a dict with **reversed** indices, to prove order-independence); exactly
**one** morph (`tongueOut`) is webcam-undriven; `_neutral` is skipped; nothing is aliased.
It is wired as `prebuild`, so a mismatch **fails `npm run build`**.

## Headless / verification path (swap out the webcam)

The webcam is only one input. A screenshot/QA harness can drive the rig with **no camera
and no MediaPipe model** by injecting blendshape frames through the *same* driver:

```js
// in a Puppeteer/Playwright page, after window.__viewer.ready resolves:
await window.__viewer.ready;                       // GLB loaded
window.__viewer.injectBlendshapes([                // one frame of {categoryName, score}
  { categoryName: 'jawOpen',        score: 0.8 },
  { categoryName: 'mouthSmileLeft', score: 0.6 },
]);
window.__viewer.setInfluence('tongueOut', 1.0);    // drive the webcam-undriven morph
// then screenshot window.__viewer.renderer.domElement (preserveDrawingBuffer is on)
window.__viewer.state;   // {glbLoaded, morphTargets, morphTargetCount, contract52, unresolved, …}
```

Other swaps:
- **Supplied video** instead of live webcam: `?video=<url>&autostart=1`, or
  `window.__viewer.loadVideoUrl(url)`.
- **Recorded blendshape trace**: feed frames from a JSON capture into `injectBlendshapes`
  on a timer — deterministic, no model, reproducible pixels.

`window.__viewer.state.unresolved` surfaces any MediaPipe `categoryName` that found no
morph target (naming drift for `qa-verifier`); for this GLB it stays empty.

## Attribution / third-party licenses

The head asset base changed from FLAME to **ICT-FaceKit (Light, MIT)** + **TripoSR (MIT)**
in the new commercial stack; the **commercial gate of record is `out/compliance_newstack.md`**.

- **In-app:** the **"Credits / Licenses"** button opens a modal crediting ICT-FaceKit (MIT,
  © USC-ICT 2020), TripoSR (MIT), MediaPipe (Apache-2.0), and three.js (MIT), with links.
- `public/licenses/*.txt` (→ `dist/licenses/`) — verbatim `mediapipe-Apache-2.0-LICENSE.txt`
  and `three.js-MIT-LICENSE.txt`.
- `public/THIRD-PARTY-NOTICES.md` still carries the FLAME-era text and is **owned by the
  licensing agent** — updating it to the ICT/TripoSR notices is that agent's task; this
  viewer change only corrects the user-facing in-app credit and points to `compliance_newstack.md`.

## Files

- `index.html` — UI (source toggle, start/stop, reset, FPS, smoothing, head-pose, tongueOut +
  full manual sliders, coverage + unresolved panels, Credits modal).
- `src/main.js` — three.js scene, multi-primitive GLB load (opaque-hardened, graceful degrade),
  MediaPipe VIDEO-mode setup, driver loop, head pose, manual sliders, headless hook.
- `src/driver.js` — **pure** name-resolution + smoothing + clamp (no three/DOM; reused by the verifier).
- `src/arkit52.js` — the 52-name ARKit contract + MediaPipe-not-emitted set (shared by app + verifier).
- `src/config.js` — asset locations + tuning (the only place you edit paths).
- `vite.config.js` — excludes the MediaPipe wasm from bundling; static build.
- `scripts/` — `copy-mediapipe-assets.mjs`, `copy-model.mjs`, `fetch-model.mjs`, `verify-names.mjs`.
