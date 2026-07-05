# Viewer / Driver Report — Track B1 (FLAME 2023 Open), Phase 5

> Owner: `viewer-driver` (Opus 4.8) · Date: 2026-07-04 · Track: **B1** · Base: **FLAME 2023
> Open (CC-BY-4.0), commercial** · Run: **COMMERCIAL**
> Mode this session: **RE-TARGET + HONESTY-OF-DOCS** (B2 → B1). No GPU, **no MediaPipe
> FaceLandmarker inference**, no webcam opened, **no real GLB required**. The web app is
> re-pointed at the B1 contract and **bundle-verified** + **static-name-verified**; the live
> run (camera/model + real `out/head_arkit.glb`) is deferred to the **GPU pod**.
>
> Cross-refs: **`out/shapes/arkit_manifest.json`** (authoritative B1 shape contract, from
> `arkit-rigger`), `out/shapes/rig_report.md`. App: `out/viewer/` (README there has full run
> docs). The legacy B2 name map `out/arkit_51_52_map.json` is retained only for back-compat.

---

## 1. What was re-targeted (B2 → B1)

The viewer's **core design was already correct and B1-compatible** — drive by exact,
case-sensitive `categoryName` → `morphTargetDictionary` lookup; skip + log any unresolved
name; no renames; no hardcoded morph order. `src/driver.js` was **not touched**. This pass
re-pointed the *contract* and *docs* from the superseded B2 target (a 51-baked MetaHuman GLB,
`tongueOut` the sole gap) to the B1 reality (a FLAME-supported subset, **4+ honest no-ops**):

| File | Change |
|---|---|
| `scripts/verify-names.mjs` | **Retargeted** to reconcile the pure driver against `out/shapes/arkit_manifest.json` (B1 supported set) instead of the B2 51-name list. Handles `run_state`: while `DEFERRED-pod-run` it verifies the **a-priori** classification and prints that the final check reruns on the pod GLB; when `measured-on-pod` it verifies the measured `supported === true` set and auto-folds any pod-demoted shape into the no-op set. |
| `package.json` | `description` rewritten from "51 baked ARKit morph targets" to the B1 reality (drive up to 52 MediaPipe ARKit categories vs the FLAME-supported morph targets; `tongueOut`/`cheekPuff`/`cheekSquint*` + any pod-demoted are honest no-ops). |
| `src/config.js` | `contractUrl` re-pointed to `out/shapes/arkit_manifest.json` (B1 contract) for the on-screen coverage/unresolved panels — **read-only, never used to remap**. Loader stays back-compatible with the legacy map. Header B2 → B1. `glbUrl` **unchanged** (`out/head_arkit.glb`). |
| `src/main.js` | Coverage + unresolved panels made **B1-aware**: any MediaPipe name that doesn't resolve is now **labeled expected-unsupported (per manifest) vs UNEXPECTED naming-drift**, so `qa-verifier` isn't false-alarmed by the known no-ops. `window.__viewer.state` now exposes `expectedUnsupported` + `contractRunState`. Driver path unchanged. |
| `vite.config.js` | Added a dev/preview mount for `/shapes/arkit_manifest.json` (served from `out/`). |
| `index.html`, `README.md` | B2 → B1 wording. |

Name resolution remains **by name only** — no hardcoded shape order, no remap/alias table
(CLAUDE.md invariant #2). The loaded GLB's `morphTargetDictionary` is the **sole source of
truth at runtime**; the manifest is consulted only to *label* honesty, never to drive.

## 2. Which of the 52 ARKit names get driven — B1 (a-priori, per manifest)

Measured by `npm run verify-names` (static; exercises the real `src/driver.js` against
`out/shapes/arkit_manifest.json`, `run_state: DEFERRED-pod-run`):

| | count | names |
|---|---|---|
| **Expected-driven** (carried by the FLAME-supported rig, resolve 1:1 by exact name) | **48 / 52** | **14 eye** (Blink/LookDown/LookIn/LookOut/LookUp/Squint/Wide × L,R), **4 jaw** (jawForward, jawLeft, jawRight, jawOpen), **23 mouth** (Close, Funnel, Pucker, Right, Left, Smile L/R, Frown L/R, Dimple L/R, Stretch L/R, RollLower, RollUpper, ShrugLower, ShrugUpper, Press L/R, LowerDown L/R, UpperUp L/R), **5 brow** (Down L/R, single bilateral **browInnerUp**, OuterUp L/R), **2 nose** (noseSneer L/R) |
| **Honestly-unsupported — a-priori** | **4** | **`tongueOut`**, **`cheekPuff`**, **`cheekSquintLeft`**, **`cheekSquintRight`** |
| **Renamed / aliased to hide a gap** | **0** | none |

**Of the 48 expected-driven, 6 are `intended: "weak-attempt"`** — `eyeSquintLeft`,
`eyeSquintRight`, `eyeWideLeft`, `eyeWideRight`, `noseSneerLeft`, `noseSneerRight`. The pod's
leakage/isolation gates adjudicate them; **any that fail are demoted to unsupported** and the
manifest is overwritten. The remaining 42 are `intended: "strong"`.

### Runtime "unresolved MediaPipe" set (the qa-verifier signal)

MediaPipe FaceLandmarker v2 emits **51 ARKit categories + `_neutral`** and **never emits
`tongueOut`**. So at live-run time against a B1 GLB, the *expected* unresolved-from-MediaPipe
set (emitted by MediaPipe but not carried by the GLB) is:

- **a-priori: `cheekPuff`, `cheekSquintLeft`, `cheekSquintRight`** (3) — honest no-ops,
  labeled *expected unsupported (B1)* in the panel and in `window.__viewer.state`.
- **`tongueOut`** does **not** appear here (MediaPipe doesn't emit it); it's unsupported in
  the GLB but simply never reaches the driver from tracking.
- **plus any pod-demoted weak shape** — if e.g. `noseSneerLeft` is demoted, it too becomes an
  expected no-op and is folded in automatically (manifest-driven; no code change).

**Anything unresolved beyond that manifest-declared set = a real upstream naming-contract
violation** (naming drift / build gap) — surfaced in the panel as **UNEXPECTED — naming
drift** and in `window.__viewer.state.unresolved` (minus `expectedUnsupported`) for
`qa-verifier` to chase.

### Honesty caveats the rig records (viewer drives all by name regardless)

- **`browInnerUp`** is a single bilateral shape (Apple/MediaPipe both ship one) — resolves
  1:1, no split, no remap.
- **`mouthClose`** is linearized at `jawOpen` (correct only *in combination* with jawOpen).
- **eye-look** shapes are eye-joint rotation only (lids don't follow — honest limitation).
- **`jawForward`** is a synthesized LBS-weighted jaw-joint translation (FLAME's jaw is
  rotation-only). These are the rigger's documented approximations; the driver just writes
  the influence by name.

## 3. Verification done this session (no inference)

| Check | Command | Result |
|---|---|---|
| Static name-resolution (retargeted) | `npm run verify-names` | **PASS (A-PRIORI)** — 48/52 carried resolve by exact name; 4 unsupported (`cheekPuff, cheekSquintLeft, cheekSquintRight, tongueOut`) no-op & report; 48/51 MediaPipe categories drive; `_neutral` skipped; no renames; influences written by name (order-independent, dict built reversed); EMA correct. Prints the `DEFERRED-pod-run` note that the final check reruns on the pod GLB. |
| Production bundle | `npm run build` (vite 6.4.3) | **PASS** — 11 modules transformed, `dist/` emitted, wasm auto-copied to `public/mediapipe/wasm` (only the pre-existing three+mediapipe chunk-size warning). |

**NOT run** (deferred to the GPU pod, per the contamination guard): live FaceLandmarker
inference, webcam capture, and loading a real GLB. Those code paths exist and bundle; they
were **not executed**. **Final, authoritative name resolution reruns against the pod-built
`out/head_arkit.glb`** — whose `morphTargetDictionary` is the source of truth — once the
manifest flips to `run_state: measured-on-pod`; re-run `npm run verify-names` there to lock it.

## 4. Run command (recap; full docs in `out/viewer/README.md`)

```bash
cd out/viewer
npm install          # already satisfied in the scaffold
npm run verify-names # static B1 name check (no GPU / no inference)
npm run build        # bundle check
npm run dev          # http://localhost:5173  (predev copies MediaPipe wasm locally)
# GLB: auto-loaded from out/head_arkit.glb (drop the pod-built head there; else graceful msg)
# Source: UI dropdown Webcam vs Video file, or ?source=webcam / ?video=<url>&autostart=1
# Model: `npm run fetch-model` for offline, else auto CDN fallback at runtime
```

## 5. Headless / verification swap (webcam → screenshot path)

The webcam is one of several inputs. A Puppeteer/Playwright harness drives the rig with **no
camera and no MediaPipe model** through the *same* name-resolving driver:

```js
await window.__viewer.ready;                    // GLB + contract loaded
window.__viewer.injectBlendshapes([             // one frame; identical driver path as live
  { categoryName: 'jawOpen', score: 0.8 },
  { categoryName: 'mouthSmileLeft', score: 0.6 },
  { categoryName: 'cheekPuff', score: 1.0 },    // B1: no-op, logged as expected-unsupported
]);
// screenshot window.__viewer.renderer.domElement (preserveDrawingBuffer is enabled)
window.__viewer.state;   // {glbLoaded, morphTargets, unresolved, expectedUnsupported, contractRunState, …}
```

Also: `?video=<url>` for a deterministic clip; `window.__viewer.setInfluence(name, v)` for a
single-shape poke. Because the injection path is the production driver, a screenshot proves
the actual name→influence mapping, not a stub. To swap the webcam for CI, feed a recorded
blendshape trace into `injectBlendshapes(frame)` on a timer — no model, reproducible pixels.

## 6b. Commercial attribution / third-party notices — NOW WIRED (compliance AC-9(b) item (b))

> Added 2026-07-04 by `viewer-driver` (Opus 4.8) at the direction of `license-compliance`, which
> flagged attribution wiring as **the one concrete remaining *build* item** for the commercial B1
> goal (it was previously **ABSENT** — see `compliance_report.md` §AC-6, §AC-9(b), and the
> "Conditions that flip Track B1 to SHIP-CLEARED: yes" list item 2). Pure web/docs work: no GPU,
> no inference. This closes the *code/build* half of that condition; the non-code halves remain open
> (see the restatement at the end of this section).

**What was bundled — `out/viewer/public/THIRD-PARTY-NOTICES.md`** (Vite copies `public/` verbatim
into `dist/`, so it ships as `dist/THIRD-PARTY-NOTICES.md`, 368 lines / ~21 KB), plus verbatim
license texts under `public/licenses/` → `dist/licenses/`:

| Notice | Content bundled | Source |
|---|---|---|
| **FLAME 2023 Open — CC-BY-4.0** | The **exact** credit + FLAME-paper citation from `compliance_report.md` §B1-1 (Max Planck Institute; link `https://creativecommons.org/licenses/by/4.0/`; "the model was fit, re-textured, and rigged — changes were made"; cite Li, Bolkart, Black, Li, Romero, "Learning a model of facial shape and expression from 4D scans," ACM TOG (SIGGRAPH Asia) 2017) | verbatim from §B1-1 |
| **MediaPipe `@mediapipe/tasks-vision` — Apache-2.0** | Full **Apache-2.0 license text** inline in the notices file **and** as `licenses/mediapipe-Apache-2.0-LICENSE.txt` (218 lines, incl. the upstream Lucent/UTF addendum), plus the MediaPipe Authors/Google LLC copyright attribution | fetched from the **upstream `google-ai-edge/mediapipe` repo `LICENSE`** (the npm tarball ships neither LICENSE nor NOTICE — confirmed: `node_modules/@mediapipe/tasks-vision/` has only `package.json`, `README.md`, the bundles, `.d.ts`, and `wasm/`). **Required because `dist/mediapipe/wasm/*` redistributes the MediaPipe binaries.** |
| **three.js — MIT** | Full MIT copyright + permission notice inline **and** as `licenses/three.js-MIT-LICENSE.txt` (copied from `node_modules/three/LICENSE`, "Copyright © 2010-2024 three.js authors") | in-bundle banner ("Copyright 2010-2024 Three.js Authors") **also** confirmed still present in `dist/assets/index-*.js` and listed explicitly |
| **PyTorch3D (BSD-3) / OpenCV (Apache-2.0) / Blender (GPL)** | Documented **recorded decision**: these are **pod-side build tools**; their notices are required only if that pod code/binary is redistributed, which the shipped product (GLB + web viewer) does **not** do — stated as an informed, recorded scoping decision, not an omission, with the trigger to revisit | — |

**MediaPipe NOTICE sourcing (important):** Apache-2.0 §4(a) (bundle a copy of the License) is
satisfied by the bundled full text. Apache-2.0 §4(d) (propagate a NOTICE file) is **not triggered**
because the upstream `google-ai-edge/mediapipe` repo **has no NOTICE file** — verified this session
by (i) the repo-root file listing (LICENSE present, no NOTICE) and (ii) a repo-wide code search for
a `NOTICE` file returning **zero** results. The notices file records this finding explicitly and
flags that a future MediaPipe NOTICE must be reproduced if one appears.

**In-app credits/licenses affordance (CC-BY-4.0 "reasonably visible" requirement).** A **"Credits /
Licenses"** button was added to the controls panel in `index.html`; it opens an in-app modal
(`#credits` panel + backdrop, Esc/close/backdrop to dismiss; wired in `src/main.js`) that surfaces
the FLAME CC-BY credit + FLAME-paper cite **as visible on-screen text** (not just a repo file), the
MediaPipe Apache-2.0 + three.js MIT attributions, the pod-tools scoping note, and links to the full
`THIRD-PARTY-NOTICES.md` and the two `licenses/*.txt`. This makes the CC-BY credit reasonably visible
in the running product, as CC-BY-4.0 requires.

**Re-verified this session (measurement, not claim):**

| Check | Command | Result |
|---|---|---|
| Production bundle | `npm run build` | **PASS (exit 0)** — 11 modules transformed; `dist/` emitted. `dist/THIRD-PARTY-NOTICES.md` (21,533 B), `dist/licenses/mediapipe-Apache-2.0-LICENSE.txt` (12,331 B) + `dist/licenses/three.js-MIT-LICENSE.txt` (1,081 B) present; three.js banner still in `dist/assets/index-*.js`; credits affordance + visible FLAME credit present in `dist/index.html`. |
| Driver name-resolution (no regression) | `npm run verify-names` | **PASS (exit 0)** — unchanged from §3: 48/52 carried resolve by exact name; 4 a-priori unsupported (`cheekPuff, cheekSquintLeft, cheekSquintRight, tongueOut`) no-op & report; 48/51 MediaPipe categories drive; no renames. The docs/notices work did **not** touch `src/driver.js`. |

**This does NOT by itself flip `SHIP-CLEARED` to yes.** Only AC-9(b) item (b) (attribution wiring)
is now done. The governing verdict in `compliance_report.md` stays **`SHIP-CLEARED: no`** for Track
B1 because the non-code conditions remain **open**:
- **(a) FLAME 2023 Open asset-provenance confirmation** — operator must download the CC-BY "FLAME
  2023 (Open)" release and confirm **in writing** that the UV template **and** landmark-embedding
  files come from the **Open (CC-BY) release itself**, not the CC-BY-NC-SA texture package or a
  DECA/EMOCA mirror (AC-9(a)). Not adjudicated here.
- **(d) Input-photo rights** — ownership + subject consent + biometric/publicity/GDPR/BIPA exposure
  in `random-person.jpeg` is a **human-legal** question qualified counsel must clear (AC-9(d)). Not
  adjudicated here.
- Plus (c) the pod-verified image-baked-only texture (`no_statistical_albedo_prior: true`). Notices
  wiring touches none of these. **Provenance + photo-rights still gate the ship.**

## 6. Handoff for `qa-verifier`

- App: `out/viewer/` (turnkey; needs the **pod-built** `out/head_arkit.glb` at live-run time).
- The **48-driven / 4-unsupported (a-priori)** split above is *measured by* `npm run
  verify-names`, not claimed — reproducible offline.
- On the pod, after the GLB + `measured-on-pod` manifest land: re-run `npm run verify-names`
  (auto-switches to the measured supported set and folds in demotions), then during the live
  run watch `window.__viewer.state`. The **only expected** unresolved-from-MediaPipe names are
  `cheekPuff`, `cheekSquintLeft`, `cheekSquintRight` plus any pod-demoted weak shape (all in
  `expectedUnsupported`). **Any name in `unresolved` but not in `expectedUnsupported` = naming
  drift / build gap to chase** — the viewer flags it *UNEXPECTED* on-screen.
