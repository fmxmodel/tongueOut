# TODO — Image → ARKit Avatar → GLB (newARC)

Snapshot of the working tree. Two pipelines live here; the **new stack** is the active one.

## Folder map
```
newARC/
├── newstack/                  # ACTIVE — TripoSR + ICT-FaceKit commercial pipeline
│   ├── run_newstack.sh        # orchestrator: STAGES="1..7", REFINE, TEX_SIZE, DRACO
│   └── pipe/
│       ├── arkit_names.py     # ARKit-52 contract + ICT→ARKit name map (52/52: 51 OBJ + 1 synth)
│       ├── mp_ibug68.py       # MediaPipe-478 → iBUG/Multi-PIE-68 correspondence
│       ├── common.py          # numpy core: OBJ io, rasterizer, weak-persp cam, Umeyama, smoothing
│       ├── ict_loader.py      # ICT FaceXModel → cache npz (neutral + id modes + 51 expr + UVs + landmarks)
│       ├── s1_landmarks.py    # MediaPipe landmarks on the photo
│       ├── s2_fit_identity.py # fit ICT identity coeffs (linear model) to the photo landmarks
│       ├── s3a_align_clay.py  # align TripoSR clay → ICT space (pytorch3d view sweep + Umeyama)
│       ├── s3b_refine_blender.py  # gated shrinkwrap onto smoothed clay (hair/proportions; face protected)
│       ├── s4_build_shapes.py # additive ARKit blendshapes from ICT deltas + gated tongueOut/gaze synth
│       ├── tongue_synth.py    # tongueOut delta from ICT's real static tongue (cKDTree select)
│       ├── gaze_synth.py      # eyeball-rotation deltas for eyeLook* (ICT OBJs move lids only)
│       ├── _selftest_tongue.py    # offline synthetic-geometry test for tongue_synth (18 checks)
│       ├── s5_bake_texture.py # bake photo → ICT UVs + eye_left/right.png (photo-derived iris)
│       ├── eye_texture.py     # iris/pupil/sclera sampled from photo; procedural pole-centered disc
│       ├── s6_export_blender.py   # GLB (52/52 morphs, HeadMat+EyeMat, opaque-hardened) + .blend
│       ├── s7_verify_glb.py   # stdlib GLB parser: morphs/names/materials-opaque vs contract
│       └── s8_render_previews.py  # proof renders from the GLB (front/back/eyes/gaze)
├── recon/ rig/ blender_build_rig.py   # ALPHA (FLAME 2023 Open pipeline; superseded, kept on branch `alpha`)
├── out/                       # reports, manifests, head_arkit.glb (FLAME), head_arkit_v2.glb (new stack)
│   ├── compliance_newstack.md # licensing: NEWSTACK-SHIP-CLEARED conditional→yes (4 build items)
│   └── ...
└── open_avatar.py             # local Blender launcher (imports a GLB)
```

## Status
- [x] New-stack env on pod (TripoSR + ICT-FaceKit + torchmcubes + rembg)
- [x] TripoSR clay from the photo (hair volume + child proportions; rough, as expected)
- [x] ICT-FaceKit Light (MIT): neutral + **51 pre-authored ARKit blendshapes** (blink/smile/frown/brows)
- [x] Full pipeline runs end-to-end → `head_arkit_v2.glb`, s7 PASS
- [x] Identity fit tight to the photo; geometry is a real win over FLAME (real head + hair, not a bald adult)
- [x] Licensing cleared (conditional): pin rembg `u2net`, ICT **Light only**, no bpy binary shipped, wire notices

## In progress / next
- [x] **s5 texture bake FIX authored** (pending pod re-run) — winding was inverted for ICT topology → dark central face. Now **measures** the camera-facing sign (like `recon/bake_texture.py`), rejects grazing texels + X-mirror fallback, exterior-priority UV rasterization so interior islands can't steal face texels, and a `central_face` sanity gate in `bake_metrics.json`. Verify `winding.facing_sign` + `central_face.pass` after re-run.
- [x] **tongueOut → 52/52** — CODED (pending pod re-run of s4+): `tongue_synth.py` selects the tongue
  from ICT's real static geometry (region `[14062:17039)` verts with cKDTree distance-to-nearest-tooth
  `[17039:21451)` > 1.0 cm, no hardcoded ids), pushes the tip `+4.5 cm` forward (`smoothstep**1.5`
  root→tip weighting, small `+0.8` y-lift) so tip z ≈ 13.6 > lips ≈ 11.9. `s4` gates it: delta EXACTLY 0
  outside the tongue set, tip final z > lip-front z (measured via mouth landmark verts 48–67), else die.
  Source `"synthesized-ict-tongue"`; manifest → 52 supported / 0 unsupported. Offline self-test
  (`_selftest_tongue.py`, synthetic geometry): 18/18 PASS. Verify on pod: `[tongue]` log line ~760 verts,
  centroid ≈ (0,−3.8,4.4), manifest `shapes.tongueOut.synth`.
- [x] **Eyes fixed** (was: blank white stare) — eyeball UVs overlap the face UVs, so eyes get
  DEDICATED photo-derived textures (`eye_texture.py`: iris color from MediaPipe iris ring
  468-472/473-477, pupil from darkest central quartile, sclera from brightest decile of the
  eye opening; procedural disc at UV (0.5,0.5) = eyeball forward pole, iris_uv_radius 0.110)
  bound to `EyeMat` in s6. ICT's transparent-purpose eye shells (lacrimal/blend/occlusion,
  verts [24591,25351)) are stripped at export — measured, they sat in FRONT of the iris and
  rendered as skin-textured lids once opaque. `eyeLook*` morphs now rotate the eyeballs
  (`gaze_synth.py`, In/Out 35°, Up 25°, Down 30°) because ICT's OBJs move lids only
  (measured eyeball delta 0.0). Proof renders: `out/renders/glb_*.png`.
- [x] **UDIM tile-1+ → RestMat** (kills the pre-existing stretched-face back of head) — ICT
  UVs are multi-tile (u up to 7); the bake fills tile 0 only and the wrapping sampler painted
  the skull back/teeth/mouth-socket with the FACE image. Polys with max corner u > 1 now use
  `RestMat` driven by s5 per-vertex colors (photo where visible, TripoSR clay hair, honest
  flat teeth/mouth defaults, eye sockets = shadowed skin) exported as COLOR_0.
- [x] **Opaque export hardened** — all materials single-sided (glTF doubleSided=false) +
  GLB JSON post-pass writes EXPLICIT `alphaMode:"OPAQUE"`; s6 re-imports the GLB and
  measures functional opacity (alpha socket unlinked & 1.0, DITHERED, culling on; fails on
  BLEND). Note: Blender 4.2's `blend_method` is a deprecated alias reading HASHED for ALL
  materials (even factory-new) — functional opacity is the real gate. s7 fails any material
  that is not explicitly OPAQUE/single-sided/textured.
- [x] Re-run `STAGES="4 5 6 7 8"` on pod — s7 PASS (52/52 names, 2 primitives, both
  materials OPAQUE), renders show iris+pupil+sclera, gaze morph moves the iris, back solid
- [x] Wire `head_arkit_v2.glb` into `out/viewer/` (three.js + MediaPipe) — loads the 52/52 GLB
  (3 opaque primitives HeadMat/EyeMat/RestMat), drives `morphTargetInfluences` by exact
  `categoryName`→`morphTargetDictionary` on ALL primitives (no remap: names ARE ARKit-52 1:1).
  `verify-names` parses the real GLB → 52/52, 51 MediaPipe categories resolve, tongueOut manual
  slider (MediaPipe never emits it); prebuild gates the build; `npm run build` PASS. UI: webcam/
  video, smoothing, head-pose toggle, FPS, tongueOut + full manual sliders. Run: `cd out/viewer
  && npm install && npm run dev`.
- [ ] Ship prep: THIRD-PARTY-NOTICES (rembg/U²-Net, TripoSR, ICT-FaceKit © USC-ICT 2020, MediaPipe, three.js, Draco)

## Decisions (don't re-litigate)
- **"Hole in the back of the head" was NOT geometry** — it was Blender's glTF importer assigning
  `blend_method=HASHED` + `show_transparent_back=True`, making the skin see-through so the internal
  teeth/tongue/eyeballs showed through. Proven: raw ICT, our pre-/post-shrinkwrap neutrals, and the
  exported GLB with a flat opaque material all render a **solid** closed back; topology is closed and
  identical to ICT (23 boundary loops) at every stage. Fixes: `open_avatar.py` forces opaque on import;
  s6 hardens the GLB to `alphaMode:"OPAQUE"` + single-sided. Diagnostic renders in `out/renders/`.
- **TRELLIS.2 rejected as the clay generator** — (a) it wouldn't fix the above (we always retopologize
  onto ICT, so the clay never becomes the output surface), and (b) it depends on `nvdiffrast`/`nvdiffrec`
  under NVIDIA's 1-Way *non-commercial* license, which breaks the commercial requirement (the exact NC
  trap avoided by choosing PyTorch3D). Keep TripoSR (MIT).

## Pod
RunPod RTX 6000 Ada. Pipeline at `/workspace/newstack/pipe/`; ICT/TripoSR at `/workspace/newstack/`; Blender 4.2.3 (headless via `xvfb-run`).
