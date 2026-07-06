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
│       ├── s4_build_shapes.py # additive ARKit blendshapes from ICT deltas + gated tongueOut synth
│       ├── tongue_synth.py    # tongueOut delta from ICT's real static tongue (cKDTree select)
│       ├── _selftest_tongue.py    # offline synthetic-geometry test for tongue_synth (18 checks)
│       ├── s5_bake_texture.py # bake photo → ICT UVs  (FIX IN PROGRESS: winding/visibility)
│       ├── s6_export_blender.py   # GLB (52/52 morphs) + .blend
│       └── s7_verify_glb.py   # stdlib GLB parser: morph count, names vs contract, texture
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
- [ ] Re-run `STAGES="4 5 6 7"`, re-render (correct front = Blender −Y), QA the real GLB (52/52)
- [ ] Wire `head_arkit_v2.glb` into `out/viewer/` (three.js + MediaPipe); ICT→ARKit driver name map
- [ ] Ship prep: THIRD-PARTY-NOTICES (rembg/U²-Net, TripoSR, ICT-FaceKit © USC-ICT 2020, MediaPipe, three.js, Draco)

## Pod
RunPod RTX 6000 Ada. Pipeline at `/workspace/newstack/pipe/`; ICT/TripoSR at `/workspace/newstack/`; Blender 4.2.3 (headless via `xvfb-run`).
