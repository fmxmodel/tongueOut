# Reconstruction Report — Track B1 (FLAME 2023 Open, COMMERCIAL)

> Owner: `face-reconstructor` (Fable) · Date: 2026-07-04, rev. 2026-07-05 · Track: **B1 [ACTIVE]**
>
> **STATUS: STAGE 1 GREEN ON THE POD; STAGES 2–4 REVISED FOR THE PKL-ONLY OPEN
> RELEASE, RERUN PENDING.** Reported from the pod run (RTX 6000 Ada): stage 1
> `recon.landmarks` completed (MediaPipe FaceLandmarker → 478 landmarks →
> `landmarks.npz`). The staged FLAME 2023 Open download was **measured to be
> pkl-only** (`flame2023.pkl` + readme; NO UV template, NO landmark embedding —
> see §0b). Stages 2–3 were revised locally (authoring box, **no compute run**:
> `python -m py_compile` only) to run from the pkl alone; numeric artifacts for
> stages 2–4 remain **DEFERRED to the pod rerun** — none was fabricated.
>
> **§0b. PKL-ONLY RELEASE + CLEAN-ROOM SUBSTITUTES (provenance, for
> `license-compliance`).** The FLAME 2023 Open (CC-BY-4.0) package ships ONLY
> `flame2023.pkl` (measured keys: `f (9976,3)`, `v_template (5023,3)`,
> `shapedirs (5023,3,400)`, `posedirs (5023,3,36)`, `weights (5023,5)`,
> `J_regressor (5,5023)`, `kintree_table`, `J`, …). The landmark embedding and
> UV layout it does NOT ship are **self-authored / generated clean-room** —
> NOT pulled from DECA/EMOCA/TF_FLAME/flame-fitting or any NC FLAME source
> (which would re-taint the commercial base, `compliance_report.md` §B1-1):
> - **Landmark anchors:** `recon/flame_landmarks.py` derives the iBUG static-51
>   anchors geometrically from the pkl's own arrays (see §4). Persisted to
>   `out/recon/lmk_embedding_static51.npz`; a staged release file is honored as
>   an optional override.
> - **UV layout:** `recon/uv_unwrap.py` generates a deterministic **xatlas
>   (MIT)** unwrap of `v_template`+`f` (see §7); persisted with a `uv_source`
>   provenance field in `uv_coords.npz`; a staged UV template obj is honored as
>   an optional override (topology-checked, hard STOP on mismatch).

## 0. Method (fixed by the licensing gate — do not substitute)

**Optimization-based FLAME 2023 Open (CC-BY-4.0) fit to MediaPipe FaceLandmarker
landmarks** — pure optimization, no learned reconstruction weights (NO
DECA/EMOCA/Arc2Avatar/InsightFace), differentiable rasterization/camera via
**PyTorch3D 0.7.8 (BSD-3-Clause)** (NO nvdiffrast, NO 3DGS).
**Texture = per-subject albedo baked from the photo** into FLAME UV space +
mirror-symmetry fill + `cv2.inpaint` (classical). **NO statistical albedo
prior** (no MPI FLAME texture space, no FLAME_albedo_from_BFM, no AlbedoMM, no
Basel). Governing licensing: `out/compliance_report.md` (TRACK B1).

## 1. Code authored (this box) vs artifacts produced (pod only)

| File (authored, local) | Role |
|---|---|
| `recon/config.py` | paths (match `scripts/pod_setup_b1.sh` exactly), FLAME dims, hyperparameters, artifact names |
| `recon/pod_guard.py` | every entrypoint refuses to compute without an NVIDIA GPU |
| `recon/mp_flame_correspondence.py` | **self-authored** MediaPipe-478 → iBUG-68 index table (§4) |
| `recon/landmarks.py` | stage 1: MediaPipe FaceLandmarker → `landmarks.npz` + verification overlay |
| `recon/flame_model.py` | **self-authored** FLAME 2023 loader (chumpy-free) + differentiable LBS decoder |
| `recon/flame_landmarks.py` | **self-authored clean-room** iBUG static-51 anchor derivation from the pkl arrays (the Open release ships no embedding) + persisted anchors for the rig |
| `recon/uv_unwrap.py` | **clean-room** UV: xatlas (MIT) unwrap of `v_template`+`f` (the Open release ships no UV), with staged-template override + per-corner realignment proof |
| `recon/fit_flame.py` | stage 2: 3-phase optimization → `neutral.ply`, `faces.npy`, `id_params.npz`, `expression_basis.npz`, `lmk_embedding_static51.npz` |
| `recon/bake_texture.py` | stage 3: photo → UV albedo (visibility, mirror, inpaint) → `albedo.png` |
| `recon/verify_outputs.py` | stage 4: measured asserts → `recon_run_manifest.json` (exit ≠ 0 on any FAIL) |
| `scripts/run_recon_b1.sh` | turnkey pod runner (GPU-guarded, per-stage or `all`) |

| Artifact (in `out/recon/`) | Status | Produced by |
|---|---|---|
| `input_image.png` (EXIF-normalized canonical pixels) | **DEFERRED (pod)** | stage 1 |
| `landmarks.npz`, `landmarks_debug.png`, `mediapipe_blendshapes_photo.json` | **DEFERRED (pod)** | stage 1 |
| `neutral.ply` (neutral head, FLAME topology, face-count-verified) | **DEFERRED (pod)** | stage 2 |
| `faces.npy` (**the topology contract**, int32 (F,3)) | **DEFERRED (pod)** | stage 2 |
| `id_params.npz` (fitted identity + photo-state + camera) | **DEFERRED (pod)** | stage 2 |
| `expression_basis.npz` + `expression_basis_notes.json` (rigger handle) | **DEFERRED (pod)** | stage 2 |
| `fit_summary.json`, `fit_debug/*.png` | **DEFERRED (pod)** | stage 2 |
| `lmk_embedding_static51.npz` (anchors the fit used; consumed by the rig) | **DEFERRED (pod)** | stage 2 |
| `flame_landmarks_selfauthored.{json,png}` (anchor derivation debug, human-check) | **DEFERRED (pod)** | stage 2 |
| `albedo.png`, `albedo_mask.png`, `uv_coords.npz` (with `uv_source`), `bake_summary.json` | **DEFERRED (pod)** | stage 3 |
| `recon_run_manifest.json` (measured verdict) | **DEFERRED (pod)** | stage 4 |

## 2. Exact GPU-run command sequence (on the pod)

```bash
# 0. copy the repo to the pod; put the photo at /workspace/inputs/random-person.jpeg
# 1. provision (idempotent; installs torch 2.4.1+cu121, pytorch3d 0.7.8, mediapipe 0.10.14)
bash scripts/pod_setup_b1.sh
# 2. MANUAL, license-gated: download FLAME 2023 (Open) ONLY -> /workspace/models/flame2023_open/
#    (models/README.md §1 — do NOT download the CC-BY-NC-SA texture package)
# 3. run the reconstruction (activates /workspace/venvs/b1 itself):
bash scripts/run_recon_b1.sh            # all 4 stages
#    or stage-by-stage:
bash scripts/run_recon_b1.sh landmarks  # MediaPipe -> landmarks.npz (+ overlay to eyeball)
bash scripts/run_recon_b1.sh fit        # optimization -> neutral.ply / faces.npy / params / basis
bash scripts/run_recon_b1.sh bake       # photo -> out/recon/albedo.png
bash scripts/run_recon_b1.sh verify     # measured asserts -> recon_run_manifest.json
```

Human checkpoint after `landmarks`: open `out/recon/landmarks_debug.png` and confirm
the 68 numbered picks sit on the right facial features (§4) before trusting the fit.

## 3. Topology contract (the guarantee downstream lives on)

- **Source of truth: `out/recon/faces.npy`** — extracted **on the pod** from the FLAME
  2023 Open pkl. **The pkl is the sole authority** (the Open release ships no template
  obj); if the operator stages one anyway, `fit_flame.py::maybe_check_template_topology`
  cross-checks it and any mismatch is a hard STOP, never papered over. The generated UV
  is likewise proven to index the contract (`uv_unwrap.py`: `vmapping[faces_uv] ==
  faces`, plus `verify_outputs.py`'s `faces_verts_idx == faces.npy` re-check).
- **Expected values — FLAME 2023 Open template (verify on pod): N_v = 5023 vertices,
  N_f = 9976 triangles.** These come from FLAME's published documentation, not from a
  local run; `verify_outputs.py` records the **measured** counts in
  `recon_run_manifest.json` and flags any divergence from the documented expectation.
  Downstream must assert against `faces.npy`, not against these prose numbers.
- Every future mesh (all 52 ARKit expression meshes, the GLB base mesh) MUST reuse
  `faces.npy` verbatim. `expression_basis.npz["faces"]` is stored identical and
  re-checked by `verify_outputs.py`. If topology drifts anywhere: STOP (CLAUDE.md
  invariant 3).
- `neutral.ply` is face-count-verified twice (at export in `fit_flame.py`, again in
  `verify_outputs.py`) — a vertex-only PLY is rejected loudly.

## 4. MediaPipe ↔ FLAME landmark correspondence (explicit, self-authored)

- FLAME side — **self-authored clean-room** (the Open release ships NO embedding;
  NC embeddings are barred): `recon/flame_landmarks.py` derives the **iBUG static-51**
  (17–67) anchors deterministically from the pkl's own arrays, in IOD-scaled units
  (IOD = eyeball-joint distance):
  - **eyes** — rest joints 3/4 are the eyeball centers (verified by measurement:
    the vertices rigidly skinned to them, `weights[:,3|4]>0.5`, centroid at the
    joint); the lid-margin ring = skin vertices hugging the eyeball sphere; canthi =
    ring x-extremes, lid points at 1/3–2/3 x-stations above/below eye height;
  - **nose** — tip = max-+z sub-eye near-midline vertex; nasion = midline z-dip at
    eye height; bridge = midline snaps between them; subnasale = midline z-concavity
    below the tip; alar wings = x-extremes of the nose protrusion zone;
  - **mouth** — the closed lips are near-coincident sheets split by JAW skinning
    weight (lower lip rides joint 2): seam pairs (w≤0.35 vs w≥0.65 within 0.08·IOD)
    give the inner-lip line; corners = seam x-extremes; outer lips = vermilion
    z-crests traced per x-station on the correct jaw-weight side;
  - **brows** — anthropometric stations (0.55·IOD above eye centers, arched, spanning
    canthus x-range) snapped to the local browridge z-crest — the lowest-weight (0.8)
    group, so approximate-but-correct-feature anchors suffice.
  Anchors are **vertex-snapped** (one-hot barycentric; ≤ ~half-edge ≈ 2–4 mm
  quantization, up to ~5 mm semantic slack on the heuristic stations — acceptable for
  a weighted smooth-L1 optimization data term). Hard measured sanity gates (laterality
  signs, vertical ordering, nose protrusion, mouth width vs IOD) STOP the run if the
  geometry defies the derivation. Human-check material: the fit overlays plus
  `out/recon/flame_landmarks_selfauthored.{json,png}` (per-anchor vertex, xyz, rule).
  Contour 0–16 is **not fabricated**: the jaw-contour term stays disabled (static-51
  only). A staged release embedding file is honored as an optional override
  (loaded variant-tolerantly by `flame_model.load_landmark_embedding`).
- **The anchors actually used are persisted** to `out/recon/lmk_embedding_static51.npz`
  (with a `source` field); `rig/build_arkit_shapes.py` consumes that file, so the fit
  and the rig share identical anchor definitions by construction.
- MediaPipe side: `recon/mp_flame_correspondence.py::MEDIAPIPE_IBUG68` maps each iBUG
  index to one of MediaPipe's 478 landmark indices. Groups: contour 0–16 (weight 0.3),
  brows 17–26 (0.8), nose 27–35 (1.5), eyes 36–47 (2.0), outer lips 48–59 (1.5),
  inner lips 60–67 (1.0). Full per-index list with semantic comments lives in that file.
- **Provenance (flagged to `license-compliance`, per `models/README.md` §3):** the
  table is **self-authored** against MediaPipe's published canonical face-mesh topology
  (Apache-2.0); no third-party correspondence file was copied. Verification is
  **visual, on the pod**, via `landmarks_debug.png` (all 68 picks drawn + numbered).
  Local check was text-level only: 68 entries, 0 duplicates, max index 454 < 478.

## 5. Fit design (stage 2) — seams made explicit

- **Camera:** OpenCV pinhole (+X right, +Y down, +Z forward), principal point at image
  center, single focal (optimized, tethered to `1.5·max(H,W)` px because one image
  cannot resolve focal/depth). The SAME convention feeds PyTorch3D in the bake via
  `cameras_from_opencv_projection` — one convention, two stages, no seam.
- **Parameterization:** FLAME global_orient (init π about X: FLAME +Y-up faces the
  OpenCV camera), transl (z init from outer-canthus pixel distance vs 0.09 m),
  betas(300), expression(100), jaw_pose. Neck/eye poses fixed at 0 (neck is redundant
  with global under a single photo; MediaPipe eyelids don't constrain gaze).
- **Stages:** A rigid+camera (stable-core subset) → B +identity → C +expression/jaw,
  Adam, smooth-L1 on residuals normalized by image diagonal, L2 priors on
  betas/expression/jaw/log-focal. **The photo's expression is fitted so it does NOT
  leak into identity**; the exported neutral is betas-only with expression=0, pose=0.
- **Identity is fixed once fitted** — downstream varies only expression/jaw
  (`id_params.npz["betas"]` is the subject; photo-state arrays are labeled non-identity).
- FLAME decoder is a from-scratch implementation of the published FLAME/SMPL LBS
  equations (`recon/flame_model.py`) — smplx/FLAME_PyTorch/DECA code is non-commercial
  and was NOT vendored. Legacy chumpy pickles load via a stub unpickler (no chumpy dep).
- Hyperparameters are documented starting points; tune on the pod against
  `fit_debug/stage_*_overlay.png` and `fit_summary.json` (RMSE px, per-landmark errors).

## 6. Handoff to `arkit-rigger` (their stage: plan §3.2)

The rigger consumes `expression_basis.npz` + `expression_basis_notes.json` (+
`id_params.npz`, `faces.npy`, and optionally `recon/flame_model.py` with the pod's
`flame2023.pkl` to reproduce exact fit-time math).

- **FLAME's expression space is NOT ARKit-named.** It is 100 PCA axes from 4D scans;
  no axis ≈ one ARKit shape. Jaw shapes (jawOpen/Left/Right/Forward) come from
  `jaw_pose` — an articulated joint through LBS — not the linear basis. Solving a
  FLAME (expression, jaw_pose) coefficient vector per ARKit name, and building deltas
  as `decode(betas_fit, expr_k, jaw_k) − decode(betas_fit, 0, 0)`, is the rigger's job.
  Name contract: `out/arkit_51_52_map.json` (52 exact, case-sensitive spellings).
- **Candidate-unsupported (rigger adjudicates — coverage NOT fabricated here):**

  | ARKit shape | Why flagged |
  |---|---|
  | `tongueOut` | FLAME has **no tongue geometry** — expect UNSUPPORTED (honest flat key) |
  | `cheekPuff` | cheek inflation poorly spanned by scan-based expression PCA — likely weak |
  | `cheekSquintLeft` / `cheekSquintRight` | weak in FLAME expression space — verify visually |
  | `noseSneerLeft` / `noseSneerRight` | nose wrinkle weakly represented — verify visually |

- `eyeLook{Up,Down,In,Out}{Left,Right}` map to FLAME **eye joint rotations** (joints
  3/4), not the PCA. **Eye-joint laterality is unverified** — rotate joint 3 alone,
  render, record which eye moved, before naming anything Left/Right.
- `mediapipe_blendshapes_photo.json` is the photo's expression state (QA reference
  only) — it says nothing about rig coverage.

## 7. Texture (stage 3) notes

- Bake = UV-space gather: self-contained UV rasterizer → per-texel 3D point →
  project into photo → visible iff depth-consistent (PyTorch3D depth buffer,
  ±5 mm), camera-facing (winding auto-measured, not assumed), non-grazing
  (cos ≥ 0.15). Then mirror-symmetry fill (template-space x→−x, KD-tree, ≤4 mm
  match) → `cv2.inpaint` (TELEA) → mean-color outside UV islands.
- `albedo_mask.png` legend: 255 direct / 170 mirror / 85 inpaint / 0 outside —
  QA can measure exactly how much of the face is photo-grounded
  (`verify_outputs.py` gates direct fraction > 10% of covered texels).
- **Honest caveat for QA:** delighting priors are license-barred, so `albedo.png` is
  the photo's *shaded appearance* used as baseColor (sRGB), not physically de-lit
  albedo.
- UV/V convention (for `blender-glb-builder`): `albedo.png` row r ↔ v = 1−(r+0.5)/T
  (OBJ v=1 is the TOP image row); UVs shipped in `uv_coords.npz`
  (`verts_uvs`, `faces_uv_idx`, `faces_verts_idx`, `uv_source`).
- **UV source (pkl-only Open release):** the Open release ships NO UV, and the NC
  packages that do are barred. `recon/uv_unwrap.py` therefore **generates the UV
  clean-room with xatlas (MIT, added to requirements-b1.txt)** from the pkl's
  `v_template`+`f` — deterministic (fixed input, default options), persisted once in
  `uv_coords.npz` as the single UV authority for bake + GLB. xatlas seam-splits and
  face reordering are realigned per-corner to `faces.npy` and PROVEN by the
  `vmapping[faces_uv] == faces` identity (hard STOP on failure; >1% degenerate UV
  triangles is also a STOP). A staged UV-bearing `head_template.obj` is honored as an
  optional override (topology-checked). We never needed FLAME's *specific* layout —
  only *one consistent* parameterization for the per-subject photo bake.

## 8. Local verification performed (structure/syntax ONLY — nothing executed)

- `bash -n scripts/run_recon_b1.sh` → OK; script marked executable.
- `python3 -m py_compile` on all 9 `recon/*.py` modules → OK (byte-compile only; no
  imports of torch/mediapipe/pytorch3d were executed).
- **Rev. 2026-07-05 (pkl-only adaptation):** `python3 -m py_compile` re-run on all
  `recon/*.py` (incl. new `flame_landmarks.py`, `uv_unwrap.py`) and
  `rig/build_arkit_shapes.py` → OK. The geometric anchor derivation and the xatlas
  unwrap have NOT executed anywhere yet — first execution is on the pod; both carry
  hard measured sanity gates + debug artifacts precisely because of that.
- Correspondence table checked by text parsing (grep/awk): 68 entries, 0 duplicates,
  max index 454 < 478.
- **NOT verified locally (impossible without compute, deferred to pod):** FLAME pkl
  key layout on the actual Open release, landmark-embedding variant, fit convergence,
  hyperparameter quality, bake visibility fractions, eye-joint laterality, measured
  topology counts.

## 9. Gate status

- **QA gate (Phase 2):** run `bash scripts/run_recon_b1.sh` on the pod, then
  `qa-verifier` reads `recon_run_manifest.json` (verdict is measured; exit ≠ 0 on any
  failed check) + eyeballs `landmarks_debug.png` and `fit_debug/` overlays.
- **License gate:** FLAME 2023 **Open** (CC-BY-4.0) confirmed as the ONLY base the
  code will load (paths hard-target `/workspace/models/flame2023_open/`; NC texture
  package never referenced). **New since rev. 2026-07-05 (flagged to
  `license-compliance`):** because the Open release is pkl-only, (a) the iBUG
  static-51 anchors are self-authored clean-room from the pkl arrays (§0b/§4), (b)
  the UV is generated clean-room with xatlas — **MIT** — added to
  `requirements-b1.txt` (§7), and (c) the MediaPipe↔iBUG correspondence table remains
  self-authored (§4, unchanged). No NC FLAME asset (embedding, template obj, texture
  space) is referenced, read, or consulted anywhere in the pipeline. CC-BY
  attribution obligations unchanged.
