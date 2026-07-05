# Rig Report — Track B1 (FLAME 2023 Open → ARKit-52)

> Owner: `arkit-rigger` (Fable) · Date: 2026-07-04 · Track: **B1 [ACTIVE]**
>
> **STATUS: RIG STAGE AUTHORED, EXECUTION DEFERRED TO THE GPU POD.** The pod
> (RTX 6000 Ada) is unreachable (SSH key absent, `out/gpu_requirements_b1.md` §8).
> Per the contamination guard, **nothing was run locally**: no torch, no FLAME
> decode, no solve, and **not one `.ply` delta mesh was fabricated**. Every
> mesh and every measured number below is **DEFERRED**. Local verification was
> syntax/structure only (`bash -n`, `python3 -m py_compile`, stdlib JSON
> cross-checks of the name contract).

## 0. What this stage is

The connective tissue between reconstruction and export: map FLAME's
**anonymous 100-axis expression PCA + jaw/eye pose joints** onto Apple's
**52 exactly-spelled ARKit blendshapes** as topology-locked delta meshes, so
`blender-glb-builder` can do index-aligned shape keys and MediaPipe driving
later works by pure name lookup. Name authority: `out/arkit_51_52_map.json`
(52 case-sensitive spellings; never renamed, never aliased).

## 1. Code authored (this box) vs artifacts produced (pod only)

| File (authored, local) | Role |
|---|---|
| `rig/config.py` | paths aliased from `recon.config` (interface cannot drift), activations, solver + gate thresholds (all `B1_RIG_*` env-overridable) |
| `rig/arkit_spec.py` | **the FLAME→ARKit correspondence** — self-authored, license-clean, pure-stdlib spec: per-shape pose recipes / landmark targets / honest unsupported declarations |
| `rig/build_arkit_shapes.py` | pod stage 1: topology asserts → laterality measurement → pose calibration → PCA ridge solves → measured gates → `expr_*.ply` + measured `arkit_manifest.json` + `shape_params.npz` |
| `rig/verify_shapes.py` | pod stage 2: independent re-read of every artifact, byte-level topology compare, verdict in `shapes_run_manifest.json`, exit ≠ 0 on any FAIL |
| `rig/author_local_manifest.py` | local (stdlib-only) author of the DEFERRED manifest from the same spec — zero drift between placeholder and pod build |
| `scripts/run_rig_b1.sh` | turnkey pod runner (GPU-guarded, recon-preflight, `build`/`verify`/`all`) |

| Artifact (in `out/shapes/`) | Status | Produced by |
|---|---|---|
| `arkit_manifest.json` | **AUTHORED (DEFERRED placeholders)** — pod overwrites with measured values | local now / pod build |
| `neutral.ply` (byte-identical pass-through of `out/recon/neutral.ply`) | **DEFERRED (pod)** | build |
| `expr_<arkitName>.ply` × (up to 48) | **DEFERRED (pod)** — none exists, none fabricated | build |
| `shape_params.npz` (solved coeffs / calibrated poses per shape) | **DEFERRED (pod)** | build |
| `shapes_run_manifest.json` (measured verdict) | **DEFERRED (pod)** | verify |

## 2. Exact pod-run command sequence

```bash
# prereq: recon completed on the pod (bash scripts/run_recon_b1.sh) so that
# out/recon/{neutral.ply,faces.npy,id_params.npz,expression_basis.npz,landmarks.npz}
# exist, and FLAME 2023 Open is at /workspace/models/flame2023_open/.
bash scripts/run_rig_b1.sh          # build + verify
#   or stage-by-stage:
bash scripts/run_rig_b1.sh build    # python -m rig.build_arkit_shapes
bash scripts/run_rig_b1.sh verify   # python -m rig.verify_shapes (exit!=0 on FAIL)
```

Human checkpoint after `build`: eyeball a few supported shapes (import
`expr_mouthSmileLeft.ply` etc. next to `neutral.ply`) and confirm the motion
is on the correct feature and the correct side before the Blender stage.

## 3. Method — FLAME→ARKit correspondence (license-clean, self-authored)

FLAME's 100 expression components are anonymous PCA axes; **no non-commercial
mapping assets were used** (no DECA/EMOCA, no third-party FLAME→ARKit
coefficient tables, no MetaHuman reference deltas). Three mechanisms:

1. **Pose shapes (12)** — `jawOpen`, `jawLeft/Right` drive FLAME's jaw joint;
   the 8 `eyeLook*` drive the eye joints — all through the reconstructor's own
   LBS math (`recon.flame_model.batch_rodrigues`/`_rigid_transform_chain`,
   imported, not reimplemented). **Axes and signs are never assumed**: the
   build decodes candidate rotations and measures which way lips/eyeballs
   moved (jaw pitch sign from lower-inner-lip drop; lateral axis chosen
   between Y/Z by measured lateral dominance; eye-look signs from measured
   front-of-eyeball displacement). `jawForward` is special: FLAME's jaw joint
   is rotation-only, so protrusion is synthesized as an **LBS-weighted +Z
   translation of the jaw joint** (`verts += w_jaw · t`, 6 mm) — a documented
   rig-style approximation, not PCA fabrication.
2. **PCA shapes (36)** — each shape is a sparse displacement target on the
   **iBUG-51 landmark set that FLAME 2023 Open itself anchors on the mesh**
   (the CC-BY landmark embedding, already used by the fit). Targets are
   authored from Apple's public textual blendshape descriptions (facts) with
   subject-relative magnitudes measured on the neutral mesh (mouth width,
   eye aperture, brow height). Solve: ridge-regularized least squares over
   the 100 axes, `min ‖W(Me−d)‖² + λ‖e‖²`, with a coefficient-norm cap
   (λ-escalation) so no shape extrapolates outside FLAME's plausible span.
   The landmark Jacobian `M` is **exact at rest pose** (expression is linear
   pre-LBS). `mouthClose` is linearized **at the jawOpen pose** via a batched
   secant Jacobian: delta = decode(jawOpen, e_seal) − decode(jawOpen, 0),
   which is correct when driven *together with* `jawOpen` (Apple's semantics).
3. **Unsupported (4 a priori)** — declared, never faked (below).

### Laterality (never assumed, measured three ways)

- Neutral-mesh check: all 8 left/right handle groups (eyes, brows, mouth
  corners, nostrils) must sit on the expected side of x=0 under the
  FLAME/SMPL "+X = subject-left" convention — hard STOP otherwise.
- Photo cross-check: the fitted photo-state is decoded and both eye groups
  projected with the fitted camera; their left/right pixel ordering must
  match MediaPipe's own left/right eyes from `landmarks.npz` — hard STOP
  otherwise (this closes the loop against the driver's laterality).
- Eye joints: joint 3 vs 4 assigned to Left/Right by the sign of their rest
  x-coordinates in `joints_neutral` (asserted opposite), resolving the
  reconstructor's flagged "eye-joint laterality unverified".

### Measured acceptance gates (honesty enforcement)

Every attempted shape must pass, at pod runtime: non-trivial delta
(max ≥ 0.5 mm), target direction cosine ≥ 0.5, amplitude ratio in [0.3, 3.0],
mirror-side leakage ≤ 0.6× target (one-sided shapes), off-target motion ≤
1.0× target. **Failing any gate demotes the shape to unsupported with the
measured numbers as the reason** — a flat, declared-unsupported shape is
honest; a fabricated one is a silent failure.

## 4. Topology invariants (hard STOPs, both stages)

- `expression_basis.npz["faces"]`, `neutral.ply` faces, and every re-read
  `expr_*.ply` faces must be **byte-identical** (`tobytes()` compare) to
  `out/recon/faces.npy`; vertex counts must equal neutral's.
- decode(0 expression, 0 pose) must reproduce `out/recon/neutral.ply` within
  1e-4 m (proves the basis decoder is the reconstructor's math).
- `out/shapes/neutral.ply` must be sha256-identical to `out/recon/neutral.ply`.
- `verify_shapes.py` re-derives all of this independently from disk and also
  asserts every **unsupported** shape has **no** mesh file (stale/fabricated
  mesh tripwire) and that `tongueOut` is unsupported.

## 5. Coverage — all 52 accounted for (final flags are POD-MEASURED)

Intended classification (prediction, not result): **48 attempted**
(42 strong + 6 weak-provisional), **4 a-priori unsupported**.

| Group | Shapes | Method | Intended |
|---|---|---|---|
| Jaw | jawOpen, jawLeft, jawRight | jaw joint (calibrated) | strong |
| Jaw | jawForward | LBS jaw-joint translation | strong |
| Eye gaze | eyeLook{Up,Down,In,Out}{Left,Right} (8) | eye joints (calibrated; eyeball-only, lids don't follow — honest limitation) | strong |
| Brows | browDownL/R, browInnerUp (single bilateral), browOuterUpL/R | PCA solve | strong |
| Eyelids | eyeBlinkL/R | PCA solve (gap targets) | strong |
| Eyelids | eyeSquintL/R, eyeWideL/R | PCA solve | **weak** — gates adjudicate |
| Mouth | smileL/R, frownL/R, dimpleL/R, stretchL/R, pressL/R, lowerDownL/R, upperUpL/R, left, right, pucker, funnel, rollLower/Upper, shrugLower/Upper (21) | PCA solve | strong |
| Mouth | mouthClose | PCA linearized at jawOpen | strong |
| Nose | noseSneerL/R | PCA solve | **weak** — laterality leakage gate decides |
| Cheeks | cheekPuff, cheekSquintL/R | — | **unsupported**: no cheek handles in iBUG-51 and cheek inflation/squint essentially unspanned by FLAME's scan-based PCA; solving against unrelated handles would fabricate geometry |
| Tongue | tongueOut | — | **unsupported**: FLAME has no tongue geometry (matches the name-contract map: MediaPipe never emits it; driver no-ops by name) |

Per-shape max-activation values: pose magnitudes in `rig/config.py`
(jawOpen 0.36 rad, jaw lateral 0.12 rad, jawForward 6 mm, eye pitch 0.35/0.45
rad, eye yaw 0.45 rad); PCA shapes use solved coefficient vectors (stored per
shape in `shape_params.npz`, norms + λ in the manifest). All are documented
starting points — tune on the pod against rendered previews.

## 6. Handoff

- To `blender-glb-builder`: `out/shapes/neutral.ply` + `expr_*.ply` +
  `arkit_manifest.json`. **Gate on `run_state == "measured-on-pod"`** and
  build shape keys only for `supported: true` names; unsupported names get NO
  key (the driver no-ops them by name — never rename to hide the gap).
- To `qa-verifier` / `viewer-driver`: reconcile GLB morph-target names against
  `arkit_manifest.json` supported set and `shapes_run_manifest.json` verdict.
- To `license-compliance`: the FLAME→ARKit correspondence is **self-authored**
  (`rig/arkit_spec.py` provenance header) from Apple's public descriptions +
  the CC-BY FLAME landmark embedding; no NC assets consulted. The MetaHuman
  reference-FBX borrowing option mentioned in the plan was **NOT used**
  (study-only license). CC-BY-4.0 attribution obligations unchanged
  (`out/compliance_report.md` B1-1).

## 7. Local verification performed (structure/syntax ONLY — nothing executed)

- `bash -n scripts/run_rig_b1.sh` → OK; marked executable.
- `python3 -m py_compile` on all 6 `rig/*.py` modules → OK (byte-compile only).
- `python3 -m rig.author_local_manifest` (stdlib JSON only, no numerics):
  spec ↔ contract cross-check passed — 52 names byte-exact, contract order
  preserved, intended counts {strong: 42, weak: 6, unsupported: 4}.
- Confirmed `out/shapes/` contains ONLY `arkit_manifest.json` (+ this report):
  **zero `.ply` files exist locally — nothing fabricated.**
- **NOT verified locally (impossible without compute, deferred to pod):**
  whether the weak-flagged shapes survive the gates, pose sign calibration,
  eye-joint laterality, solve quality of every PCA shape, all measured
  delta magnitudes, topology byte-compares against the real `faces.npy`.
