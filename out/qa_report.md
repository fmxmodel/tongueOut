# QA / Verification Report — Track B1 (FLAME 2023 Open), SCAFFOLD-ONLY (no-compute)

> Gate: `qa-verifier` (Opus 4.8), blocking authority (GOVERNANCE.md §4 Gate 2)
> Date: 2026-07-04 · Track: **B1 (FLAME 2023 Open → optimization fit → 52 ARKit → GLB → three.js)** · Intent: **COMMERCIAL**
> Mode: **SCAFFOLD-ONLY / no GPU install / no CPU model compute** (explicit user constraint; migrate to RTX 6000 Ada pod after).
>
> **This is a SCAFFOLD acceptance.** By design there is NO real GLB, mesh, params, or texture this
> session — that is CORRECT, not a failure. All real-artifact invariants are **DEFERRED to the pod
> run** and MUST be re-verified there (see §Deferred). Every verdict below is by measurement of the
> artifact, not by trusting a stage's own success claim.
>
> _(Supersedes the prior Track-B2 QA report; the run pivoted B2 → B1. B2 record retained in git/history.)_

## Per-check results (measured)

| # | Invariant | Measured evidence | Verdict |
|---|---|---|---|
| 1a | `head_arkit.glb` / `head_rigged.blend` absent | both absent | **PASS** |
| 1b | `out/recon/` = only `recon_report.md` | no `neutral.ply`/`faces.npy`/`id_params.npz`/`*.npz`/`albedo.png` | **PASS** |
| 1c | `out/shapes/` = only `arkit_manifest.json` + `rig_report.md` | no `neutral.ply`, no `expr_*.ply` | **PASS** |
| 1d | No stray mesh/texture under `out/` | mesh/texture scan (excl. node_modules): 0 hits | **PASS** |
| 1e | No report claims compute/mesh/GLB/inference ran | every grep hit is a negation or DEFERRED | **PASS** |
| 2a | Manifest = 52 names, exact Apple case | set-equal to Apple-52; 0 missing/extra/dup | **PASS** |
| 2b | Same order as `out/arkit_51_52_map.json` | `man_order == map_order` exactly | **PASS** |
| 2c | 0 renamed/aliased | none | **PASS** |
| 2d | `run_state == "DEFERRED-pod-run"` | confirmed; supported=`null` pending pod | **PASS** |
| 2e | 4 a-priori-unsupported flagged, no mesh | `{tongueOut, cheekPuff, cheekSquintLeft, cheekSquintRight}` → `supported:false`, `method:none`, `ply:null`, reason present | **PASS** |
| 2f | Counts block honest | 52 = 42 strong + 6 weak + 4 unsupported | **PASS** |
| 2g | Downstream gates on `run_state=="measured-on-pod"` | `blender_build_rig.py:324`; `verify-names.mjs:47` | **PASS** |
| 2h | Blender builds morph targets ONLY for `supported:true`, exact names | `:332` supported-only; `:457` key set == supported; `:219` GLB morph names == supported, no extras | **PASS** |
| 3 | `faces.npy` single topology authority, byte-equal each hop, STOP-on-drift | recon `assert_ply_has_faces`+topology check; `verify_outputs` array_equal; rig `tobytes()` + `decode(0)` repro + neutral sha256 + per-expr byte-compare + unsupported-no-mesh tripwire; blender faces-sha256-vs-manifest + uv-faces byte-compare | **PASS** |
| 4a | No forbidden NC dep imported/installed | 0 real import/install of nvdiffrast/DECA/EMOCA/insightface/arcface/smplx/FLAME_PyTorch/AlbedoMM/BFM; token hits are negations/provenance comments | **PASS** |
| 4b | FLAME loader + FLAME→ARKit spec self-authored | provenance headers ("does NOT vendor/copy" NC code); `requirements-b1.txt` all permissive | **PASS** |
| 5a | `cd out/viewer && npm run build` | exit 0 | **PASS** |
| 5b | `npm run verify-names` | exit 0 (48/52 carried resolve; 3 MediaPipe no-ops; tongueOut not emitted; 0 renames) | **PASS** |
| 5c | `py_compile` recon/ + rig/ + `blender_build_rig.py` | exit 0 | **PASS** |
| 5d | `bash -n scripts/*_b1.sh` | all OK | **PASS** |
| 5e | Scaffold files exist | `pod_setup_b1.sh`, `run_{recon,rig,glb}_b1.sh`, `requirements-b1.txt`, `models/README.md`, `out/gpu_requirements_b1.md` present + nonempty | **PASS** |
| 5f | Every pipeline entrypoint `pod_guard`-gated | `require_pod()`/`require_cuda_torch()` called in all 6 pod entrypoints; scripts refuse without `nvidia-smi`; Blender stage refuses on missing inputs + non-measured manifest | **PASS** |
| 6 | Commercial `SHIP-CLEARED` honest for a no-ship scaffold | `compliance_report.md` → `SHIP-CLEARED: no` (nothing shipped; flip-to-yes itself deferred) | **PASS (expected)** |

## Violations found
Honesty: **none.** Naming: **none.** Topology: **none.** License: **none.**

## Non-blocking notes (routed to owners)
- `out/arkit_51_52_map.json` was authored by `metahuman-route` (B2); its `driver_resolution` marks `cheekPuff`/`cheekSquint*` as `in_glb_morphtargets: true` (a MetaHuman-baking fact). The B1 manifest correctly declares those 4 **unsupported**. The map is consumed only as the ordered 52-name authority (verified byte-exact), so this is not a contract conflict — but nobody should read the map's B2 `driver_resolution` as B1 coverage.

## Invariants DEFERRED to the GPU pod (MUST be re-verified there — none provable now)
1. `out/recon/neutral.ply` has `element face N>0` (reject vertex-only); record measured V/F. → `face-reconstructor`
2. `out/recon/faces.npy` exists, int32 (F,3), indices in range, count == PLY faces. → `face-reconstructor`
3. `faces.npy` byte-equal across neutral + every `expr_*.ply` + `expression_basis.npz` + the GLB base mesh; vertex counts equal. → `arkit-rigger` / `blender-glb-builder`
4. `decode(0,0)` reproduces neutral within tol (basis == reconstructor math). → `arkit-rigger`
5. Supported shapes: non-zero delta (max ≥ 0.5 mm) + pass rig gates (direction cosine, amplitude ratio, mirror leakage, off-target); weak shapes (eyeSquint/eyeWide L/R, noseSneer L/R) survive or are honestly demoted; laterality measured 3 ways. → `arkit-rigger`
6. Manifest flips to `run_state: "measured-on-pod"` with real counts. → `arkit-rigger`
7. `out/head_arkit.glb` parses, non-trivial size, carries EXACTLY the measured-supported morph names (correct case, `browInnerUp` single/bilateral, unsupported absent); albedo texture / vertex colors survive. → `blender-glb-builder`
8. Live MediaPipe run: 52 `categoryName`s reconcile against the real GLB `morphTargetDictionary`; `window.__viewer.state.unresolved` minus `expectedUnsupported` is empty. → `viewer-driver`
9. `SHIP-CLEARED` flips to `yes` (Open-release asset provenance + input-photo rights confirmed; attribution wired ✓). → `license-compliance`

---

**ACCEPT: yes** — SCAFFOLD acceptance only. The B1 scaffold is honest (no fabricated artifacts; all reports DEFERRED), contract-consistent (52 exact ARKit names, order-locked to the map, 4 unsupported declared with reasons), topology-locked (`faces.npy` byte-equality asserted at every hop with STOP-on-drift), license-clean (no NC deps; self-authored FLAME loader + FLAME→ARKit spec; PyTorch3D/BSD over nvdiffrast), buildable (npm build + verify-names + py_compile + bash -n all exit 0), and GPU-ready (every entrypoint `pod_guard`-gated). Real-artifact QA and the SHIP gate are DEFERRED to and MUST be re-run on the pod.
