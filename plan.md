# plan.md — executable TODO for the Image → ARKit Avatar → GLB pipeline

> The living checklist `/execute-all` drives. Each task names its **owning agent** and the
> **source-plan section**. Owner-model: **F** = Fable (`claude-fable-5`, hard stages),
> **O** = Opus 4.8 (`claude-opus-4-8`). Keep boxes in sync with the orchestrator's TodoWrite.
> Gates `[GATE]` are hard stops. Full spec: `Revised_Image_to_ARKit_Avatar_GLB_Plan.md`.
> Rules of authority: `GOVERNANCE.md`.

> **PIVOT → TRACK B1 (2026-07-04, later same day) — GOVERNING.** Run moved from B2 to **B1 (FLAME
> 2023 Open, COMMERCIAL)**. Method locked: **optimization-based FLAME 2023 Open fit to MediaPipe
> landmarks** (Apache-2.0, **no non-commercial weights** — explicitly NOT DECA/EMOCA/Arc2Avatar/InsightFace,
> no 3DGS) + **per-subject albedo baked from the input photo** `random-person.jpeg` (NO MPI FLAME texture
> [CC-BY-NC-SA], no Basel/AlbedoMM statistical prior). Target: Linux **RTX 6000 Ada** RunPod pod.
> **GPU execution BLOCKED** — the SSH private key (`runpod_ssh_key`) is absent on this machine; the pod
> is unreachable until the user supplies it. B1 licensing: `SHIP-CLEARED: no` until fitter provenance +
> attribution notices are wired (textures ARE viable). Building B1 pipeline files locally now; GPU run
> deferred to the pod. Governing licensing doc = `out/compliance_report.md` (TRACK B1 section).
> **STATUS:** B1 pipeline files BUILT locally (env → recon → rig → GLB export → viewer, attribution wired)
> and **scaffold-accepted (`qa_report.md` → ACCEPT: yes)**. Only the GPU run remains — **blocked on the
> missing `runpod_ssh_key`**. Ship still `SHIP-CLEARED: no` (FLAME-Open provenance + input-photo rights open).
>
> ---
> **RUN STATUS (2026-07-04) — Track B2, SCAFFOLD-ONLY / NO-COMPUTE. [SUPERSEDED by B1 pivot — historical]** User directive: do NOT
> install any GPU systems, do NOT run CPU inference ("contamination"), migrate to GPU after all
> files are built locally. Consequence: this box has no NVIDIA GPU / no UE 5.7 / no Blender 5.x,
> so all B2 *compute* (Image→MetaHuman→GLB) is **DEFERRED to a Windows+NVIDIA GPU box**. This run
> built every file/runbook and passed **scaffold** QA (`ACCEPT: yes`, scope=scaffold). Shipping is
> **BLOCKED**: `SHIP-CLEARED: no` pending human-lawyer Epic EULA (or switch to B1/FLAME-2023-Open).
> `[x]` below = scaffold complete; real-artifact + ship gates re-run at the GPU stage (see `out/qa_report.md`).

## Choose the track first
- [x] Decide **Track A** vs **Track B** → **Track B (commercial)** chosen.
- [x] Chose **B2** (MetaHuman — higher quality; drags in UE 5.7 + Epic EULA → human-lawyer gate).

---

## Phase 0 — Intent & licensing gate `[GATE]`  · owner: `license-compliance` (O) · §6  · **DONE**
- [x] Read run intent (track) and set commercial vs internal → **B2, commercial**.
- [x] Commercial run: base is MetaHuman route (NOT FaceVerse / standard FLAME). ✓
- [x] Re-verified live licenses (MetaHuman liberalized Jun 2025; UE 5.7 Nov 2025). `out/compliance_report.md` written.
- [x] Verdict emitted → **`SHIP-CLEARED: no`** pending human-lawyer Epic EULA (or B1/FLAME-2023-Open fallback).

## Phase 1 — Environment  · owner: `env-provisioner` (O) · plan §0–6  · **SCAFFOLD ✓ / GPU-compute DEFERRED** (`out/env_report.md`, `out/gpu_requirements.md`)
- [ ] Python + torch (CUDA if GPU, else CPU) importable; record `torch.cuda.is_available()`.
- [ ] FaceVerse checkout + model `.npy` (incl. `faceverse_simple_v2.npy` with `exp_name_list`).
- [ ] Headless Blender 4.x/5.x on PATH (glTF exporter + shape keys present).
- [ ] Viewer scaffold: npm project able to load `three` + `@mediapipe/tasks-vision`.
- [ ] Write `out/env_report.md` (PASS/FAIL + versions). **Do not proceed on FAIL.**

## Phase 2 — Reconstruction `HARD` · owner: `face-reconstructor` (**F**) · §3.1–3.2 / §4.1  · **SKIPPED (B2)** — replaced by MetaHuman route below
- [ ] Fit image → neutral head mesh. Track A: FaceVerse `run.py --save_ply True`.
      Track B1: FLAME 2023 Open + DECA/EMOCA fit (flag fitter license to compliance).
- [ ] Verify PLY has faces (`element face N`, N>0) — reject vertex-only meshes.
- [ ] Emit `out/recon/`: `neutral.ply`, `id_params.npz`, `faces.npy`, expression-basis handle
      (`exp_name_list.json` / FLAME basis), `recon_report.md` (per-shape coverage).
- [ ] `[GATE]` QA (`qa-verifier`, O): faces exist, topology contract present, basis present.

## Phase 3 — Rigging `HARD` · owner: `arkit-rigger` (**F**) · §3.2  · **SKIPPED (B2)** — MetaHuman bakes 51 ARKit shapes natively
- [ ] For each of 52 ARKit names: max-activate its axis, identity fixed, rebuild mesh,
      write `expr_<name>.ply` in the **same topology** as neutral.
- [ ] Mark shapes with no clean axis (e.g. `tongueOut`) **unsupported** — never fabricate.
- [ ] Emit `out/shapes/` + `arkit_manifest.json` (all 52 accounted for, exact spelling).
- [ ] `[GATE]` QA: identical topology across all meshes; 52 names exact; non-zero deltas.

## Phase 4 — Blender assembly + GLB export  · owner: `blender-glb-builder` (O) · §3.3–3.4  · **SUBSUMED (B2)** — GLB produced by metahuman-to-glb converter at GPU stage
- [ ] `blender_build_rig.py`: base + one shape key per ARKit name (index-aligned copy).
- [ ] Export `out/head_arkit.glb` (`export_morph=True`, `export_morph_normal=True`, Draco opt).
- [ ] Save `out/head_rigged.blend`; write `out/glb_report.md`.
- [ ] `[GATE]` QA: GLB carries correctly-named morph targets; textures survived.

## Phase 5 — Viewer + driving  · owner: `viewer-driver` (O) · §3.5  · **BUILT + bundle-verified ✓ / live MediaPipe run DEFERRED** (`out/viewer/`, `out/viewer_report.md`) — 51/52 driven, `tongueOut` honest no-op
- [ ] three.js loads GLB; MediaPipe FaceLandmarker (VIDEO, blendshapes on) drives it by
      name via `morphTargetDictionary` / `morphTargetInfluences`.
- [ ] Support webcam + video file; add influence smoothing; document run command.
- [ ] Write `out/viewer_report.md` (which of 52 got driven; log unresolved category names).
- [ ] `[GATE]` QA: every MediaPipe `categoryName` resolves (except honest-unsupported).

## Phase 6 — Final acceptance `[GATE]`  · owners: `license-compliance` + `qa-verifier` (O)  · **SCAFFOLD ACCEPT: yes** (`out/qa_report.md`) / **SHIP-CLEARED: no** (`out/compliance_report.md`) — real-artifact + ship gates re-run at GPU stage
- [ ] `license-compliance`: re-verify; attribution (B1: credit FLAME 2023 Open) / EULA (B2)
      obligations recorded. `out/compliance_report.md` → **`SHIP-CLEARED: yes|no`**.
- [ ] `qa-verifier`: end-to-end reconcile (rig manifest ↔ GLB ↔ MediaPipe). `out/qa_report.md`
      → **`ACCEPT: yes|no`**.
- [ ] **Done** = `ACCEPT: yes` (Track A: internal-only) **and**, if commercial, `SHIP-CLEARED: yes`.

---

## Track B2 branch (MetaHuman) — replaces Phases 2–3 · owner: `metahuman-route` (O) · §4.2  · **SCAFFOLD ✓ / compute DEFERRED to Windows+NVIDIA GPU box** (`out/metahuman_report.md`, `out/arkit_51_52_map.json`)
- [ ] Front half: Image → MetaHuman via Epic Mesh-to-MetaHuman / MHC (needs mesh/sculpt; not one-click).
- [ ] Back half: `smorchj/metahuman-to-glb` (UE 5.7 + `.uproject` + MetaHumanCharacter + Blender 5.x) → GLB w/ 51 ARKit baked.
- [ ] Reconcile 51↔52 naming (`browInnerUp` folding). Hand GLB to Phase 5.
- [ ] `[GATE]` EULA: standalone-GLB-outside-Unreal reviewed by a human lawyer via `license-compliance`.

## Gaps filled from research (2026-07-04)
- FaceVerse **v4** = direct ResNet50 param prediction; v2+ expression comps fit to Apple 52
  (`exp_name_list` in `faceverse_simple_v2.npy`). Repo: `LizhenWangT/FaceVerse` (non-commercial).
- **FLAME 2023 Open** confirmed **CC-BY-4.0**, released **Nov 2025** — the commercial-clearable
  FLAME base (standard FLAME remains non-commercial).
- **MediaPipe FaceLandmarker** returns 52 ARKit-named blendshape `categoryName`s → maps 1:1 to
  three.js `morphTargetDictionary`; driver is a pure name lookup.
