# Environment Report — Track B2 (MetaHuman → GLB)

> Owner: `env-provisioner` (Opus 4.8) · Date: 2026-07-04 · Track: **B2 (MetaHuman route)**
>
> **Session constraint (honored exactly):** No GPU stack was installed — no CUDA toolkit,
> no GPU driver, no GPU-specific PyTorch. **No model inference or numeric/model compute was
> run on CPU** (contamination guard). Only trivial version/`--version`/file-presence checks
> were performed. GPU/UE/compute items are **DEFERRED (GPU stage)**, not FAIL and not faked.
> Full migration handoff: `out/gpu_requirements.md`.

## Hardware detected (this box)
- GPU: **Intel UHD Graphics (Raptor Lake-P), integrated** — `lspci` VGA only.
- **No NVIDIA GPU**, no `/dev/nvidia*`, no `nvidia-smi`, no `nvcc` (CUDA absent). Confirmed.
- Consequence: UE 5.7 + Blender GPU stages cannot run here; they are DEFERRED to the GPU box.

---

## Deliverable status

| # | Piece | Status | Version / detail |
|---|---|---|---|
| 1 | `smorchj/metahuman-to-glb` repo cloned | **PASS** | `vendor/metahuman-to-glb` @ HEAD (depth-1 clone), MIT scripts. README + prereqs recorded. |
| 2 | Unreal Engine 5.7 | **DEFERRED (GPU stage)** | Not present anywhere on box. Required: UE **5.7**, `UnrealEditor-Cmd`, MetaHuman + RigLogic. Windows GPU box. |
| 3 | Blender 5.x (glTF exporter + shape keys) | **DEFERRED (GPU stage)** | No Blender on PATH / `/opt` / `/usr/local` / flatpak / snap / apt. Required: Blender **5.x** on the GPU box. |
| 4 | Viewer scaffold (`three` + `@mediapipe/tasks-vision`) | **PASS** | `out/viewer/` npm project; `npm install` OK (15 pkgs). Versions below. |
| 5 | GPU-migration requirements manifest | **PASS** | `out/gpu_requirements.md` written. |

---

## 1. `smorchj/metahuman-to-glb` — PASS (cloned; no compute)
- **Location:** `/home/darpa/Desktop/newARC/vendor/metahuman-to-glb`
- **License:** MIT (scripts only; MetaHuman **assets** governed by Epic EULA — Gate 1).
- **What it is (honest):** a **MetaHuman → GLB back-half converter**, NOT image→avatar. It
  exports an **already-built UE 5.7 MetaHumanCharacter** to a web GLB with **51** ARKit
  blendshapes. The front half (image → MetaHuman via Epic Mesh-to-MetaHuman / MHC) is a
  separate GPU/UE step owned by `metahuman-route`.
- **Exact prerequisites (from its README/CONTEXT/config):**
  - UE **5.7** + `.uproject` + `UnrealEditor-Cmd(.exe)` (stages 00, 01; UE embedded Python).
  - Blender **5.x** + `blender.exe` (stages 02, 03; Blender bundled Python — `bpy`,
    `mathutils.kdtree`, `numpy`, all built-in; **no pip install**).
  - `_config/pipeline.yaml` (copy of `pipeline.example.yaml`) with project/editor/blender paths.
  - Stage launchers are **PowerShell `.ps1`** → pipeline is **Windows-native as shipped**.
  - No `requirements.txt`; stage 04 `build_site.py` is **stdlib-only** Python 3.

## 2. Unreal Engine 5.7 — DEFERRED (GPU stage)
- Detection: no `UnrealEditor*` binary on PATH or in common install roots. **Not present.**
- Not installable on this box without a GPU; a source build was **not** attempted (per rules).
- Requirement recorded in `out/gpu_requirements.md` §3 with version, binary path, and plugins.

## 3. Blender 5.x — DEFERRED (GPU stage)
- Detection: `blender --version` → not found; absent from PATH, `/opt`, `/usr/local`,
  flatpak, snap, apt. **Not present.**
- No GPU-dependent build installed here (per constraint). The **glTF 2.0 exporter + shape-key
  system are Blender built-ins** — they will exist once Blender 5.x is installed on the GPU box.
- Requirement recorded in `out/gpu_requirements.md` §4.

## 4. Viewer scaffold — PASS (pure web; safe now)
- **Location:** `/home/darpa/Desktop/newARC/out/viewer/`
  - `package.json`, `index.html`, `src/main.js` (imports both deps; **no** GLB load, **no**
    webcam, **no** MediaPipe inference — that is `viewer-driver`'s Phase 5 job), `.gitignore`.
- **`npm install` succeeded** (15 packages) — dependency fetch only, no compute.
- **Installed versions (read from `node_modules`, not executed):**

  | Package | Installed | Why pinned |
  |---|---|---|
  | `three` | **0.170.0** | Pinned to match the B2 converter's own tested viewer (`04-webview-build/templates/viewer.html` importmap uses `three@0.170.0`), so `viewer-driver` inherits a known-good version. |
  | `@mediapipe/tasks-vision` | **0.10.14** | Pinned to match the converter's `viewer.js` (`MEDIAPIPE_VERSION = '0.10.14'`) — same ARKit-52 `categoryName` set the driver relies on. |
  | `vite` | **6.4.3** | Dev server / bundler; caret range `^6.0.0`, npm resolved 6.4.3. Not pinned. |
- Verified present: `three/examples/jsm/loaders/GLTFLoader.js`, `@mediapipe/tasks-vision/vision_bundle.mjs`.
- **Toolchain:** node **v24.14.0**, npm **11.9.0** (system, pre-existing).
- Note: MediaPipe wasm + `face_landmarker.task` model are fetched from CDN **in-browser at
  runtime** by `viewer-driver`; nothing model-related runs in this provisioning session.

## 5. GPU-migration manifest — PASS
- `out/gpu_requirements.md` lists every deferred item (host OS, GPU/driver, UE 5.7, Blender
  5.x, config, exact stage commands, 51↔52 naming, EULA gate).

---

## Pre-existing tools (not installed by me; version checks only)
- git **2.43.0** · node **v24.14.0** · npm **11.9.0** · python3 **3.12.3** · Disk free ~73 GB.

## Gate-relevant flags for downstream agents
- **License (Gate 1):** MetaHuman assets → Epic EULA; standalone-GLB-outside-Unreal shipping
  is a **human-lawyer** question. Not self-clearable. See `license-compliance`.
- **Naming (Gate 2 / QA):** converter bakes **51** ARKit shapes; reconcile **51 ↔ 52**
  (`browInnerUp` bilateral fold; `tongueOut` needs non-default rig) at Phase 5.

## Bottom line
- **READY now (local, no GPU, no compute):** B2 repo cloned + prereqs documented; viewer
  scaffold with three 0.170.0 + tasks-vision 0.10.14 installed; migration manifest written.
- **DEFERRED to GPU box:** UE 5.7, Blender 5.x, both Unreal stages + both Blender stages
  (all GPU-bound), and the front-half image→MetaHuman step.
- **Do not proceed on a FAIL** — there are none. Deferred items are intentional handoffs.

**No GPU stack was installed. No model or numeric compute was run. By design.**

---
---

# Environment Report — Track B1 (FLAME 2023 Open, COMMERCIAL)  [ACTIVE]

> Added 2026-07-04 by `env-provisioner` (Opus 4.8). The run pivoted from B2 to **Track B1**:
> optimization-based **FLAME 2023 Open** fit to **MediaPipe FaceLandmarker** landmarks
> (Apache-2.0, no non-commercial weights) + per-subject albedo **baked from the input photo**
> (no statistical albedo prior). Compute TARGET = a **Linux RTX 6000 Ada RunPod pod**, NOT
> this box.
>
> **Session constraint (honored exactly):** **NO GPU stack installed** locally (no CUDA, no GPU
> torch) and **NO compute/inference run** locally (contamination guard). This box only *authored
> files*. GPU-pod items are **DEFERRED (GPU pod)** — real, not faked. The pod is currently
> **unreachable** (SSH private key `runpod_ssh_key` absent per `ssh.md`), so it was not
> provisioned live; instead the provisioning is authored to be **turnkey once connected**.

## B1 deliverable status

| # | Piece | Status | Detail |
|---|---|---|---|
| 1 | Pod env spec + setup script | **PASS (authored locally)** | `scripts/pod_setup_b1.sh` (executable, `bash -n` syntax-checked, **not executed**) + `requirements-b1.txt`. |
| 2 | FLAME 2023 Open handling doc | **PASS (authored locally)** | `models/README.md`: what to download, where (`/workspace/models/flame2023_open/`), and the ⛔ do-not-download NC texture package. |
| 3 | B1 GPU-migration manifest | **PASS (authored locally)** | `out/gpu_requirements_b1.md` (pod GPU/driver/CUDA, command sequence, model download, input path). |
| 4 | Viewer scaffold reuse note | **PASS (reused, not rebuilt)** | `out/viewer/` (three 0.170.0 + tasks-vision 0.10.14) is track-agnostic; only the driven ARKit-name subset may change (rigger decides). |
| — | Torch + CUDA install / import / `torch.cuda.is_available()` | **DEFERRED (GPU pod)** | Runs only inside `scripts/pod_setup_b1.sh` on the pod. Not installed or run here. |
| — | FLAME 2023 Open download + fit + albedo bake | **DEFERRED (GPU pod)** | License-gated manual download + GPU optimization job. Not run here. |

## Differentiable-renderer decision — **PyTorch3D** (licensing-driven)

- **Chosen: PyTorch3D (BSD-3-Clause)** — commercial-clean, and it bundles cameras +
  differentiable rasterizer (landmark fit), `TexturesUV` + UV rasterization (photo→UV albedo
  bake), and mesh regularizers, matching the B1 method in one library.
- **Rejected: nvdiffrast** — its **NVIDIA Source Code License (1-Way Commercial) is
  NON-COMMERCIAL** ("only ... non-commercially"), so it is **BARRED** from this commercial run,
  exactly like DECA/EMOCA/BFM. Verified 2026-07-04.
- Install caveat on Ada (sm_89): prebuilt CUDA-12 pytorch3d wheels are spotty, so the script
  tries the wheel then **builds from source** with `FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.9+PTX"`
  (needs a CUDA-`-devel` pod image with nvcc 12.1).

## Pinned versions (and why) — full rationale in `out/gpu_requirements_b1.md` §3

- Python **3.10** (pytorch3d wheel coverage), torch **2.4.1**/torchvision **0.19.1** **+cu121**
  (newest torch pytorch3d 0.7.8 supports; cu121 carries Ada sm_89), pytorch3d **0.7.8**,
  numpy **<2** (mediapipe C-ABI), mediapipe **0.10.14** (matches viewer's tasks-vision).
- All B1 deps are commercially permissive (BSD/MIT/Apache/HPND); **none non-commercial** —
  audited in `requirements-b1.txt`.

## B1 bottom line

- **READY now (local, no GPU, no compute):** pod setup script + requirements, FLAME model
  handling doc, B1 GPU manifest, viewer reuse confirmed.
- **DEFERRED to the pod:** the entire torch/CUDA install, `torch.cuda.is_available()` check,
  FLAME download, and the fit + albedo bake — all gated on the pod being reachable
  (private SSH key still absent).
- **Nothing was installed or run locally.** No GPU pretended. By design.

