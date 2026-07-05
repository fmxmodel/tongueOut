# GPU-Migration Manifest — Track B1 (FLAME 2023 Open, COMMERCIAL)

> Handoff document for the **Linux RTX 6000 Ada RunPod pod**. Everything here is **DEFERRED
> to the pod** by explicit constraint. This authoring session installed **no GPU stack** and
> ran **no compute** locally (contamination guard). Local-authored files: `scripts/pod_setup_b1.sh`,
> `requirements-b1.txt`, `models/README.md`. Companion status: `out/env_report.md` (Track B1 section).
>
> Generated 2026-07-04 · owner `env-provisioner` (Opus 4.8) · supersedes the B2 manifest
> `out/gpu_requirements.md` for the active run. B1 licensing: `out/compliance_report.md` (TRACK B1).

---

## 0. What B1 is (so the pod builds the right thing)

**Reconstruction = optimization-based FLAME 2023 Open fit to MediaPipe FaceLandmarker
landmarks.** Pure optimization + permissive/self-authored code. **No learned non-commercial
weights** — explicitly NO DECA, NO EMOCA, NO Arc2Avatar/Arc2Face, NO InsightFace/ArcFace, NO
3D Gaussian Splatting.
**Texture = per-subject albedo baked from `random-person.jpeg`** projected onto the FLAME UV +
mirror-symmetry + classical inpainting. **No statistical albedo prior** (no BFM / AlbedoMM /
MPI FLAME texture space).

Unlike B2 (which needed UE + Blender, no torch), **B1 IS a PyTorch/CUDA optimization job.**

## 1. Pod GPU / driver / CUDA expectations

| Item | Requirement | Notes |
|---|---|---|
| GPU | **NVIDIA RTX 6000 Ada Generation**, 48 GB, **Ada Lovelace = compute capability sm_89** | Target pod. |
| Driver | NVIDIA Linux driver **≥ 525.60.13** (CUDA 12.1 runtime floor); RunPod Ada pods ship 550+ | Verify with `nvidia-smi`. |
| CUDA (runtime) | Provided by the **torch cu121 wheel** — no separate CUDA install needed to *run* | sm_89 kernels are in the cu121 build. |
| CUDA (dev / nvcc) | **Only if pytorch3d must build from source** — use a CUDA **`-devel`** pod image (nvcc 12.1) | See §3. Not needed if the prebuilt wheel lands. |
| Ada support | sm_89 is supported by CUDA ≥ 11.8; cu121 covers it natively | No custom torch build required. |

## 2. Differentiable-renderer choice — **PyTorch3D**, not nvdiffrast (licensing-driven)

| Candidate | License | Commercial? | Verdict |
|---|---|---|---|
| **PyTorch3D** | **BSD-3-Clause** | ✅ yes | **CHOSEN** |
| nvdiffrast | NVIDIA Source Code License (1-Way Commercial) — "only ... non-commercially" | ❌ **NON-COMMERCIAL** | **BARRED** |

**Rationale:** (1) B1 is a commercial run — nvdiffrast's non-commercial license bars it exactly
as DECA/EMOCA/BFM are barred; PyTorch3D's BSD-3 is clean. (2) PyTorch3D bundles everything the
B1 method needs in one library: camera models + differentiable rasterizer for the landmark fit,
`TexturesUV` + UV rasterization for the photo→UV albedo bake, and mesh regularizers
(Laplacian/edge) for the fit prior. Caveat: PyTorch3D compiles CUDA extensions and is
ABI-coupled to the exact torch+CUDA build — handled in §3.

## 3. Pinned stack + why each pin

| Component | Pin | Why pinned (else unpinned) |
|---|---|---|
| Python | **3.10** | Best prebuilt-pytorch3d wheel coverage (`py310`). 3.11 works only via source build. |
| torch / torchvision | **2.4.1 / 0.19.1 (cu121)** | 2.4.1 is the newest torch supported by pytorch3d 0.7.8; cu121 carries sm_89 for Ada. |
| pytorch3d | **0.7.8** | Latest release; last to support torch 2.4.1. BSD-3. |
| numpy | **< 2** | mediapipe 0.10.x links numpy-1.x C-ABI. Compatibility pin, not preference. |
| mediapipe | 0.10.14 | Matches the viewer's `@mediapipe/tasks-vision` 0.10.14 → same ARKit blendshape set. |
| scipy / trimesh / opencv-python / Pillow / imageio / scikit-image / fvcore / iopath / ninja | unpinned | Take latest; all permissive (see `requirements-b1.txt` license audit). |

**pytorch3d install caveat on Ada:** the setup script tries the prebuilt wheel index
`py310_cu121_pyt241` first; CUDA-12 wheel coverage is historically spotty, so it **falls back
to a source build** with `FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.9+PTX"`. The source build
**requires nvcc 12.1**, i.e. a CUDA **`-devel`** pod image (build ~15-40 min). Choose a
`runpod/pytorch:2.4.1-*-devel` (or equivalent CUDA-12.1-devel) base to guarantee the fallback.

**License audit (all commercial-permissive; none non-commercial — verified 2026-07-04):**
torch/torchvision BSD-3 · pytorch3d BSD-3 · mediapipe Apache-2.0 · numpy/scipy/scikit-image BSD ·
trimesh MIT · opencv-python MIT (OpenCV lib Apache-2.0) · Pillow HPND · imageio BSD · fvcore
Apache-2.0 · iopath MIT · ninja Apache-2.0.

## 4. Setup command sequence (on the pod)

```bash
# 0. copy this repo (or at least scripts/, requirements-b1.txt, models/) to the pod
# 1. provision (idempotent; refuses to run without an NVIDIA GPU)
bash scripts/pod_setup_b1.sh
# 2. activate
source /workspace/venvs/b1/bin/activate
# 3. verify (also printed by the script)
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 5. Model-download step (operator, license-gated)

- **FLAME 2023 Open (CC-BY-4.0):** register + accept the license at `flame.is.tue.mpg.de`,
  download the **"FLAME 2023 (Open)"** release **only**, unpack to
  **`/workspace/models/flame2023_open/`**. Needed files (shape model + UV template + landmark
  embedding) and the ⛔ **do-not-download** texture/albedo package are enumerated in
  `models/README.md`. Not redistributable by us; not automatable.
- **MediaPipe `face_landmarker.task` (Apache-2.0):** auto-downloaded by the setup script to
  `/workspace/models/mediapipe/`.

## 6. Where the input image goes

Copy this repo's `random-person.jpeg` → **`/workspace/inputs/random-person.jpeg`**. It is the
sole subject for both the landmark fit and the albedo bake.

## 6b. Blender — the GLB assembly stage (NEW, CPU-only)

The Track B1 export stage (`blender_build_rig.py`, run via `scripts/run_glb_b1.sh`)
transcribes the rig's topology-locked delta PLYs into `out/head_arkit.glb` — one glTF
morph target per SUPPORTED ARKit name — using **Blender's shape-key system (= the glTF
morph-target system)**. This is a **new pod requirement** beyond the PyTorch stack above.

| Item | Requirement | Notes |
|---|---|---|
| Blender | **4.2 LTS** (any 4.x/5.x works) headless | Ships the built-in `io_scene_gltf2` exporter with `export_morph` / `export_morph_normal` and Draco flags. Pinned `BLENDER_VER=4.2.3` in `pod_setup_b1.sh` (override if a patch URL 404s). |
| GPU | **NOT needed** | glTF export + shape-key assembly are pure CPU. Blender runs `--background` (no display). The stage runs on the pod only because its *inputs* come from the GPU recon+rig stages. |
| Python deps | **none extra** | Uses Blender's bundled Python + bundled numpy. `bpy` is never `pip install`-ed and is never present on the authoring box. |
| Draco | optional | `libextern_draco.so` ships inside the Blender build; enable with `B1_GLB_DRACO=1`. Default **OFF** for three.js-viewer compatibility (viewer has no DRACO decoder wired). |

**Where the pod gets it:** `scripts/pod_setup_b1.sh` **step 6b** downloads a portable
`blender-<ver>-linux-x64.tar.xz` to `$WORKSPACE/blender/` (idempotent; warn-not-die if
offline). `scripts/run_glb_b1.sh` auto-detects Blender on `PATH`, at `$BLENDER`, or at
`$WORKSPACE/blender/blender`. Manual alternative: extract any Blender 4.x/5.x build and
`export BLENDER=/path/to/blender`.

**Run order on the pod:** `run_recon_b1.sh` → `run_rig_b1.sh` → `run_glb_b1.sh`. The GLB
stage **refuses to run** until the six real inputs exist AND
`out/shapes/arkit_manifest.json` has `run_state == "measured-on-pod"` (it never fabricates
an empty GLB). Outputs: `out/head_arkit.glb`, `out/head_rigged.blend`, `out/glb_report.md`.

## 7. Viewer — REUSE the existing scaffold (do not rebuild)

`out/viewer/` (three **0.170.0** + `@mediapipe/tasks-vision` **0.10.14**, vite) is track-agnostic
and **reusable as-is for B1**. It loads a GLB and drives morph targets by ARKit name via
`morphTargetDictionary`. Only the **driven-name target set may change** — B2 baked 51 ARKit
shapes; the B1 FLAME fit supports whatever ARKit subset the **`arkit-rigger`** decides. That is a
name-list decision inside the viewer, not a scaffold rebuild.

## 8. What is still BLOCKED / open (not env issues)

- **Pod unreachable:** SSH **private** key (`runpod_ssh_key`) is absent on the authoring box
  (`ssh.md` documents only the public key). The pod cannot be provisioned live until the
  operator supplies the private key. All of the above is turnkey once connected.
- **Ship gate (`license-compliance`):** `SHIP-CLEARED: no` until (a) the MediaPipe↔FLAME
  landmark-correspondence provenance is cleared, (b) FLAME UV+landmark assets are confirmed
  from the Open release, (c) no Basel/AlbedoMM/MPI albedo anywhere, (d) attribution wired
  (FLAME 2023 Open CC-BY credit + FLAME cite; MediaPipe Apache NOTICE). See
  `out/compliance_report.md` §B1-4/§B1-5.
