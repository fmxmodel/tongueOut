# newstack env report — brand-new RunPod (RTX 6000 Ada)

Provisioned a fresh, empty pod (previous /workspace volume NOT attached). Rebuilt the
entire newstack toolchain from scratch. All heavy artifacts persist under `/workspace`.

**STATUS: READY**  (measured, not claimed — see verification below)

Pod: `root@195.26.233.87 -p 37551`  ·  date 2026-07-06

---

## 1. Pod GPU / driver / base
| Item | Value |
|------|-------|
| GPU | NVIDIA RTX 6000 Ada Generation, 49 GB (49140 MiB) |
| Driver | 550.127.05 |
| CUDA (driver) | 12.4 |
| CUDA toolkit | 12.4.131 at `/usr/local/cuda/bin/nvcc` (NOT on PATH by default) |
| Base python | 3.11.10 |
| /workspace | fresh MooseFS mount, 160 TB free |

GPU is present and healthy — FaceVerse-style/TripoSR fitting runs on GPU (fast), not CPU.

## 2. Python / torch / CUDA (venv = /workspace/env, --system-site-packages)
Verification (`import torch,pytorch3d,scipy,mediapipe,rembg,trimesh; torch.cuda.is_available()`):

```
REQUIRED verify: True
torch 2.4.1+cu124   cuda 12.4   avail True   device NVIDIA RTX 6000 Ada Generation
pytorch3d 0.7.8
torchmcubes: import OK ; marching_cubes ran on cuda -> ([3117,3],[1039,3])
numpy 2.4.6   scipy 1.17.1
transformers 4.35.0   tokenizers 0.14.1   huggingface_hub 0.17.3
mediapipe 0.10.35   trimesh 4.12.2   onnxruntime 1.27.0
cv2 5.0.0   PIL 12.3.0   omegaconf 2.3.1   einops 0.8.2   rembg 2.0.76
```

Base image torch 2.4.1+cu124 was **reused** (venv `--system-site-packages`); torch NOT
reinstalled.

### Pins (why)
- `transformers==4.35.0` + `tokenizers<0.15` (=0.14.1) — the known TripoSR checkpoint /
  state_dict break on newer transformers/tokenizers.
- `huggingface_hub` left UNPINNED → resolver chose **0.17.3**. Pinning it to 0.19.4 caused
  `ResolutionImpossible` because tokenizers 0.14 caps huggingface_hub `<0.18`.
- torchmcubes + pytorch3d built from source with `FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=8.9`
  (sm_89 = Ada). Both required `--no-build-isolation` + explicit `Torch_DIR` and
  `pybind11_DIR` on the CMake line — pip build-isolation otherwise hides the venv's
  torch/pybind11 from CMake (`find_package(Torch)` / `find_package(pybind11)` failures).

## 3. Asset paths + sizes (all under /workspace, persisted)
| Asset | Path | Size / count |
|-------|------|--------------|
| git repo (main) | `/workspace/newarc` | 66 MB |
| photo | `/workspace/inputs/random-person.jpeg` | 519 KB |
| ICT-FaceKit FaceXModel (MIT) | `/workspace/newstack/ICT-FaceKit/FaceXModel` | **160 files, 389 MB** (no LFS pointers left) |
| TripoSR clay | `/workspace/newstack/out_triposr/0/mesh.obj` | 8.88 MB — **81,812 verts / 163,346 faces**, vertex-colored |
| MediaPipe task | `/workspace/models/mediapipe/face_landmarker.task` | 3.76 MB |
| Blender 4.2.3 LTS | `/workspace/blender/blender-4.2.3-linux-x64/blender` | 2.1 GB (headless verified) |
| venv | `/workspace/env` | 3.0 GB |
| HF weight cache | `/workspace/cache/huggingface` (TripoSR `model.ckpt`) | 1.6 GB |
| TripoSR repo | `/workspace/TripoSR` | 91 MB |

Blender headless OK: `xvfb-run -a .../blender --background --version` → `Blender 4.2.3 LTS`.

## 4. TripoSR clay generation
`run.py` default path (built-in rembg background removal + resize) on
`/workspace/inputs/random-person.jpeg`, `--device cuda:0 --mc-resolution 256`, obj format.
`--bake-texture` deliberately NOT passed (moderngl `create_context` → XOpenDisplay crash
headless). Result: `out_triposr/0/mesh.obj`, **81,812 v / 163,346 f**, non-trivial, vertex
colors present. HF weights cached under `/workspace/cache/huggingface` (persist via HF_HOME).

## 5. s1/s2 smoke-test (env WIRING only — stages 1 & 2)
Command: `ROOT=/workspace/newstack PIPE=/workspace/newarc/newstack/pipe
PHOTO=/workspace/inputs/random-person.jpeg STAGES="1 2" bash run_newstack.sh`
(venv active, `HF_HOME=/workspace/cache/huggingface`). **RC=0.**

- **s1 (landmarks):** 478 MediaPipe landmarks (expected 478). Wrote
  `out/landmarks/landmarks.npz`, `overlay.jpg`, `mp_blendshapes_photo.json`. 2.2 s.
  (Required apt `libgles2 libegl1` — mediapipe 0.10.35 needs `libGLESv2.so.2` at import.)
- **s2 (identity fit):** converged (loss 0.000210). Landmark reproj error mean **22.30 px**,
  max 87.61 px (eyes 5–7 px, contour ~40 px). `id_abs_max=0.013`, expression fit on. Wrote
  `out/fit/fitted_neutral.npy/.obj`, `camera.json`, `fit_metrics.json`, `fit_debug.jpg`. 12.7 s.

Stages 4–7 intentionally NOT run (pipe code owned by a parallel agent).

## 6. Exact end-to-end run command
```bash
source /workspace/env/bin/activate
export HF_HOME=/workspace/cache/huggingface
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
ROOT=/workspace/newstack \
PIPE=/workspace/newarc/newstack/pipe \
PHOTO=/workspace/inputs/random-person.jpeg \
ICT=/workspace/newstack/ICT-FaceKit \
CLAY=/workspace/newstack/out_triposr/0/mesh.obj \
MP_TASK=/workspace/models/mediapipe/face_landmarker.task \
BLENDER=/workspace/blender/blender-4.2.3-linux-x64/blender \
STAGES="1 2 3 4 5 6 7" \
bash /workspace/newarc/newstack/run_newstack.sh
```
Output GLB: `/workspace/newstack/out/export/head_arkit_v2.glb`. Blender stages auto-wrap in
`xvfb-run -a`. Rerun rig only: `STAGES="4 5 6 7"`. Pure-ICT (skip clay): `REFINE=0`.

## 7. Persistence / restore
- `/workspace/restore_env.sh` — idempotent rebuild (apt libs + git-lfs always; venv,
  torchmcubes, pytorch3d, TripoSR weights, ICT, Blender, clay rebuilt if missing).
- `/workspace/RESTART.md` — layout, activate line, HF_HOME, run command, pins, gotchas.
- Container FS (apt, PATH) is wiped each pod start → restore_env.sh re-runs the apt block +
  `git lfs install` every time. Everything else lives under `/workspace`.

## 8. License flag (for license-compliance gate)
- ICT-FaceKit **FaceXModel = MIT** — commercial OK. Only the MIT "Light" model was pulled;
  no non-MIT extras.
- **TripoSR weights** license must be cleared before any commercial ship (this pod = Track A).

---
### PASS / FAIL per deliverable
| # | Deliverable | Result |
|---|-------------|--------|
| 1 | SSH + GPU verified (nvidia-smi, python, torch) | **PASS** |
| 2 | apt headless-Blender libs + tools (+ libgles2/libegl1 for mediapipe) | **PASS** |
| 3 | tongueOut repo cloned, photo copied | **PASS** |
| 4 | venv + full python stack (pytorch3d 0.7.8, torchmcubes CUDA, scipy, mediapipe, rembg) | **PASS** |
| 5 | TripoSR clay `out_triposr/0/mesh.obj` (81,812 v / 163,346 f) | **PASS** |
| 6 | ICT-FaceKit FaceXModel (160 files, MIT) | **PASS** |
| 7 | MediaPipe task + Blender 4.2.3 headless verified | **PASS** |
| 8 | s1/s2 smoke-test wiring (RC=0) | **PASS** |
| 9 | restore_env.sh + RESTART.md written | **PASS** |

---

## 9. TripoSG geometry source (added 2026-07-07) — replaces TripoSR clay

> NOTE: this section was produced on pod `root@195.26.233.74 -p 27697` (the live pod for
> this task), not the `.87` pod named at the top of this report. GPU is the same model
> (RTX 6000 Ada, 49 GB, CUDA 12.4, torch 2.4.1+cu124).

### License gate (HARD BLOCKER) — PASS: MIT for code AND weights
- **Code** — `github.com/VAST-AI-Research/TripoSG/LICENSE` (fetched raw), verbatim opening:
  `MIT License / Copyright (c) 2025 VAST-AI-Research and contributors.` Full permission grant
  including "to use, copy, modify, merge, publish, distribute, sublicense, and/or **sell**".
- **Weights** — `huggingface.co/VAST-AI/TripoSG` model card metadata: `license: mit`.
- Verdict: **commercial use OK**. TripoSG is a clean commercial-grade replacement for the
  non-commercial-encumbered FaceVerse and for TripoSR (whose weights still need clearing).
- **CAVEAT flagged for license-compliance:** the stock `scripts/inference_triposg.py`
  auto-downloads **briaai/RMBG-1.4** (license `bria-rmbg-1.4` = **NON-commercial**) for
  background removal. We DID NOT use it. The input was pre-matted with `rembg`/u2net
  (Apache-2.0) and passed as an RGBA image with valid alpha; `prepare_image(rmbg_net=None)`
  then skips RMBG. No RMBG weights touch the clay. The repo `NOTICE` also credits code
  derived from HunyuanDiT / FlashVDM (Tencent Community Licenses) — code-provenance notes,
  not shipped weights, but worth a human-lawyer glance for a commercial ship.

### Install (isolated venv — main `/workspace/env` untouched)
Dedicated venv `/workspace/env_triposg` (python 3.11.10) because TripoSG's deps conflict with
the main stack (it pins `numpy==1.22.3`, needs recent `diffusers`; main env has numpy 2.4.6 +
transformers 4.35 + hf_hub 0.17.3 that the TripoSR/pytorch3d stack depends on). Pins and why:

| pkg | version | why pinned |
|-----|---------|-----------|
| torch / torchvision | 2.4.1+cu124 / 0.19.1+cu124 | match pod CUDA 12.4 (same as main env) |
| diffusers | 0.30.0 | 0.39 (latest) registers a flash-attn `custom_op` with PEP-604 `float\|None` hints that torch 2.4.1 `infer_schema` cannot parse -> import crash. 0.30 has the APIs TripoSG needs (FP32LayerNorm, apply_rotary_emb, FlowMatchEulerDiscreteScheduler, PeftAdapterMixin) and imports clean on torch 2.4.1 |
| transformers | 4.44.2 | v5.x (latest) risks moving Dinov2Model/BitImageProcessor; 4.44 is TripoSG-era and provides both |
| huggingface_hub | 0.25.2 | diffusers 0.30 needs `errors.LocalEntryNotFoundError` (>=0.25) and peft 0.19 needs >=0.25; still has `cached_download` (<0.26) |
| numpy | 2.4.6 | KEPT at 2.4.6 (not the repo's 1.22.3): 1.22.3 has no py3.11 wheel, and diso's CUDA ext was compiled against 2.4.6 (downgrading would ABI-break it) |
| diso | 0.1.4 | built from sdist with `--no-build-isolation` (needs torch present); provides the 505^3 dual-MC surface extractor |
| pymeshlab | 2023.12.post3 | decimation only |

- Weights persisted (survive pod kill): `/workspace/pretrained_weights/TripoSG` = **7.5 GB**
  (transformer + vae + dinov2 encoder + feature_extractor). `HF_HOME=/workspace/cache/huggingface`.
- Repo clone: `/workspace/TripoSG`. Runner: `/workspace/TripoSG/run_triposg_clay.py`
  (my wrapper; does NOT touch `newstack/pipe/*.py`). pip freeze: `/workspace/logs/triposg_pipfreeze.txt`.

### Exact run command
```
source /workspace/env_triposg/bin/activate
export CUDA_HOME=/usr/local/cuda HF_HOME=/workspace/cache/huggingface
# pre-matte with permissive u2net -> input_rgba.png (bypasses non-commercial RMBG)
cd /workspace/TripoSG
python run_triposg_clay.py \
  --rgba /workspace/newstack/out_triposg/input_rgba.png \
  --outdir /workspace/newstack/out_triposg --seed 42 --steps 50 --guidance 7.0
```

### Output (measured) — location `/workspace/newstack/out_triposg/`
| file | verts | faces | size | color |
|------|------:|------:|-----:|-------|
| random-person_triposg.glb (full, primary) | 3,652,472 | 2,581,396 | 74.8 MB | **NONE** |
| random-person_triposg.obj / .ply (full) | 3,652,472 | 2,581,396 | 199 / 77 MB | NONE |
| random-person_triposg_300k.glb (decimated companion) | 150,084 | 300,000 | 7.2 MB | NONE |

- **Format:** GLB (also OBJ + PLY). Watertight. Surface = diso dual-MC on a **505^3** SDF grid.
- **Scale:** normalized to ~[-1,1]; bbox extent 1.785 x 1.864 x 1.810.
- **COLOR = NONE (geometry-only).** `visual.kind=None`, no per-vertex color, no UV, no texture
  image. TripoSG is shape-only; **there is no official TripoSG texture model in the repo.**
  Downstream color options: (a) transfer per-vertex color from the TripoSR clay
  `out_triposr/0/mesh.obj` (HAS RGBA, e.g. row0 [111,97,85,255]) via nearest-surface sampling;
  (b) an MV-Adapter / texture-gen pipeline; (c) reproject the front photo + infer back/side.
- **VRAM:** peak 6.56 GB. **Time:** 18.2 s (50 steps) on RTX 6000 Ada.

### TripoSR vs TripoSG
| metric | TripoSR (kept, fallback color) | TripoSG full | TripoSG 300k |
|--------|-------------------------------:|-------------:|-------------:|
| verts | 81,812 | 3,652,472 | 150,084 |
| faces | 163,346 | 2,581,396 | 300,000 |
| vertex color | YES (RGBA) | NO | NO |
| bbox extent | 0.86 x 0.74 x 0.98 | 1.79 x 1.86 x 1.81 | (same as full) |
| surface | triplane NeRF -> MC | 505^3 SDF -> diso MC | decimated |

TripoSG full = ~15.8x TripoSR's face count over a comparable/larger head volume, i.e. far
higher surface sampling -> sharper detail and a tighter hairline. This is grounded in
geometry resolution/topology (measured), **not** a rendered pixel diff. TripoSR clay is left
UNMODIFIED and remains the fallback color source.

### PASS / FAIL (TripoSG additions)
| # | Deliverable | Result |
|---|-------------|--------|
| 10 | License gate (code+weights MIT, quoted) | **PASS** |
| 11 | TripoSG installed, isolated venv, weights persisted (7.5 GB) | **PASS** |
| 12 | Clay generated `out_triposg/random-person_triposg.glb` (2.58M f, watertight) | **PASS** |
| 13 | Output format/color reported (GLB, geometry-only, no texture) | **PASS** |
| 14 | TripoSR clay kept intact; pipe/*.py untouched | **PASS** |

**STATUS: READY** (clay at `/workspace/newstack/out_triposg/random-person_triposg.glb`,
color=none, license=MIT)
