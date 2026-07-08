# newARC — Single Image → ARKit Avatar → GLB

Turn a **single photo** into a **rigged 3D head** with **52 ARKit blendshapes**, exported as a **GLB** — commercially permissive (all MIT/Apache/BSD components).

**Branch `minimal`**: simplified TripoSR-only pipeline. No TripoSG, no TRELLIS, no photo-projection complexity. Photo provides front face color; TripoSR provides consistent 360° fill.

---

## Pipeline overview

```
photo
  ↓ rembg / U²-Net              background removal
  ↓ TripoSR                      3D geometry + vertex colors
  ↓ MediaPipe landmarks          s1 — facial landmark detection
  ↓ ICT identity fit              s2 — fit to photo landmarks
  ↓ TripoSR → ICT alignment       s3a — align clay to ICT space
  ↓ Blender shrinkwrap            s3b — retopologise onto ICT topology
  ↓ verify refinement             s3c — topology/reprojection gates
  ↓ 52 ARKit blendshapes          s4 — ICT deltas + tongueOut + gaze
  ↓ texture bake                  s5 — photo front + TripoSR fill
  ↓ Blender GLB export            s6 — opaque, doubleSided, 52 morphs
  ↓ verify GLB                    s7 — contract measurement gates
  ↓ three.js + MediaPipe viewer   live webcam-driven 52/52
GLB avatar
```

---

## Requirements

### GPU (recommended)
- NVIDIA GPU with 16 GB+ VRAM (RTX 6000 Ada used in development)
- CUDA 12.4
- Linux (Ubuntu 22.04 used in development)

### Software
- Python 3.11+
- Blender 4.2.3 LTS (headless-capable)
- Node.js 18+ (for the three.js viewer)

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/fmxmodel/tongueOut.git
cd tongueOut
git checkout minimal
```

### 2. Install Python dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install mediapipe trimesh scipy opencv-python pillow scikit-image imageio
pip install fvcore iopath ninja
pip install "rembg[cpu]" huggingface-hub
```

### 3. Install PyTorch3D

```bash
git clone https://github.com/facebookresearch/pytorch3d.git /tmp/pytorch3d
pip install /tmp/pytorch3d
```

### 4. Install Blender 4.2.3 LTS

```bash
wget https://mirror.clarkson.edu/blender/release/Blender4.2/blender-4.2.3-linux-x64.tar.xz
tar xf blender-4.2.3-linux-x64.tar.xz -C /opt/
export PATH="/opt/blender-4.2.3-linux-x64:$PATH"
```

Install X11 dependencies for headless Blender:
```bash
apt-get install -y libxi6 libxrender1 libxfixes3 libegl1-mesa libgles2-mesa xvfb
```

### 5. Download models

```bash
# ICT-FaceKit (MIT)
git clone https://github.com/ICT-VGL/ICT-FaceKit.git /workspace/newstack/ICT-FaceKit

# MediaPipe face landmarker
mkdir -p /workspace/models/mediapipe
wget -O /workspace/models/mediapipe/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

# TripoSR model weights (downloaded automatically on first run)
```

### 6. Install TripoSR

```bash
cd /workspace
wget https://github.com/VAST-AI-Research/TripoSR/archive/refs/heads/main.zip
unzip main.zip && mv TripoSR-main TripoSR && rm main.zip
cd TripoSR
CUDACXX=/usr/local/cuda/bin/nvcc pip install git+https://github.com/tatsy/torchmcubes.git
pip install omegaconf einops transformers imageio gradio moderngl xatlas
```

### 7. Set up viewer

```bash
cd out/viewer
npm install
```

---

## Quick start (one command)

Place your input photo at `inputs/random-person.jpeg`, then:

```bash
# 1. Generate TripoSR clay
cd /workspace/TripoSR
CUDA_HOME=/usr/local/cuda python3 run.py /workspace/inputs/random-person.jpeg \
  --output-dir /workspace/newstack/out_triposr/0 --model-save-format obj

# 2. Set up paths
ln -sf /workspace/newstack/out_triposr/0/0/mesh.obj /workspace/newstack/out_triposr/0/mesh.obj

# 3. Run the full pipeline
cd /workspace/newstack
bash run_newstack.sh
```

Outputs:
- `out/export/head_arkit_v2.glb` — final rigged avatar (52 morph targets)
- `out/renders/glb_*.png` — proof renders (front/profile/back/eyes/tongue)
- `out/rig/arkit_manifest.json` — blendshape manifest

---

## Stage-by-stage execution

Each stage can be run independently:

```bash
STAGES="1"   bash run_newstack.sh   # MediaPipe landmarks
STAGES="2"   bash run_newstack.sh   # ICT identity fit
STAGES="3"   bash run_newstack.sh   # clay align + shrinkwrap
STAGES="4"   bash run_newstack.sh   # 52 ARKit shapes
STAGES="5"   bash run_newstack.sh   # texture bake
STAGES="6"   bash run_newstack.sh   # Blender GLB export
STAGES="7"   bash run_newstack.sh   # verify GLB
STAGES="8"   bash run_newstack.sh   # proof renders
STAGES="5 6 7" bash run_newstack.sh  # re-texture + export + verify
```

---

## Viewing the result

### three.js web viewer

```bash
cd out/viewer
npm run dev
# Opens at http://localhost:5173
```

The viewer loads `head_arkit_v2.glb` and drives all 52 morph targets live from webcam via MediaPipe FaceLandmarker.

### Blender (local inspection)

```bash
/path/to/blender --python open_avatar.py
```

Imports the GLB with all 52 shape keys, opaque materials, and Material Preview mode.

---

## Project structure

```
newARC/
├── newstack/                   # Active pipeline
│   ├── run_newstack.sh         # Orchestrator
│   └── pipe/                   # Stage scripts
│       ├── s1_landmarks.py     # MediaPipe landmarks
│       ├── s2_fit_identity.py  # ICT identity fit
│       ├── s3a_align_clay.py   # TripoSR → ICT alignment
│       ├── s3b_refine_blender.py  # Blender shrinkwrap
│       ├── s3c_verify_refine.py   # Topology verification
│       ├── s4_build_shapes.py  # 52 ARKit blendshapes
│       ├── s5_bake_texture.py  # Texture bake (photo + TripoSR)
│       ├── s6_export_blender.py   # GLB export
│       ├── s7_verify_glb.py    # GLB contract verification
│       ├── s8_render_previews.py # Proof renders
│       ├── common.py           # Shared utilities
│       ├── arkit_names.py      # ARKit-52 contract
│       ├── ict_loader.py       # ICT-FaceKit model loader
│       ├── tongue_synth.py     # tongueOut synthesis
│       ├── gaze_synth.py       # eyeLook rotation synthesis
│       └── eye_texture.py      # Photo-derived eye textures
├── out/                        # Pipeline outputs
│   ├── export/head_arkit_v2.glb  # Final avatar
│   ├── viewer/                 # three.js + MediaPipe viewer
│   └── renders/                # Proof render images
├── open_avatar.py              # Blender launcher script
└── README.md                   # This file
```

---

## Licensing

All components are commercially permissive:
- **TripoSR** — MIT (code + weights)
- **U²-Net / rembg** — MIT / Apache
- **ICT-FaceKit** — MIT © USC-ICT 2020 (Light model only)
- **MediaPipe** — Apache 2.0
- **three.js** — MIT
- **PyTorch3D** — BSD
- **Blender** — GPL (build-time only, never shipped)

See `out/compliance_newstack.md` for the full compliance report.
