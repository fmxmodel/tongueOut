#!/usr/bin/env bash
# ============================================================================
# pod_setup_b1.sh  —  turnkey provisioner for Track B1 (FLAME 2023 Open, COMMERCIAL)
# ----------------------------------------------------------------------------
# WHERE THIS RUNS:  the GPU POD ONLY  (Linux, NVIDIA RTX 6000 Ada / sm_89).
# WHERE IT MUST NOT RUN:  the authoring laptop (no GPU, contamination guard).
#   -> The first thing it does is REFUSE to run if no NVIDIA GPU is present.
#
# WHAT IT DOES (idempotent — checks first, installs only what's missing):
#   1. venv at $VENV  (records activation path)
#   2. torch 2.4.1 + cu121  (Ada sm_89 is in the cu121 build)
#   3. requirements-b1.txt  (mediapipe, numpy<2, scipy, trimesh, opencv, Pillow, ...)
#   4. pytorch3d 0.7.8  (try prebuilt wheel -> else source build for sm_89)
#   5. MediaPipe face_landmarker.task model
#   6. prints the MANUAL FLAME 2023 Open download step (cannot be automated:
#      it is gated behind registration + CC-BY license acceptance)
#   7. verifies the stack imports and that torch.cuda.is_available() == True
#
# The B1 METHOD this toolchain must match EXACTLY:
#   reconstruction = optimization-based FLAME 2023 Open fit to MediaPipe
#   FaceLandmarker landmarks (pure optimization; NO learned NC weights).
#   texture = per-subject albedo baked from the input photo onto the FLAME UV
#   + mirror-symmetry + CLASSICAL inpainting (cv2.inpaint). NO statistical
#   albedo prior (no BFM / AlbedoMM / MPI FLAME texture space).
#
# DIFFERENTIABLE-RENDERER CHOICE: PyTorch3D (BSD-3-Clause), NOT nvdiffrast.
#   nvdiffrast is under the NVIDIA Source Code License (1-Way Commercial) =
#   NON-COMMERCIAL -> BARRED from this commercial run. PyTorch3D is BSD and also
#   bundles cameras / TexturesUV / mesh regularizers we need for the fit+bake.
#
# Usage on the pod:
#   bash scripts/pod_setup_b1.sh
#   source /workspace/venvs/b1/bin/activate
# ============================================================================
set -euo pipefail

# ---- config (override via env if the pod layout differs) -------------------
WORKSPACE="${WORKSPACE:-/workspace}"
VENV="${VENV:-$WORKSPACE/venvs/b1}"
PYBIN="${PYBIN:-python3.10}"                 # pin: 3.10 = best pytorch3d wheel coverage
TORCH_VER="${TORCH_VER:-2.4.1}"             # pin: last torch supported by pytorch3d 0.7.8
TV_VER="${TV_VER:-0.19.1}"                  # torchvision paired with torch 2.4.1
CUDA_TAG="${CUDA_TAG:-cu121}"              # cu121 build ships sm_89 kernels for Ada
PYT3D_VER="${PYT3D_VER:-0.7.8}"
ARCH_LIST="${ARCH_LIST:-8.9+PTX}"          # Ada Lovelace = sm_89; +PTX for forward-compat
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REQ_FILE="${REQ_FILE:-$REPO_ROOT/requirements-b1.txt}"

MODELS_DIR="$WORKSPACE/models"
FLAME_DIR="$MODELS_DIR/flame2023_open"
MP_DIR="$MODELS_DIR/mediapipe"
INPUT_DIR="$WORKSPACE/inputs"
MP_TASK_URL="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

# Blender for the GLB assembly stage (blender_build_rig.py). Headless glTF
# export is CPU-only (no GPU needed); we install a portable build so the pod is
# self-contained. 4.2 LTS ships the modern io_scene_gltf2 exporter with
# export_morph / export_morph_normal / Draco flags. Override BLENDER_VER if a
# patch release URL 404s (see https://download.blender.org/release/).
BLENDER_VER="${BLENDER_VER:-4.2.3}"
BLENDER_DIR="$WORKSPACE/blender"

log()  { printf '\n\033[1;36m[b1-setup]\033[0m %s\n' "$*"; }
warn() { printf '\n\033[1;33m[b1-setup WARN]\033[0m %s\n' "$*"; }
die()  { printf '\n\033[1;31m[b1-setup FATAL]\033[0m %s\n' "$*" >&2; exit 1; }

# ---- 0. CONTAMINATION GUARD: refuse to run without an NVIDIA GPU ------------
if ! command -v nvidia-smi >/dev/null 2>&1; then
  die "no 'nvidia-smi' found. This script must run ON THE GPU POD, never on the
       CPU-only authoring box. Aborting to honor the no-local-compute rule."
fi
nvidia-smi -L || die "nvidia-smi present but no GPU enumerated."
log "GPU detected:"; nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv || true

# ---- 1. venv ---------------------------------------------------------------
if [ ! -x "$VENV/bin/python" ]; then
  command -v "$PYBIN" >/dev/null 2>&1 || die "$PYBIN not found. Install Python 3.10 (or set PYBIN=python3.11)."
  log "creating venv at $VENV"
  "$PYBIN" -m venv "$VENV"
else
  log "venv already exists at $VENV (reusing)"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# ---- 2. torch (CUDA build for Ada) -----------------------------------------
if python -c "import torch,sys; sys.exit(0 if torch.version.cuda else 1)" 2>/dev/null; then
  log "torch already installed: $(python -c 'import torch;print(torch.__version__)')"
else
  log "installing torch $TORCH_VER + torchvision $TV_VER ($CUDA_TAG)"
  pip install "torch==${TORCH_VER}" "torchvision==${TV_VER}" \
      --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
fi

# ---- 3. permissive python deps ---------------------------------------------
log "installing requirements-b1.txt (mediapipe / numpy<2 / scipy / trimesh / opencv / Pillow / ...)"
[ -f "$REQ_FILE" ] || die "requirements file not found: $REQ_FILE"
pip install -r "$REQ_FILE"

# ---- 4. pytorch3d 0.7.8: prebuilt wheel first, else source build -----------
if python -c "import pytorch3d" 2>/dev/null; then
  log "pytorch3d already installed: $(python -c 'import pytorch3d;print(pytorch3d.__version__)')"
else
  PYTAG="$(python -c 'import sys;print(f"py3{sys.version_info.minor}")')"
  PT_SHORT="$(python -c 'import torch;print("".join(torch.__version__.split("+")[0].split(".")))')"
  WHEEL_IDX="https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/${PYTAG}_${CUDA_TAG}_pyt${PT_SHORT}/download.html"
  log "attempting pytorch3d prebuilt wheel from: $WHEEL_IDX"
  if pip install "pytorch3d==${PYT3D_VER}" -f "$WHEEL_IDX"; then
    log "pytorch3d installed from prebuilt wheel"
  else
    warn "no matching prebuilt wheel — building pytorch3d ${PYT3D_VER} from source for sm_89.
          This needs a CUDA *devel* image (nvcc matching torch's CUDA ${CUDA_TAG}).
          Build takes ~15-40 min on the pod."
    command -v nvcc >/dev/null 2>&1 || die "nvcc not found. Use a CUDA-devel pod image
      (e.g. runpod/pytorch:2.4.1-...-devel) or install the CUDA 12.1 toolkit, then re-run."
    FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="$ARCH_LIST" \
      pip install --no-build-isolation \
      "git+https://github.com/facebookresearch/pytorch3d.git@V${PYT3D_VER}"
  fi
fi

# ---- 5. MediaPipe FaceLandmarker model -------------------------------------
mkdir -p "$MP_DIR"
if [ -s "$MP_DIR/face_landmarker.task" ]; then
  log "face_landmarker.task already present at $MP_DIR"
else
  log "downloading face_landmarker.task -> $MP_DIR"
  ( command -v wget >/dev/null 2>&1 && wget -q -O "$MP_DIR/face_landmarker.task" "$MP_TASK_URL" ) \
    || curl -fsSL -o "$MP_DIR/face_landmarker.task" "$MP_TASK_URL" \
    || warn "could not fetch face_landmarker.task (offline?). Fetch manually from:
             $MP_TASK_URL"
fi

# ---- 6. FLAME 2023 Open — MANUAL, license-gated ----------------------------
mkdir -p "$FLAME_DIR" "$INPUT_DIR"
if [ -n "$(ls -A "$FLAME_DIR" 2>/dev/null || true)" ]; then
  log "FLAME 2023 Open assets present in $FLAME_DIR"
else
  warn "FLAME 2023 Open model is NOT present and CANNOT be auto-downloaded (it is gated
        behind registration + CC-BY license acceptance, and is NOT redistributable by us).
        OPERATOR ACTION REQUIRED — see models/README:
          1. Register + accept the CC-BY-4.0 license at flame.is.tue.mpg.de
          2. Download the 'FLAME 2023 (Open)' release ONLY.
             Do NOT download the FLAME texture / albedo package (CC-BY-NC-SA = BARRED).
          3. Place files into: $FLAME_DIR/
             (flame2023.pkl — the Open release is PKL-ONLY; the landmark anchors
             and UV are self-authored/generated clean-room by the pipeline.
             See models/README section 1.)
          4. Put the input photo at: $INPUT_DIR/random-person.jpeg"
fi

# ---- 6b. Blender (portable, headless) for the GLB assembly stage -----------
# Consumed by blender_build_rig.py / scripts/run_glb_b1.sh. CPU-only; needs no
# CUDA. Idempotent: skips if a usable Blender is already reachable.
if command -v blender >/dev/null 2>&1 || [ -x "$BLENDER_DIR/blender" ]; then
  log "Blender already present ($(command -v blender || echo "$BLENDER_DIR/blender"))"
else
  BLENDER_SERIES="${BLENDER_VER%.*}"
  BLENDER_TARBALL="blender-${BLENDER_VER}-linux-x64.tar.xz"
  BLENDER_URL="https://download.blender.org/release/Blender${BLENDER_SERIES}/${BLENDER_TARBALL}"
  log "installing portable Blender ${BLENDER_VER} -> $BLENDER_DIR"
  mkdir -p "$BLENDER_DIR"
  if ( command -v wget >/dev/null 2>&1 && wget -q -O "/tmp/${BLENDER_TARBALL}" "$BLENDER_URL" ) \
       || curl -fsSL -o "/tmp/${BLENDER_TARBALL}" "$BLENDER_URL"; then
    tar -xJf "/tmp/${BLENDER_TARBALL}" -C "$BLENDER_DIR" --strip-components=1
    rm -f "/tmp/${BLENDER_TARBALL}"
    log "Blender installed. Add to PATH:  export PATH=\"$BLENDER_DIR:\$PATH\""
    log "(scripts/run_glb_b1.sh also auto-detects \$WORKSPACE/blender/blender)"
  else
    warn "could not fetch Blender ${BLENDER_VER} from $BLENDER_URL (offline / bad patch
          version?). Install manually: download a Blender 4.x/5.x linux-x64 build from
          https://www.blender.org/download/ , extract to $BLENDER_DIR, OR set
          \$BLENDER=/path/to/blender before running scripts/run_glb_b1.sh."
  fi
fi

# ---- 7. verify (imports + CUDA) --------------------------------------------
log "verifying the stack (imports + torch.cuda.is_available)"
python - <<'PY'
import torch, importlib
print("torch            :", torch.__version__, "| cuda build:", torch.version.cuda)
print("cuda available   :", torch.cuda.is_available())
print("device count     :", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0         :", torch.cuda.get_device_name(0),
          "| cc", ".".join(map(str, torch.cuda.get_device_capability(0))))
for m in ("pytorch3d","mediapipe","cv2","trimesh","scipy","numpy","PIL","skimage"):
    try:
        mod = importlib.import_module(m)
        print(f"{m:16s} : OK  {getattr(mod,'__version__','')}")
    except Exception as e:
        print(f"{m:16s} : FAIL  {e!r}")
PY

# Blender (GLB stage) — report the version that run_glb_b1.sh will use
if command -v blender >/dev/null 2>&1; then
  log "blender          : $(blender --version 2>/dev/null | head -1)"
elif [ -x "$BLENDER_DIR/blender" ]; then
  log "blender          : $("$BLENDER_DIR/blender" --version 2>/dev/null | head -1) ($BLENDER_DIR)"
else
  warn "blender          : NOT installed — the GLB stage (scripts/run_glb_b1.sh) will
        refuse until Blender is present. Re-run this script online, or install manually."
fi

log "DONE. Activate with:  source $VENV/bin/activate"
log "venv activation path recorded: $VENV/bin/activate"
log "GLB stage (after recon+rig):  bash scripts/run_glb_b1.sh"
