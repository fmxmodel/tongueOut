#!/usr/bin/env bash
# ============================================================================
# run_recon_b1.sh  —  turnkey runner for the Track B1 reconstruction stage
# ----------------------------------------------------------------------------
# WHERE THIS RUNS:  the GPU POD ONLY (Linux, RTX 6000 Ada). Refuses otherwise.
# PREREQ:  bash scripts/pod_setup_b1.sh   (venv, torch cu121, pytorch3d 0.7.8,
#          mediapipe + face_landmarker.task)  +  the MANUAL FLAME 2023 Open
#          download into /workspace/models/flame2023_open/ (models/README.md)
#          +  the input photo at /workspace/inputs/random-person.jpeg.
#
# STAGES (each is also runnable alone: pass landmarks|fit|bake|verify):
#   1. python -m recon.landmarks       MediaPipe FaceLandmarker -> landmarks.npz
#   2. python -m recon.fit_flame       optimization fit -> neutral.ply,
#                                      faces.npy, id_params.npz, expression_basis.npz
#   3. python -m recon.bake_texture    photo -> FLAME UV albedo.png (no prior)
#   4. python -m recon.verify_outputs  measured asserts -> recon_run_manifest.json
#
# Usage on the pod (from the repo root):
#   bash scripts/run_recon_b1.sh          # all stages
#   bash scripts/run_recon_b1.sh fit      # one stage
# ============================================================================
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
VENV="${VENV:-$WORKSPACE/venvs/b1}"
FLAME_DIR="${B1_FLAME_DIR:-$WORKSPACE/models/flame2023_open}"
MP_TASK="${B1_MP_TASK:-$WORKSPACE/models/mediapipe/face_landmarker.task}"
INPUT_IMG="${B1_INPUT:-$WORKSPACE/inputs/random-person.jpeg}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${1:-all}"

log() { printf '\n\033[1;36m[b1-recon]\033[0m %s\n' "$*"; }
die() { printf '\n\033[1;31m[b1-recon FATAL]\033[0m %s\n' "$*" >&2; exit 1; }

# ---- contamination guard: GPU pod only --------------------------------------
command -v nvidia-smi >/dev/null 2>&1 \
  || die "no nvidia-smi. This runs ONLY on the GPU pod, never on the authoring box."
nvidia-smi -L >/dev/null 2>&1 || die "nvidia-smi present but no GPU enumerated."

# ---- venv --------------------------------------------------------------------
[ -x "$VENV/bin/python" ] || die "venv missing at $VENV — run scripts/pod_setup_b1.sh first."
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# ---- preflight: required assets ----------------------------------------------
[ -s "$MP_TASK" ] || die "face_landmarker.task missing at $MP_TASK (pod_setup_b1.sh step 5)."
[ -s "$INPUT_IMG" ] || die "input photo missing at $INPUT_IMG (models/README.md section 5)."
[ -n "$(ls -A "$FLAME_DIR" 2>/dev/null || true)" ] \
  || die "FLAME 2023 Open assets missing in $FLAME_DIR — MANUAL license-gated download
          (models/README.md section 1). Do NOT download the NC texture package."
python - <<'PY' || die "python stack broken — re-run scripts/pod_setup_b1.sh"
import torch, pytorch3d, mediapipe, cv2, trimesh, scipy  # noqa: F401
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
print("[b1-recon] stack OK:", torch.__version__, "| pytorch3d", pytorch3d.__version__)
PY

cd "$REPO_ROOT"
run_stage() { log "stage: $1"; python -m "recon.$1"; }

case "$STAGE" in
  all)
    run_stage landmarks
    run_stage fit_flame
    run_stage bake_texture
    run_stage verify_outputs
    ;;
  landmarks)  run_stage landmarks ;;
  fit)        run_stage fit_flame ;;
  bake)       run_stage bake_texture ;;
  verify)     run_stage verify_outputs ;;
  *) die "unknown stage '$STAGE' (use: all|landmarks|fit|bake|verify)" ;;
esac

log "recon stage complete. Artifacts in out/recon/ — manifest:"
[ -f "$REPO_ROOT/out/recon/recon_run_manifest.json" ] \
  && cat "$REPO_ROOT/out/recon/recon_run_manifest.json" \
  || echo "(manifest not present — run the verify stage)"
