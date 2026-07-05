#!/usr/bin/env bash
# ============================================================================
# run_rig_b1.sh  —  turnkey runner for the Track B1 ARKit rigging stage
# ----------------------------------------------------------------------------
# WHERE THIS RUNS:  the GPU POD ONLY (Linux, RTX 6000 Ada). Refuses otherwise.
# PREREQ:  scripts/pod_setup_b1.sh (same venv as recon) + a COMPLETED recon
#          run (bash scripts/run_recon_b1.sh) so out/recon/ holds neutral.ply,
#          faces.npy, id_params.npz, expression_basis.npz, landmarks.npz
#          + the FLAME 2023 Open dir (landmark embedding is read from it).
#
# STAGES (each runnable alone: pass build|verify):
#   1. python -m rig.build_arkit_shapes   solve FLAME->ARKit, write
#                                         out/shapes/expr_*.ply + manifest
#   2. python -m rig.verify_shapes        independent measured re-check
#                                         (exit != 0 on any failure)
#
# Usage on the pod (from the repo root):
#   bash scripts/run_rig_b1.sh            # both stages
#   bash scripts/run_rig_b1.sh build      # one stage
# ============================================================================
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
VENV="${VENV:-$WORKSPACE/venvs/b1}"
FLAME_DIR="${B1_FLAME_DIR:-$WORKSPACE/models/flame2023_open}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${1:-all}"

log() { printf '\n\033[1;35m[b1-rig]\033[0m %s\n' "$*"; }
die() { printf '\n\033[1;31m[b1-rig FATAL]\033[0m %s\n' "$*" >&2; exit 1; }

# ---- contamination guard: GPU pod only --------------------------------------
command -v nvidia-smi >/dev/null 2>&1 \
  || die "no nvidia-smi. This runs ONLY on the GPU pod, never on the authoring box."
nvidia-smi -L >/dev/null 2>&1 || die "nvidia-smi present but no GPU enumerated."

# ---- venv --------------------------------------------------------------------
[ -x "$VENV/bin/python" ] || die "venv missing at $VENV — run scripts/pod_setup_b1.sh first."
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# ---- preflight: the recon stage must be complete -------------------------------
for f in neutral.ply faces.npy id_params.npz expression_basis.npz landmarks.npz; do
  [ -s "$REPO_ROOT/out/recon/$f" ] \
    || die "out/recon/$f missing — run bash scripts/run_recon_b1.sh first."
done
[ -s "$REPO_ROOT/out/arkit_51_52_map.json" ] \
  || die "out/arkit_51_52_map.json (the 52-name contract) is missing."
[ -n "$(ls -A "$FLAME_DIR" 2>/dev/null || true)" ] \
  || die "FLAME 2023 Open assets missing in $FLAME_DIR (landmark embedding is
          read from there — models/README.md section 1)."
python - <<'PY' || die "python stack broken — re-run scripts/pod_setup_b1.sh"
import torch, trimesh, numpy  # noqa: F401
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
print("[b1-rig] stack OK: torch", torch.__version__)
PY

cd "$REPO_ROOT"
run_stage() { log "stage: $1"; python -m "rig.$1"; }

case "$STAGE" in
  all)
    run_stage build_arkit_shapes
    run_stage verify_shapes
    ;;
  build)  run_stage build_arkit_shapes ;;
  verify) run_stage verify_shapes ;;
  *) die "unknown stage '$STAGE' (use: all|build|verify)" ;;
esac

log "rig stage complete. Artifacts in out/shapes/ — measured manifest:"
[ -f "$REPO_ROOT/out/shapes/shapes_run_manifest.json" ] \
  && cat "$REPO_ROOT/out/shapes/shapes_run_manifest.json" \
  || echo "(shapes_run_manifest.json not present — run the verify stage)"
