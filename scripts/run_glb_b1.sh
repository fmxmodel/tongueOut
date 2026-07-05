#!/usr/bin/env bash
# ============================================================================
# run_glb_b1.sh  —  turnkey runner for the Track B1 Blender assembly + GLB stage
# ----------------------------------------------------------------------------
# WHERE THIS RUNS:  the GPU POD (or any Linux box with Blender + the real
#   recon/rig artifacts). It does NOT need a GPU — glTF export is CPU-only — but
#   the inputs it consumes are produced by the GPU recon+rig stages, so in
#   practice it runs on the pod after them.
#
# PREREQ (all produced on the pod, upstream of this stage):
#   out/shapes/arkit_manifest.json   run_state == "measured-on-pod"  (rig)
#   out/shapes/neutral.ply           (rig)
#   out/shapes/expr_<arkitName>.ply  one per SUPPORTED shape          (rig)
#   out/recon/faces.npy              topology contract                (recon)
#   out/recon/albedo.png             baked albedo                     (recon)
#   out/recon/uv_coords.npz          FLAME UV layout                  (recon)
#   + Blender 4.x/5.x on PATH (see scripts/pod_setup_b1.sh step 8 /
#     out/gpu_requirements_b1.md "Blender").
#
# WHAT IT DOES:
#   1. locate a headless Blender (PATH, or $BLENDER, or $WORKSPACE/blender/blender)
#   2. preflight: the six real inputs exist AND the manifest is measured-on-pod
#   3. blender --background --python blender_build_rig.py
#      -> out/head_arkit.glb, out/head_rigged.blend, out/glb_report.md
#
# Usage on the pod (from the repo root):
#   bash scripts/run_glb_b1.sh
#   B1_GLB_DRACO=1 bash scripts/run_glb_b1.sh      # smaller GLB (needs DRACO
#                                                  # decoder configured in viewer)
# ============================================================================
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

log() { printf '\n\033[1;34m[b1-glb]\033[0m %s\n' "$*"; }
die() { printf '\n\033[1;31m[b1-glb FATAL]\033[0m %s\n' "$*" >&2; exit 1; }

# ---- 1. locate Blender ------------------------------------------------------
BLENDER_BIN=""
if [ -n "${BLENDER:-}" ] && [ -x "${BLENDER}" ]; then
  BLENDER_BIN="$BLENDER"
elif command -v blender >/dev/null 2>&1; then
  BLENDER_BIN="$(command -v blender)"
elif [ -x "$WORKSPACE/blender/blender" ]; then
  BLENDER_BIN="$WORKSPACE/blender/blender"
fi
[ -n "$BLENDER_BIN" ] || die "Blender not found. Install it (scripts/pod_setup_b1.sh step 8),
        set \$BLENDER=/path/to/blender, or put it on PATH. See out/gpu_requirements_b1.md."
log "using Blender: $BLENDER_BIN"
"$BLENDER_BIN" --version | head -1 || true

# ---- 2. preflight: the real inputs must exist -------------------------------
for f in shapes/arkit_manifest.json shapes/neutral.ply \
         recon/faces.npy recon/albedo.png recon/uv_coords.npz; do
  [ -s "$REPO_ROOT/out/$f" ] \
    || die "out/$f missing — run recon (scripts/run_recon_b1.sh) and rig
            (scripts/run_rig_b1.sh) on the pod FIRST. This stage is DEFERRED until then."
done

# ---- 3. gate on a MEASURED manifest (belt-and-suspenders; the python re-checks)
python3 - "$REPO_ROOT/out/shapes/arkit_manifest.json" <<'PY' || die "manifest gate failed"
import json, sys
m = json.load(open(sys.argv[1]))
rs = m.get("run_state")
if rs != "measured-on-pod":
    sys.exit(f"[b1-glb FATAL] manifest run_state == {rs!r}, not 'measured-on-pod'. "
             "The rig stage has NOT run on the pod yet — nothing to build.")
sup = [n for n, v in m.get("shapes", {}).items() if v.get("supported") is True]
print(f"[b1-glb] manifest measured-on-pod: {len(sup)} supported ARKit shapes")
PY

# ---- 4. build ---------------------------------------------------------------
cd "$REPO_ROOT"
log "running: $BLENDER_BIN --background --python blender_build_rig.py"
"$BLENDER_BIN" --background --python "$REPO_ROOT/blender_build_rig.py"

# ---- 5. summary -------------------------------------------------------------
[ -s "$REPO_ROOT/out/head_arkit.glb" ] \
  || die "blender_build_rig.py finished but out/head_arkit.glb is missing/empty."
log "GLB stage complete:"
ls -lh "$REPO_ROOT/out/head_arkit.glb" "$REPO_ROOT/out/head_rigged.blend" 2>/dev/null || true
[ -f "$REPO_ROOT/out/glb_report.md" ] && cat "$REPO_ROOT/out/glb_report.md"
