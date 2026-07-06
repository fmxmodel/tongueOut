#!/usr/bin/env bash
# =============================================================================
# newstack fusion pipeline: single photo -> commercial GLB (ICT Light, MIT)
# with hair volume (TripoSR clay), photo texture, ARKit-52 morph targets
# (51 from ICT expression OBJs + tongueOut synthesized from ICT's real
# static tongue geometry -- see pipe/tongue_synth.py).
#
# Pod layout (defaults; override via env):
#   pipe scripts   /workspace/newstack/pipe
#   ICT-FaceKit    /workspace/newstack/ICT-FaceKit    (FaceXModel only, MIT)
#   TripoSR clay   /workspace/newstack/out_triposr/0/mesh.obj
#   photo          /workspace/inputs/random-person.jpeg
#   outputs        /workspace/newstack/out
#
# Usage:
#   bash /workspace/newstack/pipe/../run_newstack.sh          # all stages
#   STAGES="4 5 6 7" bash run_newstack.sh                     # rerun from rig
#   REFINE=0 bash run_newstack.sh                             # skip clay (A/B)
#   TEX_SIZE=2048 DRACO=1 bash run_newstack.sh
#
# Stages: 1 landmarks | 2 identity fit | 3 clay align + shrinkwrap refine
#         4 ARKit shapes | 5 texture bake | 6 Blender GLB export | 7 verify
# Every stage is independently rerunnable; artifacts live under $OUT/<stage>/.
# =============================================================================
set -euo pipefail

ROOT=${ROOT:-/workspace/newstack}
# locate the stage scripts relative to this file (works for any scp layout)
SELF_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [ -z "${PIPE:-}" ]; then
  if   [ -f "$SELF_DIR/s1_landmarks.py" ];      then PIPE=$SELF_DIR
  elif [ -f "$SELF_DIR/pipe/s1_landmarks.py" ]; then PIPE=$SELF_DIR/pipe
  else PIPE=$ROOT/pipe
  fi
fi
OUT=${OUT:-$ROOT/out}
PY=${PY:-python3}
BLENDER=${BLENDER:-/workspace/blender/blender-4.2.3-linux-x64/blender}

PHOTO=${PHOTO:-/workspace/inputs/random-person.jpeg}
ICT=${ICT:-$ROOT/ICT-FaceKit}
CLAY=${CLAY:-$ROOT/out_triposr/0/mesh.obj}
MP_TASK=${MP_TASK:-/workspace/models/mediapipe/face_landmarker.task}

STAGES=${STAGES:-"1 2 3 4 5 6 7"}
REFINE=${REFINE:-1}          # 0 = skip clay align + shrinkwrap (pure ICT fit)
TEX_SIZE=${TEX_SIZE:-1024}
DRACO=${DRACO:-0}
S2_ARGS=${S2_ARGS:-}         # e.g. "--lam-id 0.1 --iters2 1200"
S3A_ARGS=${S3A_ARGS:-}       # e.g. "--force-bbox --fallback-up z_up"
S3B_ARGS=${S3B_ARGS:-}       # e.g. "--max-disp 4 --face-weight 0.2"
S5_ARGS=${S5_ARGS:-}         # e.g. "--no-expression"

export PYTHONUNBUFFERED=1
mkdir -p "$OUT/logs"

blender_run() {  # blender_run <script> [args...]
  local script=$1; shift
  local prefix=()
  if [ -z "${DISPLAY:-}" ] && command -v xvfb-run >/dev/null 2>&1; then
    prefix=(xvfb-run -a)
  fi
  "${prefix[@]}" "$BLENDER" --background --factory-startup \
      --python "$script" -- "$@"
}

want() { [[ " $STAGES " == *" $1 "* ]]; }

log_run() {  # log_run <tag> <cmd...>
  local tag=$1; shift
  echo ""
  echo "=== [$tag] $(date +%H:%M:%S) :: $* ==="
  "$@" 2>&1 | tee "$OUT/logs/$tag.log"
}

# Which neutral feeds the rig: refined (clay-fused) or fitted (pure ICT).
if [ "$REFINE" = "1" ]; then
  NEUTRAL="$OUT/refine/refined_neutral.npy"
else
  NEUTRAL="$OUT/fit/fitted_neutral.npy"
fi

if want 1; then
  log_run s1 "$PY" "$PIPE/s1_landmarks.py" \
      --photo "$PHOTO" --task "$MP_TASK" --out "$OUT"
fi

if want 2; then
  # shellcheck disable=SC2086
  log_run s2 "$PY" "$PIPE/s2_fit_identity.py" \
      --ict "$ICT" --out "$OUT" $S2_ARGS
fi

if want 3; then
  if [ "$REFINE" = "1" ]; then
    # shellcheck disable=SC2086
    log_run s3a "$PY" "$PIPE/s3a_align_clay.py" \
        --clay "$CLAY" --task "$MP_TASK" --out "$OUT" $S3A_ARGS
    # shellcheck disable=SC2086
    log_run s3b blender_run "$PIPE/s3b_refine_blender.py" --out "$OUT" $S3B_ARGS
  else
    echo "=== [s3] REFINE=0 -- skipping clay align + shrinkwrap ==="
  fi
fi

if want 4; then
  log_run s4 "$PY" "$PIPE/s4_build_shapes.py" \
      --ict "$ICT" --out "$OUT" --neutral "$NEUTRAL"
fi

if want 5; then
  # shellcheck disable=SC2086
  log_run s5 "$PY" "$PIPE/s5_bake_texture.py" \
      --out "$OUT" --size "$TEX_SIZE" $S5_ARGS
fi

if want 6; then
  DR=()
  [ "$DRACO" = "1" ] && DR=(--draco)
  log_run s6 blender_run "$PIPE/s6_export_blender.py" --out "$OUT" "${DR[@]}"
fi

if want 7; then
  log_run s7 "$PY" "$PIPE/s7_verify_glb.py" --out "$OUT"
fi

echo ""
echo "=== newstack done. GLB: $OUT/export/head_arkit_v2.glb ==="
echo "=== manifest: $OUT/rig/arkit_manifest.json  verify: $OUT/export/verify_report.json ==="
