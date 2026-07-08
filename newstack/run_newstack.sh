#!/usr/bin/env bash
# =============================================================================
# newstack fusion pipeline — SIMPLIFIED: TripoSR-only pipeline
#
# Single photo -> commercial GLB (ICT Light, MIT)
# With hair volume (TripoSR clay), TripoSR vertex colors, ARKit-52 morph targets
# (51 from ICT expression OBJs + tongueOut synthesized from ICT's real
# static tongue geometry -- see pipe/tongue_synth.py).
#
# Removed: TripoSG, TRELLIS, photo-projection texture, multi-source blending
# Color source: TripoSR vertex colors only (simple k-NN bake to ICT UVs)
#
# Pod layout (defaults; override via env):
#   pipe scripts   /workspace/newstack/pipe        (or /workspace/newARC/newstack/pipe)
#   ICT-FaceKit    /workspace/newstack/ICT-FaceKit (FaceXModel only, MIT)
#   TripoSR clay   /workspace/newstack/out_triposr/0/mesh.obj
#   photo          /workspace/inputs/random-person.jpeg
#   outputs        /workspace/newstack/out
#
# Usage:
#   bash /workspace/newstack/run_newstack.sh             # all stages
#   STAGES="4 5 6 7" bash run_newstack.sh                # rerun from rig
#   TEX_SIZE=2048 DRACO=1 bash run_newstack.sh
#
# Stages: 1 landmarks | 2 identity fit | 3 clay align + shrinkwrap refine
#         4 ARKit shapes | 5 texture bake (TripoSR only) | 6 Blender GLB export
#         7 verify | 8 render proof images from the GLB
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

TEX_SIZE=${TEX_SIZE:-1024}
DRACO=${DRACO:-0}
S2_ARGS=${S2_ARGS:-}
S3A_ARGS=${S3A_ARGS:-}
S3B_ARGS=${S3B_ARGS:-}
S3C_ARGS=${S3C_ARGS:-}
S5_ARGS=${S5_ARGS:-}

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
NEUTRAL="$OUT/refine/refined_neutral.npy"
if [ ! -f "$NEUTRAL" ]; then
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
  # shellcheck disable=SC2086
  log_run s3a "$PY" "$PIPE/s3a_align_clay.py" \
      --clay "$CLAY" --task "$MP_TASK" --out "$OUT" $S3A_ARGS
  # shellcheck disable=SC2086
  log_run s3b blender_run "$PIPE/s3b_refine_blender.py" --out "$OUT" $S3B_ARGS
  # shellcheck disable=SC2086
  log_run s3c "$PY" "$PIPE/s3c_verify_refine.py" --out "$OUT" $S3C_ARGS
fi

if want 4; then
  log_run s4 "$PY" "$PIPE/s4_build_shapes.py" \
      --ict "$ICT" --out "$OUT" --neutral "$NEUTRAL"
fi

if want 5; then
  # SIMPLIFIED: TripoSR-only texture bake. No photo projection, no TRELLIS,
  # no multi-source blending. TripoSR vertex colors sampled at each texel
  # 3D position via k-NN, with interior defaults + eye textures.
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

if want 8; then
  log_run s8 blender_run "$PIPE/s8_render_previews.py" --out "$OUT"
fi

echo ""
echo "=== newstack done. GLB: $OUT/export/head_arkit_v2.glb ==="
echo "=== manifest: $OUT/rig/arkit_manifest.json  verify: $OUT/export/verify_report.json ==="
