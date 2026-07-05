"""Configuration for the Track B1 ARKit rigging stage (arkit-rigger).

INPUTS (authored by face-reconstructor; paths come from recon.config so the
interface can never drift):
  out/recon/neutral.ply            base mesh (fitted identity, expr=0, pose=0)
  out/recon/faces.npy              THE topology contract, int32 (F,3)
  out/recon/id_params.npz          fitted betas + photo-state + camera
  out/recon/expression_basis.npz   expr_dirs/posedirs/j_regressor/lbs_weights/
                                   parents/faces/v_neutral/joints_neutral
  out/recon/landmarks.npz          MediaPipe photo landmarks (laterality check)
  out/recon/lmk_embedding_static51.npz  iBUG static-51 surface anchors -- the
                                   ones the FIT actually used. The FLAME 2023
                                   Open release is pkl-only (no embedding
                                   file); recon self-authors the anchors
                                   clean-room (recon/flame_landmarks.py) and
                                   persists them here for the rig.

OUTPUTS (out/shapes/):
  neutral.ply                      byte-identical pass-through of the base
  expr_<arkitName>.ply             one per SUPPORTED shape, same topology
  arkit_manifest.json              all 52 names, measured supported/unsupported
  shape_params.npz                 solved coefficients / poses per shape
  shapes_run_manifest.json         verify stage's measured verdict
  rig_report.md                    method + coverage (authored, then pod-audited)

All numeric values below are STARTING POINTS chosen from anatomy-scale
reasoning (FLAME is in meters); tune ON THE POD against rendered previews.
Nothing here was validated locally (no-local-compute contamination guard).
Everything is env-overridable via B1_RIG_* so pod tuning needs no code edits.
"""

import os
from pathlib import Path

from recon import config as RC  # single source of truth for the input paths

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
SHAPES_DIR = Path(os.environ.get("B1_RIG_OUT_DIR", str(_REPO_ROOT / "out" / "shapes")))

NEUTRAL_OUT_PLY = SHAPES_DIR / "neutral.ply"          # pass-through copy
ARKIT_MANIFEST_JSON = SHAPES_DIR / "arkit_manifest.json"
SHAPE_PARAMS_NPZ = SHAPES_DIR / "shape_params.npz"
SHAPES_RUN_MANIFEST_JSON = SHAPES_DIR / "shapes_run_manifest.json"
RIG_REPORT_MD = SHAPES_DIR / "rig_report.md"

# The ARKit name contract (canonical Apple-52 spelling, case-sensitive).
NAME_CONTRACT_JSON = Path(
    os.environ.get("B1_RIG_NAME_CONTRACT", str(_REPO_ROOT / "out" / "arkit_51_52_map.json"))
)

# Inputs (aliased from recon.config -- do NOT redefine, only reference)
RECON_NEUTRAL_PLY = RC.NEUTRAL_PLY
RECON_FACES_NPY = RC.FACES_NPY
RECON_ID_PARAMS_NPZ = RC.ID_PARAMS_NPZ
RECON_EXPR_BASIS_NPZ = RC.EXPR_BASIS_NPZ
RECON_LANDMARKS_NPZ = RC.LANDMARKS_NPZ

DEVICE = os.environ.get("B1_RIG_DEVICE", RC.DEVICE)


def _f(env: str, default: float) -> float:
    return float(os.environ.get(env, str(default)))


# --------------------------------------------------------------------------
# Pose-based shape activations (axis-angle magnitudes, radians / meters).
# SIGNS AND AXES ARE NEVER ASSUMED: build_arkit_shapes.py calibrates them by
# measurement (decode, look at which way the geometry moved) before naming
# anything Left/Right/Open. These are magnitudes only.
# --------------------------------------------------------------------------
JAW_OPEN_RAD = _f("B1_RIG_JAW_OPEN_RAD", 0.36)        # ~21 deg jaw drop
JAW_LAT_RAD = _f("B1_RIG_JAW_LAT_RAD", 0.12)          # ~7 deg lateral swing
JAW_FWD_M = _f("B1_RIG_JAW_FWD_M", 0.006)             # 6 mm protrusion (LBS translate)
EYE_PITCH_UP_RAD = _f("B1_RIG_EYE_UP_RAD", 0.35)      # ~20 deg gaze up
EYE_PITCH_DOWN_RAD = _f("B1_RIG_EYE_DOWN_RAD", 0.45)  # ~26 deg gaze down
EYE_YAW_RAD = _f("B1_RIG_EYE_YAW_RAD", 0.45)          # ~26 deg gaze in/out

# --------------------------------------------------------------------------
# PCA ridge solve (expression shapes)
# --------------------------------------------------------------------------
RIDGE_LAMBDA_REL = _f("B1_RIG_LAMBDA_REL", 1e-2)  # x mean(diag(A^T W^2 A))
E_NORM_CAP = _f("B1_RIG_E_CAP", 4.0)              # ||e||_2 cap (PCA-coeff units)
LAMBDA_ESCALATION = 10.0                          # lambda *= this until under cap
LAMBDA_MAX_TRIES = 8
STABILIZER_WEIGHT = _f("B1_RIG_STAB_W", 0.25)     # vs target weight 1.0
SECANT_EPS = _f("B1_RIG_SECANT_EPS", 1.0)         # finite-diff step, PCA units
                                                  # (exact at rest: pre-LBS linear)

# --------------------------------------------------------------------------
# Measured acceptance gates (fail => shape demoted to unsupported, with the
# measured numbers recorded as the reason -- coverage is never fabricated)
# --------------------------------------------------------------------------
GATE_MIN_MAX_DELTA_M = _f("B1_RIG_GATE_MIN_DELTA", 5e-4)  # non-trivial delta
GATE_TARGET_COS = _f("B1_RIG_GATE_COS", 0.5)      # achieved vs requested direction
GATE_AMP_MIN = _f("B1_RIG_GATE_AMP_MIN", 0.3)     # achieved/requested magnitude
GATE_AMP_MAX = _f("B1_RIG_GATE_AMP_MAX", 3.0)
GATE_LEAK_RATIO = _f("B1_RIG_GATE_LEAK", 0.6)     # mirror-side / target-side RMS
GATE_STAB_RATIO = _f("B1_RIG_GATE_STAB", 1.0)     # stabilizer / target RMS
GATE_JAW_DROP_MIN_M = _f("B1_RIG_GATE_JAW_DROP", 3e-3)
GATE_JAW_LAT_MIN_M = _f("B1_RIG_GATE_JAW_LAT", 1.5e-3)
GATE_JAW_FWD_MIN_M = _f("B1_RIG_GATE_JAW_FWD", 2e-3)
GATE_EYE_MIN_VERTS = int(os.environ.get("B1_RIG_GATE_EYE_VERTS", "8"))
GATE_EYE_MOVE_MIN_M = _f("B1_RIG_GATE_EYE_MOVE", 1e-3)

# Neutral reproduction: decode(0,0) must match neutral.ply to this tolerance
# (PLY stores float32; FLAME head spans ~0.25 m, so 1e-4 m is generous).
NEUTRAL_REPRO_TOL_M = _f("B1_RIG_NEUTRAL_TOL", 1e-4)


def ensure_out_dirs() -> None:
    SHAPES_DIR.mkdir(parents=True, exist_ok=True)
