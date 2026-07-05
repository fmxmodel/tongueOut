"""Central configuration for the Track B1 reconstruction pipeline.

All pod paths match scripts/pod_setup_b1.sh / models/README.md exactly:
  /workspace/venvs/b1                      venv (activated by run_recon_b1.sh)
  /workspace/models/flame2023_open/        FLAME 2023 Open (CC-BY-4.0), manual download
  /workspace/models/mediapipe/face_landmarker.task
  /workspace/inputs/random-person.jpeg     the single input photo

Everything is overridable via B1_* environment variables so a differently
laid-out pod does not require code edits.

Numeric hyperparameters are STARTING POINTS chosen from standard practice for
landmark-based FLAME fits; they are expected to be tuned ON THE POD against
the debug overlays (out/recon/fit_debug/). None of them has been validated
locally -- no compute ran on the authoring box.
"""

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths (pod layout; env-overridable)
# --------------------------------------------------------------------------
WORKSPACE = Path(os.environ.get("B1_WORKSPACE", "/workspace"))

FLAME_DIR = Path(os.environ.get("B1_FLAME_DIR", str(WORKSPACE / "models" / "flame2023_open")))
MP_TASK_PATH = Path(
    os.environ.get("B1_MP_TASK", str(WORKSPACE / "models" / "mediapipe" / "face_landmarker.task"))
)
INPUT_IMAGE = Path(os.environ.get("B1_INPUT", str(WORKSPACE / "inputs" / "random-person.jpeg")))

# Repo-relative output dir (the artifact bus): <repo>/out/recon
_REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(os.environ.get("B1_OUT_DIR", str(_REPO_ROOT / "out" / "recon")))
FIT_DEBUG_DIR = OUT_DIR / "fit_debug"

# Candidate filenames inside FLAME_DIR, matched BY ROLE (models/README.md:
# "names vary slightly by release"). First existing file wins.
#
# MEASURED ON THE POD (2026-07-05): the FLAME 2023 Open (CC-BY-4.0) release is
# PKL-ONLY -- it ships flame2023.pkl + a readme and NOTHING else. There is NO
# UV template and NO landmark embedding in the Open package (those live in
# NC-licensed packages/repos, which are BARRED from this commercial run).
# Therefore only the MODEL pkl is required; the TEMPLATE and EMBEDDING lists
# below are OPTIONAL OVERRIDES (resolved via find_optional_flame_file). When
# absent -- the expected case -- the pipeline substitutes clean-room,
# self-authored equivalents:
#   landmark embedding -> recon/flame_landmarks.py (geometric derivation from
#                         v_template/weights/joints; persisted to
#                         out/recon/lmk_embedding_static51.npz)
#   UV layout          -> recon/uv_unwrap.py (deterministic xatlas [MIT] unwrap
#                         of v_template+faces; persisted to out/recon/uv_coords.npz)
FLAME_MODEL_CANDIDATES = [
    "flame2023.pkl",
    "FLAME2023.pkl",
    "flame2023_no_jaw.pkl",  # some releases ship a no-jaw variant; jaw variant preferred above
    "generic_model.pkl",
]
FLAME_TEMPLATE_CANDIDATES = [  # OPTIONAL override -- not in the Open release
    "head_template.obj",
    "flame_template.obj",
    "head_template_mesh.obj",
]
FLAME_LMK_EMBEDDING_CANDIDATES = [  # OPTIONAL override -- not in the Open release
    "landmark_embedding_with_eyes.npy",
    "landmark_embedding.npy",
    "flame_static_embedding.pkl",
]

# --------------------------------------------------------------------------
# Canonical artifact names under OUT_DIR (the handoff contract)
# --------------------------------------------------------------------------
CANONICAL_IMAGE = OUT_DIR / "input_image.png"       # EXIF-normalized decode, single source of pixels
LANDMARKS_NPZ = OUT_DIR / "landmarks.npz"
LANDMARKS_DEBUG_PNG = OUT_DIR / "landmarks_debug.png"
BLENDSHAPES_JSON = OUT_DIR / "mediapipe_blendshapes_photo.json"  # photo-state reference only
NEUTRAL_PLY = OUT_DIR / "neutral.ply"
FACES_NPY = OUT_DIR / "faces.npy"
ID_PARAMS_NPZ = OUT_DIR / "id_params.npz"
EXPR_BASIS_NPZ = OUT_DIR / "expression_basis.npz"
EXPR_BASIS_NOTES = OUT_DIR / "expression_basis_notes.json"
FIT_SUMMARY_JSON = OUT_DIR / "fit_summary.json"
ALBEDO_PNG = OUT_DIR / "albedo.png"
ALBEDO_MASK_PNG = OUT_DIR / "albedo_mask.png"
UV_COORDS_NPZ = OUT_DIR / "uv_coords.npz"
# The landmark anchors the fit ACTUALLY used (release file or self-authored) --
# persisted so rig/build_arkit_shapes.py consumes the identical anchors.
LMK_EMBEDDING_NPZ = OUT_DIR / "lmk_embedding_static51.npz"
FLAME_LMK_SELF_JSON = OUT_DIR / "flame_landmarks_selfauthored.json"
FLAME_LMK_SELF_PNG = OUT_DIR / "flame_landmarks_selfauthored.png"
# On any derivation failure: measured diagnostics (region counts, y-bands,
# jaw-weight histogram) land here so the failure is actionable, not just FATAL.
FLAME_LMK_FAIL_JSON = OUT_DIR / "flame_landmarks_failure.json"
BAKE_SUMMARY_JSON = OUT_DIR / "bake_summary.json"
RUN_MANIFEST_JSON = OUT_DIR / "recon_run_manifest.json"

# --------------------------------------------------------------------------
# FLAME dimensions (asserted at load time against the actual model file)
# --------------------------------------------------------------------------
# FLAME 2020/2023 convention: shapedirs has 300 identity + 100 expression
# components. flame_model.py asserts shapedirs.shape[2] == N_SHAPE + N_EXPR
# and STOPS loudly if the downloaded release differs (never silently adapts).
N_SHAPE = int(os.environ.get("B1_N_SHAPE", "300"))
N_EXPR = int(os.environ.get("B1_N_EXPR", "100"))

# How many components the optimizer actually uses (<= the model's capacity).
N_SHAPE_FIT = int(os.environ.get("B1_N_SHAPE_FIT", "300"))
N_EXPR_FIT = int(os.environ.get("B1_N_EXPR_FIT", "100"))

# Expected FLAME topology, per FLAME's published documentation. These are
# EXPECTATIONS to verify on the pod, not measurements -- nothing was run
# locally. fit_flame.py measures the real counts and verify_outputs.py
# records them in recon_run_manifest.json. A mismatch is a STOP, not a warn.
EXPECTED_N_VERTS = 5023
EXPECTED_N_FACES = 9976

# --------------------------------------------------------------------------
# Fit hyperparameters (tune on pod; see module docstring)
# --------------------------------------------------------------------------
SEED = 0
DEVICE = os.environ.get("B1_DEVICE", "cuda")

STAGE_A_ITERS = int(os.environ.get("B1_STAGE_A_ITERS", "400"))   # rigid + camera
STAGE_B_ITERS = int(os.environ.get("B1_STAGE_B_ITERS", "500"))   # + shape
STAGE_C_ITERS = int(os.environ.get("B1_STAGE_C_ITERS", "600"))   # + expression/jaw
STAGE_A_LR = 1e-2
STAGE_B_LR = 5e-3
STAGE_C_LR = 5e-3

# Regularizer weights (loss operates on residuals normalized by image diagonal
# and scaled x100, so these are relative to O(1) data terms).
SHAPE_REG_W = float(os.environ.get("B1_SHAPE_REG", "1e-3"))
EXPR_REG_W = float(os.environ.get("B1_EXPR_REG", "1e-3"))
JAW_REG_W = float(os.environ.get("B1_JAW_REG", "1e-2"))
FOCAL_REG_W = float(os.environ.get("B1_FOCAL_REG", "1e-2"))

# Single-image focal/depth ambiguity: focal is optimized but tethered to init.
FOCAL_INIT_FACTOR = 1.5          # f_init = 1.5 * max(H, W) px, a portrait-lens prior
INTEROCULAR_3D_M = 0.09          # approx outer-canthus distance (m) used ONLY for z init

# --------------------------------------------------------------------------
# Texture bake
# --------------------------------------------------------------------------
TEX_RES = int(os.environ.get("B1_TEX_RES", "1024"))
DEPTH_TOL_M = 5e-3               # visibility: |texel depth - zbuffer| tolerance (m)
COS_MIN = 0.15                   # reject grazing-angle texels (quality gate)
MIRROR_MATCH_TOL_M = 4e-3        # max template-space distance for a mirror-texel match
INPAINT_RADIUS = 5               # cv2.inpaint radius (classical TELEA; no learned prior)

# Mask value legend for albedo_mask.png (QA reads this):
MASK_DIRECT = 255                # texel sampled directly from the photo
MASK_MIRROR = 170                # filled by bilateral mirror symmetry
MASK_INPAINT = 85                # filled by cv2.inpaint (classical)
MASK_OUTSIDE = 0                 # outside every UV island (filled with mean valid color)


def ensure_out_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIT_DEBUG_DIR.mkdir(parents=True, exist_ok=True)


def find_flame_file(candidates, role: str) -> Path:
    """Resolve a FLAME asset by role from the candidate name list; die loudly."""
    for name in candidates:
        p = FLAME_DIR / name
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"[config FATAL] no {role} found in {FLAME_DIR} (tried: {candidates}). "
        "FLAME 2023 Open is a MANUAL, license-gated download -- see models/README.md "
        "section 1 and scripts/pod_setup_b1.sh step 6."
    )


def find_optional_flame_file(candidates):
    """Like find_flame_file, but returns None when nothing matches.

    Used for the assets the FLAME 2023 Open release does NOT ship (measured on
    the pod: the release is pkl-only): the UV template obj and the landmark
    embedding. Absence is the expected, fully supported case -- the pipeline
    then self-authors/generates clean-room equivalents (recon/flame_landmarks.py
    and recon/uv_unwrap.py). A staged file acts as an optional override and,
    for the template, as an extra topology cross-check.
    """
    for name in candidates:
        p = FLAME_DIR / name
        if p.is_file():
            return p
    return None
