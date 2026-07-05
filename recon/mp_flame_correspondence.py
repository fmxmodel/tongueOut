"""MediaPipe-478 <-> FLAME landmark correspondence (SELF-AUTHORED).

WHY THIS FILE EXISTS
--------------------
FLAME 2023 Open ships a landmark embedding that anchors the iBUG/dlib-68
landmark set onto the FLAME surface as (face_index, barycentric) points
(models/README.md section 1). MediaPipe FaceLandmarker emits 478 landmarks in
its own indexing. The bridge between the two is NOT part of either download
(models/README.md section 3) -- it must be authored with clean provenance for
this COMMERCIAL run. This file IS that bridge.

PROVENANCE (flagged to `license-compliance`)
--------------------------------------------
The index list below is SELF-AUTHORED for this project by reading off vertex
indices from MediaPipe's published canonical face mesh topology
(mediapipe/modules/face_geometry/data/canonical_face_model.obj and the
uv-visualization card, Apache-2.0) against the published iBUG-68 point
definitions (semantic point locations; the iBUG *annotation scheme* is a
factual point layout). No third-party correspondence file was copied.
It MUST be visually verified on the pod: recon.landmarks writes
out/recon/landmarks_debug.png with all 68 picks drawn and numbered on the
input photo; a human (or qa-verifier) confirms each pick sits on the right
facial feature before trusting the fit.

CONVENTIONS
-----------
- "Left"/"Right" below are SUBJECT-anatomical (subject's right eye appears on
  the LEFT side of an unmirrored photo).
- iBUG-68 indices are 0-based: 0-16 jaw contour, 17-21 right brow,
  22-26 left brow, 27-30 nose bridge (30 = tip), 31-35 nose base,
  36-41 right eye, 42-47 left eye, 48-59 outer lips, 60-67 inner lips.
- FLAME's *static* embedding covers iBUG 17..67 (51 points, no jaw contour);
  the *full/dynamic* embeddings add the 17 contour points. The fit uses the
  static 51 as its primary data term and the contour (if the downloaded
  embedding provides it) at low weight -- MediaPipe's face-oval points are the
  visible-silhouette boundary, which only approximates the iBUG jawline.
"""

import numpy as np

# --------------------------------------------------------------------------
# iBUG-68 -> MediaPipe-478 index table (self-authored; verify via overlay)
# --------------------------------------------------------------------------
MEDIAPIPE_IBUG68 = np.array(
    [
        # --- jaw / face contour, subject-RIGHT ear down to chin to subject-LEFT (0-16)
        234,  # 0  contour start, subject-right, temple/ear level
        93,   # 1
        132,  # 2
        58,   # 3
        172,  # 4
        136,  # 5
        150,  # 6
        176,  # 7
        152,  # 8  chin center (menton)
        400,  # 9
        379,  # 10
        365,  # 11
        397,  # 12
        288,  # 13
        361,  # 14
        323,  # 15
        454,  # 16 contour end, subject-left, temple/ear level
        # --- subject-RIGHT eyebrow, outer -> inner (17-21)
        70,   # 17 outer tail
        63,   # 18
        105,  # 19 mid
        66,   # 20
        107,  # 21 inner head
        # --- subject-LEFT eyebrow, inner -> outer (22-26)
        336,  # 22 inner head
        296,  # 23
        334,  # 24 mid
        293,  # 25
        300,  # 26 outer tail
        # --- nose bridge, nasion -> tip (27-30)
        168,  # 27 nasion (between eyes)
        6,    # 28
        197,  # 29
        4,    # 30 nose tip (pronasale)
        # --- nose base row, subject-right alar -> subject-left alar (31-35)
        98,   # 31 subject-right nostril wing
        97,   # 32
        2,    # 33 subnasale (center under the nose)
        326,  # 34
        327,  # 35 subject-left nostril wing
        # --- subject-RIGHT eye, outer corner CCW (36-41)
        33,   # 36 outer canthus
        160,  # 37 upper lid, outer third
        158,  # 38 upper lid, inner third
        133,  # 39 inner canthus
        153,  # 40 lower lid, inner third
        144,  # 41 lower lid, outer third
        # --- subject-LEFT eye, inner corner CW (42-47)
        362,  # 42 inner canthus
        385,  # 43 upper lid, inner third
        387,  # 44 upper lid, outer third
        263,  # 45 outer canthus
        373,  # 46 lower lid, outer third
        380,  # 47 lower lid, inner third
        # --- OUTER lips, subject-right corner across the top, then bottom (48-59)
        61,   # 48 subject-right mouth corner
        40,   # 49 upper lip, right quarter
        37,   # 50 upper lip, right of philtrum
        0,    # 51 upper lip center (cupid's bow)
        267,  # 52 upper lip, left of philtrum
        270,  # 53 upper lip, left quarter
        291,  # 54 subject-left mouth corner
        321,  # 55 lower lip, left quarter
        314,  # 56 lower lip, left of center
        17,   # 57 lower lip center
        84,   # 58 lower lip, right of center
        91,   # 59 lower lip, right quarter
        # --- INNER lips (60-67)
        78,   # 60 subject-right inner corner
        81,   # 61 upper inner, right
        13,   # 62 upper inner center
        311,  # 63 upper inner, left
        308,  # 64 subject-left inner corner
        402,  # 65 lower inner, left
        14,   # 66 lower inner center
        178,  # 67 lower inner, right
    ],
    dtype=np.int64,
)
assert MEDIAPIPE_IBUG68.shape == (68,), "iBUG-68 table must have exactly 68 entries"
assert len(set(MEDIAPIPE_IBUG68.tolist())) == 68, "iBUG-68 table has duplicate MediaPipe indices"

# iBUG index groups (0-based), used for per-group loss weights + reporting.
IBUG_GROUPS = {
    "contour": list(range(0, 17)),
    "brow_right": list(range(17, 22)),
    "brow_left": list(range(22, 27)),
    "nose_bridge": list(range(27, 31)),
    "nose_base": list(range(31, 36)),
    "eye_right": list(range(36, 42)),
    "eye_left": list(range(42, 48)),
    "lips_outer": list(range(48, 60)),
    "lips_inner": list(range(60, 68)),
}

# Per-group data-term weights (contour is soft: silhouette != iBUG jawline).
IBUG_GROUP_WEIGHTS = {
    "contour": 0.3,
    "brow_right": 0.8,
    "brow_left": 0.8,
    "nose_bridge": 1.5,
    "nose_base": 1.5,
    "eye_right": 2.0,
    "eye_left": 2.0,
    "lips_outer": 1.5,
    "lips_inner": 1.0,
}

# Expression-insensitive subset for the rigid/camera stage (eye corners,
# nose, mouth corners). All within the static-51 range (>= 17).
STABLE_CORE_IBUG = [36, 39, 42, 45, 27, 28, 29, 30, 33, 48, 54]

# FLAME's static embedding covers iBUG 17..67 in order.
STATIC51_IBUG = list(range(17, 68))


def per_landmark_weights() -> np.ndarray:
    """(68,) float32 weight vector from IBUG_GROUP_WEIGHTS."""
    w = np.zeros(68, dtype=np.float32)
    for group, idxs in IBUG_GROUPS.items():
        w[idxs] = IBUG_GROUP_WEIGHTS[group]
    assert (w > 0).all(), "every iBUG landmark must belong to a weighted group"
    return w
