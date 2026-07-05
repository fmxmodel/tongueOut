"""FLAME -> ARKit-52 correspondence specification (SELF-AUTHORED, license-clean).

WHY THIS FILE EXISTS
--------------------
FLAME's 100 expression components are anonymous PCA axes from 4D scans; none
is ARKit-named. Apple's 52 blendshapes are defined only as short prose
descriptions of facial movements (developer.apple.com, ARFaceAnchor.
BlendShapeLocation). This file expresses each ARKit shape as either:

  (a) a FLAME POSE recipe   -- jaw joint / eye joints, magnitudes from
      rig.config, axes+signs CALIBRATED BY MEASUREMENT on the pod (never
      assumed), or
  (b) a sparse GEOMETRIC TARGET on the iBUG-68 landmark set that FLAME 2023
      Open itself anchors on the mesh (the CC-BY landmark embedding), which
      build_arkit_shapes.py turns into a ridge-regularized least-squares
      solve over the 100 expression axes, or
  (c) an honest UNSUPPORTED declaration (reasoned, never faked).

PROVENANCE (flagged to license-compliance)
------------------------------------------
Targets are authored from Apple's PUBLIC textual descriptions of each
blendshape (facts about facial anatomy, not copyrightable expression) plus
the iBUG-68 point semantics already used by the reconstruction. NO
DECA/EMOCA/ARKit-blendshape mesh assets, NO non-commercial FLAME->ARKit
coefficient tables, NO MetaHuman reference deltas were consulted or copied.
The only model data touched at runtime is the FLAME 2023 Open (CC-BY-4.0)
download + the reconstructor's out/recon/ artifacts.

HONESTY / APPROXIMATION NOTES
-----------------------------
- These are APPROXIMATIONS: quality is bounded by (i) how much of the shape
  FLAME's scan-based PCA actually spans, (ii) the sparsity of 51 landmark
  handles. Every solve passes MEASURED gates (rig.config) on the pod;
  failures are demoted to unsupported with the numbers in the manifest.
- "intended" below is a PREDICTION ("strong"/"weak"), not a result. The
  manifest's supported flag is decided by pod measurement only.
- Laterality: iBUG semantics say 17-21/36-41/48/31 are the subject's RIGHT
  side and 22-26/42-47/54/35 the subject's LEFT (recon/mp_flame_
  correspondence.py). build_arkit_shapes.py RE-MEASURES this on the neutral
  mesh (FLAME/SMPL +X = subject-left) and cross-checks against the photo
  via the fitted camera before trusting any Left/Right name.

This module is PURE PYTHON (stdlib only) so the local manifest author can
import it without touching numpy/torch (contamination guard).
"""

# --------------------------------------------------------------------------
# iBUG-68 landmark handles (indices into the iBUG-68 scheme; the FLAME static
# embedding covers 17..67, row = ibug_index - 17). Semantics per
# recon/mp_flame_correspondence.py: suffix _R = subject's RIGHT, _L = LEFT.
# --------------------------------------------------------------------------
HANDLES = {
    "brow_R": [17, 18, 19, 20, 21],
    "brow_L": [22, 23, 24, 25, 26],
    "brow_inner": [20, 21, 22, 23],          # inner halves of both brows
    "brow_outer_R": [17, 18, 19],
    "brow_outer_L": [24, 25, 26],
    "eye_R_upper": [37, 38],                  # upper eyelid
    "eye_R_lower": [40, 41],                  # lower eyelid
    "eye_L_upper": [43, 44],
    "eye_L_lower": [46, 47],
    "eye_R_all": [36, 37, 38, 39, 40, 41],
    "eye_L_all": [42, 43, 44, 45, 46, 47],
    "nose_bridge": [27, 28, 29, 30],
    "nostril_R": [31],
    "nostril_L": [35],
    "mouth_corner_R": [48, 60],               # outer + inner corner
    "mouth_corner_L": [54, 64],
    "lip_upper_outer": [49, 50, 51, 52, 53],
    "lip_upper_outer_R": [49, 50],
    "lip_upper_outer_L": [52, 53],
    "lip_lower_outer": [55, 56, 57, 58, 59],
    "lip_lower_outer_R": [58, 59],
    "lip_lower_outer_L": [55, 56],
    "lip_upper_inner": [61, 62, 63],
    "lip_lower_inner": [65, 66, 67],
    "lip_upper_inner_R": [61],
    "lip_upper_inner_L": [63],
    "lip_lower_inner_R": [67],
    "lip_lower_inner_L": [65],
    "lips_all": list(range(48, 68)),
    "lips_outer_all": list(range(48, 60)),
}

# Upper<->lower eyelid counterparts (for "gap" targets: blink/squint/wide).
LID_COUNTERPART = {37: 41, 38: 40, 43: 47, 44: 46,
                   41: 37, 40: 38, 47: 43, 46: 44}

# Inner-lip upper<->lower counterparts (for the mouthClose seal solve).
INNER_LIP_PAIRS = [(61, 67), (62, 66), (63, 65)]

# --------------------------------------------------------------------------
# Direction tokens, resolved on the pod against the NEUTRAL geometry
# (FLAME canonical frame: +X = subject-left, +Y = up, +Z = forward/out of
# the face; the +X axiom is re-verified by measurement + photo cross-check):
#   up/down/fwd/back/left/right : canonical axes
#   to_center / from_center     : horizontal unit vector toward/away from the
#                                 mouth center (midpoint of iBUG 48 & 54) --
#                                 side-agnostic lateral in/out
#   gap                         : the vector from this eyelid point to its
#                                 LID_COUNTERPART (unnormalized; factor
#                                 scales it; negative factor = open wider)
# Units (measured on the neutral mesh, meters):
#   MW  = mouth width |lmk54-lmk48|         BH_L/BH_R = brow-to-eye height
#   GAP = the counterpart vector itself      ABS = factor is already meters
# --------------------------------------------------------------------------

def _t(handle, direction, factor, unit):
    return {"handle": handle, "dir": direction, "factor": factor, "unit": unit}


def _mirror_handle(h):
    if h.endswith("_L"):
        return h[:-2] + "_R"
    if h.endswith("_R"):
        return h[:-2] + "_L"
    if "_L_" in h:
        return h.replace("_L_", "_R_")
    if "_R_" in h:
        return h.replace("_R_", "_L_")
    return h


# --- one-sided PCA templates: "{S}" is instantiated as L and R ------------
_SIDED_PCA = {
    "browDown{SIDE}": {
        "intended": "strong",
        "targets": [_t("brow_{S}", "down", 0.30, "BH_{S}")],
        "free": ["eye_{S}_upper"],   # brow drop drags the upper lid; allow it
        "notes": "Brow depressor; strongly present in FLAME expression scans.",
    },
    "browOuterUp{SIDE}": {
        "intended": "strong",
        "targets": [_t("brow_outer_{S}", "up", 0.30, "BH_{S}")],
        "free": ["eye_{S}_upper"],
        "notes": "Outer brow raise.",
    },
    "eyeBlink{SIDE}": {
        "intended": "strong",
        "targets": [_t("eye_{S}_upper", "gap", 0.85, "GAP"),
                    _t("eye_{S}_lower", "gap", 0.10, "GAP")],
        "free": [],
        "notes": "Upper lid closes 85% of the measured aperture, lower rises "
                 "10%. Lid closure exists in FLAME 2023's expression space; "
                 "gates verify per-eye isolation.",
    },
    "eyeSquint{SIDE}": {
        "intended": "weak",
        "targets": [_t("eye_{S}_lower", "gap", 0.35, "GAP"),
                    _t("eye_{S}_upper", "gap", 0.10, "GAP")],
        "free": [],
        "notes": "Lower-lid raise; flagged weak in FLAME PCA by the "
                 "reconstructor -- gates adjudicate.",
    },
    "eyeWide{SIDE}": {
        "intended": "weak",
        "targets": [_t("eye_{S}_upper", "gap", -0.45, "GAP")],
        "free": [],
        "notes": "Upper lid opens beyond neutral (negative gap factor); "
                 "flagged weak -- gates adjudicate.",
    },
    "mouthSmile{SIDE}": {
        "intended": "strong",
        "targets": [_t("mouth_corner_{S}", "up", 0.12, "MW"),
                    _t("mouth_corner_{S}", "from_center", 0.08, "MW"),
                    _t("lip_upper_outer_{S}", "up", 0.05, "MW"),
                    _t("lip_lower_outer_{S}", "up", 0.04, "MW")],
        "free": [],
        "notes": "Corner up + lateral; smiles dominate FLAME's scan corpus.",
    },
    "mouthFrown{SIDE}": {
        "intended": "strong",
        "targets": [_t("mouth_corner_{S}", "down", 0.10, "MW"),
                    _t("mouth_corner_{S}", "from_center", 0.03, "MW"),
                    _t("lip_lower_outer_{S}", "down", 0.04, "MW")],
        "free": [],
        "notes": "Corner depressor.",
    },
    "mouthDimple{SIDE}": {
        "intended": "strong",
        "targets": [_t("mouth_corner_{S}", "back", 0.05, "MW"),
                    _t("mouth_corner_{S}", "from_center", 0.04, "MW")],
        "free": [],
        "notes": "Corner retracts toward the cheek (buccinator); subtle.",
    },
    "mouthStretch{SIDE}": {
        "intended": "strong",
        "targets": [_t("mouth_corner_{S}", "from_center", 0.10, "MW"),
                    _t("mouth_corner_{S}", "down", 0.07, "MW")],
        "free": [],
        "notes": "Corner out+down (risorius/platysma direction).",
    },
    "mouthPress{SIDE}": {
        "intended": "strong",
        "targets": [_t("mouth_corner_{S}", "back", 0.03, "MW"),
                    _t("lip_upper_outer_{S}", "down", 0.02, "MW"),
                    _t("lip_lower_outer_{S}", "up", 0.02, "MW")],
        "free": [],
        "notes": "One-sided lip press (lips thin against each other).",
    },
    "mouthLowerDown{SIDE}": {
        "intended": "strong",
        "targets": [_t("lip_lower_outer_{S}", "down", 0.07, "MW"),
                    _t("lip_lower_inner_{S}", "down", 0.05, "MW")],
        "free": [],
        "notes": "Lower-lip depressor on one side.",
    },
    "mouthUpperUp{SIDE}": {
        "intended": "strong",
        "targets": [_t("lip_upper_outer_{S}", "up", 0.07, "MW"),
                    _t("lip_upper_inner_{S}", "up", 0.05, "MW")],
        "free": [],
        "notes": "Upper-lip levator on one side.",
    },
    "noseSneer{SIDE}": {
        "intended": "weak",
        "targets": [_t("nostril_{S}", "up", 0.004, "ABS"),
                    _t("lip_upper_outer_{S}", "up", 0.003, "ABS")],
        "free": [],
        "notes": "Nostril-wing + upper-lip raise. Nose wrinkling is weakly "
                 "represented in FLAME PCA and one-sidedness is dubious -- "
                 "the leakage gate decides honestly.",
    },
}

# --- bilateral / special PCA shapes ----------------------------------------
_BILATERAL_PCA = {
    "browInnerUp": {
        "intended": "strong",
        "targets": [_t("brow_inner", "up", 0.35, "BH_MEAN")],
        "free": [],
        "leak_pair": None,
        "notes": "Single bilateral inner-brow raise (Apple ships ONE shape).",
    },
    "mouthLeft": {
        "intended": "strong",
        "targets": [_t("lips_all", "left", 0.08, "MW")],
        "free": [],
        "leak_pair": None,
        "notes": "Whole mouth shifts to the SUBJECT's left (+X, measured).",
    },
    "mouthRight": {
        "intended": "strong",
        "targets": [_t("lips_all", "right", 0.08, "MW")],
        "free": [],
        "leak_pair": None,
        "notes": "Whole mouth shifts to the SUBJECT's right.",
    },
    "mouthPucker": {
        "intended": "strong",
        "targets": [_t("lips_outer_all", "to_center", 0.12, "MW"),
                    _t("mouth_corner_L", "to_center", 0.15, "MW"),
                    _t("mouth_corner_R", "to_center", 0.15, "MW"),
                    _t("lips_all", "fwd", 0.05, "MW")],
        "free": [],
        "leak_pair": None,
        "notes": "Lips compress toward center and protrude (kiss).",
    },
    "mouthFunnel": {
        "intended": "strong",
        "targets": [_t("lips_all", "fwd", 0.08, "MW"),
                    _t("lips_outer_all", "to_center", 0.04, "MW"),
                    _t("lip_upper_inner", "up", 0.03, "MW"),
                    _t("lip_lower_inner", "down", 0.03, "MW")],
        "free": [],
        "leak_pair": None,
        "notes": "Open-O: lips forward, inner aperture opens.",
    },
    "mouthRollLower": {
        "intended": "strong",
        "targets": [_t("lip_lower_outer", "back", 0.05, "MW"),
                    _t("lip_lower_outer", "up", 0.02, "MW"),
                    _t("lip_lower_inner", "back", 0.04, "MW")],
        "free": [],
        "leak_pair": None,
        "notes": "Lower lip rolls inward over the teeth (landmark-level "
                 "approximation; true roll needs volume FLAME lacks).",
    },
    "mouthRollUpper": {
        "intended": "strong",
        "targets": [_t("lip_upper_outer", "back", 0.05, "MW"),
                    _t("lip_upper_outer", "down", 0.02, "MW"),
                    _t("lip_upper_inner", "back", 0.04, "MW")],
        "free": [],
        "leak_pair": None,
        "notes": "Upper lip rolls inward (same approximation caveat).",
    },
    "mouthShrugLower": {
        "intended": "strong",
        "targets": [_t("lip_lower_outer", "up", 0.06, "MW"),
                    _t("lip_lower_outer", "fwd", 0.03, "MW"),
                    _t("lip_lower_inner", "up", 0.04, "MW")],
        "free": [],
        "leak_pair": None,
        "notes": "Chin/mentalis pushes the lower lip up and out.",
    },
    "mouthShrugUpper": {
        "intended": "strong",
        "targets": [_t("lip_upper_outer", "up", 0.05, "MW"),
                    _t("lip_upper_outer", "fwd", 0.03, "MW"),
                    _t("lip_upper_inner", "up", 0.03, "MW")],
        "free": [],
        "leak_pair": None,
        "notes": "Upper lip pushes up/out (philtrum shrug).",
    },
}

# mouthClose is special: its delta is linearized AT THE JAW-OPEN POSE --
# solve expression coeffs that re-seal the inner lips under jawOpen, delta =
# decode(jaw_open, e_seal) - decode(jaw_open, 0). Driving jawOpen+mouthClose
# together then approximates "jaw open, lips sealed" (Apple's definition:
# lip closure independent of jaw). Targets are computed procedurally from
# the jaw-open geometry (INNER_LIP_PAIRS midpoints), not from this table.
_MOUTH_CLOSE = {
    "intended": "strong",
    "targets": "procedural: inner-lip pairs re-meet at their jaw-open midpoints",
    "free": ["lips_outer_all", "mouth_corner_L", "mouth_corner_R"],
    "leak_pair": None,
    "notes": "Linearized at jawOpen (secant Jacobian, batched decode). "
             "Correct only in combination with jawOpen -- documented.",
}

# --- pose-based shapes (FLAME joints; axes/signs calibrated by measurement) -
_POSE = {
    "jawOpen": {"method": "pose_jaw_open", "intended": "strong",
                "notes": "Jaw joint pitch; sign chosen by measuring that the "
                         "lower inner lip drops. Magnitude JAW_OPEN_RAD."},
    "jawLeft": {"method": "pose_jaw_lat", "dir": "left", "intended": "strong",
                "notes": "Jaw joint lateral axis (Y or Z, measured); sign "
                         "chosen so the chin/lower lip moves to subject-left."},
    "jawRight": {"method": "pose_jaw_lat", "dir": "right", "intended": "strong",
                 "notes": "Mirror of jawLeft (same measured axis, opposite sign)."},
    "jawForward": {"method": "pose_jaw_fwd", "intended": "strong",
                   "notes": "FLAME's jaw joint is rotation-only; protrusion is "
                            "synthesized as an LBS-weighted +Z TRANSLATION of "
                            "the jaw joint (verts += w_jaw * t). Self-authored, "
                            "rig-style approximation -- documented, not PCA."},
}
for _side, _letter in (("Left", "L"), ("Right", "R")):
    for _dir in ("Up", "Down", "In", "Out"):
        _POSE[f"eyeLook{_dir}{_side}"] = {
            "method": "pose_eye", "side": _letter, "dir": _dir.lower(),
            "intended": "strong",
            "notes": "Eye JOINT rotation (eyeball only; lids do not follow -- "
                     "honest limitation). Joint<->side mapping and rotation "
                     "signs are measured on the pod, never assumed.",
        }

# --- honestly unsupported (a priori: no geometry / no handles / unspanned) --
_UNSUPPORTED = {
    "tongueOut": "FLAME 2023 has no tongue geometry; no expression axis or "
                 "pose can produce it. Declared unsupported (flat, no mesh) -- "
                 "matches out/arkit_51_52_map.json (MediaPipe never emits it).",
    "cheekPuff": "No cheek handles exist in the iBUG-51 set and cheek "
                 "inflation is essentially unspanned by FLAME's scan-based "
                 "expression PCA; any solve would fabricate geometry against "
                 "unrelated handles. Declared unsupported.",
    "cheekSquintLeft": "No cheek handles; the infraorbital bulge of cheek "
                       "squint is not reliably spanned by FLAME PCA and would "
                       "duplicate eyeSquintLeft dishonestly. Unsupported.",
    "cheekSquintRight": "Mirror of cheekSquintLeft. Unsupported.",
}


def _instantiate_sided():
    out = {}
    for tmpl_name, tmpl in _SIDED_PCA.items():
        for side_word, s in (("Left", "L"), ("Right", "R")):
            name = tmpl_name.replace("{SIDE}", side_word)
            targets = []
            for t in tmpl["targets"]:
                targets.append({
                    "handle": t["handle"].replace("{S}", s),
                    "dir": t["dir"],
                    "factor": t["factor"],
                    "unit": t["unit"].replace("{S}", s),
                })
            free = [h.replace("{S}", s) for h in tmpl["free"]]
            tgt_handles = sorted({t["handle"] for t in targets})
            leak_pair = (tgt_handles, [_mirror_handle(h) for h in tgt_handles])
            out[name] = {
                "method": "pca",
                "intended": tmpl["intended"],
                "targets": targets,
                "free": free,
                "leak_pair": leak_pair,
                "notes": tmpl["notes"],
            }
    return out


def build_spec():
    """The full 52-shape specification: {arkitName: spec_dict}."""
    spec = {}
    spec.update(_instantiate_sided())
    for name, s in _BILATERAL_PCA.items():
        spec[name] = {"method": "pca", **s}
    spec["mouthClose"] = {"method": "pca_jawopen", **_MOUTH_CLOSE}
    for name, s in _POSE.items():
        spec[name] = dict(s)
    for name, reason in _UNSUPPORTED.items():
        spec[name] = {"method": "none", "intended": "unsupported",
                      "reason": reason}
    return spec


SPEC = build_spec()

# Self-check at import (pure counting -- no numerics): 52 shapes total.
assert len(SPEC) == 52, f"spec has {len(SPEC)} shapes, expected 52"
_counts = {"strong": 0, "weak": 0, "unsupported": 0}
for _s in SPEC.values():
    _counts[_s["intended"]] += 1
INTENDED_COUNTS = dict(_counts)   # {'strong': 42, 'weak': 6, 'unsupported': 4}
assert INTENDED_COUNTS == {"strong": 42, "weak": 6, "unsupported": 4}, INTENDED_COUNTS
