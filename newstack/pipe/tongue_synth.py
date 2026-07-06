"""Synthesize the ARKit `tongueOut` delta from ICT-FaceKit's REAL tongue.

ICT-FaceKit ships no tongue *blendshape*, but the FaceXModel topology contains
real static tongue geometry inside the published "Gums and tongue" vertex
region. This module selects the tongue by GEOMETRY, never by hardcoded vertex
ids: a region vertex is tongue iff its distance to the nearest TOOTH vertex
exceeds TOOTH_CLEARANCE_CM. Gums hug the teeth; the tongue body lies on the
floor of the mouth well clear of them. On the generic ICT neutral this yields
~760 central verts: centroid ~(0, -3.8, 4.4) cm, x in [-3.2, 3.2],
y in [-5.4, -0.8], z in [1.0, 9.1].

The delta pushes the tongue FORWARD (+z, toward the viewer; front teeth
z ~ 10.8, lips z ~ 11.9, tongue tip z ~ 9.1 on the generic neutral) with a
root->tip smoothstep**1.5 falloff, so the root stays anchored and the
blade/tip does the travel, plus a small +y lift so the blade clears the lower
lip. AMOUNT_Z_CM is chosen so the tip lands ~4.5 cm forward (~13.6, past the
lips). Stage 4 gates this: tip final z MUST exceed the neutral lip-front z.

HARD invariant (asserted here AND re-checked by s4): every vertex outside the
selected tongue set gets EXACTLY 0.0 displacement. Same vertex count/order as
every other shape; faces untouched; additive like every other ICT delta.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import smoothstep  # noqa: E402

# ICT-FaceKit FaceXModel published vertex regions (0-based, END-EXCLUSIVE
# slices). Per the ICT-FaceKit vertex-order documentation:
#   "Gums and tongue" = vertex ids 14062 .. 17038 inclusive
#   "Teeth"           = vertex ids 17039 .. 21450 inclusive (ends at 21451)
# common.ICT_REGIONS agrees on the teeth range (17039, 21451).
GUMS_TONGUE_START = 14062
GUMS_TONGUE_END = 17039     # end-exclusive (last gums/tongue id: 17038)
TEETH_START = 17039
TEETH_END = 21451           # end-exclusive (last tooth id: 21450)

# Selection: region verts farther than this from the nearest tooth vertex are
# tongue (gums stay glued to the teeth; the tongue floor is > 1 cm away).
TOOTH_CLEARANCE_CM = 1.0

# Delta shaping (ICT model space is centimeters, +y up, +z toward viewer).
AMOUNT_Z_CM = 4.5    # forward push at the tip (w=1): tip ~9.1 -> ~13.6 > lips ~11.9
AMOUNT_Y_CM = 0.8    # small upward lift so the blade clears the lower lip
WEIGHT_POWER = 1.5   # sharpen smoothstep: root anchored, tip does the travel

# Soft sanity bounds on the selection size (generic ICT neutral gives ~760).
EXPECT_SEL_MIN, EXPECT_SEL_MAX = 200, 2500


def _log(msg):
    print(f"[tongue] {msg}")


def _zero_result(n, return_stats):
    delta = np.zeros((n, 3), dtype=np.float64)
    if not return_stats:
        return delta
    return delta, {"available": False, "n_selected": 0,
                   "indices": np.zeros(0, dtype=np.int64)}


def _nearest_tooth_dist(region_verts, teeth_verts):
    """Per-vertex distance to the nearest tooth vertex. cKDTree when scipy is
    present (it is a real pipeline dependency); chunked brute force otherwise
    so the module stays importable in minimal environments."""
    try:
        from scipy.spatial import cKDTree
        return cKDTree(teeth_verts).query(region_verts, k=1)[0]
    except ImportError:
        _log("WARN scipy unavailable -- chunked brute-force nearest-tooth distances")
        dist = np.empty(len(region_verts), dtype=np.float64)
        for s in range(0, len(region_verts), 256):
            blk = region_verts[s:s + 256]
            d = np.linalg.norm(blk[:, None, :] - teeth_verts[None, :, :], axis=2)
            dist[s:s + 256] = d.min(axis=1)
        return dist


def synth_tongue_out_delta(neutral_verts, return_stats=False):
    """Build the tongueOut additive delta on a neutral of the ICT topology.

    neutral_verts : (N,3) float array, ICT model space (cm).
    return_stats  : if True, return (delta, stats) where stats carries the
                    selected indices + geometry numbers s4 needs for its gates
                    and the manifest.

    Returns delta (N,3) float64 -- EXACTLY zero outside the selected tongue
    set. If the mesh is too small to contain the tongue/teeth region (or the
    selection degenerates), returns an all-zero delta and logs that tongueOut
    is unavailable instead of crashing; callers that require 52/52 must check
    stats["available"] / the delta magnitude and fail loudly themselves.
    """
    v = np.asarray(neutral_verts, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"neutral_verts must be (N,3), got {v.shape}")
    n = len(v)

    if n < TEETH_END:
        _log(f"mesh has {n} verts < teeth range end {TEETH_END} -- this "
             "topology lacks the tongue/teeth region; tongueOut UNAVAILABLE "
             "(zero delta)")
        return _zero_result(n, return_stats)

    region_idx = np.arange(GUMS_TONGUE_START, GUMS_TONGUE_END, dtype=np.int64)
    dist = _nearest_tooth_dist(v[region_idx], v[TEETH_START:TEETH_END])
    tongue_idx = region_idx[dist > TOOTH_CLEARANCE_CM]
    n_sel = len(tongue_idx)

    if n_sel == 0:
        _log(f"selected 0 tongue verts (no gums+tongue vert clears the teeth "
             f"by {TOOTH_CLEARANCE_CM} cm) -- tongueOut UNAVAILABLE (zero delta)")
        return _zero_result(n, return_stats)
    if not (EXPECT_SEL_MIN <= n_sel <= EXPECT_SEL_MAX):
        _log(f"WARN selection count {n_sel} outside expected "
             f"[{EXPECT_SEL_MIN}, {EXPECT_SEL_MAX}] (generic ICT gives ~760) "
             "-- inspect the neutral")

    tz = v[tongue_idx, 2]
    z_root, z_tip = float(tz.min()), float(tz.max())
    if z_tip - z_root < 1e-6:
        _log("degenerate tongue z-span (z_tip ~ z_root) -- tongueOut "
             "UNAVAILABLE (zero delta)")
        return _zero_result(n, return_stats)

    # Root->tip weighting: w = smoothstep((z - z_root)/(z_tip - z_root))**1.5
    w = smoothstep(tz, z_root, z_tip) ** WEIGHT_POWER
    delta = np.zeros((n, 3), dtype=np.float64)
    delta[tongue_idx, 1] = w * AMOUNT_Y_CM
    delta[tongue_idx, 2] = w * AMOUNT_Z_CM

    # HARD invariant: exactly 0.0 on every vertex outside the tongue set.
    outside = np.ones(n, dtype=bool)
    outside[tongue_idx] = False
    assert not delta[outside].any(), \
        "tongueOut delta leaked outside the selected tongue set"

    c = v[tongue_idx].mean(axis=0)
    _log(f"selected {n_sel}/{len(region_idx)} gums+tongue verts as tongue; "
         f"centroid=({c[0]:+.2f}, {c[1]:+.2f}, {c[2]:+.2f}) cm, "
         f"z_root={z_root:.2f}, z_tip={z_tip:.2f} -> tip pushed to "
         f"{z_tip + AMOUNT_Z_CM:.2f} (+{AMOUNT_Z_CM:.1f} fwd, "
         f"+{AMOUNT_Y_CM:.1f} lift)")

    if not return_stats:
        return delta
    return delta, {
        "available": True,
        "n_selected": int(n_sel),
        "n_region": int(len(region_idx)),
        "indices": tongue_idx,
        "centroid": [round(float(x), 3) for x in c],
        "z_root": z_root,
        "z_tip": z_tip,
        "tip_final_z": z_tip + AMOUNT_Z_CM,
        "clearance_cm": TOOTH_CLEARANCE_CM,
        "amount_z_cm": AMOUNT_Z_CM,
        "amount_y_cm": AMOUNT_Y_CM,
        "weight_power": WEIGHT_POWER,
    }
