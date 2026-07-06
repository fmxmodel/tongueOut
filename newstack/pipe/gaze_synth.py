"""Synthesize eyeball-rotation deltas for the 8 ARKit eyeLook shapes.

MEASURED (out/rig/arkit_deltas.npz before this synth): ICT's eyeLook*
expression OBJs move the LIDS only -- max |delta| over the eyeball vertex
range is exactly 0.0 for all eight shapes. A gaze morph that never moves the
iris is useless for driving, so the standard rig behavior (rotate the eyeball
about its own center) is baked INTO the morph target as an additive delta:

    delta = R @ (v - c) + c - v      for v in that eye's vertex range only

where c is the eyeball centroid and R a fixed-angle rotation. Linear morph
interpolation traces the chord instead of the arc (the pole dips ~1-cos(a/2)
of the radius at weight 0.5, ~0.05 cm here) -- the accepted trade-off of
blendshape gaze rigs.

Conventions (MEASURED, not assumed): ICT space is cm, +Y up, +Z front,
subject's LEFT = +X (ICT "Eyeball left" [21451:23021) has centroid x=+3.2 and
eyeBlinkLeft moves verts with mean x=+3.2). ARKit's Left/Right are the
subject's. "Out" = temporal (away from the nose), "In" = nasal.

Rotation axes: yaw about +Y by a>0 moves the +Z pole toward +X; pitch about
+X by b>0 moves the +Z pole toward -Y (down). Every synthesized delta is
GATED by measuring the forward pole's actual displacement direction.
"""

import numpy as np

# subject-left / subject-right eyeball vertex ranges (ICT README #7/#8)
EYEBALL = {"Left": (21451, 23021), "Right": (23021, 24591)}

# gaze extremes (degrees) baked at morph weight 1.0 -- typical human/rig range
GAZE_DEG = {"In": 35.0, "Out": 35.0, "Up": 25.0, "Down": 30.0}


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def parse_eyelook(name):
    """'eyeLookOutLeft' -> ('Out', 'Left'). Raises on anything else."""
    assert name.startswith("eyeLook"), name
    for side in ("Left", "Right"):
        if name.endswith(side):
            kind = name[len("eyeLook"):-len(side)]
            assert kind in GAZE_DEG, f"unknown gaze kind {kind!r} in {name}"
            return kind, side
    raise AssertionError(f"eyeLook name without Left/Right suffix: {name}")


def synth_gaze_delta(name, neutral, return_stats=False):
    """Additive eyeball-rotation delta (N,3) float64 for one eyeLook shape.

    Zero everywhere except that eye's vertex range (by construction, and
    asserted). The forward pole's displacement direction is MEASURED and
    gated against the ARKit semantics of the shape name.
    """
    kind, side = parse_eyelook(name)
    v0, v1 = EYEBALL[side]
    verts = np.asarray(neutral, dtype=np.float64)
    eye = verts[v0:v1]
    c = eye.mean(axis=0)

    ang = np.deg2rad(GAZE_DEG[kind])
    # temporal direction: +X for the subject-left eye, -X for the right
    temporal = 1.0 if side == "Left" else -1.0
    if kind == "Out":
        R = _rot_y(temporal * ang)
    elif kind == "In":
        R = _rot_y(-temporal * ang)
    elif kind == "Up":
        R = _rot_x(-ang)   # b<0 moves the +Z pole toward +Y
    else:  # Down
        R = _rot_x(ang)

    delta = np.zeros_like(verts)
    delta[v0:v1] = (eye - c) @ R.T + c - eye

    # ---- gates (measure, don't trust the algebra above)
    pole_i = int(np.argmax(eye[:, 2]))          # forward (+z) pole
    pd = delta[v0 + pole_i]
    expect = {"Out": (0, temporal), "In": (0, -temporal),
              "Up": (1, 1.0), "Down": (1, -1.0)}[kind]
    axis, sign = expect
    assert pd[axis] * sign > 0.1, \
        (f"{name}: pole moved {pd.round(3)} -- expected axis {axis} "
         f"sign {sign:+.0f}; rotation direction wrong")
    mag = float(np.linalg.norm(pd))
    assert abs(pd[axis]) >= 0.7 * mag, \
        f"{name}: pole motion {pd.round(3)} not dominated by axis {axis}"
    radius = float(np.linalg.norm(eye - c, axis=1).max())
    expected_mag = 2.0 * radius * np.sin(ang / 2.0)
    assert 0.5 * expected_mag < mag < 1.5 * expected_mag, \
        f"{name}: pole |delta|={mag:.3f} vs expected ~{expected_mag:.3f} cm"

    if not return_stats:
        return delta
    return delta, {
        "kind": kind, "side": side, "angle_deg": GAZE_DEG[kind],
        "vertex_range": [v0, v1], "centroid_cm": c.round(3).tolist(),
        "eyeball_radius_cm": round(radius, 3),
        "pole_delta_cm": pd.round(3).tolist(),
        "pole_moved_cm": round(mag, 3),
    }
