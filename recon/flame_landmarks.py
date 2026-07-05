"""SELF-AUTHORED FLAME static-51 landmark embedding (clean-room, commercial).

WHY THIS FILE EXISTS (measured on the pod, 2026-07-05)
-------------------------------------------------------
The FLAME 2023 Open (CC-BY-4.0) release ships ONLY `flame2023.pkl` (+ readme).
It contains NO landmark embedding. The embeddings floating around in
DECA / EMOCA / TF_FLAME / flame-fitting / smplx are NC-licensed packages and
are BARRED from this commercial run (out/compliance_report.md B1-1 caveat,
models/README.md section 1). This module therefore DERIVES the iBUG static-51
(iBUG 17..67; no jaw contour) landmark anchors GEOMETRICALLY from the pkl's
own arrays -- nothing is copied from any third-party embedding file, and no
index list from an NC repo was consulted.

CALIBRATED AGAINST THE REAL MESH (pod measurements, 2026-07-05)
----------------------------------------------------------------
V=5023, F=9976. Axes: +x lateral (eye joints x=±0.031), +y UP (eyes y=+0.023,
jaw joint y=-0.0155, mesh bottom y=-0.187), +z FRONT (nose tip = global max z,
midline y=-0.01 z=+0.075). IOD (eye-joint distance) = 0.0631 m. Jaw skinning
weight = weights[:,2]; ~0 on the nose/upper lip, high on lower lip/chin; the
mouth sits at roughly y in [-0.04, -0.075].

METHOD (deterministic; pure numpy + scipy; no learned weights)
--------------------------------------------------------------
FLAME's topology is fixed, its template bilaterally symmetric, canonically
oriented (+X = subject's anatomical LEFT -- SMPL/FLAME convention, re-checked
downstream by rig/build_arkit_shapes.py's measured laterality gate) and
metrically scaled. Landmarks are then detectable properties of the geometry:

  eyes    rest joints 3/4 are the eyeball centers (verified: the vertices
          rigidly skinned to them centroid at the joints). Subject-RIGHT eye =
          the negative-x joint. Lid-margin ring = skin vertices hugging the
          eyeball sphere in front of its center; canthi = ring x-extremes;
          upper/lower lid points at 1/3 / 2/3 x-stations.
  mouth   the LIP SEAM y is LOCATED FROM THE MESH: scan the midline strip
          below the nose and take the topmost jaw-weighted (w>=0.5) vertex --
          that is the top of the lower lip (jaw weight transitions ~0 -> high
          exactly at the seam). CORNERS (commissures) = the lateral-most point
          where the upper-lip and lower-lip sheets CONVERGE (near-coincident
          vertex pair across the seam), searched ONLY inside anthropometric
          bounds: |x| <= min(0.48*IOD, eye half-width) and frontal z (near
          z_seam) -- never the raw seam-strip x-extreme, which was measured on
          the pod to bleed into the cheek (|x|=1.63*IOD/2, z 30 mm behind the
          seam). Inner-lip points = seam-line members at fixed width
          fractions; outer-lip points = local +z vermilion crests above (low
          jaw weight) / below (high jaw weight) the seam. Jaw weight only
          DISAMBIGUATES upper vs lower lip -- never a hard gate that can empty
          a candidate set.
  nose    tip (30) = most-protrusive (+z) sub-eye near-midline vertex;
          nasion (27) = midline z-dip at eye height; bridge 28/29 = midline
          snaps between them; subnasale (33) = midline z-concavity midway
          between the tip and the lip seam; alar wings (31/35) = x-extremes of
          the nose protrusion zone (nearest-point fallback at anthropometric
          stations); 32/34 = midpoint snaps.
  brows   anthropometric stations (0.55*IOD above eye centers, arched,
          spanning the canthus x-range) snapped to the local browridge
          z-crest. Lowest fit weight (0.8) -- approximate anchors suffice.

ROBUSTNESS CONTRACT (rev 2 + rev 2.1, after two pod runs)
----------------------------------------------------------
Rev 1 hard-filtered candidate sets and died on empty ones (pod run 1:
`empty candidate set for upper lip center (51)`). Rev 2's corner pick took raw
seam-strip x-extremes and landed on the cheek (pod run 2, caught by the
mouth-width gate: 0.1027 m = 1.63*IOD); rev 2.1 replaces it with the bounded
lip-convergence construction above and tightens the gate to the anthropometric
0.5-1.05*IOD band. EVERY anchor resolves through guaranteed pickers over
provably NON-EMPTY base sets:
k-nearest-in-(x,y)-then-extreme-z ("local crest/dip"), nearest-to-3D-target,
or extremes of an asserted-nonempty region. Weight/region preferences only
reorder or softly restrict candidates, with automatic fallback to the broader
base. If a BASE set is somehow empty (deformed template), the failure dumps
measured diagnostics -- region vertex counts, y-bands, a jaw-weight histogram
-- to stdout AND out/recon/flame_landmarks_failure.json before dying.

EXPECTED ACCURACY (recorded honestly)
-------------------------------------
Vertex-snapped one-hot barycentrics: <= ~half a local edge (~2-4 mm on the
face region) quantization + up to ~5 mm semantic slack on heuristic stations
(brows, outer-lip spread, 32/34). Acceptable: the anchors feed a weighted
smooth-L1 landmark OPTIMIZATION (recon/fit_flame.py), and the rigger reuses
the identical anchors via the persisted npz, so residual bias is consistent
pipeline-wide. Hard measured sanity gates (laterality, vertical ordering,
nose protrusion, mouth width vs IOD) still STOP the run -- with diagnostics.

OUTPUT CONTRACT (unchanged schema -- the rig consumes it)
---------------------------------------------------------
build_static51_embedding() returns {static_faces (51,) int64, static_bary
(51,3) float64 (one-hot), vertex_ids, points, source, rules}. NO full/contour
embedding is fabricated -- the fit's contour term stays disabled.
persist_embedding() writes out/recon/lmk_embedding_static51.npz (static_faces,
static_bary, source, vertex_ids) -- identical keys to rev 1.

Standalone (pod-gated) debug run:  python -m recon.flame_landmarks
"""

import json
import sys

import numpy as np

from . import config as C

SELF_AUTHORED_SOURCE = "self-authored-geometric/2.1 (recon/flame_landmarks.py, clean-room)"

# iBUG static-51 = indices 17..67 (0-based; contour 0..16 excluded by design).
IBUG_STATIC = list(range(17, 68))

# Anthropometric stations, all in IOD units (IOD = eyeball-center distance).
_BROW_HEIGHT = 0.55          # brow line above eye-center height
_BROW_ARCH_R = (0.02, 0.07, 0.09, 0.07, 0.04)   # iBUG 17..21 (outer -> inner)
_BROW_ARCH_L = (0.04, 0.07, 0.09, 0.07, 0.02)   # iBUG 22..26 (inner -> outer)
_LIP_FRAC_UP = (0.23, 0.37, 0.63, 0.77)         # iBUG 49,50,52,53 across the top
_LIP_FRAC_LO = (0.77, 0.63, 0.37, 0.23)         # iBUG 55,56,58,59 across the bottom
_INNER_FRAC = {60: 0.08, 61: 0.30, 62: 0.50, 63: 0.70, 64: 0.92,
               65: 0.70, 66: 0.50, 67: 0.30}


def _ids(mask: np.ndarray) -> np.ndarray:
    return np.nonzero(mask)[0]


# --------------------------------------------------------------------------
# the derivation
# --------------------------------------------------------------------------
def build_static51_embedding(v_template, faces, lbs_weights, rest_joints,
                             debug_json=None, debug_png=None):
    """Derive the static-51 iBUG embedding from the FLAME pkl arrays alone.

    v_template  (V,3) float  faces (F,3) int  lbs_weights (V,J) float
    rest_joints (J,3) float  -- J_regressor @ v_template
    Returns {static_faces (51,) int64, static_bary (51,3) float64,
             vertex_ids (51,) int64, points (51,3) float64, source, rules}.
    """
    V = np.asarray(v_template, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    W = np.asarray(lbs_weights, dtype=np.float64)
    J = np.asarray(rest_joints, dtype=np.float64)
    w_jaw = W[:, 2] if W.shape[1] > 2 else np.zeros(V.shape[0])

    # measured context, grows as the derivation proceeds; dumped on ANY failure
    ctx = {"n_vertices": int(V.shape[0]), "n_faces": int(F.shape[0]),
           "bbox_min_m": V.min(axis=0).tolist(), "bbox_max_m": V.max(axis=0).tolist()}

    # ---- failure diagnostics (requirement: actionable numbers, not just FATAL)
    def diag_die(msg, **extra):
        hist, edges = np.histogram(w_jaw, bins=10, range=(0.0, 1.0))
        doc = {
            "schema": "b1-flame-landmarks-failure/1.0",
            "error": msg,
            "measured_context": ctx,
            "jaw_weight_histogram": {
                "bin_edges": [round(float(e), 3) for e in edges],
                "counts": [int(h) for h in hist],
            },
            "extra": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                      for k, v in extra.items()},
        }
        try:
            with open(C.FLAME_LMK_FAIL_JSON, "w", encoding="utf-8") as f:
                json.dump(doc, f, indent=2)
            where = str(C.FLAME_LMK_FAIL_JSON)
        except OSError:
            where = "(could not write failure json)"
        print("[flame_landmarks DIAG] " + json.dumps(doc, indent=2))
        sys.exit(f"[flame_landmarks FATAL] {msg} -- diagnostics dumped to {where}")

    def require(ids, what, **extra):
        ids = np.asarray(ids)
        if ids.size == 0:
            diag_die(f"empty BASE vertex set for {what}", **extra)
        return ids

    # ---- guaranteed pickers (never operate on possibly-empty filtered sets) --
    def knn_xy(ids, x_t, y_t, k):
        """k nearest of `ids` in the (x,y) plane to the target station."""
        d2 = (V[ids, 0] - x_t) ** 2 + (V[ids, 1] - y_t) ** 2
        k = min(int(k), ids.size)
        return ids[np.argsort(d2)[:k]]

    def local_max_z(ids, x_t, y_t, k, what):
        sub = knn_xy(require(ids, what), x_t, y_t, k)
        return int(sub[np.argmax(V[sub, 2])])

    def local_min_z(ids, x_t, y_t, k, what):
        sub = knn_xy(require(ids, what), x_t, y_t, k)
        return int(sub[np.argmin(V[sub, 2])])

    def nearest3(ids, p, what):
        ids = require(ids, what)
        d = np.linalg.norm(V[ids] - np.asarray(p, dtype=np.float64), axis=1)
        return int(ids[np.argmin(d)])

    def nearest_xy(ids, x_t, y_t, what):
        ids = require(ids, what)
        d2 = (V[ids, 0] - x_t) ** 2 + (V[ids, 1] - y_t) ** 2
        return int(ids[np.argmin(d2)])

    def prefer(primary_ids, fallback_ids, what):
        """Soft restriction: use the preferred subset when populated, else the
        guaranteed base -- never an empty set."""
        primary_ids = np.asarray(primary_ids)
        if primary_ids.size > 0:
            return primary_ids
        print(f"[flame_landmarks WARN] preference set empty for {what}; "
              "falling back to the broad base set")
        return require(fallback_ids, what)

    # ---- joint semantics, verified by measurement (never assumed) ------------
    if W.shape[1] != 5 or J.shape[0] != 5:
        diag_die(f"expected the 5-joint FLAME rig [global,neck,jaw,eyeA,eyeB]; "
                 f"got weights {W.shape}, joints {J.shape}")
    ball_a, ball_b = W[:, 3] > 0.5, W[:, 4] > 0.5
    ctx["eyeball_vert_counts"] = [int(ball_a.sum()), int(ball_b.sum())]
    if int(ball_a.sum()) < 20 or int(ball_b.sum()) < 20:
        diag_die("joints 3/4 do not rigidly own eyeball vertices "
                 "(counts above) -- not eyeball joints")
    for ball, j in ((ball_a, J[3]), (ball_b, J[4])):
        off = float(np.linalg.norm(V[ball].mean(axis=0) - j))
        if off > 0.05:
            diag_die(f"eyeball-vertex centroid is {off:.4f} m from its rest "
                     "joint -- joints 3/4 are not eyeball centers")
    skin = ~(ball_a | ball_b)

    # subject-RIGHT eye = negative-x joint (+X = subject-left, SMPL/FLAME frame)
    if J[3][0] < J[4][0]:
        cR, cL, ballR, ballL = J[3], J[4], ball_a, ball_b
    else:
        cR, cL, ballR, ballL = J[4], J[3], ball_b, ball_a
    iod = float(np.linalg.norm(cL - cR))
    ctx.update({"iod_m": iod, "eye_R_joint": cR.tolist(), "eye_L_joint": cL.tolist(),
                "jaw_joint": J[2].tolist()})
    if not (0.01 < iod < 0.3):
        diag_die(f"implausible eye-joint separation {iod:.4f} m")
    if abs(cR[0] + cL[0]) > 0.2 * iod or abs(cR[1] - cL[1]) > 0.2 * iod:
        diag_die("eye joints not bilaterally symmetric (see measured_context)")
    eye_y = 0.5 * (cR[1] + cL[1])
    eye_z = max(cR[2], cL[2])
    rR = float(np.median(np.linalg.norm(V[ballR] - cR, axis=1)))
    rL = float(np.median(np.linalg.norm(V[ballL] - cL, axis=1)))
    ctx.update({"eye_y": eye_y, "eyeball_radii_m": [rR, rL]})

    jaw_verts = w_jaw > 0.5
    if not jaw_verts.any() or float(V[jaw_verts, 1].mean()) >= eye_y:
        diag_die("jaw-weighted (col 2) vertex centroid is not below eye level "
                 "-- joint-order assumption broken",
                 jaw_centroid=(V[jaw_verts].mean(axis=0) if jaw_verts.any() else "none"))

    # base sets -- asserted nonempty ONCE, with diagnostics
    front = V[:, 2] >= eye_z - 0.1 * iod
    skin_ids = require(_ids(skin), "skin (all non-eyeball vertices)")
    face_ids = require(_ids(skin & front), "front-of-face skin",
                       front_threshold_z=eye_z - 0.1 * iod)
    mid = skin & front & (np.abs(V[:, 0]) <= 0.10 * iod)
    if int(mid.sum()) < 30:
        mid = skin & front & (np.abs(V[:, 0]) <= 0.15 * iod)
    mid_ids = require(_ids(mid), "midline front strip",
                      strip_halfwidth_m=0.15 * iod)
    ctx.update({"n_skin": int(skin_ids.size), "n_face_front": int(face_ids.size),
                "n_midline_strip": int(mid_ids.size)})

    picks, rules = {}, {}

    def put(ib, vid, rule):
        picks[ib] = int(vid)
        rules[ib] = rule

    # ---- nose tip (30): guaranteed -- argmax z over a soft-preferred band ------
    tip_band = mid_ids[(V[mid_ids, 1] < eye_y)
                       & (V[mid_ids, 1] > eye_y - 1.5 * iod)]
    tip_base = prefer(tip_band, face_ids[V[face_ids, 1] < eye_y], "nose tip band")
    tip = int(tip_base[np.argmax(V[tip_base, 2])])
    put(30, tip, "max +z sub-eye midline skin vertex (pronasale)")
    tip_p = V[tip]
    ctx["nose_tip"] = tip_p.tolist()

    # ---- lip seam: LOCATED FROM THE MESH (midline jaw-weight 0 -> high
    # transition), not from fixed y thresholds ----------------------------------
    below_nose = mid_ids[(V[mid_ids, 1] <= tip_p[1] - 0.05 * iod)
                         & (V[mid_ids, 1] >= tip_p[1] - 2.2 * iod)]
    below_nose = require(below_nose, "midline strip below the nose",
                         y_band=[tip_p[1] - 2.2 * iod, tip_p[1] - 0.05 * iod])
    hi = below_nose[(w_jaw[below_nose] >= 0.5)
                    & (V[below_nose, 1] <= tip_p[1] - 0.2 * iod)]
    if hi.size == 0:
        # transition softer than 0.5 on this mesh: take the midline vertices
        # above half the local peak jaw weight below the nose as the seam marker
        wsub = w_jaw[below_nose]
        if float(wsub.max()) < 0.05:
            diag_die("no jaw-weighted midline vertices below the nose -- cannot "
                     "locate the lip seam",
                     midline_below_nose_count=int(below_nose.size),
                     midline_jaw_w_max=float(wsub.max()))
        thr = 0.5 * float(wsub.max())
        hi = below_nose[wsub >= thr]
        print(f"[flame_landmarks WARN] midline jaw weights peak at "
              f"{wsub.max():.2f}; using transition threshold {thr:.2f}")
    seam_top_vid = int(hi[np.argmax(V[hi, 1])])
    y_seam = float(V[seam_top_vid, 1])
    z_seam = float(V[seam_top_vid, 2])
    ctx.update({"y_seam": y_seam, "z_seam": z_seam,
                "seam_top_vertex": int(seam_top_vid)})
    print(f"[flame_landmarks] lip seam located at y={y_seam:+.4f} m "
          f"(z={z_seam:+.4f} m) from the midline jaw-weight transition")

    # ---- rest of the nose (needs the seam for a robust subnasale band) ---------
    y_sub_target = 0.5 * (y_seam + tip_p[1]) + 0.05 * iod   # midway tip<->seam
    sub = local_min_z(mid_ids, 0.0, y_sub_target, 30, "subnasale (iBUG 33)")
    put(33, sub, "midline z-concavity midway between nose tip and lip seam")
    sub_p = V[sub]

    nas = local_min_z(mid_ids, 0.0, eye_y + 0.15 * iod, 30, "nasion (iBUG 27)")
    put(27, nas, "midline z-dip at eye height (nasion)")
    for ib, t in ((28, 1.0 / 3.0), (29, 2.0 / 3.0)):
        tgt = (1 - t) * V[nas] + t * tip_p
        put(ib, nearest3(mid_ids, tgt, f"nose bridge (iBUG {ib})"),
            f"midline snap at lerp(nasion,tip,{t:.2f})")

    # alar wings: zone extremes preferred; nearest-to-station fallback guaranteed
    nose_zone = face_ids[(V[face_ids, 1] >= sub_p[1] - 0.1 * iod)
                         & (V[face_ids, 1] <= tip_p[1] + 0.1 * iod)
                         & (np.abs(V[face_ids, 0]) <= 0.40 * iod)
                         & (V[face_ids, 2] >= tip_p[2] - 0.50 * iod)]
    y_alar = 0.5 * (sub_p[1] + tip_p[1])
    if nose_zone.size >= 4:
        alar_r = int(nose_zone[np.argmin(V[nose_zone, 0])])
        alar_l = int(nose_zone[np.argmax(V[nose_zone, 0])])
        rule_alar = "x-extreme of the nose protrusion zone"
    else:
        print("[flame_landmarks WARN] nose-wing zone sparse "
              f"({nose_zone.size} verts); using nearest-to-station fallback")
        alar_r = nearest3(face_ids, (-0.28 * iod, y_alar, tip_p[2] - 0.35 * iod),
                          "right alar (iBUG 31)")
        alar_l = nearest3(face_ids, (+0.28 * iod, y_alar, tip_p[2] - 0.35 * iod),
                          "left alar (iBUG 35)")
        rule_alar = "nearest vertex to the anthropometric alar station (fallback)"
    put(31, alar_r, f"{rule_alar} (subject-right)")
    put(35, alar_l, f"{rule_alar} (subject-left)")
    put(32, nearest3(face_ids, 0.5 * (sub_p + V[alar_r]), "iBUG 32"),
        "snap at midpoint(subnasale, right alar)")
    put(34, nearest3(face_ids, 0.5 * (sub_p + V[alar_l]), "iBUG 34"),
        "snap at midpoint(subnasale, left alar)")

    # ---- eyes: ring with guaranteed shell fallback ------------------------------
    def eye_ring_ids(c, r, what):
        d = np.linalg.norm(V - c[None, :], axis=1)
        for margin in (0.06, 0.10, 0.16):
            ids = _ids(skin & (d <= r + margin * iod) & (V[:, 2] > c[2]))
            if ids.size >= 12:
                if margin > 0.06:
                    print(f"[flame_landmarks WARN] eye ring for {what} needed "
                          f"margin {margin:.2f}*IOD ({ids.size} verts)")
                return ids
        base = _ids(skin & (V[:, 2] > c[2]))
        if base.size == 0:
            base = skin_ids
        print(f"[flame_landmarks WARN] eye-ring shells all sparse for {what}; "
              "using the 40 skin vertices closest to the eyeball sphere")
        return base[np.argsort(np.abs(d[base] - r))[:40]]

    def lid_point(ring, x_t, upper, c_y, what):
        side = ring[(V[ring, 1] > c_y) if upper else (V[ring, 1] < c_y)]
        side = prefer(side, ring, what)
        return int(side[np.argmin(np.abs(V[side, 0] - x_t))])

    ringR = eye_ring_ids(cR, rR, "subject-right eye")
    ringL = eye_ring_ids(cL, rL, "subject-left eye")
    outR = int(ringR[np.argmin(V[ringR, 0])])   # temporal side of right eye = -x
    innR = int(ringR[np.argmax(V[ringR, 0])])
    innL = int(ringL[np.argmin(V[ringL, 0])])   # nasal side of left eye
    outL = int(ringL[np.argmax(V[ringL, 0])])
    put(36, outR, "right lid-ring min-x (outer canthus)")
    put(39, innR, "right lid-ring max-x (inner canthus)")
    put(42, innL, "left lid-ring min-x (inner canthus)")
    put(45, outL, "left lid-ring max-x (outer canthus)")

    def x_station(a, b, t):
        return V[a, 0] + t * (V[b, 0] - V[a, 0])

    put(37, lid_point(ringR, x_station(outR, innR, 1 / 3), True, cR[1], "iBUG 37"),
        "right upper lid @ outer-third x-station")
    put(38, lid_point(ringR, x_station(outR, innR, 2 / 3), True, cR[1], "iBUG 38"),
        "right upper lid @ inner-third x-station")
    put(40, lid_point(ringR, x_station(outR, innR, 2 / 3), False, cR[1], "iBUG 40"),
        "right lower lid @ inner-third x-station")
    put(41, lid_point(ringR, x_station(outR, innR, 1 / 3), False, cR[1], "iBUG 41"),
        "right lower lid @ outer-third x-station")
    put(43, lid_point(ringL, x_station(innL, outL, 1 / 3), True, cL[1], "iBUG 43"),
        "left upper lid @ inner-third x-station")
    put(44, lid_point(ringL, x_station(innL, outL, 2 / 3), True, cL[1], "iBUG 44"),
        "left upper lid @ outer-third x-station")
    put(46, lid_point(ringL, x_station(innL, outL, 2 / 3), False, cL[1], "iBUG 46"),
        "left lower lid @ outer-third x-station")
    put(47, lid_point(ringL, x_station(innL, outL, 1 / 3), False, cL[1], "iBUG 47"),
        "left lower lid @ inner-third x-station")

    # ---- mouth: everything anchored to the MEASURED seam ------------------------
    lip_ids = face_ids[(V[face_ids, 1] >= y_seam - 0.45 * iod)
                       & (V[face_ids, 1] <= y_seam + 0.40 * iod)
                       & (V[face_ids, 2] >= z_seam - 0.5 * iod)]
    lip_ids = require(lip_ids, "lip band around the measured seam",
                      y_band=[y_seam - 0.45 * iod, y_seam + 0.40 * iod],
                      z_floor=z_seam - 0.5 * iod)
    ctx["n_lip_band"] = int(lip_ids.size)

    # jaw weight DISAMBIGUATES sides; geometric side split is the fallback
    upper_ids = prefer(lip_ids[(w_jaw[lip_ids] < 0.5)
                               & (V[lip_ids, 1] > y_seam - 0.10 * iod)],
                       lip_ids[V[lip_ids, 1] >= y_seam], "upper lip side")
    lower_ids = prefer(lip_ids[w_jaw[lip_ids] >= 0.5],
                       lip_ids[V[lip_ids, 1] <= y_seam], "lower lip side")
    ctx.update({"n_upper_lip": int(upper_ids.size), "n_lower_lip": int(lower_ids.size)})

    # corners (iBUG 48/54) = the COMMISSURES. NEVER the raw seam-strip
    # x-extremes: measured on the pod (rev 2 failure), the strip bleeds
    # laterally into cheek/jaw -- raw extremes landed at |x|=0.051 (1.63*IOD,
    # wider than the eyes) and ~30 mm BEHIND the seam in z. Constraints:
    #   anthropometric: neutral mouth width ~0.6-0.9*IOD, so each commissure
    #     sits at |x| <~ 0.45*IOD and ALWAYS inside the eye width;
    #   geometric: the commissure is FRONTAL (z near z_seam, cheeks recede).
    # PRIMARY construction: the lateral-most point where the upper-lip and
    # lower-lip sheets CONVERGE (near-coincident vertex pair across the seam)
    # inside those bounds. Fallbacks stay bounded and guaranteed.
    eye_half_x = 0.5 * (abs(cR[0]) + abs(cL[0]))
    x_bound = min(0.48 * iod, 0.98 * eye_half_x)
    z_front_lip = z_seam - 0.30 * iod
    ctx.update({"corner_x_bound": x_bound, "corner_z_floor": z_front_lip})

    def corner_candidates(ids):
        return ids[(np.abs(V[ids, 0]) <= x_bound)
                   & (np.abs(V[ids, 1] - y_seam) <= 0.15 * iod)
                   & (V[ids, 2] >= z_front_lip)]

    up_c, lo_c = corner_candidates(upper_ids), corner_candidates(lower_ids)
    ctx.update({"n_corner_cand_upper": int(up_c.size),
                "n_corner_cand_lower": int(lo_c.size)})
    vid48 = vid54 = None
    n_pairs = 0
    if up_c.size >= 2 and lo_c.size >= 2:
        from scipy.spatial import cKDTree

        d, nn = cKDTree(V[lo_c]).query(V[up_c], k=1)
        paired = d <= 0.10 * iod          # lips are closed: sheets nearly touch
        n_pairs = int(paired.sum())
        if n_pairs >= 6:
            pu = up_c[paired]
            vid48 = int(pu[np.argmin(V[pu, 0])])
            vid54 = int(pu[np.argmax(V[pu, 0])])
            rule_corner = ("lateral-most upper/lower lip convergence pair "
                           "(commissure), bounded |x|<=min(0.48*IOD, eye "
                           "half-width) and frontal z")
    ctx["n_corner_pairs"] = n_pairs
    if vid48 is None:
        strip = corner_candidates(lip_ids)
        if strip.size >= 2:
            print(f"[flame_landmarks WARN] only {n_pairs} lip-convergence pairs; "
                  "corners from the BOUNDED frontal seam strip instead")
            vid48 = int(strip[np.argmin(V[strip, 0])])
            vid54 = int(strip[np.argmax(V[strip, 0])])
            rule_corner = "x-extreme of the bounded frontal seam strip (fallback)"
        else:
            print("[flame_landmarks WARN] bounded seam strip empty; corners "
                  "snapped to the anthropometric commissure stations")
            vid48 = nearest3(lip_ids, (-0.35 * iod, y_seam, z_seam),
                             "right mouth corner (iBUG 48)")
            vid54 = nearest3(lip_ids, (+0.35 * iod, y_seam, z_seam),
                             "left mouth corner (iBUG 54)")
            rule_corner = ("nearest vertex to the anthropometric commissure "
                           "station (last-resort fallback)")
    put(48, vid48, f"{rule_corner} (subject-right)")
    put(54, vid54, f"{rule_corner} (subject-left)")
    x_r, x_l = V[vid48, 0], V[vid54, 0]
    mw = x_l - x_r
    ctx.update({"mouth_width_m": mw, "mouth_width_over_iod": mw / iod,
                "corner_right_xyz": V[vid48].tolist(),
                "corner_left_xyz": V[vid54].tolist()})
    # anthropometric gate: neutral mouth width ~0.6-0.9*IOD; 0.5-1.05 leaves
    # honest headroom without ever re-admitting the measured 1.63*IOD cheek bug
    if not (0.5 * iod <= mw <= 1.05 * iod):
        diag_die(f"implausible mouth width {mw:.4f} m = {mw / iod:.2f}*IOD "
                 f"(anthropometric gate 0.5-1.05*IOD)",
                 corner_right=V[vid48], corner_left=V[vid54],
                 n_corner_pairs=n_pairs,
                 n_corner_cand_upper=int(up_c.size),
                 n_corner_cand_lower=int(lo_c.size))

    # inner lips: seam-line members at width fractions (guaranteed nearest picks)
    for ib, frac in _INNER_FRAC.items():
        side = upper_ids if ib <= 64 else lower_ids
        near_seam = side[np.abs(V[side, 1] - y_seam) <= 0.15 * iod]
        base = prefer(near_seam, side, f"inner lip (iBUG {ib})")
        put(ib, nearest_xy(base, x_r + frac * mw, y_seam, f"inner lip (iBUG {ib})"),
            f"seam-line {'upper' if ib <= 64 else 'lower'} member @ "
            f"{frac:.2f} of mouth width")

    # outer lips: local vermilion z-crests (k-NN in (x,y), then max z -- never empty)
    put(51, local_max_z(upper_ids, 0.0, y_seam + 0.18 * iod, 24,
                        "upper lip center (iBUG 51)"),
        "local +z crest above the seam on the midline (upper vermilion)")
    put(57, local_max_z(lower_ids, 0.0, y_seam - 0.18 * iod, 24,
                        "lower lip center (iBUG 57)"),
        "local +z crest below the seam on the midline (lower vermilion)")
    for ib, frac in zip((49, 50, 52, 53), _LIP_FRAC_UP):
        put(ib, local_max_z(upper_ids, x_r + frac * mw, y_seam + 0.12 * iod, 16,
                            f"iBUG {ib}"),
            f"upper vermilion local +z crest @ {frac:.2f} of mouth width")
    for ib, frac in zip((55, 56, 58, 59), _LIP_FRAC_LO):
        put(ib, local_max_z(lower_ids, x_r + frac * mw, y_seam - 0.15 * iod, 16,
                            f"iBUG {ib}"),
            f"lower vermilion local +z crest @ {frac:.2f} of mouth width")

    # ---- brows: anthropometric stations snapped to the local browridge crest ----
    brow_y = eye_y + _BROW_HEIGHT * iod
    xo_r, xi_r = V[outR, 0] - 0.08 * iod, V[innR, 0]
    xi_l, xo_l = V[innL, 0], V[outL, 0] + 0.08 * iod
    for k, (ib, t) in enumerate(zip(range(17, 22), (0.0, 0.25, 0.5, 0.75, 1.0))):
        x_t = xo_r + t * (xi_r - xo_r)
        y_t = brow_y + _BROW_ARCH_R[k] * iod
        put(ib, local_max_z(face_ids, x_t, y_t, 24, f"right brow (iBUG {ib})"),
            f"browridge local +z crest @ station {t:.2f} (outer->inner), "
            f"y=eye+{_BROW_HEIGHT}*IOD+arch")
    for k, (ib, t) in enumerate(zip(range(22, 27), (0.0, 0.25, 0.5, 0.75, 1.0))):
        x_t = xi_l + t * (xo_l - xi_l)
        y_t = brow_y + _BROW_ARCH_L[k] * iod
        put(ib, local_max_z(face_ids, x_t, y_t, 24, f"left brow (iBUG {ib})"),
            f"browridge local +z crest @ station {t:.2f} (inner->outer), "
            f"y=eye+{_BROW_HEIGHT}*IOD+arch")

    # ---- measured sanity gates (STOP, never paper over -- with diagnostics) ------
    missing = [i for i in IBUG_STATIC if i not in picks]
    if missing:
        diag_die(f"internal error: iBUG indices not derived: {missing}")
    P = {i: V[picks[i]] for i in IBUG_STATIC}
    checks = {
        "canthi order x36<x39<0": P[36][0] < P[39][0] < 0.0,
        "canthi order 0<x42<x45": 0.0 < P[42][0] < P[45][0],
        "nostrils x31<0<x35": P[31][0] < 0.0 < P[35][0],
        "mouth corners x48<0<x54": P[48][0] < 0.0 < P[54][0],
        "brow above eye line": min(P[19][1], P[24][1]) > eye_y,
        "nose tip below eye line": P[30][1] < eye_y,
        "subnasale below tip": P[33][1] < P[30][1],
        "upper lip below subnasale": P[51][1] < P[33][1],
        "lower lip below upper lip": P[57][1] < P[51][1],
        "nose tip is most-protrusive pick": abs(
            P[30][2] - max(p[2] for p in P.values())) < 1e-9,
        "inner lips inside corners": P[48][0] < P[62][0] < P[54][0],
    }
    bad = {k: v for k, v in checks.items() if not v}
    if bad:
        diag_die("geometric sanity gates FAILED: " + "; ".join(sorted(bad)),
                 pick_36=P[36], pick_45=P[45], pick_30=P[30], pick_33=P[33],
                 pick_48=P[48], pick_54=P[54], pick_51=P[51], pick_57=P[57])
    n_dup = 51 - len({picks[i] for i in IBUG_STATIC})
    if n_dup:
        print(f"[flame_landmarks WARN] {n_dup} landmark pair(s) snapped to the "
              "same vertex (harmless for the fit; recorded in the debug json)")

    # ---- vertex -> (incident face, one-hot barycentric) ---------------------------
    vert2face = np.full(V.shape[0], -1, dtype=np.int64)
    vert2face[F.reshape(-1)] = np.repeat(np.arange(F.shape[0]), 3)
    static_faces = np.empty(51, dtype=np.int64)
    static_bary = np.zeros((51, 3), dtype=np.float64)
    vertex_ids = np.empty(51, dtype=np.int64)
    for row, ib in enumerate(IBUG_STATIC):
        vid = picks[ib]
        f = int(vert2face[vid])
        if f < 0:
            diag_die(f"landmark vertex {vid} (iBUG {ib}) is not referenced by "
                     "any face -- cannot anchor")
        corner = int(np.nonzero(F[f] == vid)[0][0])
        static_faces[row] = f
        static_bary[row, corner] = 1.0
        vertex_ids[row] = vid

    emb = {
        "static_faces": static_faces,
        "static_bary": static_bary,
        "vertex_ids": vertex_ids,
        "points": np.stack([V[picks[i]] for i in IBUG_STATIC]),
        "source": SELF_AUTHORED_SOURCE,
        "rules": {str(i): rules[i] for i in IBUG_STATIC},
    }
    print(f"[flame_landmarks] self-authored static-51 embedding derived "
          f"(IOD={iod * 1e3:.1f} mm, seam y={y_seam * 1e3:+.1f} mm, "
          f"mouth width={mw * 1e3:.1f} mm, eyeball r={rR * 1e3:.1f}/{rL * 1e3:.1f} mm)")

    if debug_json is not None:
        _dump_debug_json(debug_json, emb, iod)
    if debug_png is not None:
        _dump_debug_png(debug_png, V, emb, iod)
    return emb


# --------------------------------------------------------------------------
# persistence: the anchors the fit ACTUALLY used (rig consumes this file)
# --------------------------------------------------------------------------
def persist_embedding(emb: dict, source: str, path=None) -> None:
    path = C.LMK_EMBEDDING_NPZ if path is None else path
    payload = {
        "static_faces": np.asarray(emb["static_faces"], dtype=np.int64),
        "static_bary": np.asarray(emb["static_bary"], dtype=np.float64),
        "source": np.array(source),
    }
    if "vertex_ids" in emb:
        payload["vertex_ids"] = np.asarray(emb["vertex_ids"], dtype=np.int64)
    if "full_faces" in emb:  # only present when a release file provided contour
        payload["full_faces"] = np.asarray(emb["full_faces"], dtype=np.int64)
        payload["full_bary"] = np.asarray(emb["full_bary"], dtype=np.float64)
    np.savez(path, **payload)
    print(f"[flame_landmarks] persisted fit anchors -> {path} (source: {source})")


def load_persisted_embedding(path=None):
    """Load out/recon/lmk_embedding_static51.npz. Returns (emb dict, source).
    This is what rig/build_arkit_shapes.py consumes -- by construction the
    SAME anchors the fit used, whether release-loaded or self-authored."""
    path = C.LMK_EMBEDDING_NPZ if path is None else path
    if not path.is_file():
        sys.exit(f"[flame_landmarks FATAL] {path} missing -- run "
                 "`python -m recon.fit_flame` first (it persists the landmark "
                 "anchors it used).")
    z = np.load(path, allow_pickle=True)
    emb = {"static_faces": np.asarray(z["static_faces"]).astype(np.int64),
           "static_bary": np.asarray(z["static_bary"], dtype=np.float64)}
    if emb["static_faces"].shape != (51,) or emb["static_bary"].shape != (51, 3):
        sys.exit(f"[flame_landmarks FATAL] malformed persisted embedding: "
                 f"static_faces {emb['static_faces'].shape}, "
                 f"static_bary {emb['static_bary'].shape}.")
    if "full_faces" in z:
        emb["full_faces"] = np.asarray(z["full_faces"]).astype(np.int64)
        emb["full_bary"] = np.asarray(z["full_bary"], dtype=np.float64)
    return emb, str(z["source"])


# --------------------------------------------------------------------------
# debug artifacts (human verification material)
# --------------------------------------------------------------------------
def _dump_debug_json(path, emb, iod) -> None:
    doc = {
        "schema": "b1-selfauthored-flame-landmarks/1.0",
        "source": emb["source"],
        "provenance": (
            "Derived geometrically from flame2023.pkl arrays only "
            "(v_template, f, weights, J_regressor). NO third-party embedding "
            "file was read or consulted; NC FLAME repos (DECA/EMOCA/TF_FLAME/"
            "smplx/flame-fitting) were NOT used. See recon/flame_landmarks.py "
            "module docstring for the full method."
        ),
        "iod_m": iod,
        "landmarks": {
            str(ib): {
                "vertex": int(emb["vertex_ids"][row]),
                "xyz_m": [float(x) for x in emb["points"][row]],
                "face": int(emb["static_faces"][row]),
                "rule": emb["rules"][str(ib)],
            }
            for row, ib in enumerate(IBUG_STATIC)
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    print(f"[flame_landmarks] debug json -> {path}")


def _dump_debug_png(path, V, emb, iod) -> None:
    try:
        import cv2
    except ImportError:
        print("[flame_landmarks WARN] cv2 unavailable; skipping debug PNG")
        return
    size = 1000
    ctr = emb["points"].mean(axis=0)
    span = 3.5 * iod

    def to_px(p):
        u = int(round((p[0] - ctr[0]) / span * size + size / 2))
        v = int(round(-(p[1] - ctr[1]) / span * size + size / 2))
        return u, v

    img = np.zeros((size, size, 3), dtype=np.uint8)
    frontish = V[:, 2] >= np.percentile(V[:, 2], 60)
    for p in V[frontish]:
        u, v = to_px(p)
        if 0 <= u < size and 0 <= v < size:
            img[v, u] = (70, 70, 70)
    for row, ib in enumerate(IBUG_STATIC):
        u, v = to_px(emb["points"][row])
        cv2.circle(img, (u, v), 4, (0, 255, 0), -1)
        cv2.putText(img, str(ib), (u + 5, v - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (0, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "self-authored FLAME static-51 (front ortho; numbers = iBUG)",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(str(path), img)
    print(f"[flame_landmarks] debug png -> {path}")


# --------------------------------------------------------------------------
# standalone (pod-gated) debug entrypoint
# --------------------------------------------------------------------------
def main() -> None:
    from .pod_guard import require_pod

    require_pod()
    C.ensure_out_dirs()
    from .flame_model import load_flame_pkl  # numpy path only; torch unused here

    model_path = C.find_flame_file(C.FLAME_MODEL_CANDIDATES, "FLAME shape model pkl")
    data = load_flame_pkl(model_path)
    v_template = np.asarray(data["v_template"], dtype=np.float64)
    faces = np.asarray(data["f"]).astype(np.int64)
    weights = np.asarray(data["weights"], dtype=np.float64)
    j_reg = np.asarray(data["J_regressor"], dtype=np.float64)
    rest_joints = j_reg @ v_template

    emb = build_static51_embedding(
        v_template, faces, weights, rest_joints,
        debug_json=C.FLAME_LMK_SELF_JSON, debug_png=C.FLAME_LMK_SELF_PNG)
    persist_embedding(emb, SELF_AUTHORED_SOURCE)
    print("[flame_landmarks] standalone derivation DONE -- eyeball "
          f"{C.FLAME_LMK_SELF_PNG} before trusting the fit.")


if __name__ == "__main__":
    main()
