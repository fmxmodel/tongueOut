"""Stage 3b -- shrinkwrap refinement of the fitted neutral onto the smoothed
clay (RUNS INSIDE BLENDER):

  xvfb-run -a blender --background --factory-startup \
      --python pipe/s3b_refine_blender.py -- [args]

Why Blender: the plan calls for Blender's Shrinkwrap (NEAREST_SURFACEPOINT)
against a low-passed clay. Noise control so the clay can't wreck ICT's clean
face, in order:
  1. the clay TARGET is smoothed first (Smooth modifier, evaluated in the
     depsgraph -- the shrinkwrap sees the smoothed surface);
  2. raw per-vertex displacements > --cutoff are discarded (clay missing
     there, e.g. below the neck);
  3. displacement magnitudes are clamped to --max-disp;
  4. the displacement FIELD is graph-Laplacian smoothed over ICT topology;
  5. a region+feature mask gates the result.

TWO weight schemes (--weights):
  legacy (TripoSR default): interior verts (>= 11248) ZERO, eye/mouth/nose
     vicinities protected with a smoothstep falloff, face gets --face-weight,
     scalp/neck 1.0 (grab the clay's hair volume everywhere).
  face  (TripoSG path): the FACE region [0,9409) gets ~1.0 (the face TAKES
     the sharp TripoSG shape), feathering to 0.0 over --feather-cm into the
     head/neck region measured from the face-region boundary -- the cranium /
     back / neck keep ICT's clean watertight topology, no shrinkwrap lumps.
     Eyes/mouth/nose vicinities and the EARS (thin structures = ICP-nearest-
     point garbage) stay protected; interior stays EXACTLY put.

VOLUME MATCHING (--prescale, face mode only): the bald ICT fit is measurably
smaller than the haired clay (head-band bbox ratios ~1.25-1.35). Before the
shrinkwrap, the fitted neutral is scaled about the exterior centroid by
per-axis factors measured on the head band (y >= chin) so its bbox matches
the clay's; the composite output is
    refined = fitted + m_pre*delta_prescale + m_sw*delta_shrinkwrap
where m_sw = m_pre * face_mask. Face verts (m_pre=m_sw~1) land exactly ON the
clay surface; the back (m_sw~0) keeps the smoothly-inflated ICT shape; the
protected features and the interior (eyeballs/teeth -- which must stay
registered with the lids/lips) stay at the FITTED positions.

Same 26719-vert topology, asserted. --no-shrinkwrap passes fitted through
unchanged (A/B fallback).

Outputs under out/refine/: refined_neutral.npy/.obj, refine_stats.json,
refine_debug.npz (masks + displacement magnitudes, consumed by s3c verify).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (EXTERIOR_END, ICT_REGIONS, P, assert_topology, die,  # noqa: E402
                    edges_from_faces, faces_as_lists, min_dist_to_points,
                    out_dir, smooth_field, smoothstep, write_obj)
from mp_ibug68 import IBUG_EYES, IBUG_MOUTH, IBUG_NOSE  # noqa: E402

import bpy  # noqa: E402

FACE_END = ICT_REGIONS["face"][1]          # 9409
HN0, HN1 = ICT_REGIONS["head_neck"]        # 9409, 11248


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(description="s3b shrinkwrap refine")
    ap.add_argument("--out", default=P.OUT)
    ap.add_argument("--clay-npz", default=None,
                    help="aligned clay npz (default out/clay/clay_aligned.npz;"
                         " pass out/clay/clay_sg_aligned.npz for TripoSG)")
    ap.add_argument("--no-shrinkwrap", action="store_true")
    ap.add_argument("--clay-smooth-iters", type=int, default=25)
    ap.add_argument("--clay-smooth-factor", type=float, default=0.5)
    ap.add_argument("--cutoff", type=float, default=6.0, help="cm; discard bigger raw deltas")
    ap.add_argument("--max-disp", type=float, default=3.0, help="cm; clamp delta magnitude")
    ap.add_argument("--smooth-iters", type=int, default=12)
    ap.add_argument("--smooth-lam", type=float, default=0.6)
    ap.add_argument("--weights", choices=("legacy", "face"), default="legacy",
                    help="legacy: scalp 1.0 / face --face-weight (TripoSR). "
                         "face: face 1.0 feathering to 0 over head/neck (TripoSG)")
    ap.add_argument("--face-weight", type=float, default=0.35,
                    help="[legacy] shrink influence on the face region")
    ap.add_argument("--feather-cm", type=float, default=3.0,
                    help="[face] feather width from the face-region boundary")
    ap.add_argument("--prescale", choices=("none", "yz", "xyz", "uniform"),
                    default="none",
                    help="[face] bbox volume-match of the fitted neutral to "
                         "the clay before shrinkwrap (measured per axis on "
                         "the head band, applied about the exterior centroid)")
    ap.add_argument("--protect-r0", type=float, default=1.2, help="cm, eyes/mouth zero radius")
    ap.add_argument("--protect-r1", type=float, default=3.0, help="cm, eyes/mouth full radius")
    ap.add_argument("--nose-r0", type=float, default=0.5)
    ap.add_argument("--nose-r1", type=float, default=1.5)
    ap.add_argument("--ear-r0", type=float, default=2.0, help="cm [face mode]")
    ap.add_argument("--ear-r1", type=float, default=3.5, help="cm [face mode]")
    return ap.parse_args(argv)


def new_object(name, verts, faces):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts.tolist(), [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def prescale_factors(fitted, clay_v, mode):
    """Per-axis bbox volume-match factors, measured on the head band
    (y >= ~chin level = 25th pct of face-region y), robust p2..p98 extents,
    applied about the exterior centroid. Returns (centroid, k(3,), y_chin)."""
    ext = fitted[:EXTERIOR_END]
    c = ext.mean(axis=0)
    y_chin = float(np.percentile(fitted[:FACE_END, 1], 25.0))
    band_i = ext[ext[:, 1] >= y_chin]
    band_c = clay_v[clay_v[:, 1] >= y_chin]
    if len(band_c) < 100:
        die("prescale: clay has <100 verts above chin level -- alignment broken?")
    e_i = np.percentile(band_i, 98, axis=0) - np.percentile(band_i, 2, axis=0)
    e_c = np.percentile(band_c, 98, axis=0) - np.percentile(band_c, 2, axis=0)
    k = e_c / np.maximum(e_i, 1e-9)
    if mode == "yz":
        k[0] = 1.0
    elif mode == "uniform":
        k[:] = float(np.sqrt(k[1] * k[2]))
    for ax, kk in zip("xyz", k):
        if not (0.7 <= kk <= 1.7):
            die(f"prescale factor k{ax}={kk:.3f} outside [0.7,1.7] -- "
                "clay/ICT band extents are not sane, refusing")
    return c, k, y_chin


def main():
    args = parse_args()
    t0 = time.time()
    od = out_dir(args.out, "refine")
    fit_dir = Path(args.out) / "fit"

    fitted = np.load(fit_dir / "fitted_neutral.npy")
    assert_topology(fitted, "fitted_neutral (s3b input)")
    topo = np.load(fit_dir / "topology.npz")
    faces_flat, faces_off = topo["faces_flat"], topo["faces_off"]
    lmk_verts = topo["lmk_verts"]

    if args.no_shrinkwrap:
        print("[s3b] --no-shrinkwrap: passing fitted neutral through unchanged")
        refined = fitted.copy()
        stats = {"mode": "passthrough"}
    else:
        if args.prescale != "none" and args.weights != "face":
            die("--prescale requires --weights face (legacy path unchanged)")
        clay_npz = Path(args.clay_npz) if args.clay_npz \
            else Path(args.out) / "clay" / "clay_aligned.npz"
        clay = np.load(clay_npz)
        cv, cf = clay["verts"].astype(np.float64), clay["faces"]
        print(f"[s3b] clay target: {clay_npz.name}  {len(cv)} v / {len(cf)} f  "
              f"weights={args.weights} prescale={args.prescale}")

        # ---- volume matching (prescale) -----------------------------------
        delta_pre = np.zeros_like(fitted)
        pre_info = {}
        if args.prescale != "none":
            c, k, y_chin = prescale_factors(fitted, cv, args.prescale)
            Vs = c + (fitted - c) * k
            delta_pre = Vs - fitted
            pre_info = {"mode": args.prescale, "k": np.round(k, 4).tolist(),
                        "centroid": np.round(c, 2).tolist(),
                        "band_y_chin": round(y_chin, 2),
                        "delta_pre_max_cm":
                            float(np.linalg.norm(delta_pre, axis=1).max())}
            print(f"[s3b] prescale k={np.round(k, 3)} about c={np.round(c, 2)} "
                  f"(band y>={y_chin:.1f})  max|delta_pre|="
                  f"{pre_info['delta_pre_max_cm']:.2f} cm")
        else:
            Vs = fitted

        # ---- shrinkwrap the (pre-scaled) fitted onto the smoothed clay ----
        bpy.ops.wm.read_factory_settings(use_empty=True)
        clay_obj = new_object("clay", cv, [tuple(int(i) for i in f) for f in cf])
        sm = clay_obj.modifiers.new("smooth", "SMOOTH")
        sm.factor = args.clay_smooth_factor
        sm.iterations = args.clay_smooth_iters
        print(f"[s3b] clay Smooth modifier: factor={sm.factor} iters={sm.iterations} "
              "(evaluated in depsgraph; shrinkwrap sees the SMOOTHED clay)")

        fit_obj = new_object("fitted", Vs, faces_as_lists(faces_flat, faces_off))
        sw = fit_obj.modifiers.new("shrink", "SHRINKWRAP")
        sw.target = clay_obj
        sw.wrap_method = "NEAREST_SURFACEPOINT"

        dg = bpy.context.evaluated_depsgraph_get()
        ev = fit_obj.evaluated_get(dg)
        me = ev.to_mesh()
        if len(me.vertices) != len(fitted):
            print(f"[s3b FATAL] evaluated mesh has {len(me.vertices)} verts "
                  f"!= {len(fitted)} -- topology drift, STOP")
            sys.exit(1)
        wrapped = np.empty(len(fitted) * 3, dtype=np.float64)
        me.vertices.foreach_get("co", wrapped)
        wrapped = wrapped.reshape(-1, 3)
        ev.to_mesh_clear()

        delta = wrapped - Vs
        mag = np.linalg.norm(delta, axis=1)
        n_cut = int((mag > args.cutoff).sum())
        delta[mag > args.cutoff] = 0.0  # clay absent there (e.g. below neck)
        mag = np.linalg.norm(delta, axis=1)
        over = mag > args.max_disp
        delta[over] *= (args.max_disp / mag[over])[:, None]

        edges = edges_from_faces(faces_flat, faces_off)
        delta_s = smooth_field(delta, edges, args.smooth_iters, args.smooth_lam)

        # ---- feature protection (both modes): eyes/mouth/nose stay FITTED
        eye_mouth = fitted[lmk_verts[IBUG_EYES + IBUG_MOUTH]]
        nose = fitted[lmk_verts[IBUG_NOSE]]
        prot = smoothstep(min_dist_to_points(fitted, eye_mouth),
                          args.protect_r0, args.protect_r1)
        prot *= smoothstep(min_dist_to_points(fitted, nose),
                           args.nose_r0, args.nose_r1)

        face_mask = None
        ear_info = {}
        if args.weights == "face":
            # ears: thin structures, nearest-surface-point garbage -- find the
            # extreme-|x| exterior verts at eye height, protect around them
            eye_y = float(fitted[lmk_verts[IBUG_EYES], 1].mean())
            band = np.where(np.abs(fitted[:EXTERIOR_END, 1] - eye_y) < 3.0)[0]
            ear_r = fitted[band[np.argmax(fitted[band, 0])]]
            ear_l = fitted[band[np.argmin(fitted[band, 0])]]
            prot *= smoothstep(min_dist_to_points(fitted, np.stack([ear_l, ear_r])),
                               args.ear_r0, args.ear_r1)
            ear_info = {"ear_left": np.round(ear_l, 2).tolist(),
                        "ear_right": np.round(ear_r, 2).tolist()}

            # face mask: 1.0 on the face region, feather to 0 over head/neck
            # measured from the face/head_neck boundary verts
            e0, e1 = edges[:, 0], edges[:, 1]
            cross = ((e0 < FACE_END) & (e1 >= HN0) & (e1 < HN1)) | \
                    ((e1 < FACE_END) & (e0 >= HN0) & (e0 < HN1))
            bverts = np.unique(np.where(e0[cross] < FACE_END,
                                        e0[cross], e1[cross]))
            face_mask = np.zeros(len(fitted))
            face_mask[:FACE_END] = 1.0
            d_hn = min_dist_to_points(fitted[HN0:HN1], fitted[bverts])
            face_mask[HN0:HN1] = 1.0 - smoothstep(d_hn, 0.0, args.feather_cm)

            m_pre = np.zeros(len(fitted))
            m_pre[:EXTERIOR_END] = prot[:EXTERIOR_END]
            m_sw = m_pre * face_mask
            m_pre = smooth_field(m_pre, edges, 5, 0.5)
            m_sw = smooth_field(m_sw, edges, 5, 0.5)
            m_pre[EXTERIOR_END:] = 0.0          # interior stays EXACTLY put
            m_sw[EXTERIOR_END:] = 0.0
        else:
            # ---- legacy mask: regions + feature protection (TripoSR path)
            w = np.zeros(len(fitted))
            w[:EXTERIOR_END] = 1.0              # face + head/neck skin only
            w[:FACE_END] *= args.face_weight    # gentle on the face region
            w *= prot
            w = smooth_field(w, edges, 5, 0.5)
            w[EXTERIOR_END:] = 0.0              # interior stays EXACTLY put
            m_sw = w
            m_pre = np.zeros(len(fitted))

        disp_sw = m_sw[:, None] * delta_s
        disp_pre = m_pre[:, None] * delta_pre
        disp = disp_pre + disp_sw
        refined = fitted + disp
        dmag = np.linalg.norm(disp, axis=1)
        sw_mag = np.linalg.norm(disp_sw, axis=1)
        pre_mag = np.linalg.norm(disp_pre, axis=1)
        # back region = head/neck verts beyond the feather band: shrinkwrap
        # displacement there must be ~0 (ICT stays clean)
        if face_mask is not None:
            back_sel = np.zeros(len(fitted), dtype=bool)
            back_sel[HN0:HN1] = face_mask[HN0:HN1] < 0.01
            back_max_sw = float(sw_mag[back_sel].max()) if back_sel.any() else 0.0
        else:
            back_max_sw = None
        stats = {
            "mode": "shrinkwrap",
            "weights": args.weights,
            "clay_npz": str(clay_npz),
            "prescale": pre_info or {"mode": "none"},
            "ears": ear_info,
            "raw_delta_discarded_over_cutoff": n_cut,
            "clamped_over_max_disp": int(over.sum()),
            "disp_mean_cm": float(dmag.mean()),
            "disp_max_cm": float(dmag.max()),
            "disp_mean_face_cm": float(dmag[:FACE_END].mean()),
            "disp_mean_scalp_cm": float(dmag[HN0:HN1].mean()),
            "sw_disp_mean_face_cm": float(sw_mag[:FACE_END].mean()),
            "sw_disp_max_cm": float(sw_mag.max()),
            "pre_disp_max_cm": float(pre_mag.max()),
            "back_region_max_sw_disp_cm": back_max_sw,
            "verts_moved_over_1mm": int((dmag > 0.1).sum()),
            "params": vars(args),
        }
        print(f"[s3b] disp: mean={stats['disp_mean_cm']:.2f}cm "
              f"max={stats['disp_max_cm']:.2f}cm  face-mean="
              f"{stats['disp_mean_face_cm']:.2f}cm scalp-mean="
              f"{stats['disp_mean_scalp_cm']:.2f}cm  "
              f">1mm: {stats['verts_moved_over_1mm']}/{len(fitted)}")
        if back_max_sw is not None:
            print(f"[s3b] back region (head/neck beyond feather): "
                  f"max shrinkwrap disp = {back_max_sw:.4f} cm")
        np.savez(od / "refine_debug.npz",
                 mask=m_sw, disp_mag=dmag,                    # legacy keys
                 m_sw=m_sw, m_pre=m_pre,
                 sw_disp_mag=sw_mag, pre_disp_mag=pre_mag,
                 **({"face_mask": face_mask} if face_mask is not None else {}))

    assert_topology(refined, "refined_neutral")
    np.save(od / "refined_neutral.npy", refined)
    write_obj(od / "refined_neutral.obj", refined, faces_flat, faces_off,
              comment="newstack s3b refined ICT neutral (cm, +Y up, +Z front)")
    with open(od / "refine_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[s3b] refined_neutral -> {od / 'refined_neutral.obj'}")
    print(f"[s3b] DONE in {time.time()-t0:.1f}s")


main()
