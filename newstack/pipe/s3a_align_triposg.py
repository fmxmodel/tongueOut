#!/usr/bin/env python3
"""Stage 3a-SG -- rigid-align the TripoSG clay (GEOMETRY-ONLY, no texture) to
ICT space via trimmed ICP against the ALREADY-ALIGNED TripoSR clay
(system python: trimesh + scipy).

Why not the landmark route (s3a_align_clay.py): TripoSG ships no vertex
colors/texture, so a pytorch3d render is a blank grey head and MediaPipe
cannot find a face on it. Instead we exploit that the TripoSR clay -- SAME
head, SAME photo, similar shape -- is already expressed in ICT space (cm,
+Y up, +Z front) by s3a_align_clay.py. TripoSG then only needs a similarity
transform onto that surface:

  0. CLEANUP: TripoSG reconstructs the photo's BACKGROUND WALL as a huge
     flat slab fused to the person (79%-of-faces component spans the full
     bbox) plus ~230 floating confetti shards and a hair-shard chaos hugging
     the wall. The wall is detected as the dominant area-weighted normal
     direction + the strongest planar-offset peak; the wall band AND
     everything behind it are dropped, then the largest connected component
     is kept. Without this the ICP centroid/scale/trim are all poisoned
     (measured: it locked 90 degrees off);
  1. global trimmed ICP from the IDENTITY-rotation init (measured on this
     pod: TripoSG's canonical frame is already +Y-up / face-toward-+Z, the
     ICT convention; scale is seeded from robust x-widths, translation from
     crown/nose/median-x anchors). The trimmed blob objective is nearly
     pose-invariant for hairy partial heads -- a 24-rotation sweep repeatedly
     locked 90-deg-off poses -- so the sweep is only a FALLBACK, and the
     winner is chosen by the direction-aware front-depth metric, never by
     ICP rms. The ICP target is the SR HEAD BAND (y >= --target-ymin), not
     the full bust, so SR's chest cannot bias the scale;
  2. fine trimmed ICP from the best init: Umeyama similarity (scale+R+t)
     refit each iteration on nearest-neighbour pairs, worst --trim quantile
     of correspondences dropped (hair interpretation + bust cropping differ
     between the two generators);
  3. FACE-POLISH ICP: the global fit registers the blob, but the two
     generators disagree most about hair, which biases the face by ~1 cm.
     A final trimmed RIGID-ONLY ICP (R+t, Kabsch -- scale stays from the
     global fit: a free scale against a partial smooth template collapses,
     measured 13.48 -> 9.70) registers the SG points that lie near the
     fitted ICT FACE region [0,9409) against those face verts -- the exact
     surface the s3b shrinkwrap will pull the ICT face onto.

GATED (dies loudly, never ships a mis-aligned clay):
  - trimmed inlier RMSE  <= --max-rms cm  (global fit)
  - fitted-ICT INNER-face landmark verts (brows/nose/eyes/mouth, iBUG 17-67)
    -> aligned-SG-surface mean distance <= --max-face-dist cm
  - FRONT-DEPTH gate (direction-aware -- nearest-vertex distance alone lets
    a big enveloping blob pass in the WRONG pose): for every inner landmark,
    the front-most SG surface directly above its (x,y) must exist and its z
    must match the fitted landmark z to --max-front-err cm on average
  - aligned bbox y-extent within sane ratio of the fitted ICT head.
Artifacts are saved BEFORE the gates fire so a failure can be inspected.

Outputs under out/clay/ (the TripoSR files are NOT touched -- TripoSR stays
the s5 color source):
  clay_sg_aligned.npz   verts float32 + tri faces int32 (s3b shrinkwrap target)
  clay_sg_aligned.ply   same mesh, for eyeballing
  clay_sg_align.json    S, R, T + ICP metrics + gate values
"""

import argparse
import sys
import time
from itertools import permutations, product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ICT_REGIONS, P, die, out_dir, save_json, umeyama  # noqa: E402

FACE_END = ICT_REGIONS["face"][1]  # 9409


def proper_rotations_24():
    """All 24 proper (det=+1) signed axis-permutation matrices."""
    mats = []
    for perm in permutations(range(3)):
        for signs in product((1.0, -1.0), repeat=3):
            M = np.zeros((3, 3))
            for row, (p, s) in enumerate(zip(perm, signs)):
                M[row, p] = s
            if np.linalg.det(M) > 0.5:
                mats.append(M)
    assert len(mats) == 24
    return mats


def strip_background_slab(v, f, eps=0.06, ang_deg=25.0):
    """Remove the reconstructed background wall (+ the hair-shard chaos that
    hugs it, + confetti) from a TripoSG mesh. Returns (verts, faces, info).
    Raw (pre-alignment) units.

    The wall is the dominant direction of the area-weighted normal outer-
    product (a wall concentrates area in ONE +-direction; a head spreads it
    over the sphere -- and decimation collapses the flat wall into FEW HUGE
    faces, so area weighting is essential). Its depth is the strongest
    area-weighted histogram peak of wall-normal face offsets. The wall is
    WARPED, so a thin plane band is not enough (measured: 8.7k of ~30k wall
    faces): orient the normal so the person (vertex-median side) is positive
    and drop EVERYTHING at or behind the wall band, then keep the largest
    connected component (kills shards/confetti)."""
    import trimesh
    m = trimesh.Trimesh(vertices=v, faces=f, process=False)
    fn = m.face_normals
    fa = m.area_faces
    M = (fn * fa[:, None]).T @ fn                  # 3x3 direction mass
    _, V_ = np.linalg.eigh(M)
    n = V_[:, -1]                                  # dominant unsigned normal
    cos_lim = np.cos(np.radians(ang_deg))
    cen = v[f].mean(axis=1)                        # face centroids
    proj = cen @ n
    wallish = np.abs(fn @ n) > cos_lim
    hist, edges_ = np.histogram(proj[wallish], bins=200,
                                weights=fa[wallish])
    d_star = 0.5 * (edges_[np.argmax(hist)] + edges_[np.argmax(hist) + 1])
    if np.median(v @ n) < d_star:                  # person on negative side
        n, d_star, proj = -n, -d_star, -proj       # -> flip toward person
    behind = proj < d_star + eps                   # wall band + all behind it
    m.update_faces(~behind)
    comps = m.split(only_watertight=False)
    if not len(comps):
        die("slab strip removed everything -- eps/angle too aggressive")
    main_c = max(comps, key=lambda c: len(c.faces))
    info = {"slab_normal_toward_person": np.round(n, 4).tolist(),
            "slab_offset": float(d_star),
            "faces_removed_at_or_behind_wall": int(behind.sum()),
            "components_after_strip": int(len(comps)),
            "kept_faces": int(len(main_c.faces)),
            "kept_verts": int(len(main_c.vertices))}
    return (np.asarray(main_c.vertices, dtype=np.float64),
            np.asarray(main_c.faces, dtype=np.int64), info)


def front_depth_errors(fitted_lmk, sg_v, rad=0.6):
    """Direction-aware registration check: for each landmark point, the
    front-most (max z) SG vertex within +-rad cm in (x,y) is compared to the
    landmark's z. Returns (|dz| per covered landmark, n_missing)."""
    errs, miss = [], 0
    for p in fitted_lmk:
        m = (np.abs(sg_v[:, 0] - p[0]) < rad) & (np.abs(sg_v[:, 1] - p[1]) < rad)
        if not m.any():
            miss += 1
            continue
        errs.append(abs(float(sg_v[m, 2].max()) - float(p[2])))
    return np.asarray(errs), miss


def rigid_fit(src, dst):
    """Kabsch: rotation+translation only (NO scale) minimizing ||R@src+t-dst||."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, _, Vt = np.linalg.svd(xd.T @ xs / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    return R, mu_d - R @ mu_s


def icp_trimmed(src, tgt, tree, s, R, t, iters, trim, tol=1e-5):
    """Trimmed point-to-point ICP; similarity refit by Umeyama each iter.
    Returns (s, R, t, inlier_rms, iters_run). src/tgt (N,3)/(M,3)."""
    rms = None
    it = 0
    for it in range(1, iters + 1):
        cur = s * (src @ R.T) + t
        d, j = tree.query(cur, workers=-1)
        thr = np.quantile(d, 1.0 - trim)
        keep = d <= thr
        new_rms = float(np.sqrt((d[keep] ** 2).mean()))
        s, R, t = umeyama(src[keep], tgt[j[keep]])
        if rms is not None and abs(rms - new_rms) < tol:
            rms = new_rms
            break
        rms = new_rms
    return s, R, t, rms, it


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clay-sg",
                    default="/workspace/newstack/out_triposg/random-person_triposg_300k.glb")
    ap.add_argument("--out", default=P.OUT)
    ap.add_argument("--target-npz", default=None,
                    help="aligned TripoSR clay npz (default out/clay/clay_aligned.npz)")
    ap.add_argument("--n-src", type=int, default=20000)
    ap.add_argument("--coarse-src", type=int, default=4000)
    ap.add_argument("--coarse-iters", type=int, default=10)
    ap.add_argument("--fine-iters", type=int, default=80)
    ap.add_argument("--target-ymin", type=float, default=-8.0,
                    help="cm; crop the SR target below this (head band) so "
                         "the chest cannot bias the ICP scale")
    ap.add_argument("--trim", type=float, default=0.25,
                    help="fraction of worst NN correspondences dropped per iter")
    ap.add_argument("--face-iters", type=int, default=40,
                    help="face-polish ICP iterations (0 = disable)")
    ap.add_argument("--face-cap", type=float, default=2.0,
                    help="cm; SG points farther than this from the fitted "
                         "face are excluded from the polish")
    ap.add_argument("--face-trim", type=float, default=0.30)
    ap.add_argument("--max-rms", type=float, default=1.2,
                    help="cm; GATE on trimmed inlier RMSE (global fit)")
    ap.add_argument("--max-face-dist", type=float, default=1.0,
                    help="cm; GATE on mean INNER-landmark->SG-surface distance")
    ap.add_argument("--max-front-err", type=float, default=1.2,
                    help="cm; GATE on mean front-depth error at inner landmarks")
    ap.add_argument("--slab-eps", type=float, default=0.06,
                    help="raw units; wall band half-width (everything at or "
                         "behind the wall is dropped)")
    ap.add_argument("--no-slab-strip", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    od = out_dir(args.out, "clay")
    from scipy.spatial import cKDTree

    tgt_npz = Path(args.target_npz) if args.target_npz \
        else Path(args.out) / "clay" / "clay_aligned.npz"
    if not tgt_npz.is_file():
        die(f"{tgt_npz} missing -- run s3a_align_clay.py (TripoSR landmark "
            "alignment) first; it is the ICP target AND the s5 color source")
    tz = np.load(tgt_npz)
    tgt_v = tz["verts"].astype(np.float64)
    print(f"[s3sg] ICP target (TripoSR aligned): {len(tgt_v)} v")

    import trimesh
    m = trimesh.load(args.clay_sg, process=False, force="mesh")
    v_raw = np.asarray(m.vertices, dtype=np.float64)
    faces = np.asarray(m.faces, dtype=np.int64)
    if len(faces) == 0:
        die(f"TripoSG clay {args.clay_sg} has NO faces")
    print(f"[s3sg] TripoSG clay: {len(v_raw)} v / {len(faces)} f  "
          f"extent={np.round(v_raw.max(0) - v_raw.min(0), 2)}")

    strip_info = {}
    if not args.no_slab_strip:
        v_raw, faces, strip_info = strip_background_slab(
            v_raw, faces, eps=args.slab_eps)
        print(f"[s3sg] slab strip: removed "
              f"{strip_info['faces_removed_at_or_behind_wall']} faces at/"
              f"behind the wall (n={strip_info['slab_normal_toward_person']}, "
              f"d={strip_info['slab_offset']:.3f}), kept largest of "
              f"{strip_info['components_after_strip']} comps -> "
              f"{strip_info['kept_verts']} v / {strip_info['kept_faces']} f  "
              f"extent={np.round(v_raw.max(0) - v_raw.min(0), 2)}")

    rng = np.random.default_rng(args.seed)
    src = v_raw[rng.choice(len(v_raw), min(args.n_src, len(v_raw)), replace=False)]
    src_c = src[rng.choice(len(src), min(args.coarse_src, len(src)), replace=False)]
    # head-band target: SR's chest must not bias the ICP scale
    tgt_head = tgt_v[tgt_v[:, 1] >= args.target_ymin]
    tree = cKDTree(tgt_head)

    fitted = np.load(Path(args.out) / "fit" / "fitted_neutral.npy")
    lmk_verts = np.load(Path(args.out) / "fit" / "topology.npz")["lmk_verts"]
    face_tgt = fitted[:FACE_END]
    tree_face = cKDTree(face_tgt)

    def face_polish(s_f, R_f, t_f):
        """Rigid-only trimmed ICP of near-face SG points onto the fitted ICT
        face. Returns (s,R,t, n_pairs, face_rms) or None if too few pairs."""
        n_pairs = 0
        for _ in range(args.face_iters):
            cur = s_f * (src @ R_f.T) + t_f
            d, j = tree_face.query(cur, workers=-1)
            keep = d <= args.face_cap
            if keep.sum() < 500:
                return None
            thr = np.quantile(d[keep], 1.0 - args.face_trim)
            keep &= d <= thr
            n_pairs = int(keep.sum())
            # rigid-only refit in the SCALED source frame (scale frozen:
            # a free scale against a partial template collapses)
            R_d, t_d = rigid_fit(s_f * (src[keep] @ R_f.T) + t_f,
                                 face_tgt[j[keep]])
            R_f = R_d @ R_f
            t_f = R_d @ t_f + t_d
        d, _ = tree_face.query(s_f * (src @ R_f.T) + t_f, workers=-1)
        d_in = d[d <= args.face_cap]
        rms = float(np.sqrt((d_in[d_in <= np.quantile(
            d_in, 1.0 - args.face_trim)] ** 2).mean()))
        return s_f, R_f, t_f, n_pairs, rms

    inner_lmk = fitted[lmk_verts[17:]]

    def align_from(s0, R0, t0):
        """Global trimmed ICP + rigid face polish; scored by the
        direction-aware front-depth mean (NOT by ICP rms)."""
        s_f, R_f, t_f, rms_f, it_f = icp_trimmed(
            src, tgt_head, tree, s0, R0, t0,
            iters=args.fine_iters, trim=args.trim)
        pol = face_polish(s_f, R_f, t_f)
        if pol is None:
            return None
        s_f, R_f, t_f, n_pairs, face_rms = pol
        fd_err, fd_miss = front_depth_errors(
            inner_lmk, s_f * (src @ R_f.T) + t_f)
        fd = float(fd_err.mean()) if len(fd_err) else np.inf
        return {"s": s_f, "R": R_f, "t": t_f, "global_rms": rms_f,
                "iters": it_f, "n_pairs": n_pairs, "face_rms": face_rms,
                "fd_mean": fd, "fd_miss": fd_miss}

    # ---- primary init: IDENTITY rotation (TripoSG canonical frame is
    # +Y-up / face-toward-+Z, measured), robust scale + anchor translation
    def widths(v):
        return np.percentile(v[:, 0], 95) - np.percentile(v[:, 0], 5)

    def anchor(v):  # median x, crown y, nose z -- stable on both meshes
        return np.array([np.median(v[:, 0]),
                         np.percentile(v[:, 1], 98),
                         np.percentile(v[:, 2], 98)])

    s0 = widths(tgt_head) / max(widths(src), 1e-12)
    t0 = anchor(tgt_head) - s0 * anchor(src)
    print(f"[s3sg] identity init: s0={s0:.3f} t0={np.round(t0, 2)}")
    result = align_from(s0, np.eye(3), t0)
    init_used = "identity"
    if result is not None:
        print(f"[s3sg] identity path: global rms={result['global_rms']:.3f} "
              f"cm  face rms={result['face_rms']:.3f} cm  front-depth mean="
              f"{result['fd_mean']:.3f} cm (miss {result['fd_miss']})")

    # ---- fallback: 24-rotation coarse sweep, only if identity fails the
    # front-depth criterion
    if result is None or result["fd_mean"] > args.max_front_err \
            or result["fd_miss"] > 0:
        print("[s3sg] identity init unsatisfying -- trying 24-rotation sweep")
        mu_s, mu_t = src.mean(0), tgt_head.mean(0)
        sig_s = float(np.linalg.norm(src - mu_s, axis=1).mean())
        sig_t = float(np.linalg.norm(tgt_head - mu_t, axis=1).mean())
        s0s = sig_t / max(sig_s, 1e-12)
        best = None
        for ci, R0 in enumerate(proper_rotations_24()):
            t_init = mu_t - s0s * (R0 @ mu_s)
            s_c, R_c, t_c, rms_c, _ = icp_trimmed(
                src_c, tgt_head, tree, s0s, R0, t_init,
                iters=args.coarse_iters, trim=0.30)
            if best is None or rms_c < best[0]:
                best = (rms_c, s_c, R_c, t_c, ci)
        rms_c, s_b, R_b, t_b, ci = best
        print(f"[s3sg] sweep coarse best: candidate #{ci} rms={rms_c:.3f} cm")
        res_b = align_from(s_b, R_b, t_b)
        if res_b is not None:
            print(f"[s3sg] sweep path: front-depth mean={res_b['fd_mean']:.3f}"
                  f" cm (miss {res_b['fd_miss']})")
        if result is None or (res_b is not None
                              and res_b["fd_mean"] < result["fd_mean"]):
            result, init_used = res_b, f"sweep#{ci}"
    if result is None:
        die("no alignment path produced a usable transform")

    s_f, R_f, t_f = result["s"], result["R"], result["t"]
    inlier_rms = result["global_rms"]
    n_face_pairs, face_rms = result["n_pairs"], result["face_rms"]
    d_all, _ = tree.query(s_f * (src @ R_f.T) + t_f, workers=-1)
    median_nn = float(np.median(d_all))
    print(f"[s3sg] chosen init: {init_used}  scale={s_f:.4f}  "
          f"face polish pairs={n_face_pairs} rms={face_rms:.3f} cm")

    # ---- final metrics on the full transform
    v_aligned = s_f * (v_raw @ R_f.T) + t_f
    # reverse direction: TripoSR sample -> aligned SG surface
    tree_sg = cKDTree(v_aligned)
    rev_idx = rng.choice(len(tgt_v), min(20000, len(tgt_v)), replace=False)
    d_rev, _ = tree_sg.query(tgt_v[rev_idx], workers=-1)
    median_rev = float(np.median(d_rev))

    # face gate: the fitted ICT landmark verts must lie ON the aligned
    # surface. INNER landmarks (brows/nose/eyes/mouth, iBUG 17-67) gate;
    # the jaw CONTOUR (0-16) may sit under hair -> report only.
    d_lmk, _ = tree_sg.query(fitted[lmk_verts], workers=-1)
    inner_mean = float(d_lmk[17:].mean())
    inner_max = float(d_lmk[17:].max())
    contour_mean = float(d_lmk[:17].mean())

    # direction-aware front-depth check at the inner landmarks
    fd_err, fd_miss = front_depth_errors(fitted[lmk_verts[17:]], v_aligned)
    fd_mean = float(fd_err.mean()) if len(fd_err) else np.inf
    fd_max = float(fd_err.max()) if len(fd_err) else np.inf

    ext_sg = v_aligned.max(0) - v_aligned.min(0)
    ext_ict = fitted[:11248].max(0) - fitted[:11248].min(0)
    y_ratio = float(ext_sg[1] / max(ext_ict[1], 1e-9))

    print(f"[s3sg] metrics: global inlier_rms={inlier_rms:.3f} cm  median_nn="
          f"{median_nn:.3f} cm  median_rev={median_rev:.3f} cm")
    print(f"[s3sg] landmarks -> SG surface: inner mean={inner_mean:.3f} cm "
          f"max={inner_max:.3f} cm  contour mean={contour_mean:.3f} cm")
    print(f"[s3sg] front-depth @inner landmarks: mean={fd_mean:.3f} cm "
          f"max={fd_max:.3f} cm  missing={fd_miss}/51")
    print(f"[s3sg] aligned bbox extent={np.round(ext_sg, 2)} cm "
          f"(ICT head y-extent ratio {y_ratio:.2f})")

    failures = []
    if inlier_rms > args.max_rms:
        failures.append(f"global ICP inlier RMSE {inlier_rms:.3f} cm > "
                        f"{args.max_rms} cm")
    if inner_mean > args.max_face_dist:
        failures.append(f"inner-landmark->SG-surface mean {inner_mean:.3f} cm "
                        f"> {args.max_face_dist} cm -- the FACE did not register")
    if fd_miss > 0:
        failures.append(f"{fd_miss} inner landmarks have NO SG surface above "
                        "them (wrong pose or missing face)")
    if fd_mean > args.max_front_err:
        failures.append(f"front-depth mean {fd_mean:.3f} cm > "
                        f"{args.max_front_err} cm -- face surface is not where "
                        "the fitted face is (pose/scale wrong)")
    if not (0.5 <= y_ratio <= 2.0):
        failures.append(f"aligned SG y-extent ratio {y_ratio:.2f} is insane")

    info = {
        "method": "slab-strip + identity-init trimmed ICP vs SR head band "
                  "(24-rot sweep fallback, chosen by front-depth) + rigid "
                  "face-polish vs fitted ICT face",
        "target_npz": str(tgt_npz),
        "init_used": init_used,
        "slab_strip": strip_info,
        "front_depth_mean_cm": fd_mean,
        "front_depth_max_cm": fd_max,
        "front_depth_missing": int(fd_miss),
        "trim": args.trim,
        "global_inlier_rms_cm": inlier_rms,
        "median_nn_cm": median_nn,
        "median_rev_cm": median_rev,
        "face_polish_pairs": n_face_pairs,
        "face_polish_inlier_rms_cm": face_rms,
        "lmk_inner_to_surface_mean_cm": inner_mean,
        "lmk_inner_to_surface_max_cm": inner_max,
        "lmk_contour_to_surface_mean_cm": contour_mean,
        "y_extent_ratio_vs_ict": y_ratio,
        "S": float(s_f), "R": R_f.tolist(), "T": np.asarray(t_f).tolist(),
        "clay_verts": int(len(v_aligned)), "clay_faces": int(len(faces)),
        "gates": {"max_rms_cm": args.max_rms,
                  "max_face_dist_cm": args.max_face_dist},
        "gates_passed": not failures,
        "failures": failures,
    }
    # save artifacts BEFORE gating so failures can be inspected
    save_json(od / "clay_sg_align.json", info)
    np.savez(od / "clay_sg_aligned.npz",
             verts=v_aligned.astype(np.float32), faces=faces.astype(np.int32))
    trimesh.Trimesh(vertices=v_aligned, faces=faces, process=False)\
        .export(od / "clay_sg_aligned.ply")
    print(f"[s3sg] aligned TripoSG -> {od / 'clay_sg_aligned.ply'} "
          f"(bbox y: {v_aligned[:, 1].min():.1f}..{v_aligned[:, 1].max():.1f} cm "
          f"vs ICT {fitted[:, 1].min():.1f}..{fitted[:, 1].max():.1f})")
    if failures:
        die("TripoSG alignment GATES FAILED: " + "; ".join(failures))
    print(f"[s3sg] DONE in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
