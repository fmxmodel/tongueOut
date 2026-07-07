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

  1. coarse init sweep: 24 proper axis-permutation rotations x centroid +
     RMS-radius scale, each scored by a short trimmed ICP (TripoSG is
     ~[-1,1]-normalized with its own axis convention -- scale AND orientation
     are unknown);
  2. fine trimmed ICP from the best init: Umeyama similarity (scale+R+t)
     refit each iteration on nearest-neighbour pairs, worst --trim quantile
     of correspondences dropped (hair interpretation + bust cropping differ
     between the two generators);
  3. FACE-POLISH ICP: the global fit registers the blob, but the two
     generators disagree most about hair, which biases the face by ~1 cm.
     A final trimmed ICP registers the SG points that lie near the fitted
     ICT FACE region [0,9409) against those face verts -- the exact surface
     the s3b shrinkwrap will pull the ICT face onto.

GATED (dies loudly, never ships a mis-aligned clay):
  - trimmed inlier RMSE  <= --max-rms cm  (global fit)
  - fitted-ICT INNER-face landmark verts (brows/nose/eyes/mouth, iBUG 17-67)
    -> aligned-SG-surface mean distance <= --max-face-dist cm  (the face MUST
    register; the jaw CONTOUR may legitimately sit under hair and is only
    reported)
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

    rng = np.random.default_rng(args.seed)
    src = v_raw[rng.choice(len(v_raw), min(args.n_src, len(v_raw)), replace=False)]
    src_c = src[rng.choice(len(src), min(args.coarse_src, len(src)), replace=False)]
    tree = cKDTree(tgt_v)

    mu_s, mu_t = src.mean(0), tgt_v.mean(0)
    sig_s = float(np.linalg.norm(src - mu_s, axis=1).mean())
    sig_t = float(np.linalg.norm(tgt_v - mu_t, axis=1).mean())
    s0 = sig_t / max(sig_s, 1e-12)

    # ---- coarse init sweep: 24 proper rotations, short trimmed ICP each
    best = None  # (rms, s, R, t, cand_idx)
    for ci, R0 in enumerate(proper_rotations_24()):
        t_init = mu_t - s0 * (R0 @ mu_s)
        s_c, R_c, t_c, rms_c, _ = icp_trimmed(
            src_c, tgt_v, tree, s0, R0, t_init,
            iters=args.coarse_iters, trim=0.30)
        if best is None or rms_c < best[0]:
            best = (rms_c, s_c, R_c, t_c, ci)
    rms_c, s_f, R_f, t_f, ci = best
    print(f"[s3sg] coarse best: candidate #{ci} rms={rms_c:.3f} cm")

    # ---- fine trimmed ICP from the best init
    s_f, R_f, t_f, rms_f, it_f = icp_trimmed(
        src, tgt_v, tree, s_f, R_f, t_f,
        iters=args.fine_iters, trim=args.trim)
    print(f"[s3sg] fine ICP: {it_f} iters  inlier rms={rms_f:.3f} cm  "
          f"scale={s_f:.4f}")

    # global-fit metric (vs TripoSR) BEFORE the face polish
    d_all, _ = tree.query(s_f * (src @ R_f.T) + t_f, workers=-1)
    inlier = d_all <= np.quantile(d_all, 1.0 - args.trim)
    inlier_rms = float(np.sqrt((d_all[inlier] ** 2).mean()))
    median_nn = float(np.median(d_all))

    # ---- face-polish ICP against the fitted ICT FACE region --------------
    fitted = np.load(Path(args.out) / "fit" / "fitted_neutral.npy")
    lmk_verts = np.load(Path(args.out) / "fit" / "topology.npz")["lmk_verts"]
    face_tgt = fitted[:FACE_END]
    tree_face = cKDTree(face_tgt)
    n_face_pairs = 0
    for _ in range(args.face_iters):
        cur = s_f * (src @ R_f.T) + t_f
        d, j = tree_face.query(cur, workers=-1)
        keep = d <= args.face_cap
        if keep.sum() < 500:
            die(f"face polish: only {int(keep.sum())} SG points within "
                f"{args.face_cap} cm of the fitted face -- global ICP is off")
        thr = np.quantile(d[keep], 1.0 - args.face_trim)
        keep &= d <= thr
        n_face_pairs = int(keep.sum())
        s_f, R_f, t_f = umeyama(src[keep], face_tgt[j[keep]])
    cur = s_f * (src @ R_f.T) + t_f
    d, _ = tree_face.query(cur, workers=-1)
    d_in = d[d <= args.face_cap]
    face_rms = float(np.sqrt((d_in[d_in <= np.quantile(d_in, 1.0 - args.face_trim)] ** 2).mean()))
    print(f"[s3sg] face polish: {n_face_pairs} pairs  "
          f"face inlier rms={face_rms:.3f} cm  scale={s_f:.4f}")

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

    ext_sg = v_aligned.max(0) - v_aligned.min(0)
    ext_ict = fitted[:11248].max(0) - fitted[:11248].min(0)
    y_ratio = float(ext_sg[1] / max(ext_ict[1], 1e-9))

    print(f"[s3sg] metrics: global inlier_rms={inlier_rms:.3f} cm  median_nn="
          f"{median_nn:.3f} cm  median_rev={median_rev:.3f} cm")
    print(f"[s3sg] landmarks -> SG surface: inner mean={inner_mean:.3f} cm "
          f"max={inner_max:.3f} cm  contour mean={contour_mean:.3f} cm")
    print(f"[s3sg] aligned bbox extent={np.round(ext_sg, 2)} cm "
          f"(ICT head y-extent ratio {y_ratio:.2f})")

    failures = []
    if inlier_rms > args.max_rms:
        failures.append(f"global ICP inlier RMSE {inlier_rms:.3f} cm > "
                        f"{args.max_rms} cm")
    if inner_mean > args.max_face_dist:
        failures.append(f"inner-landmark->SG-surface mean {inner_mean:.3f} cm "
                        f"> {args.max_face_dist} cm -- the FACE did not register")
    if not (0.5 <= y_ratio <= 2.0):
        failures.append(f"aligned SG y-extent ratio {y_ratio:.2f} is insane")

    info = {
        "method": "icp-to-aligned-triposr (24-rot coarse init + trimmed "
                  "umeyama ICP) + face-polish ICP vs fitted ICT face",
        "target_npz": str(tgt_npz),
        "coarse_candidate": int(ci),
        "coarse_rms_cm": float(rms_c),
        "fine_iters_run": int(it_f),
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
