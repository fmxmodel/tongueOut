#!/usr/bin/env python3
"""Stage 3a-SG -- align the TripoSG clay (GEOMETRY-ONLY, no texture) to ICT
space (system python: trimesh + scipy + [via s3a_align_clay] pytorch3d +
mediapipe).

Pipeline:
  1. CLEANUP: TripoSG reconstructs the photo's BACKGROUND WALL as a huge
     warped slab fused to the person (79%-of-faces component spanning the
     full bbox) plus ~230 confetti shards and a hair-shard chaos hugging the
     wall. The wall is the dominant direction of the area-weighted normal
     outer-product; its depth is the strongest planar-offset histogram peak.
     The normal is oriented toward the person (vertex-median side), the wall
     band AND everything behind it are dropped, and the largest connected
     component is kept. Without this every downstream step is poisoned.
  2. LANDMARK ALIGNMENT: one orthographic FRONT render of the cleaned clay
     (Blender Workbench studio+cavity, exact pixel<->world mapping via
     render_neutral.py --square --dump-cam) -> MediaPipe landmarks on the
     grey sculpture (measured to detect fine on THIS render; the pytorch3d
     Phong grey render stays a near-black silhouette and never detects) ->
     each landmark pixel is unprojected by columnar max-z on the mesh
     itself -> trimmed Umeyama similarity against the fitted ICT landmark
     VERTICES. Landmark correspondences are SEMANTIC, so the scale is
     well-posed -- unlike blob ICP, which was measured to be pose-degenerate
     on hairy partial heads (90-deg-off optima, scale drift 13.5-18.9).
     The front view is TripoSG's canonical convention (+Y-up, face->+Z,
     verified on this pod); no detection => DIE, no silent fallback.
  3. FACE-POLISH: rigid-only trimmed ICP (R+t, Kabsch; scale frozen -- a
     free scale against a partial smooth template collapses, measured
     13.48 -> 9.70) of the near-face SG points onto the fitted ICT face
     region [0,9409) -- the exact surface s3b will shrinkwrap the face onto.

GATED (dies loudly, never ships a mis-aligned clay):
  - MediaPipe must detect a face on the front render and >= --min-lmk
    landmarks must unproject onto the mesh
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
  clay_sg_cleaned.ply   slab-stripped TripoSG in RAW coords (debug)
  clay_sg_aligned.npz   verts float32 + tri faces int32 (s3b shrinkwrap target)
  clay_sg_aligned.ply   same mesh, for eyeballing
  clay_sg_align.json    S, R, T + metrics + gate values
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ICT_REGIONS, P, detect_face, die, landmarks_to_np,  # noqa: E402
                    out_dir, save_json, umeyama)
from mp_ibug68 import MEDIAPIPE_IBUG68  # noqa: E402

FACE_END = ICT_REGIONS["face"][1]  # 9409
BLENDER_DEFAULT = "/workspace/blender/blender-4.2.3-linux-x64/blender"


def strip_background_slab(v, f, eps=0.06, ang_deg=25.0):
    """Remove the reconstructed background wall (+ the hair-shard chaos that
    hugs it, + confetti) from a TripoSG mesh. Returns (verts, faces, info).
    Raw (pre-alignment) units. See module docstring, step 1."""
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clay-sg",
                    default="/workspace/newstack/out_triposg/random-person_triposg_300k.glb")
    ap.add_argument("--task", default=P.MP_TASK)
    ap.add_argument("--out", default=P.OUT)
    ap.add_argument("--blender", default=os.environ.get("BLENDER", BLENDER_DEFAULT))
    ap.add_argument("--min-conf", type=float, default=0.3)
    ap.add_argument("--min-lmk", type=int, default=40,
                    help="GATE: minimum unprojected landmarks for umeyama")
    ap.add_argument("--lmk-trim", type=float, default=0.15,
                    help="fraction of worst landmark pairs dropped in pass 2")
    ap.add_argument("--slab-eps", type=float, default=0.06,
                    help="raw units; wall band half-width (everything at or "
                         "behind the wall is dropped)")
    ap.add_argument("--no-slab-strip", action="store_true")
    ap.add_argument("--face-iters", type=int, default=40,
                    help="face-polish rigid ICP iterations (0 = disable)")
    ap.add_argument("--face-cap", type=float, default=2.0,
                    help="cm; SG points farther than this from the fitted "
                         "face are excluded from the polish")
    ap.add_argument("--face-trim", type=float, default=0.30)
    ap.add_argument("--n-src", type=int, default=20000)
    ap.add_argument("--max-face-dist", type=float, default=1.0,
                    help="cm; GATE on mean INNER-landmark->SG-surface distance")
    ap.add_argument("--max-front-err", type=float, default=1.2,
                    help="cm; GATE on mean front-depth error at inner landmarks")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    od = out_dir(args.out, "clay")
    from scipy.spatial import cKDTree

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

    cleaned_ply = od / "clay_sg_cleaned.ply"
    trimesh.Trimesh(vertices=v_raw, faces=faces, process=False)\
        .export(cleaned_ply)
    cleaned_npz = od / "clay_sg_cleaned.npz"
    np.savez(cleaned_npz, verts=v_raw.astype(np.float32),
             faces=faces.astype(np.int32))

    # ---- one ortho FRONT render (Blender Workbench; exact px<->world map)
    cam_json = od / "clay_sg_detect_cam.json"
    cmd = [args.blender, "--background", "--factory-startup", "--python",
           str(Path(__file__).resolve().parent / "render_neutral.py"), "--",
           "--out", str(args.out), "--tag", "sg_detect",
           "--mesh", str(cleaned_npz), "--views", "front", "--res", "900",
           "--square", "--dump-cam", str(cam_json)]
    if not os.environ.get("DISPLAY") and shutil.which("xvfb-run"):
        cmd = ["xvfb-run", "-a"] + cmd
    print(f"[s3sg] front render: {' '.join(cmd)}")
    rc = subprocess.run(cmd, stdout=subprocess.DEVNULL).returncode
    if rc != 0:
        die(f"Blender front render failed (exit {rc})")
    with open(cam_json, encoding="utf-8") as fjs:
        cam = json.load(fjs)["front"]

    # ---- MediaPipe on the grey render + columnar max-z unprojection ------
    import cv2
    img = cv2.imread(cam["png"])
    if img is None:
        die(f"render {cam['png']} unreadable")
    rgb = np.ascontiguousarray(img[..., ::-1])
    res = detect_face(rgb, args.task, args.min_conf)
    if res is None:
        die("MediaPipe found NO face on the TripoSG front render -- "
            "cannot landmark-align (no silent fallback)")
    lm = landmarks_to_np(res)  # normalized [0,1]

    fitted = np.load(Path(args.out) / "fit" / "fitted_neutral.npy")
    lmk_verts = np.load(Path(args.out) / "fit" / "topology.npz")["lmk_verts"]
    ict_lmk = fitted[lmk_verts]

    cx, cy = cam["center"][0], cam["center"][1]
    S_o = cam["ortho_scale"]
    rad = S_o / 120.0
    sg_pts, ict_pts, used = [], [], []
    dbg = img.copy()
    N_px = cam["res_x"]
    for i, mp_idx in enumerate(MEDIAPIPE_IBUG68):
        u, v_ = float(lm[mp_idx, 0]), float(lm[mp_idx, 1])
        x = cx + (u - 0.5) * S_o
        y = cy + (0.5 - v_) * S_o
        sel = (np.abs(v_raw[:, 0] - x) < rad) & (np.abs(v_raw[:, 1] - y) < rad)
        if not sel.any():
            continue
        z = float(v_raw[sel, 2].max())
        sg_pts.append((x, y, z))
        ict_pts.append(ict_lmk[i])
        used.append(i)
        cv2.circle(dbg, (int(u * N_px), int(v_ * N_px)), 3, (0, 255, 0), -1)
    cv2.imwrite(str(od / "clay_sg_debug_landmarks.png"), dbg)
    if len(used) < args.min_lmk:
        die(f"only {len(used)} landmarks unprojected onto the TripoSG mesh "
            f"(need >= {args.min_lmk})")
    sg_pts = np.asarray(sg_pts)
    ict_pts = np.asarray(ict_pts)

    # trimmed Umeyama similarity (semantic correspondences => scale well-posed)
    S, R, T = umeyama(sg_pts, ict_pts)
    resid = np.linalg.norm((S * (R @ sg_pts.T).T + T) - ict_pts, axis=1)
    keep = resid <= np.quantile(resid, 1.0 - args.lmk_trim)
    S, R, T = umeyama(sg_pts[keep], ict_pts[keep])
    resid2 = np.linalg.norm((S * (R @ sg_pts[keep].T).T + T) - ict_pts[keep],
                            axis=1)
    lmk_rms = float(np.sqrt((resid2 ** 2).mean()))
    base_info = {"n_landmarks_used": int(len(used)),
                 "n_kept_after_trim": int(keep.sum()),
                 "residual_rms_cm": lmk_rms,
                 "residual_max_cm": float(resid2.max())}
    v_aligned = S * (v_raw @ R.T) + T
    print(f"[s3sg] landmark umeyama: {len(used)} unprojected, "
          f"{int(keep.sum())} kept, rms={lmk_rms:.2f} cm  scale={S:.4f}")

    # ---- rigid face polish onto the fitted ICT face ----------------------
    tree_face = cKDTree(fitted[:FACE_END])
    rng = np.random.default_rng(args.seed)
    src = v_aligned[rng.choice(len(v_aligned),
                               min(args.n_src, len(v_aligned)), replace=False)]
    R_p, t_p = np.eye(3), np.zeros(3)
    n_pairs, face_rms = 0, None
    for _ in range(args.face_iters):
        cur = src @ R_p.T + t_p
        d, j = tree_face.query(cur, workers=-1)
        keep = d <= args.face_cap
        if keep.sum() < 500:
            die(f"face polish: only {int(keep.sum())} SG points within "
                f"{args.face_cap} cm of the fitted face -- alignment is off")
        thr = np.quantile(d[keep], 1.0 - args.face_trim)
        keep &= d <= thr
        n_pairs = int(keep.sum())
        R_d, t_d = rigid_fit(cur[keep], fitted[:FACE_END][j[keep]])
        R_p = R_d @ R_p
        t_p = R_d @ t_p + t_d
    if args.face_iters > 0:
        d, _ = tree_face.query(src @ R_p.T + t_p, workers=-1)
        d_in = d[d <= args.face_cap]
        face_rms = float(np.sqrt((d_in[d_in <= np.quantile(
            d_in, 1.0 - args.face_trim)] ** 2).mean()))
        print(f"[s3sg] face polish: {n_pairs} pairs  rigid  "
              f"face inlier rms={face_rms:.3f} cm")
        v_aligned = v_aligned @ R_p.T + t_p
        R = R_p @ R
        T = R_p @ T + t_p

    # ---- final metrics ----------------------------------------------------
    tree_sg = cKDTree(v_aligned)
    tgt_npz = od / "clay_aligned.npz"
    median_rev = None
    if tgt_npz.is_file():
        sr_v = np.load(tgt_npz)["verts"].astype(np.float64)
        rev_idx = rng.choice(len(sr_v), min(20000, len(sr_v)), replace=False)
        d_rev, _ = tree_sg.query(sr_v[rev_idx], workers=-1)
        median_rev = float(np.median(d_rev))

    d_lmk, _ = tree_sg.query(fitted[lmk_verts], workers=-1)
    inner_mean = float(d_lmk[17:].mean())
    inner_max = float(d_lmk[17:].max())
    contour_mean = float(d_lmk[:17].mean())

    fd_err, fd_miss = front_depth_errors(fitted[lmk_verts[17:]], v_aligned)
    fd_mean = float(fd_err.mean()) if len(fd_err) else np.inf
    fd_max = float(fd_err.max()) if len(fd_err) else np.inf

    ext_sg = v_aligned.max(0) - v_aligned.min(0)
    ext_ict = fitted[:11248].max(0) - fitted[:11248].min(0)
    y_ratio = float(ext_sg[1] / max(ext_ict[1], 1e-9))

    print(f"[s3sg] landmarks -> SG surface: inner mean={inner_mean:.3f} cm "
          f"max={inner_max:.3f} cm  contour mean={contour_mean:.3f} cm")
    print(f"[s3sg] front-depth @inner landmarks: mean={fd_mean:.3f} cm "
          f"max={fd_max:.3f} cm  missing={fd_miss}/51")
    print(f"[s3sg] aligned bbox extent={np.round(ext_sg, 2)} cm "
          f"(ICT head y-extent ratio {y_ratio:.2f})"
          + (f"  median SR->SG dist={median_rev:.2f} cm" if median_rev else ""))

    failures = []
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
        "method": "slab-strip + front-ortho-render mediapipe landmarks + "
                  "columnar-max-z unprojection + trimmed umeyama + rigid "
                  "face-polish vs fitted ICT face",
        "landmark_align": base_info,
        "slab_strip": strip_info,
        "face_polish_pairs": n_pairs,
        "face_polish_inlier_rms_cm": face_rms,
        "lmk_inner_to_surface_mean_cm": inner_mean,
        "lmk_inner_to_surface_max_cm": inner_max,
        "lmk_contour_to_surface_mean_cm": contour_mean,
        "front_depth_mean_cm": fd_mean,
        "front_depth_max_cm": fd_max,
        "front_depth_missing": int(fd_miss),
        "median_rev_cm": median_rev,
        "y_extent_ratio_vs_ict": y_ratio,
        "S": S, "R": R.tolist(), "T": np.asarray(T).tolist(),
        "clay_verts": int(len(v_aligned)), "clay_faces": int(len(faces)),
        "gates": {"max_face_dist_cm": args.max_face_dist,
                  "max_front_err_cm": args.max_front_err},
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
