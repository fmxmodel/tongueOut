#!/usr/bin/env python3
"""Stage 3c -- verify the refined neutral BY MEASUREMENT (system python).

Independent gate after s3b (works for both the TripoSR/legacy and the
TripoSG/face-weighted paths):

  1. topology: exactly 26719 verts, no NaN/Inf, boundary-loop count == 23
     (same closed topology as raw ICT -- the "hole in the head" class of
     failure is a topology break, catch it here);
  2. photo fidelity: reproject the 68 ICT landmark VERTICES of the refined
     neutral (+ the fitted expression offset, same basis as s2's metric)
     through the s2 weak-perspective camera and compare against the MediaPipe
     photo landmarks -- mean px error must stay <= --max-reproj and within
     --reproj-slack px of the s2 fitted baseline (~22 px on this photo);
  3. back region (face mode only): shrinkwrap displacement over head/neck
     verts beyond the feather band must be ~0 (ICT cranium stayed clean);
  4. face-to-clay tightness (face mode, report only): mean distance from
     strongly-shrinkwrapped verts (m_sw > 0.7) to the clay surface.

Writes out/refine/refine_verify.json + reproj overlay; exits non-zero on any
gate failure.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ICT_REGIONS, N_VERTS, P, die, out_dir,  # noqa: E402
                    project_weak_persp, save_json)
from mp_ibug68 import IBUG_GROUPS  # noqa: E402

HN0, HN1 = ICT_REGIONS["head_neck"]


def boundary_loops(faces_flat, faces_off):
    """Count boundary loops: edges used by exactly one polygon, then the
    connected components of the boundary-edge graph."""
    count = {}
    off = np.asarray(faces_off)
    ff = np.asarray(faces_flat)
    for p in range(len(off) - 1):
        idx = ff[off[p]:off[p + 1]]
        n = len(idx)
        for k in range(n):
            a, b = int(idx[k]), int(idx[(k + 1) % n])
            e = (a, b) if a < b else (b, a)
            count[e] = count.get(e, 0) + 1
    bnd = [e for e, c in count.items() if c == 1]
    adj = {}
    for a, b in bnd:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    seen, loops = set(), 0
    for s in adj:
        if s in seen:
            continue
        loops += 1
        stack = [s]
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            stack.extend(adj[u])
    return loops, len(bnd)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=P.OUT)
    ap.add_argument("--max-reproj", type=float, default=30.0,
                    help="px; hard gate on mean landmark reprojection error")
    ap.add_argument("--reproj-slack", type=float, default=5.0,
                    help="px; refined mean may exceed the s2 fitted baseline "
                         "by at most this much")
    ap.add_argument("--expected-loops", type=int, default=23)
    ap.add_argument("--back-max-cm", type=float, default=0.05,
                    help="cm; max allowed shrinkwrap disp beyond the feather")
    args = ap.parse_args()
    t0 = time.time()
    od = out_dir(args.out, "refine")
    fit_dir = Path(args.out) / "fit"

    refined = np.load(od / "refined_neutral.npy")
    fitted = np.load(fit_dir / "fitted_neutral.npy")
    topo = np.load(fit_dir / "topology.npz")
    lmk_verts = topo["lmk_verts"]
    with open(od / "refine_stats.json", encoding="utf-8") as f:
        stats = json.load(f)

    report = {"mode": stats.get("mode"), "weights": stats.get("weights")}
    failures = []

    # ---- 1. topology
    if len(refined) != N_VERTS:
        failures.append(f"vert count {len(refined)} != {N_VERTS}")
    if not np.isfinite(refined).all():
        failures.append("refined_neutral contains NaN/Inf")
    loops, n_bnd = boundary_loops(topo["faces_flat"], topo["faces_off"])
    report["boundary_loops"] = loops
    report["boundary_edges"] = n_bnd
    if loops != args.expected_loops:
        failures.append(f"boundary loops {loops} != {args.expected_loops} "
                        "(raw ICT)")
    print(f"[s3c] topology: {len(refined)} verts, {loops} boundary loops "
          f"({n_bnd} boundary edges) -- expected {args.expected_loops}")

    # ---- 2. photo reprojection (same basis as s2 fit_metrics: + expr offset)
    with open(fit_dir / "camera.json", encoding="utf-8") as f:
        cam = json.load(f)
    s, R, t = cam["s_px_per_cm"], np.asarray(cam["R"]), np.asarray(cam["t_px"])
    target = np.load(Path(args.out) / "landmarks" / "landmarks.npz")["ibug68_px"]
    expr_off = np.load(fit_dir / "expression_offset.npy")

    def reproj_err(verts):
        uv, _ = project_weak_persp(verts[lmk_verts] + expr_off[lmk_verts], s, R, t)
        return uv, np.linalg.norm(uv - target, axis=1)

    uv_r, err_r = reproj_err(refined)
    _, err_f = reproj_err(fitted)
    report["reproj_px_mean_refined"] = float(err_r.mean())
    report["reproj_px_max_refined"] = float(err_r.max())
    report["reproj_px_mean_fitted"] = float(err_f.mean())
    report["reproj_px_by_group"] = {g: float(err_r[i].mean())
                                    for g, i in IBUG_GROUPS.items()}
    print(f"[s3c] reprojection px: refined mean={err_r.mean():.2f} "
          f"max={err_r.max():.2f}  (s2 fitted baseline mean={err_f.mean():.2f})")
    for g, e in report["reproj_px_by_group"].items():
        print(f"      {g:12s} {e:6.2f}")
    if err_r.mean() > args.max_reproj:
        failures.append(f"reproj mean {err_r.mean():.2f}px > {args.max_reproj}px")
    if err_r.mean() > err_f.mean() + args.reproj_slack:
        failures.append(f"reproj mean {err_r.mean():.2f}px exceeds fitted "
                        f"baseline {err_f.mean():.2f}px by more than "
                        f"{args.reproj_slack}px")

    # overlay (best-effort)
    try:
        import cv2
        img = cv2.imread(str(Path(args.out) / "landmarks" / "input_image.png"))
        for (gx, gy), (rx, ry) in zip(target, uv_r):
            cv2.circle(img, (int(gx), int(gy)), 3, (0, 255, 0), -1)
            cv2.circle(img, (int(rx), int(ry)), 3, (0, 0, 255), 1)
        cv2.imwrite(str(od / "reproj_debug.jpg"), img)
        print(f"[s3c] overlay -> {od / 'reproj_debug.jpg'}")
    except Exception as e:
        print(f"[s3c WARN] reproj_debug.jpg skipped: {e}")

    # ---- 3./4. face-mode measurements from refine_debug.npz
    dbg_path = od / "refine_debug.npz"
    if stats.get("mode") == "shrinkwrap" and dbg_path.is_file():
        dbg = np.load(dbg_path)
        if "face_mask" in dbg:
            fm = dbg["face_mask"]
            sw_mag = dbg["sw_disp_mag"]
            back_sel = np.zeros(len(refined), dtype=bool)
            back_sel[HN0:HN1] = fm[HN0:HN1] < 0.01
            back_max = float(sw_mag[back_sel].max()) if back_sel.any() else 0.0
            report["back_region_verts"] = int(back_sel.sum())
            report["back_region_max_sw_disp_cm"] = back_max
            print(f"[s3c] back region: {int(back_sel.sum())} verts beyond "
                  f"feather, max shrinkwrap disp = {back_max:.4f} cm")
            if back_max > args.back_max_cm:
                failures.append(f"back-region shrinkwrap disp {back_max:.3f}cm "
                                f"> {args.back_max_cm}cm -- cranium not clean")
            # face-to-clay tightness (report only)
            try:
                from scipy.spatial import cKDTree
                clay = np.load(stats["clay_npz"])
                tree = cKDTree(clay["verts"])
                core = dbg["m_sw"] > 0.7
                if core.any():
                    d, _ = tree.query(refined[core], workers=-1)
                    report["face_core_verts"] = int(core.sum())
                    report["face_to_clay_mean_cm"] = float(d.mean())
                    report["face_to_clay_p95_cm"] = float(np.percentile(d, 95))
                    print(f"[s3c] face core (m_sw>0.7): {int(core.sum())} verts, "
                          f"dist to clay mean={d.mean():.3f}cm "
                          f"p95={np.percentile(d, 95):.3f}cm")
            except Exception as e:
                print(f"[s3c WARN] face-to-clay check skipped: {e}")
        else:
            print("[s3c] legacy-weights run: back-region/face-core checks n/a")

    report["failures"] = failures
    report["pass"] = not failures
    save_json(od / "refine_verify.json", report)
    if failures:
        die("refine verification FAILED: " + "; ".join(failures))
    print(f"[s3c] ALL GATES PASS in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
