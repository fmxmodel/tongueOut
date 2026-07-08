#!/usr/bin/env python3
"""Stage 5 -- bake the diffuse texture onto ICT's OWN UVs (system python).
SIMPLIFIED HYBRID: TripoSR back/sides + photo projection for front face.

Color sources:
  FRONT: photo pixels projected through the weak-persp camera onto the
         expressed refined mesh. Visibility = z-buffer + N.V ramp.
         Grazing texels rejected (stretched photo pixels look bad).
         Procedural person mask prevents backdrop bleeding.
  BACK:  TripoSR vertex colors sampled via k-NN (consistent, no hallucination)
  INTERIOR: honest flat defaults (teeth/eyeballs/mouth)
  EYES:  photo-derived iris textures (not procedural)

No TRELLIS, no hair-zone recolor, no multi-source blending over the hair.
The photo only paints where it can see well (front face, N·V > threshold);
everything else gets TripoSR colors. This gives good skin tones on the face
while keeping the consistent TripoSR look everywhere else.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ICT_REGIONS, P, bilinear_sample, die, out_dir,
                    project_weak_persp, rasterize, save_json, smoothstep,
                    triangulate, vertex_normals)

REGION_BOUNDS = np.array([ICT_REGIONS["face"][1], ICT_REGIONS["head_neck"][1],
                          ICT_REGIONS["interior_a"][1], ICT_REGIONS["teeth"][1],
                          ICT_REGIONS["eyeballs"][1]])
COL_TEETH = np.array([0.85, 0.82, 0.75])
COL_SCLERA = np.array([0.93, 0.92, 0.90])
COL_MOUTH = np.array([0.40, 0.18, 0.16])
LUM_W = np.array([0.299, 0.587, 0.114])


def build_person_mask(photo, tol=0.12, erode_px=3, blur_sigma=3.0):
    """Soft person mask in [0,1]: 1 on subject, 0 on backdrop."""
    import cv2
    H, W = photo.shape[:2]
    border = np.concatenate([photo[0, :], photo[1, :], photo[:, 0],
                             photo[:, 1], photo[:, W - 2], photo[:, W - 1]])
    bg_rgb = np.median(border, axis=0)
    cand = (np.linalg.norm(photo - bg_rgb, axis=2) < tol).astype(np.uint8)
    _, lab = cv2.connectedComponents(cand, connectivity=4)
    touch = np.unique(np.concatenate([lab[0, :], lab[:, 0], lab[:, W - 1]]))
    touch = touch[touch != 0]
    person_hard = ~np.isin(lab, touch)
    k = 2 * int(erode_px) + 1
    person = cv2.erode(person_hard.astype(np.uint8), np.ones((k, k), np.uint8))
    person = cv2.GaussianBlur(person.astype(np.float64), (0, 0), blur_sigma)
    return person, {"enabled": True, "bg_rgb": np.round(bg_rgb, 3).tolist()}


def sample_clay(tree, ccols, pts, k):
    """Inverse-distance mean color of k nearest clay vertices."""
    from scipy.spatial import cKDTree
    dist, idx = tree.query(pts, k=k)
    if k == 1:
        return dist, ccols[idx]
    w = 1.0 / np.maximum(dist, 1e-6)
    col = (ccols[idx] * w[..., None]).sum(axis=1) / w.sum(axis=1)[..., None]
    return dist[:, 0], col


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=P.OUT)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--clay-knn", type=int, default=8)
    ap.add_argument("--clay-max-dist", type=float, default=6.0)
    ap.add_argument("--gutter", type=int, default=12)
    ap.add_argument("--zbuf-eps", type=float, default=0.4)
    ap.add_argument("--ndotv-lo", type=float, default=0.08)
    ap.add_argument("--ndotv-hi", type=float, default=0.60)
    args = ap.parse_args()
    t0 = time.time()
    od = out_dir(args.out, "tex")
    T = args.size

    # Load ICT mesh + UVs
    z = np.load(Path(args.out) / "rig" / "arkit_deltas.npz")
    verts = z["refined_neutral"].astype(np.float64)
    faces_flat, faces_off = z["faces_flat"], z["faces_off"]
    corner_vt, vt = z["corner_vt"], z["vt"].astype(np.float64)
    if (corner_vt < 0).any():
        die("mesh has corners without vt -- cannot bake on ICT UVs")

    # Load expression offset + camera for photo projection
    expr_off = np.load(Path(args.out) / "fit" / "expression_offset.npy")
    expressed = verts + expr_off
    cam = json.loads((Path(args.out) / "fit" / "camera.json").read_text())
    s, R, t2 = cam["s_px_per_cm"], np.asarray(cam["R"]), np.asarray(cam["t_px"])

    from PIL import Image
    photo = np.asarray(Image.open(Path(args.out) / "landmarks" / "input_image.png")
                       .convert("RGB"), dtype=np.float64) / 255.0
    H, W = photo.shape[:2]
    print(f"[s5] photo {W}x{H}, texture {T}x{T}")
    pmask, _ = build_person_mask(photo)
    pmask3 = pmask[..., None]

    # Load aligned TripoSR clay
    clay_ply = Path(args.out) / "clay" / "clay_aligned.ply"
    if not clay_ply.is_file():
        die(f"clay not found at {clay_ply}")
    import trimesh
    from scipy.spatial import cKDTree
    cm = trimesh.load(clay_ply, process=False, force="mesh")
    cverts = np.asarray(cm.vertices, dtype=np.float64)
    ccols = np.asarray(cm.visual.vertex_colors, dtype=np.float64)[:, :3] / 255.0
    ctree = cKDTree(cverts)
    print(f"[s5] TripoSR clay: {len(cverts)} verts")

    # UV rasterization
    tri_c = triangulate(faces_flat, faces_off)
    tri_v = faces_flat[tri_c]
    tri_vt = corner_vt[tri_c]
    normals = vertex_normals(expressed, tri_v)
    tri_reg = np.searchsorted(REGION_BOUNDS, tri_v[:, 0], side="right")

    vt_px = np.stack([vt[:, 0] * T, (1.0 - vt[:, 1]) * T], axis=1)
    pos2d_w = vt_px[tri_vt.reshape(-1)]
    attrs_w = np.concatenate([expressed[tri_v.reshape(-1)],
                              normals[tri_v.reshape(-1)]], axis=1)
    tris_w = np.arange(len(pos2d_w), dtype=np.int64).reshape(-1, 3)
    prio_w = np.repeat((tri_reg <= 1).astype(np.float64), 3)
    print(f"[s5] UV rasterization {T}x{T} ...")
    _, tid, abuf = rasterize(pos2d_w, tris_w, T, T, depth=prio_w, attrs=attrs_w)
    covered = tid >= 0
    K = int(covered.sum())
    print(f"[s5] texels covered: {K} ({100.0*K/(T*T):.1f}%)")

    pos = abuf[..., :3][covered]
    nrm = abuf[..., 3:6][covered]
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    reg = tri_reg[tid[covered]]

    # ---- Photo projection (front face only)
    uv_t, _ = project_weak_persp(pos, s, R, t2)
    view = R.T @ np.array([0.0, 0.0, 1.0])
    ndotv = nrm @ view
    inb = ((uv_t[:, 0] >= 0) & (uv_t[:, 0] < W)
           & (uv_t[:, 1] >= 0) & (uv_t[:, 1] < H))
    mval = bilinear_sample(pmask3, uv_t)[:, 0]
    visible = inb & (ndotv > args.ndotv_lo) & (mval > 0.05)
    w_photo = smoothstep(ndotv, args.ndotv_lo, args.ndotv_hi) * np.clip(mval, 0.0, 1.0)
    w_photo = np.clip(w_photo * 1.3, 0.0, 1.0)  # boost photo influence
    w_photo[~visible] = 0.0
    photo_col = bilinear_sample(photo, uv_t) if visible.any() else np.zeros_like(pos)
    n_photo = int((w_photo > 0).sum())
    print(f"[s5] photo-visible texels: {n_photo} (front face)")

    # ---- TripoSR clay fill (back/sides/hair)
    dist, clay_col = sample_clay(ctree, ccols, pos, args.clay_knn)
    clay_ok = dist < args.clay_max_dist

    # ---- Build albedo: photo where visible, clay elsewhere, defaults for interior
    albedo = np.zeros((T, T, 3), dtype=np.float32)
    # Exterior regions (face + head/neck)
    exterior = (reg <= 1) & clay_ok
    # Blend photo + clay on front face
    w = w_photo[:, None]
    blended = np.where(w > 0, w * photo_col + (1 - w) * clay_col, clay_col)
    albedo.reshape(-1, 3)[covered.ravel()] = np.where(
        exterior[:, None], blended,
        np.where((reg == 3)[:, None], COL_TEETH[None, :],
                 np.where((reg == 4)[:, None], COL_SCLERA[None, :],
                          np.where((reg >= 2)[:, None], COL_MOUTH[None, :],
                                   clay_col))))

    # UV gutter padding
    from scipy.ndimage import binary_dilation
    mask = tid >= 0
    gutter = binary_dilation(mask, iterations=args.gutter)
    fill = ~mask & gutter
    if fill.any():
        ry, rx = np.where(fill)
        for i in range(len(ry)):
            y, x = ry[i], rx[i]
            patch = albedo[max(0, y-2):y+3, max(0, x-2):x+3]
            albedo[y, x] = patch.mean(axis=(0, 1))
    print(f"[s5] gutter padding: {int(fill.sum())} texels")

    # Vertex colors for RestMat (UDIM tile-1+ polys)
    vdist, vcols = sample_clay(ctree, ccols, verts, args.clay_knn)
    interior_mask = np.zeros(len(verts), dtype=bool)
    interior_mask[ICT_REGIONS["interior_a"][0]:ICT_REGIONS["interior_b"][1]] = True
    interior_mask[ICT_REGIONS["eyeballs"][0]:ICT_REGIONS["eyeballs"][1]] = False
    vcols[interior_mask] = COL_MOUTH
    vcols[ICT_REGIONS["teeth"][0]:ICT_REGIONS["teeth"][1]] = COL_TEETH
    vcols[ICT_REGIONS["eyeballs"][0]:ICT_REGIONS["eyeballs"][1]] = COL_SCLERA
    vertex_colors = np.clip(vcols * 255, 0, 255).astype(np.uint8)

    # Write outputs
    # Boost saturation slightly (the TripoSR clay is naturally desaturated)
    from PIL import ImageEnhance
    albedo_u8 = np.clip(albedo * 255, 0, 255).astype(np.uint8)
    albedo_pil = Image.fromarray(albedo_u8)
    enhancer = ImageEnhance.Color(albedo_pil)
    albedo_pil = enhancer.enhance(2.0)  # 2x saturation boost
    albedo_pil.save(od / "albedo.png")
    print(f"[s5] albedo -> {od}/albedo.png ({T}x{T})")
    np.save(od / "vertex_colors.npy", vertex_colors)
    print(f"[s5] vertex_colors -> {od}/vertex_colors.npy")

    # Eye textures (from photo via eye_texture.py)
    from eye_texture import build_eye_textures
    eye_l, eye_r, _ = build_eye_textures(
        photo, str(Path(args.out) / "landmarks" / "landmarks.npz"),
        size=args.size // 2)
    # eye_texture returns uint8 [0,255] arrays
    Image.fromarray(eye_l).save(od / "eye_left.png")
    Image.fromarray(eye_r).save(od / "eye_right.png")
    print(f"[s5] eye textures -> {od}/eye_*.png")

    # Metrics
    metrics = {"albedo_size": T, "texels_covered": int(K),
               "photo_visible": int(n_photo),
               "photo_frac_front": round(float(n_photo) / max(K, 1), 4),
               "clay_source": "triposr", "time_s": round(time.time()-t0, 1)}
    save_json(od / "bake_metrics.json", metrics)
    print(f"[s5] done in {metrics['time_s']}s")

    summary = {"clay_only": True, "albedo": str(od / "albedo.png"),
               "vertex_colors": str(od / "vertex_colors.npy"),
               "eye_textures": {"left": str(od / "eye_left.png"),
                                "right": str(od / "eye_right.png")}}
    (Path(args.out) / "tex" / "bake_summary.json").write_text(json.dumps(summary))


if __name__ == "__main__":
    main()
