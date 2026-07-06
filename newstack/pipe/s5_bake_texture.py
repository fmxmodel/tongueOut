#!/usr/bin/env python3
"""Stage 5 -- bake the diffuse texture onto ICT's OWN UVs (system python).

One diffuse map, no statistical/NC albedo prior:
  FRONT   photo pixels projected through the fitted weak-persp camera onto the
          EXPRESSED refined mesh (expression_offset applied so a smiling photo
          lands on smiling geometry; UVs are topology-fixed, so the colors map
          back onto the neutral correctly). Visibility = z-buffer + N.V ramp,
          where the camera-facing SIGN of dot(n,view) is MEASURED from the
          depth-passing texels (mirroring recon/bake_texture.py) -- never
          assumed from ICT's OBJ winding. Grazing texels (|cos| below
          --ndotv-lo) are REJECTED, not stretched.
  BACK    TripoSR clay vertex colors (hair!) sampled at each texel's 3D
          position via nearest clay vertex (KD-tree).
  MIRROR  exterior texels with neither photo nor clay try the X-mirrored
          position through the same camera (faces are ~symmetric) before
          falling back to inpaint.
  SEAMS   blended by the N.V weight; leftovers classically inpainted (TELEA);
          UV island gutters padded by mean-dilation.
  INTERIOR teeth/eyeballs/mouth-socket texels that the photo cannot see get
          honest flat defaults (ivory / sclera / dark mouth) instead of
          projection garbage that would show when jawOpen plays.
  EYES    the eyeball UV islands overlap the face UVs (they span the whole
          atlas), so eyeballs CANNOT live on the shared albedo -- they get
          dedicated textures: out/tex/eye_left.png (+ eye_right.png when the
          measured L/R iris colors differ), built photo-derived by
          eye_texture.py with the iris disc at UV (0.5,0.5) = the eyeball's
          forward pole. s6 binds them via separate eye material(s).

Reads out/rig/arkit_deltas.npz (single source of export geometry + UVs),
out/fit/camera.json + expression_offset.npy, out/landmarks/input_image.png,
out/clay/clay_aligned.ply. Writes out/tex/albedo.png + debug maps + metrics.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ICT_REGIONS, P, bilinear_sample, die, out_dir,  # noqa: E402
                    project_weak_persp, rasterize, save_json, smoothstep,
                    triangulate, vertex_normals)

REGION_BOUNDS = np.array([ICT_REGIONS["face"][1], ICT_REGIONS["head_neck"][1],
                          ICT_REGIONS["interior_a"][1], ICT_REGIONS["teeth"][1],
                          ICT_REGIONS["eyeballs"][1]])
# region ids: 0 face, 1 head_neck, 2 interior_a, 3 teeth, 4 eyeballs, 5 interior_b
COL_TEETH = np.array([0.85, 0.82, 0.75])
COL_SCLERA = np.array([0.93, 0.92, 0.90])
COL_MOUTH = np.array([0.40, 0.18, 0.16])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=P.OUT)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--no-expression", action="store_true",
                    help="project onto the neutral instead of the expressed fit")
    ap.add_argument("--ndotv-lo", type=float, default=0.15)
    ap.add_argument("--ndotv-hi", type=float, default=0.50)
    ap.add_argument("--zbuf-eps", type=float, default=0.4, help="cm depth tolerance")
    ap.add_argument("--zbuf-max-px", type=int, default=1200)
    ap.add_argument("--clay-max-dist", type=float, default=6.0,
                    help="cm; farther nearest-clay-vertex = no clay color")
    ap.add_argument("--gutter", type=int, default=12, help="UV gutter padding px")
    ap.add_argument("--no-mirror", action="store_true",
                    help="disable X-mirrored photo fallback for occluded texels")
    ap.add_argument("--central-radius", type=float, default=5.0,
                    help="cm around the nose tip for the central-face sanity gate")
    ap.add_argument("--strict", action="store_true",
                    help="exit nonzero if the central-face sanity gate fails")
    ap.add_argument("--match-clay-color", action="store_true",
                    help="gain-match clay fill to photo skin tone (off: TripoSR "
                         "colors already come from the same photo)")
    ap.add_argument("--eye-size", type=int, default=512,
                    help="eye texture resolution (px)")
    ap.add_argument("--iris-frac", type=float, default=None,
                    help="visible-iris radius as a fraction of the eyeball's "
                         "full UV radius (default: eye_texture.IRIS_FRAC)")
    args = ap.parse_args()
    t0 = time.time()
    od = out_dir(args.out, "tex")
    T = args.size

    z = np.load(Path(args.out) / "rig" / "arkit_deltas.npz")
    verts = z["refined_neutral"].astype(np.float64)
    faces_flat, faces_off = z["faces_flat"], z["faces_off"]
    corner_vt, vt = z["corner_vt"], z["vt"].astype(np.float64)
    if (corner_vt < 0).any():
        die("mesh has corners without vt -- cannot bake on ICT UVs")

    expr_off = np.load(Path(args.out) / "fit" / "expression_offset.npy")
    expressed = verts if args.no_expression else verts + expr_off
    import json
    cam = json.loads((Path(args.out) / "fit" / "camera.json").read_text())
    s, R, t2 = cam["s_px_per_cm"], np.asarray(cam["R"]), np.asarray(cam["t_px"])

    from PIL import Image
    photo = np.asarray(Image.open(Path(args.out) / "landmarks" / "input_image.png")
                       .convert("RGB"), dtype=np.float64) / 255.0
    H, W = photo.shape[:2]
    print(f"[s5] photo {W}x{H}, texture {T}x{T}, expression "
          f"{'OFF' if args.no_expression else 'ON'}")

    tri_c = triangulate(faces_flat, faces_off)      # (M,3) into flat corners
    tri_v = faces_flat[tri_c]                       # vertex ids per corner
    tri_vt = corner_vt[tri_c]                       # vt ids per corner
    normals = vertex_normals(expressed, tri_v)
    tri_reg = np.searchsorted(REGION_BOUNDS, tri_v[:, 0], side="right")

    # ---- photo-space z-buffer on the expressed mesh
    zscale = min(1.0, args.zbuf_max_px / max(H, W))
    uv_all, depth_all = project_weak_persp(expressed, s, R, t2)
    zw, zh = int(np.ceil(W * zscale)), int(np.ceil(H * zscale))
    print(f"[s5] z-buffer {zw}x{zh} over {len(tri_v)} tris ...")
    zbuf, _, _ = rasterize(uv_all * zscale, tri_v, zw, zh, depth=depth_all)

    # ---- UV-space rasterization (wedge arrays: corners own their vt AND vertex)
    vt_px = np.stack([vt[:, 0] * T, (1.0 - vt[:, 1]) * T], axis=1)
    pos2d_w = vt_px[tri_vt.reshape(-1)]                       # (3M,2)
    attrs_w = np.concatenate([expressed[tri_v.reshape(-1)],
                              normals[tri_v.reshape(-1)]], axis=1)  # (3M,6)
    tris_w = np.arange(len(pos2d_w), dtype=np.int64).reshape(-1, 3)
    # priority "depth": exterior skin tris ALWAYS beat interior tris (teeth,
    # eyeballs, sockets) wherever UV islands overlap -- interior geometry must
    # never steal face texels and drag dark defaults onto the face.
    prio_w = np.repeat((tri_reg <= 1).astype(np.float64), 3)
    print(f"[s5] UV rasterization {T}x{T} ...")
    _, tid, abuf = rasterize(pos2d_w, tris_w, T, T, depth=prio_w, attrs=attrs_w)
    covered = tid >= 0
    K = int(covered.sum())
    print(f"[s5] texels covered: {K} ({100.0 * K / (T * T):.1f}%)")

    pos = abuf[..., :3][covered]
    nrm = abuf[..., 3:6][covered]
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    reg = tri_reg[tid[covered]]

    # ---- photo projection + visibility
    uv_t, d_t = project_weak_persp(pos, s, R, t2)
    view = R.T @ np.array([0.0, 0.0, 1.0])
    ndotv = nrm @ view
    xi = np.clip((uv_t[:, 0] * zscale).astype(np.int64), 0, zw - 1)
    yi = np.clip((uv_t[:, 1] * zscale).astype(np.int64), 0, zh - 1)
    inb = ((uv_t[:, 0] >= 0) & (uv_t[:, 0] < W)
           & (uv_t[:, 1] >= 0) & (uv_t[:, 1] < H))
    depth_ok = inb & (d_t >= zbuf[yi, xi] - args.zbuf_eps)

    # Winding auto-orientation (MEASURED, mirroring recon/bake_texture.py):
    # texels that survive the z-buffer are the front surface by construction,
    # so the sign of dot(n,view) that dominates THERE is the camera-facing
    # sign -- regardless of how ICT's OBJ winding oriented our cross products.
    ext_probe = depth_ok & (reg <= 1)
    probe = ndotv[ext_probe] if ext_probe.any() else ndotv[depth_ok]
    frac_neg = float((probe < 0).mean()) if len(probe) else 0.0
    facing_sign = -1.0 if frac_neg > 0.5 else 1.0
    ndotv = facing_sign * ndotv
    print(f"[s5] winding auto-orientation: {100.0 * frac_neg:.1f}% of "
          f"depth-passing exterior texels have dot(n,view)<0 -> "
          f"facing sign {facing_sign:+.0f} (measured, not assumed)")

    # grazing texels (cos < ndotv_lo) get w=0 -> filled from clay/mirror/
    # inpaint instead of stretching silhouette photo pixels across them
    visible = depth_ok & (ndotv > args.ndotv_lo)
    w_photo = smoothstep(ndotv, args.ndotv_lo, args.ndotv_hi)
    w_photo[~visible] = 0.0
    photo_col = bilinear_sample(photo, uv_t)
    print(f"[s5] photo-visible texels: {int((w_photo > 0).sum())} "
          f"(depth-passing {int(depth_ok.sum())}, grazing-rejected "
          f"{int((depth_ok & (ndotv > 0) & ~visible).sum())})")

    # ---- clay vertex-color fill (hair + back of head)
    clay_col = np.zeros_like(photo_col)
    clay_ok = np.zeros(K, dtype=bool)
    clay_ply = Path(args.out) / "clay" / "clay_aligned.ply"
    if clay_ply.is_file():
        import trimesh
        from scipy.spatial import cKDTree
        cm = trimesh.load(clay_ply, process=False, force="mesh")
        cverts = np.asarray(cm.vertices)
        ccols = np.asarray(cm.visual.vertex_colors, dtype=np.float64)[:, :3] / 255.0
        dist, idx = cKDTree(cverts).query(pos, k=1)
        clay_ok = dist < args.clay_max_dist
        clay_col = ccols[idx]
        if args.match_clay_color:
            both = clay_ok & (w_photo > 0.5) & (reg <= 1)
            if both.sum() > 500:
                gain = (photo_col[both].mean(0) + 1e-6) / (clay_col[both].mean(0) + 1e-6)
                clay_col = np.clip(clay_col * np.clip(gain, 0.5, 2.0), 0, 1)
                print(f"[s5] clay color gain-matched: {np.round(gain, 3)}")
        print(f"[s5] clay-fillable texels: {int(clay_ok.sum())}")
    else:
        print(f"[s5 WARN] {clay_ply} missing -- no clay fill (holes -> inpaint)")

    # ---- compose per source
    col = np.zeros((K, 3))
    filled = np.zeros(K, dtype=bool)
    src = np.zeros(K, dtype=np.uint8)  # 1 photo 2 clay 3 blend 4 default 5 mirror
    ext = reg <= 1
    m = ext & (w_photo > 0) & clay_ok
    col[m] = w_photo[m, None] * photo_col[m] + (1 - w_photo[m, None]) * clay_col[m]
    filled[m], src[m] = True, 3
    m = ext & (w_photo > 0) & ~clay_ok
    col[m], filled[m], src[m] = photo_col[m], True, 1
    m = ext & (w_photo <= 0) & clay_ok
    col[m], filled[m], src[m] = clay_col[m], True, 2

    # X-mirrored photo fallback: exterior texels with neither photo nor clay
    # try their reflection about the symmetry plane through the same camera
    # (rejected grazing/occluded texels get plausible skin, not smears).
    need = ext & ~filled
    if not args.no_mirror and need.any():
        pos_m = pos[need] * np.array([-1.0, 1.0, 1.0])
        nrm_m = nrm[need] * np.array([-1.0, 1.0, 1.0])
        uv_m, d_m = project_weak_persp(pos_m, s, R, t2)
        xm = np.clip((uv_m[:, 0] * zscale).astype(np.int64), 0, zw - 1)
        ym = np.clip((uv_m[:, 1] * zscale).astype(np.int64), 0, zh - 1)
        inb_m = ((uv_m[:, 0] >= 0) & (uv_m[:, 0] < W)
                 & (uv_m[:, 1] >= 0) & (uv_m[:, 1] < H))
        nv_m = facing_sign * (nrm_m @ view)
        ok_m = (inb_m & (d_m >= zbuf[ym, xm] - args.zbuf_eps)
                & (nv_m > args.ndotv_lo))
        idx = np.where(need)[0][ok_m]
        col[idx] = bilinear_sample(photo, uv_m)[ok_m]
        filled[idx], src[idx] = True, 5
        print(f"[s5] mirror-filled texels: {len(idx)}")

    for rid, default, vis_th in ((3, COL_TEETH, 0.35), (4, COL_SCLERA, 0.35),
                                 (2, COL_MOUTH, 0.60), (5, COL_MOUTH, 0.60)):
        m = (reg == rid) & (w_photo > vis_th)
        col[m], filled[m], src[m] = photo_col[m], True, 1
        m = (reg == rid) & (w_photo <= vis_th)
        col[m], filled[m], src[m] = default, True, 4
    holes = ~filled
    print(f"[s5] sources: photo={int((src==1).sum())} clay={int((src==2).sum())} "
          f"blend={int((src==3).sum())} default={int((src==4).sum())} "
          f"mirror={int((src==5).sum())} holes={int(holes.sum())}")

    # ---- central-face sanity gate (MEASURED): the front-facing patch around
    # the nose must be photo-dominated and roughly photo-bright.
    lmk_verts = z["lmk_verts"]
    nose = expressed[lmk_verts[30]]  # iBUG 30 = nose tip
    central = (np.linalg.norm(pos - nose, axis=1) < args.central_radius) & depth_ok
    uv_l, _ = project_weak_persp(expressed[lmk_verts], s, R, t2)
    x0 = int(np.clip(uv_l[:, 0].min(), 0, W - 1))
    x1 = int(np.clip(uv_l[:, 0].max(), 0, W - 1))
    y0 = int(np.clip(uv_l[:, 1].min(), 0, H - 1))
    y1 = int(np.clip(uv_l[:, 1].max(), 0, H - 1))
    photo_face_bright = (float(photo[y0:y1 + 1, x0:x1 + 1].mean())
                         if (x1 > x0 and y1 > y0) else float(photo.mean()))
    if central.any():
        cf = {
            "texels": int(central.sum()),
            "photo_frac": float(np.isin(src[central], (1, 3)).mean()),
            "clay_frac": float((src[central] == 2).mean()),
            "default_frac": float((src[central] == 4).mean()),
            "mirror_frac": float((src[central] == 5).mean()),
            "hole_frac_pre_inpaint": float(holes[central].mean()),
            "mean_brightness": float(col[central].mean()),
            "photo_face_brightness": photo_face_bright,
        }
        cf["brightness_ratio"] = cf["mean_brightness"] / max(photo_face_bright, 1e-6)
        sanity_pass = (cf["photo_frac"] >= 0.8 and cf["default_frac"] <= 0.02
                       and cf["brightness_ratio"] >= 0.6)
    else:
        cf, sanity_pass = {"texels": 0}, False
    cf["pass"] = sanity_pass
    print(f"[s5] central-face gate (r<{args.central_radius}cm of nose tip, "
          f"front-facing): {cf}")
    if not sanity_pass:
        print("[s5 WARN] === CENTRAL-FACE SANITY FAILED -- the face patch is "
              "not photo-dominated/bright; inspect debug_w_photo.png, "
              "debug_ndotv.png, debug_sources.png ===")

    # ---- to image; inpaint holes; pad gutters
    import cv2
    img = np.zeros((T, T, 3))
    valid = np.zeros((T, T), dtype=bool)
    img[covered] = col
    valid[covered] = filled
    img8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    hole_mask = np.zeros((T, T), dtype=np.uint8)
    hole_mask[covered] = holes.astype(np.uint8)
    if hole_mask.any():
        img8 = cv2.inpaint(img8, hole_mask, 3, cv2.INPAINT_TELEA)
        valid[hole_mask.astype(bool)] = True
    vf = valid.astype(np.float32)
    imgf = img8.astype(np.float32)
    for _ in range(args.gutter):  # mean-dilation gutter padding
        acc = cv2.blur(imgf * vf[..., None], (3, 3))
        cnt = cv2.blur(vf, (3, 3))
        grow = (cnt > 1e-6) & (vf < 0.5)
        imgf[grow] = acc[grow] / cnt[grow, None]
        vf[grow] = 1.0
    img8 = np.clip(imgf, 0, 255).astype(np.uint8)

    Image.fromarray(img8).save(od / "albedo.png")

    # ---- dedicated eye textures (eyeball UVs overlap the face UVs, so the
    # eyes cannot share the albedo -- see eye_texture.py). Pairing of the
    # model's x<0 / x>=0 eyeball to the photo's two irises is MEASURED by
    # projecting the eyeball centroids through the fitted camera.
    from eye_texture import IRIS, IRIS_FRAC, build_eye_textures
    eb0, eb1 = ICT_REGIONS["eyeballs"]
    eyeballs = expressed[eb0:eb1]
    cen = np.stack([eyeballs[eyeballs[:, 0] < 0].mean(0),
                    eyeballs[eyeballs[:, 0] >= 0].mean(0)])  # [x<0, x>=0]
    uv_eyes, _ = project_weak_persp(cen, s, R, t2)
    lmk_npz = Path(args.out) / "landmarks" / "landmarks.npz"
    lmk = np.load(lmk_npz)
    iris_px = {side: lmk["lmk478_px"][IRIS[side]["center"]] for side in IRIS}
    d_xneg = {side: float(np.linalg.norm(uv_eyes[0] - iris_px[side]))
              for side in IRIS}
    d_xpos = {side: float(np.linalg.norm(uv_eyes[1] - iris_px[side]))
              for side in IRIS}
    left_side = min(d_xneg, key=d_xneg.get)
    if min(d_xpos, key=d_xpos.get) == left_side:  # degenerate pairing
        print("[s5 WARN] eyeball<->iris pairing degenerate; defaulting "
              "x<0 -> subject_right (frontal-photo convention)")
        left_side = "subject_right"
    iris_frac = IRIS_FRAC if args.iris_frac is None else args.iris_frac
    img_l, img_r, eye_metrics = build_eye_textures(
        photo, lmk_npz, size=args.eye_size,
        left_subject_side=left_side, iris_frac=iris_frac)
    Image.fromarray(img_l).save(od / "eye_left.png")
    eye_right = od / "eye_right.png"
    if eye_metrics["shared"]:
        eye_right.unlink(missing_ok=True)  # stale separate texture = drift
    else:
        Image.fromarray(img_r).save(eye_right)
    eye_metrics["pairing"] = {
        "model_xneg_feeds": left_side,
        "proj_px": {"xneg": uv_eyes[0].round(1).tolist(),
                    "xpos": uv_eyes[1].round(1).tolist()},
        "dist_px_xneg": d_xneg, "dist_px_xpos": d_xpos,
    }
    print(f"[s5] eye textures -> eye_left.png"
          f"{'' if eye_metrics['shared'] else ' + eye_right.png'} "
          f"(shared={eye_metrics['shared']}, model x<0 iris from {left_side}, "
          f"iris_uv_radius={eye_metrics['iris_uv_radius']:.3f})")

    # ---- per-vertex colors for the UDIM tiles the albedo cannot carry.
    # MEASURED: ICT UVs are multi-tile (face spans u in [0,2], head_neck
    # [1,2], mouth socket [1,3], teeth [3,4], lashes up to [0,7]); this bake
    # only fills tile 0, and a wrapping sampler would paint every tile-1+
    # surface with the FACE image (the infamous stretched-face back of head).
    # s6 gives those polys a vertex-colored material instead: photo where
    # visible, TripoSR clay (hair!) where near, honest flat interior defaults.
    vcol = np.zeros((len(verts), 3))
    vreg = np.searchsorted(REGION_BOUNDS, np.arange(len(verts)), side="right")
    vcol[vreg == 2] = COL_MOUTH          # mouth socket + eye sockets
    vcol[vreg == 3] = COL_TEETH
    vcol[vreg == 4] = COL_SCLERA         # eyeballs (unused -- EyeMat wins)
    vcol[vreg == 5] = [0.23, 0.16, 0.12]  # lashes/lacrimal: dark hair tone
    ext_v = vreg <= 1                    # face + head_neck skin
    nv_v = facing_sign * (normals @ view)
    xi_v = np.clip((uv_all[:, 0] * zscale).astype(np.int64), 0, zw - 1)
    yi_v = np.clip((uv_all[:, 1] * zscale).astype(np.int64), 0, zh - 1)
    inb_v = ((uv_all[:, 0] >= 0) & (uv_all[:, 0] < W)
             & (uv_all[:, 1] >= 0) & (uv_all[:, 1] < H))
    vis_v = (inb_v & (depth_all >= zbuf[yi_v, xi_v] - args.zbuf_eps)
             & (nv_v > args.ndotv_lo))
    w_v = smoothstep(nv_v, args.ndotv_lo, args.ndotv_hi)
    w_v[~vis_v] = 0.0
    photo_v = bilinear_sample(photo, uv_all)
    skin_mean = (photo_v[ext_v & vis_v].mean(0)
                 if (ext_v & vis_v).any() else np.array([0.75, 0.6, 0.55]))
    vcol[ext_v] = skin_mean
    from common import EYE_SOCKETS
    vcol[EYE_SOCKETS[0]:EYE_SOCKETS[1]] = skin_mean * 0.8  # shadowed lid skin
    if clay_ply.is_file():
        from scipy.spatial import cKDTree
        dist_v, idx_v = cKDTree(cverts).query(verts[ext_v], k=1)
        near = dist_v < args.clay_max_dist
        tmp = vcol[ext_v]
        tmp[near] = ccols[idx_v[near]]
        vcol[ext_v] = tmp
    m_v = ext_v & (w_v > 0)
    vcol[m_v] = (w_v[m_v, None] * photo_v[m_v]
                 + (1 - w_v[m_v, None]) * vcol[m_v])
    np.save(od / "vertex_colors.npy", vcol.astype(np.float32))
    print(f"[s5] per-vertex tile-fallback colors -> vertex_colors.npy "
          f"(photo-visible {int((ext_v & (w_v > 0)).sum())} skin verts, "
          f"skin mean {np.round(skin_mean, 3)})")
    wmap = np.zeros((T, T))
    wmap[covered] = w_photo
    Image.fromarray((wmap * 255).astype(np.uint8)).save(od / "debug_w_photo.png")
    smap = np.zeros((T, T), dtype=np.uint8)
    smap[covered] = src * 50
    Image.fromarray(smap).save(od / "debug_sources.png")
    nmap = np.zeros((T, T))
    nmap[covered] = (ndotv + 1.0) * 0.5  # sign-corrected: bright = camera-facing
    Image.fromarray((np.clip(nmap, 0, 1) * 255).astype(np.uint8)).save(
        od / "debug_ndotv.png")
    region_names = ["face", "head_neck", "interior_a", "teeth", "eyeballs",
                    "interior_b"]
    save_json(od / "bake_metrics.json", {
        "texture_size": T, "texels_covered": K,
        "photo": int((src == 1).sum()), "clay": int((src == 2).sum()),
        "blend": int((src == 3).sum()), "interior_default": int((src == 4).sum()),
        "mirror": int((src == 5).sum()),
        "inpainted_holes": int(holes.sum()),
        "winding": {"frac_ndotv_neg_on_depth_pass": frac_neg,
                    "facing_sign": facing_sign,
                    "note": "sign measured from depth-passing exterior texels, "
                            "never assumed from OBJ winding"},
        "grazing_rejected": int((depth_ok & (ndotv > 0)
                                 & (ndotv <= args.ndotv_lo)).sum()),
        "region_texels": {n: int((reg == i).sum())
                          for i, n in enumerate(region_names)},
        "central_face": cf,
        "sanity_pass": sanity_pass,
        "expression_applied": not args.no_expression,
        "eyes": eye_metrics,
    })
    print(f"[s5] albedo -> {od / 'albedo.png'}")
    print(f"[s5] DONE in {time.time()-t0:.1f}s")
    if args.strict and not sanity_pass:
        sys.exit(2)  # after all artifacts are written, so they stay inspectable


if __name__ == "__main__":
    main()
