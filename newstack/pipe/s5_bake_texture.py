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
  MASK    photo samples must land on the PERSON: the near-uniform backdrop is
          detected procedurally (flood-fill by color distance to the border
          median from the top/left/right image borders), eroded and feathered.
          Without it, silhouette-grazing texels paint the white backdrop onto
          the scalp/sides -- the MEASURED "white bonnet" halo. Masked-out
          texels fall through to the TripoSR fill instead.
  BACK    TripoSR clay vertex colors (its 360-degree hallucination: real hair/
          skin tone) sampled at each texel's 3D position via k-NN inverse-
          distance average over the aligned clay (dense: ~0.15 cm spacing),
          then PALETTE-MATCHED to the photo in a LUMA/CHROMA decomposition
          (--clay-transfer lumachroma, the default): luma gets a residual-
          trimmed affine fit onto the photo's luma over texels where photo AND
          clay overlap (brightness lands right -> no hairline/jaw step), while
          CHROMA is PRESERVED -- scaled per channel by the photo/clay chroma
          std ratio, not replaced -- so the clay's per-vertex color variation
          (the illusion of strands/shading) survives instead of being
          flattened by the legacy per-channel RGB affine (--clay-transfer
          affine, kept for A/B; MEASURED gain ~[0.96,0.90,0.94] = desaturate).
          The HAIR ZONE (scalp cap above the hairline + skull back above ear
          level, both measured from the fitted landmarks) is special:
          MEASURED, TripoSR hallucinates the hidden crown desaturated
          grey-beige, out-of-distribution vs ALL photo-visible clay (its
          chroma is even inverted), so no fitted distribution transfer can
          recover the hair BASE color from it. There the base palette comes
          from the MEASURED photo hair (mean color of cap texels the photo
          does see -- real hair; or scalp skin if bald, which self-corrects);
          TripoSR contributes its LUMINANCE variation (shading detail) plus
          its LOCAL CHROMA DEVIATION about the zone mean (--hair-chroma),
          sampled with a sharper k-NN (--hair-knn, default 2 vs 8 elsewhere)
          so the crown has tonal variation, not a flat brown. Feathered below
          the hairline / around the ears. Finally a controlled SATURATION
          lift (--back-sat) counters the residual desaturation over the
          hair/back zone (feathered by backfacing-ness, so the photo-lit
          front is untouched).
  MIRROR  exterior texels with neither photo nor clay try the X-mirrored
          position through the same camera (faces are ~symmetric; person-mask
          enforced there too) before falling back to inpaint.
  SEAMS   final = w_photo*photo + (1-w_photo)*mapped_clay with w_photo the
          N.V smoothstep feather times the soft person mask; leftovers
          classically inpainted (TELEA); UV gutters padded by mean-dilation.
  INTERIOR teeth/eyeballs/mouth-socket texels that the photo cannot see get
          honest flat defaults (ivory / sclera / dark mouth) instead of
          projection garbage that would show when jawOpen plays.
  EYES    the eyeball UV islands overlap the face UVs (they span the whole
          atlas), so eyeballs CANNOT live on the shared albedo -- they get
          dedicated textures: out/tex/eye_left.png (+ eye_right.png when the
          measured L/R iris colors differ), built photo-derived by
          eye_texture.py with the iris disc at UV (0.5,0.5) = the eyeball's
          forward pole. s6 binds them via separate eye material(s).

The same masked photo weights + mapped TripoSR fill feed BOTH the tile-0
albedo AND vertex_colors.npy (RestMat = the UDIM tile-1+ scalp/back polys), so
the back of the head is TripoSR-hair-colored, not a flat default.

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
LUM_W = np.array([0.299, 0.587, 0.114])  # Rec.601 luma


def build_person_mask(photo, tol, erode_px, blur_sigma):
    """Soft person mask in [0,1]: 1 on the subject, 0 on the backdrop.

    Procedural + honest (no learned matting): the backdrop is the near-uniform
    color connected to the top/left/right image borders (bottom excluded -- the
    subject's torso usually exits there). Candidate pixels within --bg-tol of
    the border-median color are connected-component labeled; components that
    touch those borders are background. The person mask is then ERODED (so
    bilinear sampling near the hair/backdrop edge cannot mix white in) and
    Gaussian-feathered (soft edge -> the photo weight ramps out smoothly).
    If the backdrop is not uniform, few candidates connect and the mask
    degrades gracefully toward all-person (= previous behavior).
    """
    import cv2
    H, W = photo.shape[:2]
    border = np.concatenate([photo[0, :], photo[1, :], photo[:, 0],
                             photo[:, 1], photo[:, W - 2], photo[:, W - 1]])
    bg_rgb = np.median(border, axis=0)
    cand = (np.linalg.norm(photo - bg_rgb, axis=2) < tol).astype(np.uint8)
    _, lab = cv2.connectedComponents(cand, connectivity=4)
    touch = np.unique(np.concatenate([lab[0, :], lab[:, 0], lab[:, W - 1]]))
    touch = touch[touch != 0]  # label 0 = non-candidate pixels
    person_hard = ~np.isin(lab, touch)
    raw_frac = float(person_hard.mean())
    k = 2 * int(erode_px) + 1
    person = cv2.erode(person_hard.astype(np.uint8), np.ones((k, k), np.uint8))
    person = cv2.GaussianBlur(person.astype(np.float64), (0, 0), blur_sigma)
    info = {"enabled": True, "bg_rgb": np.round(bg_rgb, 3).tolist(),
            "tol": tol, "erode_px": int(erode_px), "blur_sigma": blur_sigma,
            "person_frac_raw": raw_frac,
            "person_frac_soft": float(person.mean())}
    return person, info


def sample_clay(tree, ccols, pts, k):
    """(dist_to_nearest, color) -- inverse-distance mean color of the k
    nearest clay vertices (mild denoise; stays local: clay ~0.15 cm spacing)."""
    dist, idx = tree.query(pts, k=k)
    if k == 1:
        return dist, ccols[idx]
    w = 1.0 / np.maximum(dist, 1e-6)
    col = (ccols[idx] * w[..., None]).sum(axis=1) / w.sum(axis=1)[..., None]
    return dist[:, 0], col


def fit_clay_to_photo(clay_c, photo_c, trim, min_pairs=2000):
    """Per-channel affine map a*c+b minimizing ||a*clay+b - photo|| over the
    texels where BOTH sources exist (front face + visible front hair). This is
    the seam killer: TripoSR's hallucinated palette is measurably washed out;
    mapping it onto the photo's palette makes the hairline/jaw blend a no-op
    color-wise. Two-pass residual-trimmed least squares; slope clamped to
    [0.4, 2.5] (offset refit after clamping). Falls back to identity when the
    overlap is too small to be trustworthy."""
    a, b = np.ones(3), np.zeros(3)
    n = int(len(clay_c))
    if n < min_pairs:
        return a, b, {"applied": False, "pairs": n,
                      "reason": f"pairs < {min_pairs}"}
    for ch in range(3):
        x, y = clay_c[:, ch], photo_c[:, ch]
        coef = np.array([1.0, 0.0])
        for _ in range(2):
            A = np.stack([x, np.ones_like(x)], axis=1)
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            r = np.abs(A @ coef - y)
            keep = r <= np.quantile(r, 1.0 - trim)
            x, y = x[keep], y[keep]
        a[ch] = float(np.clip(coef[0], 0.4, 2.5))
        b[ch] = float(np.mean(y) - a[ch] * np.mean(x))
    return a, b, {"applied": True, "pairs": n,
                  "gain": np.round(a, 3).tolist(),
                  "offset": np.round(b, 3).tolist()}


def fit_clay_luma_chroma(clay_c, photo_c, trim, min_pairs=2000):
    """Chroma-PRESERVING clay->photo transfer (the --clay-transfer lumachroma
    default). The legacy per-channel RGB affine flattens TripoSR's already-
    washed palette further (measured gain ~[0.96,0.90,0.94] -> desaturation).
    Here the color is decomposed into luma (Rec.601) + chroma (RGB - luma):
      LUMA    residual-trimmed affine LSQ onto the photo's luma over the
              overlap texels -- brightness lands on the photo's, so the
              hairline/jaw feather still has no brightness step;
      CHROMA  kept from the clay, scaled per channel by the photo/clay chroma
              std ratio (clamped [0.5, 3.5]) + mean-offset onto the photo's
              chroma -- TripoSR's per-vertex color VARIATION survives and is
              amplified to photo-like saturation instead of being flattened.
    Falls back to no-op (params None) when the overlap is too small."""
    n = int(len(clay_c))
    if n < min_pairs:
        return None, {"applied": False, "mode": "lumachroma", "pairs": n,
                      "reason": f"pairs < {min_pairs}"}
    x, y = clay_c @ LUM_W, photo_c @ LUM_W
    coef = np.array([1.0, 0.0])
    for _ in range(2):
        A = np.stack([x, np.ones_like(x)], axis=1)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        r = np.abs(A @ coef - y)
        keep = r <= np.quantile(r, 1.0 - trim)
        x, y = x[keep], y[keep]
    aL = float(np.clip(coef[0], 0.4, 2.5))
    bL = float(np.mean(y) - aL * np.mean(x))
    dev_c = clay_c - (clay_c @ LUM_W)[:, None]
    dev_p = photo_c - (photo_c @ LUM_W)[:, None]
    g, o = np.ones(3), np.zeros(3)
    for ch in range(3):
        g[ch] = np.clip(float(dev_p[:, ch].std())
                        / max(float(dev_c[:, ch].std()), 1e-6), 0.5, 3.5)
        o[ch] = float(dev_p[:, ch].mean()) - g[ch] * float(dev_c[:, ch].mean())
    params = {"aL": aL, "bL": bL, "g": g, "o": o}
    return params, {"applied": True, "mode": "lumachroma", "pairs": n,
                    "luma_gain": round(aL, 3), "luma_offset": round(bL, 3),
                    "chroma_gain": np.round(g, 3).tolist(),
                    "chroma_offset": np.round(o, 3).tolist()}


def apply_luma_chroma(params, c):
    lum = c @ LUM_W
    dev = c - lum[:, None]
    lum_m = params["aL"] * lum + params["bL"]
    return np.clip(lum_m[:, None] + dev * params["g"] + params["o"], 0.0, 1.0)


def sat_metrics(c):
    """HSV-style saturation + chroma-magnitude stats (the flat-vs-detailed
    back proof: mean_sat = how colorful, chroma_std = tonal variation)."""
    if len(c) == 0:
        return {"mean_sat": None, "chroma_mean": None, "chroma_std": None}
    mx, mn = c.max(axis=1), c.min(axis=1)
    dev = np.linalg.norm(c - (c @ LUM_W)[:, None], axis=1)
    return {"mean_sat": float(((mx - mn) / np.maximum(mx, 1e-6)).mean()),
            "chroma_mean": float(dev.mean()), "chroma_std": float(dev.std())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=P.OUT)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--no-expression", action="store_true",
                    help="project onto the neutral instead of the expressed fit")
    ap.add_argument("--ndotv-lo", type=float, default=0.08,
                    help="photo feather start (cos of the grazing angle); "
                         "0.08/0.60 spans ~53..85 deg -- a wide soft band so "
                         "the jaw/hairline photo->clay transition never reads "
                         "as a stripe (was 0.15/0.50)")
    ap.add_argument("--ndotv-hi", type=float, default=0.60)
    ap.add_argument("--zbuf-eps", type=float, default=0.4, help="cm depth tolerance")
    ap.add_argument("--zbuf-max-px", type=int, default=1200)
    ap.add_argument("--clay-max-dist", type=float, default=6.0,
                    help="cm; farther nearest-clay-vertex = no clay color")
    ap.add_argument("--clay-knn", type=int, default=8,
                    help="k nearest clay verts averaged per color sample")
    ap.add_argument("--gutter", type=int, default=12, help="UV gutter padding px")
    ap.add_argument("--no-mirror", action="store_true",
                    help="disable X-mirrored photo fallback for occluded texels")
    ap.add_argument("--central-radius", type=float, default=5.0,
                    help="cm around the nose tip for the central-face sanity gate")
    ap.add_argument("--strict", action="store_true",
                    help="exit nonzero if the central-face sanity gate fails")
    ap.add_argument("--no-bg-mask", action="store_true",
                    help="disable the procedural backdrop mask (photo pixels "
                         "may then paint the backdrop onto silhouette texels)")
    ap.add_argument("--bg-tol", type=float, default=0.12,
                    help="color distance to the border median that counts as "
                         "backdrop candidate")
    ap.add_argument("--bg-erode-px", type=int, default=3)
    ap.add_argument("--bg-blur-sigma", type=float, default=3.0,
                    help="person-mask feather; 3.0 = softer silhouette edge "
                         "(was 2.0)")
    ap.add_argument("--no-clay-match", action="store_true",
                    help="skip the clay->photo palette transfer entirely (raw "
                         "TripoSR colors are measurably washed out)")
    ap.add_argument("--clay-transfer", choices=("lumachroma", "affine"),
                    default="lumachroma",
                    help="lumachroma (default) matches the clay's LUMA to the "
                         "photo but PRESERVES+rescales the clay's per-vertex "
                         "chroma (hair keeps color variation); affine is the "
                         "legacy per-channel RGB fit (desaturates, kept for A/B)")
    ap.add_argument("--hair-knn", type=int, default=2,
                    help="sharper k-NN for clay sampling inside the hair zone "
                         "(k=2 keeps strand-scale tonal detail; elsewhere "
                         "--clay-knn=8 denoises)")
    ap.add_argument("--hair-chroma", type=float, default=1.2,
                    help="scale of TripoSR's LOCAL chroma deviation (about the "
                         "hair-zone mean) added to the photo-hair base color "
                         "in the hair zone; 0 = legacy flat palette")
    ap.add_argument("--back-sat", type=float, default=1.35,
                    help="saturation lift over the hair/back zone (feathered "
                         "by backfacing-ness; counters the residual TripoSR "
                         "desaturation); 1.0 = off")
    ap.add_argument("--match-min-w", type=float, default=0.4,
                    help="min photo weight for a texel to join the palette fit")
    ap.add_argument("--match-trim", type=float, default=0.2,
                    help="fraction of worst residuals dropped per fit pass")
    ap.add_argument("--no-cap-match", action="store_true",
                    help="skip the scalp-cap hair-palette transfer")
    ap.add_argument("--cap-lo", type=float, default=0.35,
                    help="cap feather start: brow_y + lo*(brow_y - chin_y)")
    ap.add_argument("--cap-hi", type=float, default=0.55,
                    help="cap feather full-on: brow_y + hi*(brow_y - chin_y)")
    ap.add_argument("--cap-min-pairs", type=int, default=1500,
                    help="min photo+clay cap texels to trust the cap transfer")
    ap.add_argument("--cap-min-w", type=float, default=0.12,
                    help="min photo weight for cap-fit texels (lower than "
                         "--match-min-w: the scalp top is grazing by nature, "
                         "and palette STATISTICS tolerate mild stretching)")
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

    # ---- person mask: keep the backdrop out of every photo sample
    if args.no_bg_mask:
        pmask, bg_info = np.ones((H, W)), {"enabled": False}
    else:
        pmask, bg_info = build_person_mask(photo, args.bg_tol,
                                           args.bg_erode_px, args.bg_blur_sigma)
    print(f"[s5] person mask: {bg_info}")
    pmask3 = pmask[..., None]

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
    # inpaint instead of stretching silhouette photo pixels across them;
    # texels whose projection lands on the BACKDROP (person mask ~0) get w=0
    # too -- that is the white-bonnet path, now closed.
    mval = bilinear_sample(pmask3, uv_t)[:, 0]
    visible = depth_ok & (ndotv > args.ndotv_lo) & (mval > 0.05)
    w_photo = smoothstep(ndotv, args.ndotv_lo, args.ndotv_hi) * np.clip(mval, 0.0, 1.0)
    w_photo[~visible] = 0.0
    photo_col = bilinear_sample(photo, uv_t)
    n_graze = int((depth_ok & (ndotv > 0) & (ndotv <= args.ndotv_lo)).sum())
    n_bg = int((depth_ok & (ndotv > args.ndotv_lo) & (mval <= 0.05)).sum())
    print(f"[s5] photo-visible texels: {int((w_photo > 0).sum())} "
          f"(depth-passing {int(depth_ok.sum())}, grazing-rejected {n_graze}, "
          f"backdrop-rejected {n_bg})")

    # ---- TripoSR clay fill (hair + back/sides of head), palette-matched
    clay_col = np.zeros_like(photo_col)
    clay_ok = np.zeros(K, dtype=bool)
    gain, offs, lc_params = np.ones(3), np.zeros(3), None
    match_info = {"applied": False, "reason": "no clay"}
    cap_info = {"applied": False, "reason": "no clay"}
    ctree, ccols = None, None
    clay_ply = Path(args.out) / "clay" / "clay_aligned.ply"
    if clay_ply.is_file():
        import trimesh
        from scipy.spatial import cKDTree
        cm = trimesh.load(clay_ply, process=False, force="mesh")
        cverts = np.asarray(cm.vertices)
        ccols = np.asarray(cm.visual.vertex_colors, dtype=np.float64)[:, :3] / 255.0
        ctree = cKDTree(cverts)
        dist, clay_raw = sample_clay(ctree, ccols, pos, args.clay_knn)
        clay_ok = dist < args.clay_max_dist
        clay_col = clay_raw
        if not args.no_clay_match:
            pairs = (reg <= 1) & clay_ok & (w_photo >= args.match_min_w)
            if args.clay_transfer == "lumachroma":
                lc_params, match_info = fit_clay_luma_chroma(
                    clay_raw[pairs], photo_col[pairs], args.match_trim)
                if lc_params is not None:
                    clay_col = apply_luma_chroma(lc_params, clay_raw)
            else:
                gain, offs, match_info = fit_clay_to_photo(
                    clay_raw[pairs], photo_col[pairs], args.match_trim)
                if match_info["applied"]:
                    clay_col = np.clip(clay_raw * gain + offs, 0.0, 1.0)
        # hair-zone palette (see module docstring): MEASURED, no fitted
        # distribution transfer can rescue TripoSR's hidden-crown grey-beige
        # (a global affine barely moves it; mean/std amplified its INVERTED
        # chroma into green-grey). In the hair zone the palette is therefore
        # the measured photo hair (cap texels the photo does see) and TripoSR
        # contributes only luminance variation. Hair zone = scalp cap above
        # the landmark-measured hairline + skull back above ear level.
        lmk_pos = expressed[z["lmk_verts"]]
        lmk_y = lmk_pos[:, 1]
        brow_y, chin_y = float(lmk_y[17:27].mean()), float(lmk_y[8])
        cap0 = brow_y + args.cap_lo * (brow_y - chin_y)
        cap1 = brow_y + args.cap_hi * (brow_y - chin_y)
        y_ear = float(lmk_y[[0, 16]].mean())   # iBUG 0/16: jaw top by the ears
        z_ear = float(lmk_pos[[0, 16], 2].mean())

        def hair_zone_w(p):
            # crown above the hairline; plus everything above ear level that
            # sits behind the temple plane (z gate keeps the face front out).
            # The fill only ever shows where the photo cannot see (final =
            # w_photo*photo + (1-w_photo)*fill), and hidden above-ear surface
            # is hair on any subject that has hair -- bald self-corrects
            # because the palette is measured from the visible cap.
            crown = smoothstep(p[:, 1], cap0, cap1)
            above_ear = (smoothstep(p[:, 1], y_ear + 0.5, y_ear + 3.0)
                         * smoothstep(z_ear + 3.0 - p[:, 2], 0.0, 2.0))
            return np.maximum(crown, above_ear)

        cap_mu_p, cap_mu_lc, cap_dev_mu = None, None, None
        w_cap = hair_zone_w(pos)
        # sharper k-NN inside the hair zone: k=8 averages away the strand-
        # scale tonal variation that sells hair; k=2 keeps it (denoise stays
        # k=8 everywhere else).
        clay_sharp = clay_raw
        hz = w_cap > 0.0
        if args.hair_knn != args.clay_knn and hz.any():
            _, cs = sample_clay(ctree, ccols, pos[hz], args.hair_knn)
            clay_sharp = clay_raw.copy()
            clay_sharp[hz] = cs
        if not args.no_cap_match:
            cpair = ((reg <= 1) & clay_ok & (w_photo >= args.cap_min_w)
                     & (w_cap > 0.5))
            n_cpair = int(cpair.sum())
            if n_cpair >= args.cap_min_pairs:
                # anchor the palette on the 20-50% luminance band of the
                # visible cap: the photo's crown is lit from the front-top,
                # so its MEAN is highlight-biased; the hidden scalp continues
                # the hairline's base/shadow tone, not the highlight.
                pc_pair = photo_col[cpair]
                lum_p = pc_pair @ LUM_W
                q20, q50 = np.quantile(lum_p, [0.2, 0.5])
                band = (lum_p >= q20) & (lum_p <= q50)
                cap_mu_p = pc_pair[band].mean(0)
                lum = clay_sharp @ LUM_W
                cap_mu_lc = float(max(lum[cpair].mean(), 1e-6))
                ratio = np.clip(lum / cap_mu_lc, 0.6, 1.4)
                cap_mapped = cap_mu_p[None] * ratio[:, None]
                if args.hair_chroma > 0:
                    # MODULATE the photo-hair base by TripoSR's LOCAL chroma
                    # deviation about the applied-zone mean (mean-free: the
                    # base hue stays the measured photo hair; the deviation
                    # adds the crown's tonal variation instead of a flat fill)
                    dev = clay_sharp - lum[:, None]
                    zone = clay_ok & (w_cap > 0.5)
                    cap_dev_mu = (dev[zone].mean(0) if zone.any()
                                  else dev[cpair].mean(0))
                    cap_mapped = cap_mapped + args.hair_chroma * (dev - cap_dev_mu)
                cap_mapped = np.clip(cap_mapped, 0.0, 1.0)
                clay_col = ((1.0 - w_cap[:, None]) * clay_col
                            + w_cap[:, None] * cap_mapped)
                cap_info = {"applied": True, "pairs": n_cpair,
                            "mode": "photo-hair palette x TripoSR luminance "
                                    "+ local chroma deviation",
                            "cap_y_cm": [round(cap0, 2), round(cap1, 2)],
                            "skull_back": {"y_ear": round(y_ear, 2),
                                           "z_ear": round(z_ear, 2)},
                            "photo_hair_mu": np.round(cap_mu_p, 3).tolist(),
                            "clay_cap_lum_mu": round(cap_mu_lc, 3),
                            "hair_knn": args.hair_knn,
                            "hair_chroma": args.hair_chroma,
                            "clay_dev_mu": (np.round(cap_dev_mu, 4).tolist()
                                            if cap_dev_mu is not None else None)}
            else:
                cap_info = {"applied": False, "pairs": n_cpair,
                            "reason": f"cap pairs < {args.cap_min_pairs}"}
        else:
            cap_info = {"applied": False, "reason": "--no-cap-match"}
        # controlled saturation lift over the hair/back zone (feathered by
        # backfacing-ness so the photo-lit front is untouched): counters the
        # residual desaturation of TripoSR's hallucinated palette. Applied to
        # the FILL only -- at the jaw feather the fill is palette-matched and
        # w_back ramps from 0, so no chroma step appears at the seam.
        if args.back_sat != 1.0:
            w_back = np.maximum(w_cap, smoothstep(-ndotv, 0.0, 0.25))
            lum_cc = clay_col @ LUM_W
            sat = 1.0 + (args.back_sat - 1.0) * w_back
            clay_col = np.clip(lum_cc[:, None]
                               + sat[:, None] * (clay_col - lum_cc[:, None]),
                               0.0, 1.0)
        print(f"[s5] clay-fillable texels: {int(clay_ok.sum())}; "
              f"palette match: {match_info}; cap match: {cap_info}; "
              f"back-sat lift: {args.back_sat}")
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
        mv_m = bilinear_sample(pmask3, uv_m)[:, 0]
        ok_m = (inb_m & (d_m >= zbuf[ym, xm] - args.zbuf_eps)
                & (nv_m > args.ndotv_lo) & (mv_m > 0.5))
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
    # s6 gives those polys a vertex-colored material instead. SAME recipe as
    # the albedo: masked photo where visible, palette-matched TripoSR clay
    # everywhere it has coverage (this IS the back/scalp -- hair, not a pale
    # default), skin_mean ONLY where the clay has no coverage (e.g. the neck
    # bottom below the clay's extent), honest flat interior defaults.
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
    mval_v = bilinear_sample(pmask3, uv_all)[:, 0]
    vis_v = (inb_v & (depth_all >= zbuf[yi_v, xi_v] - args.zbuf_eps)
             & (nv_v > args.ndotv_lo) & (mval_v > 0.05))
    w_v = smoothstep(nv_v, args.ndotv_lo, args.ndotv_hi) * np.clip(mval_v, 0.0, 1.0)
    w_v[~vis_v] = 0.0
    photo_v = bilinear_sample(photo, uv_all)
    skin_mean = (photo_v[ext_v & vis_v].mean(0)
                 if (ext_v & vis_v).any() else np.array([0.75, 0.6, 0.55]))
    # per-vertex fill source: 0 interior-default, 1 skin-default (no clay
    # coverage), 2 triposr clay -- photo participation tracked via w_v
    v_src = np.zeros(len(verts), dtype=np.uint8)
    vcol[ext_v] = skin_mean
    v_src[ext_v] = 1
    from common import EYE_SOCKETS
    vcol[EYE_SOCKETS[0]:EYE_SOCKETS[1]] = skin_mean * 0.8  # shadowed lid skin
    if ctree is not None:
        # SAME improved recipe as the albedo: lumachroma transfer, sharper
        # hair-zone k-NN + chroma-deviation modulation, back-sat lift --
        # RestMat (tile-1+ scalp/back polys) must stay consistent with tile 0.
        pos_v = expressed[ext_v]
        dist_v, ccol_raw_v = sample_clay(ctree, ccols, pos_v, args.clay_knn)
        if lc_params is not None:
            ccol_v = apply_luma_chroma(lc_params, ccol_raw_v)
        else:
            ccol_v = np.clip(ccol_raw_v * gain + offs, 0.0, 1.0)
        w_cap_v = hair_zone_w(pos_v)
        sharp_v = ccol_raw_v
        hz_v = w_cap_v > 0.0
        if args.hair_knn != args.clay_knn and hz_v.any():
            _, cs_v = sample_clay(ctree, ccols, pos_v[hz_v], args.hair_knn)
            sharp_v = ccol_raw_v.copy()
            sharp_v[hz_v] = cs_v
        if cap_info.get("applied"):
            lum_v = sharp_v @ LUM_W
            ratio_v = np.clip(lum_v / cap_mu_lc, 0.6, 1.4)
            cap_v = cap_mu_p[None] * ratio_v[:, None]
            if args.hair_chroma > 0 and cap_dev_mu is not None:
                cap_v = cap_v + args.hair_chroma * (
                    (sharp_v - lum_v[:, None]) - cap_dev_mu)
            cap_v = np.clip(cap_v, 0.0, 1.0)
            ccol_v = ((1.0 - w_cap_v[:, None]) * ccol_v
                      + w_cap_v[:, None] * cap_v)
        if args.back_sat != 1.0:
            w_back_v = np.maximum(w_cap_v,
                                  smoothstep(-nv_v[ext_v], 0.0, 0.25))
            lum_cv = ccol_v @ LUM_W
            sat_v = 1.0 + (args.back_sat - 1.0) * w_back_v
            ccol_v = np.clip(lum_cv[:, None]
                             + sat_v[:, None] * (ccol_v - lum_cv[:, None]),
                             0.0, 1.0)
        near = dist_v < args.clay_max_dist
        tmp = vcol[ext_v]
        tmp[near] = ccol_v[near]
        vcol[ext_v] = tmp
        tsrc = v_src[ext_v]
        tsrc[near] = 2
        v_src[ext_v] = tsrc
    m_v = ext_v & (w_v > 0)
    vcol[m_v] = (w_v[m_v, None] * photo_v[m_v]
                 + (1 - w_v[m_v, None]) * vcol[m_v])
    np.save(od / "vertex_colors.npy", vcol.astype(np.float32))
    print(f"[s5] per-vertex tile-fallback colors -> vertex_colors.npy "
          f"(photo-visible {int((ext_v & (w_v > 0)).sum())} skin verts, "
          f"skin mean {np.round(skin_mean, 3)})")

    # ---- back/scalp coverage metrics (the proof the back is TripoSR-colored,
    # not pale-default). Vertex level = RestMat (all head_neck polys are UDIM
    # tile-1+); texel level = backfacing exterior texels on the tile-0 albedo.
    hn0, hn1 = ICT_REGIONS["head_neck"]
    hn_nophoto = (w_v[hn0:hn1] == 0)
    hn_src = v_src[hn0:hn1]
    back_scalp = np.zeros(len(verts), dtype=bool)
    back_scalp[hn0:hn1] = True
    back_scalp &= verts[:, 2] < 0.0  # behind the ears
    vertex_fill = {
        "ext_verts": int(ext_v.sum()),
        "photo_blend_frac": float((w_v[ext_v] > 0).mean()),
        "head_neck": {
            "verts": int(hn1 - hn0),
            "triposr_frac": float((hn_src == 2).mean()),
            "default_frac": float((hn_src == 1).mean()),
            "photo_touched_frac": float((w_v[hn0:hn1] > 0).mean()),
            "no_photo_triposr_frac":
                float((hn_src[hn_nophoto] == 2).mean()) if hn_nophoto.any() else None,
            "no_photo_default_frac":
                float((hn_src[hn_nophoto] == 1).mean()) if hn_nophoto.any() else None,
        },
        "back_scalp_mean_rgb": np.round(vcol[back_scalp].mean(0), 3).tolist()
                               if back_scalp.any() else None,
        "back_scalp_saturation": sat_metrics(vcol[back_scalp]),
        "skin_mean_rgb": np.round(skin_mean, 3).tolist(),
    }
    print(f"[s5] vertex fill: {vertex_fill}")
    back_t = ext & (ndotv <= 0.0)
    if back_t.any():
        back_region = {
            "texels": int(back_t.sum()),
            "triposr_frac": float((src[back_t] == 2).mean()),
            "blend_frac": float((src[back_t] == 3).mean()),
            "photo_frac": float((src[back_t] == 1).mean()),
            "mirror_frac": float((src[back_t] == 5).mean()),
            "default_frac": float((src[back_t] == 4).mean()),
            "hole_frac_pre_inpaint": float(holes[back_t].mean()),
            "mean_rgb": np.round(col[back_t].mean(0), 3).tolist(),
            "saturation": sat_metrics(col[back_t]),
        }
    else:
        back_region = {"texels": 0}
    print(f"[s5] back-region texels (exterior, dot(n,view)<=0): {back_region}")
    if ctree is not None:
        hz_t = ext & clay_ok & (w_cap > 0.5)
        hair_zone = {"texels": int(hz_t.sum()),
                     "mean_rgb": (np.round(col[hz_t].mean(0), 3).tolist()
                                  if hz_t.any() else None),
                     "saturation": sat_metrics(col[hz_t])}
    else:
        hair_zone = {"texels": 0}
    print(f"[s5] hair-zone texels (composed): {hair_zone}")
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
    Image.fromarray((np.clip(pmask, 0, 1) * 255).astype(np.uint8)).save(
        od / "debug_person_mask.png")
    region_names = ["face", "head_neck", "interior_a", "teeth", "eyeballs",
                    "interior_b"]
    save_json(od / "bake_metrics.json", {
        "texture_size": T, "texels_covered": K,
        "photo": int((src == 1).sum()), "clay": int((src == 2).sum()),
        "blend": int((src == 3).sum()), "interior_default": int((src == 4).sum()),
        "mirror": int((src == 5).sum()),
        "inpainted_holes": int(holes.sum()),
        "sources_frac": {  # of covered texels; clay/blend = TripoSR-sourced
            "photo": float((src == 1).mean()),
            "triposr": float((src == 2).mean()),
            "blend": float((src == 3).mean()),
            "interior_default": float((src == 4).mean()),
            "mirror": float((src == 5).mean()),
            "holes_pre_inpaint": float(holes.mean()),
        },
        "person_mask": bg_info,
        "backdrop_rejected_texels": n_bg,
        "params": {"clay_transfer": args.clay_transfer,
                   "hair_knn": args.hair_knn, "clay_knn": args.clay_knn,
                   "hair_chroma": args.hair_chroma, "back_sat": args.back_sat,
                   "ndotv_lo": args.ndotv_lo, "ndotv_hi": args.ndotv_hi,
                   "bg_blur_sigma": args.bg_blur_sigma},
        "clay_match": match_info,
        "cap_match": cap_info,
        "winding": {"frac_ndotv_neg_on_depth_pass": frac_neg,
                    "facing_sign": facing_sign,
                    "note": "sign measured from depth-passing exterior texels, "
                            "never assumed from OBJ winding"},
        "grazing_rejected": n_graze,
        "region_texels": {n: int((reg == i).sum())
                          for i, n in enumerate(region_names)},
        "central_face": cf,
        "back_region": back_region,
        "hair_zone": hair_zone,
        "vertex_fill": vertex_fill,
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
