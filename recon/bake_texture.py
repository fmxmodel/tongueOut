"""Stage 3 -- bake a per-subject albedo from the photo into FLAME UV space.

POD-ONLY (pod_guard: CUDA torch + PyTorch3D). METHOD FIXED by the licensing
gate (out/compliance_report.md B1-2/B1-3): the texture is baked from the input
photo ONLY. NO statistical albedo prior -- no MPI FLAME texture space, no
FLAME_albedo_from_BFM, no AlbedoMM, no Basel. Occlusion completion uses
bilateral MIRROR SYMMETRY + CLASSICAL inpainting (cv2.inpaint/TELEA).
Honest caveat (recorded for QA): what the photo provides is shaded appearance,
not physically de-lit albedo; de-lighting priors are barred, so the baked map
IS the lit appearance. glTF baseColor consumes it as-is (sRGB).

PIPELINE
  1. Rebuild the PHOTO-STATE mesh (fitted betas + photo expression/jaw/global/
     transl from id_params.npz) -> camera-space vertices (OpenCV convention,
     identical to fit_flame.py).
  2. PyTorch3D rasterization of that mesh from the fitted camera
     (cameras_from_opencv_projection with R=I, t=0 since the mesh is already
     camera-space) -> per-pixel depth for occlusion testing.
  3. UV-space rasterization (small self-contained numpy scanline rasterizer,
     zero renderer-convention risk): texel -> (face, barycentric).
  4. Per texel: 3D point -> project into the photo -> visible iff in-bounds,
     depth-consistent (DEPTH_TOL), camera-facing (auto-oriented winding check),
     non-grazing (COS_MIN). Visible texels sample the photo bilinearly.
  5. Invisible texels: mirror-symmetry fill (template-space reflection x->-x,
     cKDTree nearest match), then cv2.inpaint for the residue, then mean-color
     fill outside all UV islands.

UV SOURCE (pkl-only FLAME 2023 Open; measured on the pod 2026-07-05): the Open
release ships NO UV. The layout is resolved by recon/uv_unwrap.py -- a staged
UV template obj if the operator provided one (optional override), else a
deterministic clean-room xatlas (MIT) unwrap of the pkl's v_template+f. The
resolved layout is persisted in uv_coords.npz (with an explicit `uv_source`
field) and is the single UV authority for blender_build_rig.py.

UV/IMAGE CONVENTION (consumed by the GLB builder):
  albedo.png row r corresponds to v = 1 - (r + 0.5)/TEX_RES (i.e. OBJ v=1 is
  the TOP row -- standard image convention). uv_coords.npz repeats this note.

Run:  python -m recon.bake_texture   (after recon.fit_flame)
"""

import json
import sys
import time

import numpy as np

from . import config as C
from .pod_guard import require_cuda_torch


# --------------------------------------------------------------------------
# self-contained UV rasterizer (numpy; ~10k faces -> fine)
# --------------------------------------------------------------------------
def rasterize_uv(verts_uvs: np.ndarray, faces_uv: np.ndarray, tex: int):
    """verts_uvs (T,2) in [0,1] (OBJ convention, v up); faces_uv (F,3).
    Returns face_map (tex,tex) int64 (-1 = uncovered) and bary_map
    (tex,tex,3) float32, in image convention (row 0 = v=1)."""
    xy = np.empty((verts_uvs.shape[0], 2), dtype=np.float64)
    xy[:, 0] = verts_uvs[:, 0] * tex - 0.5              # u -> x (col)
    xy[:, 1] = (1.0 - verts_uvs[:, 1]) * tex - 0.5      # v -> y (row), v=1 at top
    tri = xy[faces_uv]                                   # (F,3,2)

    face_map = np.full((tex, tex), -1, dtype=np.int64)
    bary_map = np.zeros((tex, tex, 3), dtype=np.float32)
    overlaps = 0
    eps = 1e-7

    for f in range(tri.shape[0]):
        (x0, y0), (x1, y1), (x2, y2) = tri[f]
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-12:
            continue  # degenerate UV triangle
        xa = max(int(np.floor(min(x0, x1, x2))), 0)
        xb = min(int(np.ceil(max(x0, x1, x2))), tex - 1)
        ya = max(int(np.floor(min(y0, y1, y2))), 0)
        yb = min(int(np.ceil(max(y0, y1, y2))), tex - 1)
        if xa > xb or ya > yb:
            continue
        xs, ys = np.meshgrid(np.arange(xa, xb + 1), np.arange(ya, yb + 1))
        w0 = ((y1 - y2) * (xs - x2) + (x2 - x1) * (ys - y2)) / denom
        w1 = ((y2 - y0) * (xs - x2) + (x0 - x2) * (ys - y2)) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -eps) & (w1 >= -eps) & (w2 >= -eps)
        if not inside.any():
            continue
        rr, cc = ys[inside], xs[inside]
        overlaps += int((face_map[rr, cc] >= 0).sum())
        face_map[rr, cc] = f
        bary_map[rr, cc, 0] = w0[inside]
        bary_map[rr, cc, 1] = w1[inside]
        bary_map[rr, cc, 2] = w2[inside]

    covered = int((face_map >= 0).sum())
    print(f"[bake] UV rasterized at {tex}x{tex}: {covered} texels covered "
          f"({100.0 * covered / tex / tex:.1f}%), {overlaps} island-overlap writes")
    return face_map, bary_map


def main() -> None:
    torch = require_cuda_torch()
    import cv2
    from PIL import Image
    from scipy.spatial import cKDTree

    C.ensure_out_dirs()
    t0 = time.time()
    device = torch.device(C.DEVICE)

    # ---- inputs ---------------------------------------------------------------
    for f, hint in [(C.ID_PARAMS_NPZ, "recon.fit_flame"), (C.CANONICAL_IMAGE, "recon.landmarks"),
                    (C.FACES_NPY, "recon.fit_flame")]:
        if not f.is_file():
            sys.exit(f"[bake FATAL] {f} missing -- run `python -m {hint}` first.")
    idp = np.load(C.ID_PARAMS_NPZ, allow_pickle=True)
    image_rgb = np.asarray(Image.open(C.CANONICAL_IMAGE).convert("RGB"))
    h, w = [int(x) for x in idp["image_hw"]]
    assert image_rgb.shape[:2] == (h, w), "canonical image / id_params desync"
    fx, fy, cx, cy = [float(x) for x in idp["camera_fx_fy_cx_cy"]]
    faces_contract = np.load(C.FACES_NPY).astype(np.int64)

    # ---- rebuild the photo-state mesh (camera space) -----------------------------
    from .flame_model import FlameModel

    model_path = C.find_flame_file(C.FLAME_MODEL_CANDIDATES, "FLAME shape model pkl")
    flame = FlameModel(model_path, n_shape=C.N_SHAPE, n_expr=C.N_EXPR, device=C.DEVICE)
    if not np.array_equal(flame.faces.astype(np.int64), faces_contract):
        sys.exit("[bake FATAL] FLAME pkl faces != out/recon/faces.npy -- topology "
                 "contract broken between fit and bake. STOP.")

    tt = lambda a: torch.tensor(np.asarray(a, dtype=np.float32)[None], device=device)
    with torch.no_grad():
        verts_cam = flame.decode(
            betas=tt(idp["betas"]), expression=tt(idp["photo_expression"]),
            global_orient=tt(idp["photo_global_orient"]), jaw_pose=tt(idp["photo_jaw_pose"]),
            transl=tt(idp["photo_transl"]),
        )[0]                                             # (V,3) camera-space, torch
    verts_np = verts_cam.cpu().numpy().astype(np.float64)
    print(f"[bake] photo-state mesh: z range [{verts_np[:,2].min():.3f}, "
          f"{verts_np[:,2].max():.3f}] m")

    # ---- UV layout (staged template override if present, else clean-room
    # xatlas unwrap of the pkl's v_template+f -- the FLAME 2023 Open release
    # ships NO UV; see recon/uv_unwrap.py for the provenance discipline) -------
    from .uv_unwrap import load_or_generate_uv

    verts_uvs, faces_uv, faces_v, uv_source = load_or_generate_uv(
        flame.np_v_template.astype(np.float64), faces_contract)
    # load_or_generate_uv asserts faces_v == faces.npy internally; re-assert
    # here so the bake's own invariant is measured, not delegated.
    if not np.array_equal(faces_v, faces_contract):
        sys.exit("[bake FATAL] UV layout geometry faces != faces.npy -- topology "
                 "contract broken. STOP.")

    # ---- image-space depth via PyTorch3D (occlusion reference) ------------------------
    from pytorch3d.ops import interpolate_face_attributes
    from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
    from pytorch3d.structures import Meshes
    from pytorch3d.utils import cameras_from_opencv_projection

    K = torch.tensor([[[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]],
                     dtype=torch.float32, device=device)
    cams = cameras_from_opencv_projection(
        R=torch.eye(3, device=device)[None],
        tvec=torch.zeros(1, 3, device=device),
        camera_matrix=K,
        image_size=torch.tensor([[h, w]], device=device),
    )
    mesh_t = Meshes(verts=[verts_cam], faces=[torch.as_tensor(faces_v, device=device)])
    raster = MeshRasterizer(
        cameras=cams,
        raster_settings=RasterizationSettings(
            image_size=(h, w), blur_radius=0.0, faces_per_pixel=1, cull_backfaces=False,
        ),
    )
    frags = raster(mesh_t)
    z_attr = verts_cam[torch.as_tensor(faces_v, device=device)][:, :, 2:3]   # (F,3,1)
    depth = interpolate_face_attributes(frags.pix_to_face, frags.bary_coords, z_attr)
    depth = depth[0, :, :, 0, 0]                                             # (H,W)
    hit = frags.pix_to_face[0, :, :, 0] >= 0
    depth_map = torch.where(hit, depth, torch.full_like(depth, float("inf")))
    depth_np = depth_map.cpu().numpy()
    print(f"[bake] rasterized depth: {int(hit.sum())} px covered of {h * w}")

    # ---- UV-space rasterization -----------------------------------------------------------
    T = C.TEX_RES
    face_map, bary_map = rasterize_uv(verts_uvs, faces_uv, T)
    cov_r, cov_c = np.nonzero(face_map >= 0)
    fidx = face_map[cov_r, cov_c]                       # (N,) face per covered texel
    bary = bary_map[cov_r, cov_c].astype(np.float64)    # (N,3)

    # per-texel geometry
    tri_v = verts_np[faces_v[fidx]]                     # (N,3,3) photo-state, camera space
    pos = np.einsum("nkc,nk->nc", tri_v, bary)          # (N,3)
    tpl_np = flame.np_v_template.astype(np.float64)
    tri_t = tpl_np[faces_v[fidx]]
    pos_tpl = np.einsum("nkc,nk->nc", tri_t, bary)      # (N,3) template space (symmetric)
    e1 = tri_v[:, 1] - tri_v[:, 0]
    e2 = tri_v[:, 2] - tri_v[:, 0]
    nrm = np.cross(e1, e2)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12

    # ---- visibility -----------------------------------------------------------------------
    z = pos[:, 2]
    u = fx * pos[:, 0] / np.maximum(z, 1e-6) + cx
    v = fy * pos[:, 1] / np.maximum(z, 1e-6) + cy
    in_bounds = (z > 1e-6) & (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
    ui = np.clip(np.round(u).astype(np.int64), 0, w - 1)
    vi = np.clip(np.round(v).astype(np.int64), 0, h - 1)
    depth_ref = depth_np[vi, ui]
    depth_ok = in_bounds & np.isfinite(depth_ref) & (z <= depth_ref + C.DEPTH_TOL_M)

    signed = np.einsum("nc,nc->n", nrm, pos)            # camera at origin
    neg_frac = float((signed[depth_ok] < 0).mean()) if depth_ok.any() else 0.0
    facing = (signed < 0) if neg_frac >= 0.5 else (signed > 0)
    print(f"[bake] winding auto-orientation: {neg_frac:.2f} of depth-passing texels have "
          f"dot(n,p)<0 -> treating {'dot<0' if neg_frac >= 0.5 else 'dot>0'} as camera-facing "
          "(FLAME winding not assumed; measured)")
    cosang = np.abs(signed) / (np.linalg.norm(pos, axis=1) + 1e-12)
    visible = depth_ok & facing & (cosang >= C.COS_MIN)
    print(f"[bake] visible texels: {int(visible.sum())} / {len(visible)} covered")
    if visible.sum() < 0.02 * len(visible):
        sys.exit("[bake FATAL] <2% of covered texels are photo-visible -- the fit or the "
                 "camera reconstruction is wrong. Inspect out/recon/fit_debug/ overlays. "
                 "Refusing to emit a garbage albedo.")

    # ---- sample the photo (explicit numpy bilinear; cv2.remap's SHRT_MAX
    # output-size limit rules it out for ~1e6 scattered samples) -----------------
    def bilinear_sample(img, xs, ys):
        xs = np.clip(xs, 0.0, img.shape[1] - 1.001)
        ys = np.clip(ys, 0.0, img.shape[0] - 1.001)
        x0 = np.floor(xs).astype(np.int64); y0 = np.floor(ys).astype(np.int64)
        x1, y1 = x0 + 1, y0 + 1
        wx = (xs - x0)[:, None]; wy = (ys - y0)[:, None]
        imgf = img.astype(np.float32)
        top = imgf[y0, x0] * (1 - wx) + imgf[y0, x1] * wx
        bot = imgf[y1, x0] * (1 - wx) + imgf[y1, x1] * wx
        return np.clip(top * (1 - wy) + bot * wy, 0, 255).astype(np.uint8)

    sampled = bilinear_sample(image_rgb, u, v)                          # (N,3) uint8

    albedo = np.zeros((T, T, 3), dtype=np.uint8)
    mask = np.full((T, T), C.MASK_OUTSIDE, dtype=np.uint8)
    albedo[cov_r[visible], cov_c[visible]] = sampled[visible]
    mask[cov_r[visible], cov_c[visible]] = C.MASK_DIRECT

    # ---- mirror-symmetry fill (template-space reflection x -> -x) ----------------------------
    invalid = ~visible
    n_mirror = 0
    if invalid.any() and visible.any():
        tree = cKDTree(pos_tpl[visible])
        q = pos_tpl[invalid] * np.array([-1.0, 1.0, 1.0])
        dist, nn = tree.query(q, k=1)
        ok = dist <= C.MIRROR_MATCH_TOL_M
        src = np.nonzero(visible)[0][nn[ok]]
        dst = np.nonzero(invalid)[0][ok]
        albedo[cov_r[dst], cov_c[dst]] = sampled[src]
        mask[cov_r[dst], cov_c[dst]] = C.MASK_MIRROR
        n_mirror = int(ok.sum())
    print(f"[bake] mirror fill: {n_mirror} texels")

    # ---- classical inpaint for the residue -----------------------------------------------------
    covered_mask = (face_map >= 0)
    hole = covered_mask & ((mask == C.MASK_OUTSIDE))
    n_hole = int(hole.sum())
    if n_hole:
        bgr = cv2.cvtColor(albedo, cv2.COLOR_RGB2BGR)
        bgr = cv2.inpaint(bgr, hole.astype(np.uint8) * 255, C.INPAINT_RADIUS,
                          cv2.INPAINT_TELEA)
        albedo = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mask[hole] = C.MASK_INPAINT
    print(f"[bake] cv2.inpaint(TELEA) filled {n_hole} texels")

    # ---- outside all UV islands: mean valid color (mip-bleed guard) ----------------------------
    valid_px = mask == C.MASK_DIRECT
    mean_col = (albedo[valid_px].reshape(-1, 3).mean(axis=0).astype(np.uint8)
                if valid_px.any() else np.array([128, 128, 128], np.uint8))
    albedo[~covered_mask] = mean_col

    # ---- outputs --------------------------------------------------------------------------------
    cv2.imwrite(str(C.ALBEDO_PNG), cv2.cvtColor(albedo, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(C.ALBEDO_MASK_PNG), mask)
    np.savez(
        C.UV_COORDS_NPZ,
        verts_uvs=verts_uvs.astype(np.float32),
        faces_uv_idx=faces_uv.astype(np.int32),
        faces_verts_idx=faces_v.astype(np.int32),
        v_convention=np.array(
            "albedo.png row r <-> v = 1 - (r+0.5)/TEX_RES; OBJ v=1 is the TOP image row"
        ),
        uv_source=np.array(uv_source),
        tex_res=np.int64(T),
    )
    summary = {
        "tex_res": T,
        "texels_covered": int(covered_mask.sum()),
        "texels_direct": int((mask == C.MASK_DIRECT).sum()),
        "texels_mirror": int((mask == C.MASK_MIRROR).sum()),
        "texels_inpaint": int((mask == C.MASK_INPAINT).sum()),
        "winding_facing_sign": "dot<0" if neg_frac >= 0.5 else "dot>0",
        "uv_source": uv_source,
        "no_statistical_albedo_prior": True,
        "note_shaded_appearance": (
            "Baked map is the photo's SHADED appearance (delighting priors are "
            "license-barred); recorded for QA."
        ),
        "runtime_s": time.time() - t0,
    }
    with open(C.BAKE_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[bake] DONE -> {C.ALBEDO_PNG}, {C.ALBEDO_MASK_PNG}, {C.UV_COORDS_NPZ} "
          f"({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
