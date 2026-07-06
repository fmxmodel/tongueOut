"""Photo-derived procedural eye textures (numpy + PIL only, commercial-clean).

WHY: the ICT eyeball UVs span the whole [0,1] atlas and OVERLAP the face UVs,
so eyeballs cannot share the face albedo (s5's exterior-priority rasterization
lets the face win those texels, leaving the eyes a flat sclera color -> blank
white stare). Instead each eyeball gets its OWN small texture + material.

MEASURED FACTS this module builds on (see the eye-fix task notes):
- Each ICT eyeball is unwrapped with the iris/forward(+z) pole at UV (0.5,0.5)
  (measured pole UVs: L (0.505,0.501), R (0.495,0.501)).
- The eyeball's full UV radius from center is ~0.69; an anatomically right
  visible iris is ~0.12-0.18 of that radius (human iris half-angle ~30 deg of
  the ~180 deg the unwrap spans). Default IRIS_FRAC = 0.16.
- MediaPipe 478 landmarks: subject-RIGHT iris center = idx 468 (ring 469-472),
  subject-LEFT iris center = idx 473 (ring 474-477). iBUG eye corners:
  36/39 subject-right, 42/45 subject-left.

Everything is sampled from the input photo (iris / pupil / sclera colors,
iris pixel radius) or synthesized procedurally (radial disc, limbal ring,
fibers) -- NO external image assets, so the result stays commercial-clean.
"""

from pathlib import Path

import numpy as np

# MediaPipe iris landmark indices (measured convention, image space)
IRIS = {
    "subject_right": {"center": 468, "ring": (469, 470, 471, 472),
                      "corners_ibug": (36, 39)},
    "subject_left":  {"center": 473, "ring": (474, 475, 476, 477),
                      "corners_ibug": (42, 45)},
}

FULL_UV_R = 0.69      # measured eyeball UV radius from (0.5, 0.5)
IRIS_FRAC = 0.16      # visible-iris radius / full UV radius (tunable 0.12-0.18)
PUPIL_FRAC = 0.35     # pupil radius / iris radius
SHARE_THRESH = 0.06   # L/R iris rgb distance below which one texture is shared


def _smoothstep(x, a, b):
    t = np.clip((np.asarray(x, dtype=np.float64) - a) / max(b - a, 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _disc_pixels(photo, cx, cy, r):
    """All photo pixels within radius r of (cx, cy). (K,3) float in [0,1]."""
    H, W = photo.shape[:2]
    x0, x1 = max(int(cx - r), 0), min(int(np.ceil(cx + r)) + 1, W)
    y0, y1 = max(int(cy - r), 0), min(int(np.ceil(cy + r)) + 1, H)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((0, 3))
    xs = np.arange(x0, x1)[None, :] + 0.5
    ys = np.arange(y0, y1)[:, None] + 0.5
    m = (xs - cx) ** 2 + (ys - cy) ** 2 <= r * r
    return photo[y0:y1, x0:x1][m]


def _annulus_median(photo, cx, cy, r0, r1):
    """Per-channel median color of the annulus r0 <= r <= r1 (robust to the
    dark pupil inside and specular highlights)."""
    H, W = photo.shape[:2]
    x0, x1 = max(int(cx - r1), 0), min(int(np.ceil(cx + r1)) + 1, W)
    y0, y1 = max(int(cy - r1), 0), min(int(np.ceil(cy + r1)) + 1, H)
    xs = np.arange(x0, x1)[None, :] + 0.5
    ys = np.arange(y0, y1)[:, None] + 0.5
    d2 = (xs - cx) ** 2 + (ys - cy) ** 2
    m = (d2 >= r0 * r0) & (d2 <= r1 * r1)
    px = photo[y0:y1, x0:x1][m]
    if len(px) == 0:
        return np.array([0.3, 0.25, 0.2])
    return np.median(px, axis=0)


LUMA_W = np.array([0.299, 0.587, 0.114])


def sample_eye_colors(photo, lmk478_px, ibug68_px, side,
                      iris_chroma_boost=1.3, sclera_desat=0.6):
    """Measure iris / pupil / sclera colors + iris pixel radius for one eye.

    photo (H,W,3) float in [0,1]; side in IRIS keys. Returns dict.

    MEASURED on the reference photo: the iris tint lives in the INNER annulus
    (0.35-0.60r); annuli beyond ~0.7r are contaminated by skin/lashes (red
    channel rises). The brightest visible sclera is warm (context makes it
    read white), so it is desaturated toward its own luma before use. Both
    adjustments are parametric and documented -- colors stay photo-derived.
    """
    spec = IRIS[side]
    c = np.asarray(lmk478_px[spec["center"]], dtype=np.float64)
    ring = np.asarray(lmk478_px[list(spec["ring"])], dtype=np.float64)
    r_iris = float(np.linalg.norm(ring - c, axis=1).mean())

    # iris color: inner annulus median (robust), mild chroma boost to undo
    # the desaturation of median-pooling small blurry pixels
    iris_rgb = _annulus_median(photo, c[0], c[1], 0.35 * r_iris, 0.60 * r_iris)
    y = float(iris_rgb @ LUMA_W)
    iris_rgb = np.clip(y + (iris_rgb - y) * iris_chroma_boost, 0.0, 1.0)

    # pupil: darkest quartile of the central disc, clamped DARK
    ctr_px = _disc_pixels(photo, c[0], c[1], 0.30 * r_iris)
    if len(ctr_px):
        luma = ctr_px @ LUMA_W
        dark = ctr_px[luma <= np.quantile(luma, 0.25)]
        pupil_rgb = dark.mean(axis=0)
    else:
        pupil_rgb = iris_rgb * 0.2
    pupil_rgb = np.clip(np.minimum(pupil_rgb, iris_rgb * 0.35), 0.01, 0.12)

    # sclera: brightest decile inside the eye-opening box spanned by the
    # iBUG corners (excluding the iris disc), then desaturated toward luma
    # (measured: the photo's "white" is warm; raw use reads salmon)
    A = np.asarray(ibug68_px[spec["corners_ibug"][0]], dtype=np.float64)
    B = np.asarray(ibug68_px[spec["corners_ibug"][1]], dtype=np.float64)
    H, W = photo.shape[:2]
    x0, x1 = int(max(min(A[0], B[0]), 0)), int(min(max(A[0], B[0]), W - 1))
    yc = 0.5 * (A[1] + B[1])
    y0 = int(max(yc - 0.7 * r_iris, 0))
    y1 = int(min(yc + 0.7 * r_iris, H - 1))
    xs = np.arange(x0, x1 + 1)[None, :] + 0.5
    ys = np.arange(y0, y1 + 1)[:, None] + 0.5
    m = np.hypot(xs - c[0], ys - c[1]) > 1.15 * r_iris
    px = photo[y0:y1 + 1, x0:x1 + 1][m] if m.any() else np.zeros((0, 3))
    if len(px):
        luma = px @ LUMA_W
        sclera_rgb = np.median(px[luma >= np.quantile(luma, 0.90)], axis=0)
    else:
        sclera_rgb = np.array([0.93, 0.92, 0.90])
    y = float(sclera_rgb @ LUMA_W)
    sclera_rgb = sclera_rgb + sclera_desat * (y - sclera_rgb)
    y = float(sclera_rgb @ LUMA_W)
    if y < 0.55:  # shadowed narrow eye opening -- lift so the white reads
        sclera_rgb = np.clip(sclera_rgb * (0.55 / max(y, 1e-6)), 0, 1)

    return {"iris_rgb": iris_rgb, "pupil_rgb": pupil_rgb,
            "sclera_rgb": np.clip(sclera_rgb, 0, 1),
            "iris_radius_px": r_iris, "center_px": c.tolist()}


def compose_eye_image(size, iris_rgb, pupil_rgb, sclera_rgb,
                      iris_frac=IRIS_FRAC, pupil_frac=PUPIL_FRAC):
    """Radially symmetric eye texture (H,W,3) uint8: sclera field, soft iris
    disc at UV center (= eyeball forward pole), limbal ring, fibers, pupil.

    Radial symmetry makes the image orientation-proof (any UV flip maps it
    identically), which is exactly what a pole-centered unwrap needs.
    """
    iris_rgb = np.asarray(iris_rgb, dtype=np.float64)
    pupil_rgb = np.asarray(pupil_rgb, dtype=np.float64)
    sclera_rgb = np.asarray(sclera_rgb, dtype=np.float64)
    iris_r = iris_frac * FULL_UV_R
    pupil_r = pupil_frac * iris_r

    ax = (np.arange(size) + 0.5) / size - 0.5
    u, v = np.meshgrid(ax, ax)
    r = np.hypot(u, v)
    theta = np.arctan2(v, u)

    # sclera with a gentle vignette toward the (never really seen) back of
    # the eyeball, so the visible white keeps a soft falloff at the edges
    img = sclera_rgb[None, None, :] * (
        1.0 - 0.22 * _smoothstep(r, 0.30, FULL_UV_R))[..., None]

    # iris: radial brightness gradient (brighter collarette near the pupil)
    g = 1.10 - 0.32 * np.clip(r / iris_r, 0.0, 1.0)
    # faint deterministic radial fibers, damped near pupil and limbus
    damp = (_smoothstep(r, 0.40 * iris_r, 0.60 * iris_r)
            * (1.0 - _smoothstep(r, 0.82 * iris_r, iris_r)))
    fib = 1.0 + 0.05 * np.sin(theta * 23.0 + 2.7 * np.sin(theta * 7.0)) * damp
    # limbal ring: darken the outer iris rim
    limbal = 1.0 - 0.45 * _smoothstep(r, 0.70 * iris_r, 1.02 * iris_r)
    iris_col = np.clip((iris_rgb[None, None, :] * (g * fib * limbal)[..., None]),
                       0.0, 1.0)

    w_iris = (1.0 - _smoothstep(r, iris_r * 0.96, iris_r * 1.10))[..., None]
    img = img * (1.0 - w_iris) + iris_col * w_iris

    w_pup = (1.0 - _smoothstep(r, pupil_r * 0.80, pupil_r * 1.15))[..., None]
    img = img * (1.0 - w_pup) + pupil_rgb[None, None, :] * w_pup

    return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def build_eye_textures(photo, landmarks_npz, size=512,
                       left_subject_side="subject_right",
                       iris_frac=IRIS_FRAC, share_thresh=SHARE_THRESH):
    """Build eye texture(s) from the photo + MediaPipe landmarks.

    photo: (H,W,3) float [0,1] array, or a path. landmarks_npz: path to
    s1's landmarks.npz. left_subject_side: which PHOTO eye feeds the model's
    x<0 ("left") eyeball -- the caller measures this by projecting the eyeball
    centroids through the fitted camera (frontal photos: model x<0 lands on
    the image-left iris = the subject's RIGHT eye).

    Returns (img_left, img_right, metrics). If the two measured iris colors
    are within share_thresh (L2, [0,1] rgb), ONE shared image is built from
    the averaged colors and returned as both (metrics["shared"]=True).
    """
    if not isinstance(photo, np.ndarray):
        from PIL import Image
        photo = np.asarray(Image.open(photo).convert("RGB"),
                           dtype=np.float64) / 255.0
    z = np.load(landmarks_npz)
    lmk478_px, ibug68_px = z["lmk478_px"], z["ibug68_px"]

    assert left_subject_side in IRIS
    right_subject_side = ("subject_left" if left_subject_side == "subject_right"
                          else "subject_right")
    cl = sample_eye_colors(photo, lmk478_px, ibug68_px, left_subject_side)
    cr = sample_eye_colors(photo, lmk478_px, ibug68_px, right_subject_side)

    dist = float(np.linalg.norm(cl["iris_rgb"] - cr["iris_rgb"]))
    shared = dist < share_thresh
    metrics = {
        "size": size, "iris_frac": iris_frac, "pupil_frac": PUPIL_FRAC,
        "full_uv_radius": FULL_UV_R,
        "iris_uv_radius": iris_frac * FULL_UV_R,
        "left_from": left_subject_side, "right_from": right_subject_side,
        "iris_rgb_distance": dist, "shared": shared,
        "left": {k: (np.round(v, 4).tolist() if isinstance(v, np.ndarray) else v)
                 for k, v in cl.items()},
        "right": {k: (np.round(v, 4).tolist() if isinstance(v, np.ndarray) else v)
                  for k, v in cr.items()},
    }
    if shared:
        img = compose_eye_image(
            size,
            (cl["iris_rgb"] + cr["iris_rgb"]) / 2,
            (cl["pupil_rgb"] + cr["pupil_rgb"]) / 2,
            (cl["sclera_rgb"] + cr["sclera_rgb"]) / 2,
            iris_frac=iris_frac)
        return img, img, metrics
    img_l = compose_eye_image(size, cl["iris_rgb"], cl["pupil_rgb"],
                              cl["sclera_rgb"], iris_frac=iris_frac)
    img_r = compose_eye_image(size, cr["iris_rgb"], cr["pupil_rgb"],
                              cr["sclera_rgb"], iris_frac=iris_frac)
    return img_l, img_r, metrics


if __name__ == "__main__":  # quick manual test: eye_texture.py photo lmk.npz out/
    import sys
    from PIL import Image
    il, ir, m = build_eye_textures(sys.argv[1], sys.argv[2])
    od = Path(sys.argv[3] if len(sys.argv) > 3 else ".")
    od.mkdir(parents=True, exist_ok=True)
    Image.fromarray(il).save(od / "eye_left.png")
    if not m["shared"]:
        Image.fromarray(ir).save(od / "eye_right.png")
    import json
    print(json.dumps(m, indent=2))
