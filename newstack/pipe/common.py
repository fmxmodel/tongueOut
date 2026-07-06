"""Shared utilities for the newstack pipeline.

IMPORTANT: this module is imported by BOTH system python and Blender's bundled
python (s3b/s6). Module-level imports are stdlib + numpy ONLY; anything heavier
(cv2, torch, mediapipe, trimesh, scipy, PIL) is imported lazily inside the
function that needs it.

Coordinate conventions (single source of truth):
- ICT model space: centimeters, +Y up, +Z toward the viewer (front).
- Photo space: pixels, origin top-left, y DOWN.
- Weak-perspective camera (fit by s2, consumed by s5):
      Xc = R @ X          (camera-space, model units)
      u  =  s * Xc.x + tx
      v  = -s * Xc.y + ty   (y flip: model y-up -> image y-down)
  depth = Xc.z, LARGER = CLOSER to camera. View vector toward the camera in
  model space is R.T @ [0,0,1].
- Blender world (s3b geometry math is orientation-agnostic; s6 export):
      blender = (x, -z, y) of ICT. The glTF exporter's Z-up -> Y-up conversion
      is (x, z, -y) of Blender, so GLB coords == ICT coords exactly
      (then scaled 0.01 cm -> m). +Y up / +Z front in the GLB, as glTF wants.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- topology
N_VERTS = 26719   # ICT-FaceKit FaceXModel vertex count -- the ONE topology
N_POLYS = 26384   # quad-dominant polygon count

# Published ICT-FaceKit vertex ranges (0-based, end-exclusive).
ICT_REGIONS = {
    "face":       (0, 9409),
    "head_neck":  (9409, 11248),
    "interior_a": (11248, 17039),   # mouth socket, eye sockets, gums area
    "teeth":      (17039, 21451),
    "eyeballs":   (21451, 24591),
    "interior_b": (24591, 26719),   # lacrimal / eye blend+occlusion / lashes
}
EXTERIOR_END = 11248  # verts [0, 11248) = face + head/neck skin (shrinkable)

# Eye sockets L+R (ICT README ordinals #3-#4): the inner lid/socket walls
# visible around the eyeball -- they should read as shadowed SKIN, not as
# mouth interior, when vertex-color fallbacks are painted (s5).
EYE_SOCKETS = (13294, 14062)

# Transparent-purpose eye shells (ICT README ordinals #9-#14: lacrimal fluid
# L/R + eye blend L/R + eye occlusion L/R -- Unreal Digital-Human style
# translucent-shader meshes). MEASURED: they sit up to z=10.6 IN FRONT of the
# eyeball forward pole (z=9.65), so in an all-OPAQUE export they hide the
# eyes; s6 strips their FACES (all 26719 verts stay in the authored mesh).
# Eyelashes [25351, 26719) are visible geometry and are KEPT.
EYE_SHELLS = (24591, 25351)


# ---------------------------------------------------------------- pod paths
class P:
    ROOT = "/workspace/newstack"
    PIPE = "/workspace/newstack/pipe"
    ICT = "/workspace/newstack/ICT-FaceKit"
    CLAY = "/workspace/newstack/out_triposr/0/mesh.obj"
    PHOTO = "/workspace/inputs/random-person.jpeg"
    MP_TASK = "/workspace/models/mediapipe/face_landmarker.task"
    OUT = "/workspace/newstack/out"


def out_dir(base, sub):
    d = Path(base) / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def die(msg):
    print(f"[FATAL] {msg}", file=sys.stderr)
    sys.exit(1)


def assert_topology(verts, what, n=N_VERTS):
    if len(verts) != n:
        die(f"topology drift in {what}: {len(verts)} verts != {n}. "
            "The one-topology invariant is broken -- STOP.")


# ---------------------------------------------------------------- OBJ io
def read_obj(path, verts_only=False):
    """Minimal, order-preserving OBJ reader.

    Returns dict with:
      v         (N,3) float64 positions, file order (NEVER reordered)
      vcols     (N,3) float64 in [0,1] if 'v x y z r g b' lines, else None
      vt        (T,2) float64 or None
      faces_flat(int32) polygon corner vertex ids, 0-based, concatenated
      faces_off (int32) polygon offsets, len P+1
      corner_vt (int32) per-corner vt ids, 0-based (-1 where absent)
    Handles tris/quads/ngons and v-lines with 3 or 6 columns (TripoSR colors).
    """
    txt = Path(path).read_text()
    v_chunks, vt_chunks, f_lines = [], [], []
    for line in txt.splitlines():
        if line.startswith("v "):
            v_chunks.append(line[2:])
        elif not verts_only and line.startswith("vt "):
            vt_chunks.append(line[3:])
        elif not verts_only and line.startswith("f "):
            f_lines.append(line[2:])
    if not v_chunks:
        die(f"read_obj: no vertices in {path}")
    ncol = len(v_chunks[0].split())
    arr = np.array(" ".join(v_chunks).split(), dtype=np.float64)
    if arr.size != len(v_chunks) * ncol:
        die(f"read_obj: inconsistent vertex-line column counts in {path} "
            f"({arr.size} floats for {len(v_chunks)} x {ncol}-col lines)")
    arr = arr.reshape(-1, ncol)
    out = {"v": arr[:, :3],
           "vcols": arr[:, 3:6].copy() if ncol >= 6 else None,
           "vt": None, "faces_flat": None, "faces_off": None, "corner_vt": None}
    if verts_only:
        return out
    if vt_chunks:
        vt = np.array(" ".join(vt_chunks).split(), dtype=np.float64)
        ncol_t = len(vt_chunks[0].split())
        out["vt"] = vt.reshape(-1, ncol_t)[:, :2]
    if f_lines:
        faces_flat, corner_vt, faces_off = [], [], [0]
        for fl in f_lines:
            for tok in fl.split():
                parts = tok.split("/")
                vi = int(parts[0])
                if vi < 0:
                    die(f"read_obj: negative (relative) index in {path} unsupported")
                faces_flat.append(vi - 1)
                vti = int(parts[1]) if len(parts) > 1 and parts[1] else 0
                corner_vt.append(vti - 1)
            faces_off.append(len(faces_flat))
        out["faces_flat"] = np.asarray(faces_flat, dtype=np.int32)
        out["faces_off"] = np.asarray(faces_off, dtype=np.int32)
        out["corner_vt"] = np.asarray(corner_vt, dtype=np.int32)
    return out


def write_obj(path, verts, faces_flat=None, faces_off=None, comment=""):
    """Write verts (+ optional polygon faces) as OBJ, order-preserving."""
    lines = [f"# {comment}"] if comment else []
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in np.asarray(verts, dtype=np.float64)]
    if faces_flat is not None and faces_off is not None:
        ff = np.asarray(faces_flat) + 1
        off = np.asarray(faces_off)
        for p in range(len(off) - 1):
            lines.append("f " + " ".join(str(i) for i in ff[off[p]:off[p + 1]]))
    Path(path).write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- mesh math
def faces_as_lists(faces_flat, faces_off):
    off = np.asarray(faces_off)
    ff = np.asarray(faces_flat)
    return [tuple(int(i) for i in ff[off[p]:off[p + 1]]) for p in range(len(off) - 1)]


def triangulate(faces_flat, faces_off):
    """Fan-triangulate polygons. Returns tri_corners (M,3) int64 indices into
    the FLAT corner arrays (so both vertex ids and vt ids can be fetched)."""
    off = np.asarray(faces_off)
    tris = []
    for p in range(len(off) - 1):
        s, e = int(off[p]), int(off[p + 1])
        for k in range(s + 1, e - 1):
            tris.append((s, k, k + 1))
    return np.asarray(tris, dtype=np.int64)


def edges_from_faces(faces_flat, faces_off):
    """Unique undirected edges (E,2) int64 from polygon loops."""
    ff = np.asarray(faces_flat, dtype=np.int64)
    off = np.asarray(faces_off, dtype=np.int64)
    a, b = [], []
    for p in range(len(off) - 1):
        s, e = int(off[p]), int(off[p + 1])
        idx = ff[s:e]
        a.append(idx)
        b.append(np.roll(idx, -1))
    a = np.concatenate(a)
    b = np.concatenate(b)
    ed = np.sort(np.stack([a, b], axis=1), axis=1)
    return np.unique(ed, axis=0)


def vertex_normals(verts, tri_v):
    """Area-weighted per-vertex normals. tri_v (M,3) VERTEX ids."""
    v = np.asarray(verts, dtype=np.float64)
    p0, p1, p2 = v[tri_v[:, 0]], v[tri_v[:, 1]], v[tri_v[:, 2]]
    fn = np.cross(p1 - p0, p2 - p0)
    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n, tri_v[:, k], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln < 1e-12] = 1.0
    return n / ln


def smooth_field(field, edges, iters, lam):
    """Graph-Laplacian low-pass of a per-vertex field (N,C) over mesh edges.
    Pure numpy -- usable inside Blender's python too."""
    f = np.asarray(field, dtype=np.float64).copy()
    single = f.ndim == 1
    if single:
        f = f[:, None]
    e0, e1 = edges[:, 0], edges[:, 1]
    deg = np.bincount(e0, minlength=len(f)) + np.bincount(e1, minlength=len(f))
    deg = np.maximum(deg, 1).astype(np.float64)[:, None]
    for _ in range(int(iters)):
        acc = np.zeros_like(f)
        np.add.at(acc, e0, f[e1])
        np.add.at(acc, e1, f[e0])
        f = (1.0 - lam) * f + lam * (acc / deg)
    return f[:, 0] if single else f


def smoothstep(d, r0, r1):
    t = np.clip((np.asarray(d, dtype=np.float64) - r0) / max(r1 - r0, 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def min_dist_to_points(verts, pts):
    """(N,) min euclidean distance from each vert to a small point set."""
    d = np.linalg.norm(np.asarray(verts)[:, None, :] - np.asarray(pts)[None, :, :], axis=2)
    return d.min(axis=1)


# ---------------------------------------------------------------- camera
def project_weak_persp(verts, s, R, t):
    """Weak-perspective projection per the module docstring convention.
    Returns (uv (N,2) px, depth (N,) larger=closer)."""
    Xc = np.asarray(verts, dtype=np.float64) @ np.asarray(R, dtype=np.float64).T
    u = s * Xc[:, 0] + t[0]
    v = -s * Xc[:, 1] + t[1]
    return np.stack([u, v], axis=1), Xc[:, 2]


def umeyama(src, dst):
    """Similarity transform (s, R, t) minimizing ||s*R@src + t - dst||^2."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var_s = (xs ** 2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / max(var_s, 1e-12))
    t = mu_d - s * R @ mu_s
    return s, R, t


# ---------------------------------------------------------------- rasterizer
def rasterize(pts2d, tris, W, H, depth=None, attrs=None):
    """Z-buffered triangle rasterizer (pure numpy; python loop over tris --
    ~30-90 s for ~52k tris at 1k resolution, fine for a bake stage).

    pts2d (P,2) float px (pixel centers at integer+0.5); tris (M,3) indices
    into pts2d; depth (P,) LARGER = CLOSER (None -> zeros, last-writer-wins);
    attrs (P,C) per-point attributes to interpolate (or None).

    Returns (zbuf (H,W) float64 init -inf, tri_id (H,W) int32 init -1,
             abuf (H,W,C) float64 or None).
    """
    pts2d = np.asarray(pts2d, dtype=np.float64)
    x, y = pts2d[:, 0], pts2d[:, 1]
    d = np.zeros(len(pts2d)) if depth is None else np.asarray(depth, dtype=np.float64)
    zbuf = np.full((H, W), -np.inf, dtype=np.float64)
    tid = np.full((H, W), -1, dtype=np.int32)
    abuf = None
    if attrs is not None:
        attrs = np.asarray(attrs, dtype=np.float64)
        abuf = np.zeros((H, W, attrs.shape[1]), dtype=np.float64)
    for m in range(len(tris)):
        i0, i1, i2 = tris[m]
        x0, x1, x2 = x[i0], x[i1], x[i2]
        y0, y1, y2 = y[i0], y[i1], y[i2]
        jmin = max(int(np.floor(min(x0, x1, x2))) - 1, 0)
        jmax = min(int(np.ceil(max(x0, x1, x2))) + 1, W - 1)
        imin = max(int(np.floor(min(y0, y1, y2))) - 1, 0)
        imax = min(int(np.ceil(max(y0, y1, y2))) + 1, H - 1)
        if jmax < jmin or imax < imin:
            continue
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(den) < 1e-12:
            continue
        PX = np.arange(jmin, jmax + 1, dtype=np.float64)[None, :] + 0.5
        PY = np.arange(imin, imax + 1, dtype=np.float64)[:, None] + 0.5
        l0 = ((y1 - y2) * (PX - x2) + (x2 - x1) * (PY - y2)) / den
        l1 = ((y2 - y0) * (PX - x2) + (x0 - x2) * (PY - y2)) / den
        l2 = 1.0 - l0 - l1
        inside = (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6)
        if not inside.any():
            continue
        z = l0 * d[i0] + l1 * d[i1] + l2 * d[i2]
        sub_z = zbuf[imin:imax + 1, jmin:jmax + 1]
        upd = inside & (z >= sub_z)
        if not upd.any():
            continue
        sub_z[upd] = z[upd]
        tid[imin:imax + 1, jmin:jmax + 1][upd] = m
        if abuf is not None:
            a = (l0[..., None] * attrs[i0] + l1[..., None] * attrs[i1]
                 + l2[..., None] * attrs[i2])
            abuf[imin:imax + 1, jmin:jmax + 1][upd] = a[upd]
    return zbuf, tid, abuf


def bilinear_sample(img, uv):
    """Bilinear sample img (H,W,C) float at uv (K,2) px coords. Clamped."""
    H, W = img.shape[:2]
    u = np.clip(np.asarray(uv)[:, 0] - 0.5, 0, W - 1.001)
    v = np.clip(np.asarray(uv)[:, 1] - 0.5, 0, H - 1.001)
    u0, v0 = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    fu, fv = (u - u0)[:, None], (v - v0)[:, None]
    c00 = img[v0, u0]
    c01 = img[v0, np.minimum(u0 + 1, W - 1)]
    c10 = img[np.minimum(v0 + 1, H - 1), u0]
    c11 = img[np.minimum(v0 + 1, H - 1), np.minimum(u0 + 1, W - 1)]
    return (c00 * (1 - fu) * (1 - fv) + c01 * fu * (1 - fv)
            + c10 * (1 - fu) * fv + c11 * fu * fv)


# ---------------------------------------------------------------- mediapipe
_LANDMARKER_CACHE = {}


def detect_face(rgb, task_path, min_conf=0.5):
    """Run MediaPipe FaceLandmarker (IMAGE mode). Returns the result object or
    None if no face. Landmarker instances are cached per (path, conf)."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    key = (str(task_path), float(min_conf))
    if key not in _LANDMARKER_CACHE:
        if not os.path.isfile(task_path):
            die(f"face_landmarker.task not found at {task_path}")
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(task_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            min_face_detection_confidence=float(min_conf),
        )
        _LANDMARKER_CACHE[key] = vision.FaceLandmarker.create_from_options(options)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=np.ascontiguousarray(rgb))
    result = _LANDMARKER_CACHE[key].detect(mp_image)
    return result if result.face_landmarks else None


def landmarks_to_np(result, face=0):
    """(478,3) normalized-coordinate landmarks from a FaceLandmarker result."""
    return np.array([[p.x, p.y, p.z] for p in result.face_landmarks[face]],
                    dtype=np.float64)


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    print(f"  wrote {path}")
