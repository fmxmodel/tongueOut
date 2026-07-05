"""Clean-room UV parameterization for the FLAME topology (Track B1, COMMERCIAL).

WHY THIS FILE EXISTS (measured on the pod, 2026-07-05)
-------------------------------------------------------
The FLAME 2023 Open (CC-BY-4.0) release is PKL-ONLY: no `head_template.obj`,
no UV coordinates. FLAME's own UV layout ships only in NC-licensed packages,
which are BARRED from this commercial run (models/README.md section 1,
out/compliance_report.md B1-1/B1-2). The albedo bake does not need FLAME's
specific layout -- it needs *one consistent* UV parameterization shared by the
bake (recon/bake_texture.py) and the GLB build (blender_build_rig.py). So:

  1. OPTIONAL OVERRIDE: if the operator staged a UV-bearing template obj in
     FLAME_DIR, load it (geometry faces MUST byte-match the topology contract
     -- mismatch is a hard STOP, not a fallback).
  2. EXPECTED PATH: generate the UV clean-room with xatlas (MIT license, both
     the C++ library and the PyPI `xatlas` binding; added to
     requirements-b1.txt) from the pkl's own `v_template` + `f`. Deterministic:
     same input arrays + default options -> same atlas; and the result is
     persisted once into out/recon/uv_coords.npz, which is the single UV
     authority every downstream consumer reads.

xatlas may split vertices along chart seams (UV vertex count > 3D vertex
count) and does NOT guarantee output face order. Both are handled explicitly:
output faces are realigned per-corner to the contract `faces` array and the
alignment is PROVEN by the identity vmapping[faces_uv] == faces (measured,
not assumed). The 3D topology contract (out/recon/faces.npy) is never touched.

NO silent quality fallback: if xatlas is missing, this module STOPS with
install instructions rather than substituting a cruder projection unwrap --
one pipeline, one UV layout.
"""

import sys

import numpy as np

from . import config as C


def _die(msg: str) -> None:
    sys.exit(f"[uv_unwrap FATAL] {msg}")


# --------------------------------------------------------------------------
# public entrypoint
# --------------------------------------------------------------------------
def load_or_generate_uv(v_template: np.ndarray, faces_contract: np.ndarray):
    """Resolve the UV layout: staged template file if present, else clean-room
    xatlas generation. Returns (verts_uvs (T,2) f64, faces_uv (F,3) i64,
    faces_v (F,3) i64 == faces_contract, source: str)."""
    faces_contract = np.asarray(faces_contract, dtype=np.int64)
    template_path = C.find_optional_flame_file(C.FLAME_TEMPLATE_CANDIDATES)
    if template_path is not None:
        got = _load_template_uv(template_path, faces_contract)
        if got is not None:
            return got
        print(f"[uv_unwrap WARN] {template_path.name} is staged but carries no "
              "UV (vt) data -- ignoring it and generating the clean-room unwrap.")
    return _generate_uv_xatlas(np.asarray(v_template, dtype=np.float64),
                               faces_contract)


# --------------------------------------------------------------------------
# optional override: a staged UV template obj
# --------------------------------------------------------------------------
def _load_template_uv(template_path, faces_contract):
    from pytorch3d.io import load_obj

    _v, tfaces, taux = load_obj(str(template_path), load_textures=False)
    if taux.verts_uvs is None or tfaces.textures_idx is None:
        return None  # caller falls through to generation, loudly
    faces_v = tfaces.verts_idx.cpu().numpy().astype(np.int64)
    if not np.array_equal(faces_v, faces_contract):
        _die(f"staged template {template_path} geometry faces != the topology "
             f"contract ({faces_v.shape} vs {faces_contract.shape}, "
             f"equal={np.array_equal(faces_v, faces_contract)}). A mismatched "
             "override is operator error, NOT a fallback case -- STOP and "
             "remove or fix the staged file.")
    verts_uvs = taux.verts_uvs.cpu().numpy().astype(np.float64)
    faces_uv = tfaces.textures_idx.cpu().numpy().astype(np.int64)
    src = f"release/staged template: {template_path}"
    print(f"[uv_unwrap] UV from staged template (override): {template_path.name} "
          f"({verts_uvs.shape[0]} UV verts)")
    return verts_uvs, faces_uv, faces_contract.copy(), src


# --------------------------------------------------------------------------
# expected path: deterministic clean-room unwrap (xatlas, MIT)
# --------------------------------------------------------------------------
def _canonical_rows(rows: np.ndarray):
    """Rotate each (a,b,c) row so its smallest vertex id comes first, keeping
    cyclic (winding) order. Returns (canonical rows, applied shift)."""
    shift = np.argmin(rows, axis=1)
    cols = (shift[:, None] + np.arange(3)[None, :]) % 3
    return np.take_along_axis(rows, cols, axis=1), shift


def _realign_faces(faces_contract, out_faces, out_uv_idx):
    """xatlas does not guarantee output face order. Realign its per-face UV
    indices to the contract face order AND per-corner vertex order."""
    F = faces_contract.shape[0]
    canon_in, shift_in = _canonical_rows(faces_contract)
    canon_out, shift_out = _canonical_rows(out_faces)

    lut = {tuple(r): i for i, r in enumerate(map(tuple, canon_in))}
    if len(lut) != F:
        _die("duplicate faces in the topology contract -- cannot realign the "
             "xatlas output unambiguously. STOP.")
    perm = np.empty(F, dtype=np.int64)
    for j, r in enumerate(map(tuple, canon_out)):
        i = lut.get(r)
        if i is None:
            _die(f"xatlas output face {j} {r} has no counterpart in the "
                 "contract (winding flip or topology change) -- STOP.")
        perm[j] = i
    if len(set(perm.tolist())) != F:
        _die("xatlas output faces do not map 1:1 onto the contract -- STOP.")

    # input face i = perm[j]; input corner c sits at canonical slot
    # k=(c-shift_in[i])%3, which is output corner (k+shift_out[j])%3.
    c = np.arange(3)[None, :]
    k = (c - shift_in[perm][:, None]) % 3
    src = (k + shift_out[:, None]) % 3
    faces_uv = np.empty((F, 3), dtype=np.int64)
    faces_uv[perm[:, None], c] = np.take_along_axis(out_uv_idx, src, axis=1)
    return faces_uv


def _generate_uv_xatlas(v_template, faces_contract):
    try:
        import xatlas
    except ImportError:
        _die("xatlas is not installed. It is REQUIRED on the pkl-only FLAME "
             "2023 Open release to generate the clean-room UV: "
             "`pip install xatlas` (MIT; listed in requirements-b1.txt). "
             "REFUSING to silently substitute a lower-quality projection "
             "unwrap -- one pipeline, one UV layout.")

    print("[uv_unwrap] generating clean-room UV with xatlas from "
          f"v_template {v_template.shape} + faces {faces_contract.shape} "
          "(FLAME 2023 Open ships no UV; NC UV packages are barred)")
    vmapping, indices, uvs = xatlas.parametrize(
        v_template.astype(np.float32), faces_contract.astype(np.uint32))
    vmapping = np.asarray(vmapping).astype(np.int64)      # (T,) new -> original vert
    indices = np.asarray(indices).astype(np.int64)        # (F,3) into uvs
    uvs = np.asarray(uvs, dtype=np.float64)               # (T,2)

    if indices.shape != faces_contract.shape:
        _die(f"xatlas returned {indices.shape} faces for a "
             f"{faces_contract.shape} contract -- topology changed. STOP.")

    out_faces = vmapping[indices]                          # original vert ids per face
    if np.array_equal(out_faces, faces_contract):
        faces_uv = indices
    else:
        print("[uv_unwrap] xatlas reordered faces/corners -- realigning to the "
              "topology contract")
        faces_uv = _realign_faces(faces_contract, out_faces, indices)

    # PROOF of alignment (measured): uv corner -> original vertex must equal
    # the contract, corner by corner.
    if not np.array_equal(vmapping[faces_uv], faces_contract):
        _die("post-realignment check vmapping[faces_uv] == faces FAILED -- the "
             "generated UV does not index the contract topology. STOP.")

    # defensive normalization (xatlas.parametrize returns normalized uvs; if a
    # binding version returns atlas-pixel coords, scale uniformly + say so)
    umin, umax = uvs.min(axis=0), uvs.max(axis=0)
    if umin.min() < -1e-4 or umax.max() > 1.0 + 1e-4:
        scale = float(max(umax[0] - min(umin[0], 0.0), umax[1] - min(umin[1], 0.0)))
        print(f"[uv_unwrap WARN] xatlas uvs outside [0,1] "
              f"(min={umin}, max={umax}); uniformly rescaling by 1/{scale:.3f}")
        uvs = (uvs - np.minimum(umin, 0.0)[None, :]) / scale

    # stats (recorded, not assumed)
    e1 = uvs[faces_uv[:, 1]] - uvs[faces_uv[:, 0]]
    e2 = uvs[faces_uv[:, 2]] - uvs[faces_uv[:, 0]]
    area2 = np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    n_degen = int((area2 < 1e-12).sum())
    n_seam = int(uvs.shape[0] - v_template.shape[0])
    ver = getattr(xatlas, "__version__", "unknown")
    src = (f"generated-clean-room:xatlas/{ver} (MIT) from flame2023.pkl "
           "v_template+f; deterministic default options")
    print(f"[uv_unwrap] xatlas UV OK: {uvs.shape[0]} UV verts "
          f"(+{n_seam} seam splits over V={v_template.shape[0]}), "
          f"{n_degen} degenerate UV tris, uv range "
          f"[{uvs.min():.4f},{uvs.max():.4f}]")
    if n_degen > 0.01 * faces_contract.shape[0]:
        _die(f"{n_degen} degenerate UV triangles (>1% of faces) -- the unwrap "
             "is unusable for baking. STOP.")
    return uvs, faces_uv, faces_contract.copy(), src
