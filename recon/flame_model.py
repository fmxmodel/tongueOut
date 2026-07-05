"""Self-authored FLAME 2023 loader + differentiable decoder (PyTorch).

PROVENANCE / LICENSING (matters for this COMMERCIAL run)
--------------------------------------------------------
This is a from-scratch implementation of the FLAME statistical head model
forward function, written from the published equations (Li et al., "Learning a
model of facial shape and expression from 4D scans", SIGGRAPH Asia 2017 --
the FLAME paper -- and the SMPL LBS formulation it builds on). It deliberately
does NOT vendor or copy code from smplx / FLAME_PyTorch / DECA / EMOCA, whose
licenses are non-commercial (out/compliance_report.md B1-4). Equations are
facts; this file is our own expression of them. The MODEL DATA it loads is
FLAME 2023 Open (CC-BY-4.0), a manual license-gated download
(models/README.md section 1).

MODEL STRUCTURE (asserted at load; nothing assumed silently)
------------------------------------------------------------
  v_template   (V, 3)        mean template, V expected 5023 (verify on pod)
  shapedirs    (V, 3, S+E)   identity (S=300) + expression (E=100) PCA dirs
  posedirs     (V, 3, 9*(J-1)) pose-corrective blendshapes
  J_regressor  (J, V)        joints from vertices, J expected 5 for FLAME:
                             [global(root), neck, jaw, eye_a, eye_b]
  weights      (V, J)        LBS skinning weights
  kintree_table(2, J)        parent table
  f            (F, 3)        triangles, F expected 9976 (verify on pod)

Joint order/laterality: the eye joints are joints 3 and 4; which is the
subject's LEFT eye must be VERIFIED ON THE POD (rotate joint 3, render, look).
The fit keeps both eye poses at zero so nothing here depends on it, but the
arkit-rigger DOES (eyeLook*) -- flagged in expression_basis_notes.json.

Pickle handling: legacy FLAME pkls reference `chumpy`. We do NOT depend on
chumpy (unmaintained, breaks on numpy>=1.24). A stub Unpickler absorbs chumpy
objects and extracts their underlying numpy arrays.
"""

import pickle
from pathlib import Path

import numpy as np
import torch


# --------------------------------------------------------------------------
# chumpy-free pickle loading
# --------------------------------------------------------------------------
class _ChumpyStub:
    """Absorbs unpickled chumpy objects; keeps their state dict for extraction."""

    def __init__(self, *args, **kwargs):
        self._stub_args = args

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.__dict__["_stub_state"] = state


class _StubUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.split(".")[0] == "chumpy":
            return _ChumpyStub
        return super().find_class(module, name)


def _to_numpy(value):
    """Best-effort conversion of a pkl entry to a dense numpy array."""
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "toarray"):  # scipy sparse (J_regressor commonly is)
        return np.asarray(value.toarray())
    if isinstance(value, _ChumpyStub):
        d = value.__dict__
        if "x" in d and isinstance(d["x"], np.ndarray):  # chumpy Ch stores data in .x
            return d["x"]
        arrays = [v for v in d.values() if isinstance(v, np.ndarray)]
        if len(arrays) == 1:
            return arrays[0]
        if arrays:  # ambiguous: take the largest (the payload), but say so
            arrays.sort(key=lambda a: a.size, reverse=True)
            print(f"[flame_model WARN] chumpy stub had {len(arrays)} arrays; "
                  f"taking the largest {arrays[0].shape}")
            return arrays[0]
        raise ValueError(f"chumpy stub with no ndarray payload: keys={list(d)}")
    if isinstance(value, (list, tuple)):
        return np.asarray(value)
    return value  # scalars / strings pass through


def load_flame_pkl(path: Path) -> dict:
    with open(path, "rb") as f:
        raw = _StubUnpickler(f, encoding="latin1").load()
    if not isinstance(raw, dict):
        raise ValueError(f"[flame_model FATAL] {path} did not unpickle to a dict")
    data = {k: _to_numpy(v) for k, v in raw.items()}
    print(f"[flame_model] loaded {path.name}; keys: {sorted(data.keys())}")
    return data


# --------------------------------------------------------------------------
# rotation utilities
# --------------------------------------------------------------------------
def batch_rodrigues(aa: torch.Tensor) -> torch.Tensor:
    """Axis-angle (N,3) -> rotation matrices (N,3,3). Standard Rodrigues."""
    angle = torch.norm(aa + 1e-8, dim=1, keepdim=True)          # (N,1)
    axis = aa / angle                                           # (N,3)
    cos = torch.cos(angle).unsqueeze(-1)                        # (N,1,1)
    sin = torch.sin(angle).unsqueeze(-1)
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    zeros = torch.zeros_like(x)
    K = torch.stack(
        [zeros, -z, y, z, zeros, -x, -y, x, zeros], dim=1
    ).view(-1, 3, 3)                                            # skew(axis)
    eye = torch.eye(3, dtype=aa.dtype, device=aa.device).unsqueeze(0)
    return eye * cos + (1.0 - cos) * torch.einsum("ni,nj->nij", axis, axis) + sin * K


def _rigid_transform_chain(rot_mats, joints, parents):
    """SMPL-style forward kinematics.

    rot_mats (B,J,3,3), joints (B,J,3), parents (J,) ->
      posed_joints (B,J,3), rel_transforms (B,J,4,4) for LBS.
    """
    B, J = rot_mats.shape[:2]
    rel_joints = joints.clone()
    rel_joints[:, 1:] = joints[:, 1:] - joints[:, parents[1:]]

    tf = torch.zeros(B, J, 4, 4, dtype=rot_mats.dtype, device=rot_mats.device)
    tf[:, :, :3, :3] = rot_mats
    tf[:, :, :3, 3] = rel_joints
    tf[:, :, 3, 3] = 1.0

    chain = [tf[:, 0]]
    for j in range(1, J):
        chain.append(torch.matmul(chain[parents[j]], tf[:, j]))
    transforms = torch.stack(chain, dim=1)                      # (B,J,4,4)
    posed_joints = transforms[:, :, :3, 3]

    # Remove the rest-pose joint locations so transforms act on rest vertices.
    joints_h = torch.cat(
        [joints, torch.zeros(B, J, 1, dtype=joints.dtype, device=joints.device)], dim=2
    )                                                           # (B,J,4) w=0
    correction = torch.matmul(transforms, joints_h.unsqueeze(-1))  # (B,J,4,1)
    rel_transforms = transforms.clone()
    rel_transforms[:, :, :3, 3] = transforms[:, :, :3, 3] - correction[:, :, :3, 0]
    return posed_joints, rel_transforms


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------
class FlameModel:
    """Differentiable FLAME decoder. Batched; float32; device-resident."""

    def __init__(self, model_path, n_shape=300, n_expr=100,
                 device="cuda", dtype=torch.float32):
        data = load_flame_pkl(Path(model_path))
        self.device, self.dtype = torch.device(device), dtype
        self.n_shape, self.n_expr = n_shape, n_expr

        required = ["v_template", "shapedirs", "posedirs", "J_regressor",
                    "weights", "kintree_table", "f"]
        missing = [k for k in required if k not in data]
        if missing:
            raise KeyError(
                f"[flame_model FATAL] FLAME pkl missing keys {missing}. "
                f"Present: {sorted(data.keys())}. Is this the FLAME 2023 Open "
                "model file? See models/README.md section 1."
            )

        v_template = np.asarray(data["v_template"], dtype=np.float64)   # (V,3)
        shapedirs = np.asarray(data["shapedirs"], dtype=np.float64)     # (V,3,S+E)
        posedirs = np.asarray(data["posedirs"], dtype=np.float64)       # (V,3,P)
        j_reg = np.asarray(data["J_regressor"], dtype=np.float64)       # (J,V)
        weights = np.asarray(data["weights"], dtype=np.float64)         # (V,J)
        kintree = np.asarray(data["kintree_table"]).astype(np.int64)    # (2,J)
        faces = np.asarray(data["f"]).astype(np.int64)                  # (F,3)

        V = v_template.shape[0]
        J = j_reg.shape[0]
        # ---- loud structural asserts (topology contract lives or dies here)
        if shapedirs.shape[2] != n_shape + n_expr:
            raise ValueError(
                f"[flame_model FATAL] shapedirs has {shapedirs.shape[2]} columns; "
                f"expected N_SHAPE+N_EXPR = {n_shape}+{n_expr}. This release does "
                "not match the FLAME 2020/2023 300+100 convention -- STOP and "
                "reconcile config.N_SHAPE/N_EXPR with the actual download. "
                "Do NOT silently re-split."
            )
        if posedirs.shape[:2] != (V, 3) or posedirs.shape[2] != 9 * (J - 1):
            raise ValueError(
                f"[flame_model FATAL] posedirs shape {posedirs.shape} does not match "
                f"(V,3,9*(J-1)) = ({V},3,{9 * (J - 1)})."
            )
        if faces.min() < 0 or faces.max() >= V:
            raise ValueError("[flame_model FATAL] face indices out of vertex range.")

        parents = kintree[0].copy()
        parents[0] = 0  # root's parent sentinel (often uint32 -1) -> self
        self.parents = parents

        t = lambda a: torch.as_tensor(a, dtype=dtype, device=self.device)
        self.v_template = t(v_template)
        self.shapedirs_id = t(shapedirs[:, :, :n_shape])
        self.shapedirs_expr = t(shapedirs[:, :, n_shape:n_shape + n_expr])
        # posedirs flattened for one matmul: (P, V*3)
        self.posedirs = t(posedirs.reshape(V * 3, -1).T)
        self.j_regressor = t(j_reg)
        self.lbs_weights = t(weights)
        self.faces = faces  # numpy int64 (F,3) -- THE topology contract
        self.n_verts, self.n_joints = V, J
        # float32 numpy copies for the expression-basis handoff (fit_flame.py)
        self.np_v_template = v_template.astype(np.float32)
        self.np_expr_dirs = shapedirs[:, :, n_shape:n_shape + n_expr].astype(np.float32)
        self.np_posedirs = posedirs.astype(np.float32)          # (V,3,9*(J-1))
        self.np_j_regressor = j_reg.astype(np.float32)
        self.np_lbs_weights = weights.astype(np.float32)
        print(f"[flame_model] V={V} F={faces.shape[0]} J={J} "
              f"shape={n_shape} expr={n_expr} parents={parents.tolist()}")

    # -- pieces ------------------------------------------------------------
    def shaped_vertices(self, betas, expression):
        """(B,S),(B,E) -> rest-pose shaped vertices (B,V,3)."""
        v = self.v_template.unsqueeze(0)
        if betas is not None and betas.shape[1] > 0:
            v = v + torch.einsum("bs,vcs->bvc", betas, self.shapedirs_id[:, :, :betas.shape[1]])
        if expression is not None and expression.shape[1] > 0:
            v = v + torch.einsum("be,vce->bvc",
                                 expression, self.shapedirs_expr[:, :, :expression.shape[1]])
        return v

    def decode(self, betas=None, expression=None, global_orient=None,
               neck_pose=None, jaw_pose=None, eye_pose_a=None, eye_pose_b=None,
               transl=None, batch_size=None):
        """Full FLAME forward. All pose args are axis-angle (B,3); eye_pose_a/b
        are joints 3/4 (laterality: VERIFY ON POD). Returns verts (B,V,3)."""
        # infer batch size from the first non-None tensor
        for x in (betas, expression, global_orient, jaw_pose, transl):
            if x is not None:
                batch_size = x.shape[0]
                break
        B = batch_size or 1
        dev, dt = self.device, self.dtype
        z3 = lambda x: x if x is not None else torch.zeros(B, 3, dtype=dt, device=dev)

        betas = betas if betas is not None else torch.zeros(B, 0, dtype=dt, device=dev)
        expression = expression if expression is not None else torch.zeros(B, 0, dtype=dt, device=dev)
        pose = torch.stack(
            [z3(global_orient), z3(neck_pose), z3(jaw_pose), z3(eye_pose_a), z3(eye_pose_b)],
            dim=1,
        )  # (B,J,3) -- FLAME joint order [global, neck, jaw, eye_a, eye_b]
        if pose.shape[1] != self.n_joints:
            raise ValueError(
                f"[flame_model FATAL] model has {self.n_joints} joints, pose built "
                f"for {pose.shape[1]}. Joint-order assumption broken -- STOP."
            )

        v_shaped = self.shaped_vertices(betas, expression)                  # (B,V,3)
        joints = torch.einsum("jv,bvc->bjc", self.j_regressor, v_shaped)    # (B,J,3)

        rot = batch_rodrigues(pose.reshape(-1, 3)).view(B, self.n_joints, 3, 3)
        eye3 = torch.eye(3, dtype=dt, device=dev)
        pose_feature = (rot[:, 1:] - eye3).reshape(B, -1)                   # (B,9*(J-1))
        v_posed = v_shaped + torch.matmul(pose_feature, self.posedirs).view(B, self.n_verts, 3)

        _, rel_tf = _rigid_transform_chain(rot, joints, self.parents)
        T = torch.einsum("vj,bjmn->bvmn", self.lbs_weights, rel_tf)         # (B,V,4,4)
        v_h = torch.cat(
            [v_posed, torch.ones(B, self.n_verts, 1, dtype=dt, device=dev)], dim=2
        )
        verts = torch.matmul(T, v_h.unsqueeze(-1))[:, :, :3, 0]
        if transl is not None:
            verts = verts + transl.unsqueeze(1)
        return verts

    def surface_points(self, verts, lmk_face_idx, lmk_bary):
        """Barycentric surface points: verts (B,V,3), lmk_face_idx (L,),
        lmk_bary (L,3) -> (B,L,3)."""
        tri = torch.as_tensor(self.faces[lmk_face_idx], device=verts.device)  # (L,3)
        pts = verts[:, tri]                                                   # (B,L,3,3)
        bary = torch.as_tensor(lmk_bary, dtype=verts.dtype, device=verts.device)
        return torch.einsum("blkc,lk->blc", pts, bary)


# --------------------------------------------------------------------------
# landmark embedding loading (release-variant tolerant; models/README sec. 1)
# NOTE: the FLAME 2023 Open release ships NO embedding file (pkl-only,
# measured on the pod 2026-07-05). This loader now serves only the OPTIONAL
# staged-file override; the expected path is the self-authored clean-room
# derivation in recon/flame_landmarks.py. Wiring: recon/fit_flame.py.
# --------------------------------------------------------------------------
def load_landmark_embedding(path: Path) -> dict:
    """Normalize the FLAME landmark-embedding variants to one dict:
      static_faces (51,), static_bary (51,3)      -- iBUG 17..67, always present
      full_faces (68,), full_bary (68,3)          -- optional (adds contour 0..16)
    Variants handled:
      * landmark_embedding[_with_eyes].npy : dict with static_lmk_* /
        full_lmk_* (DECA-era layout, allow_pickle)
      * flame_static_embedding.pkl         : {lmk_face_idx, lmk_b_coords} (51)
    Dynamic (yaw-binned) contour embeddings are NOT used -- recorded honestly;
    contour is fit only when a 'full' 68-point embedding exists."""
    path = Path(path)
    out = {}
    if path.suffix == ".npy":
        raw = np.load(path, allow_pickle=True)
        d = raw.item() if raw.dtype == object and raw.shape == () else raw
        if not isinstance(d, dict):
            raise ValueError(f"[flame_model FATAL] unexpected embedding layout in {path}")
        norm = {k.lower(): v for k, v in d.items()}
        if "static_lmk_faces_idx" in norm:
            out["static_faces"] = np.asarray(norm["static_lmk_faces_idx"]).reshape(-1).astype(np.int64)
            out["static_bary"] = np.asarray(norm["static_lmk_bary_coords"], dtype=np.float64).reshape(-1, 3)
        if "full_lmk_faces_idx" in norm:
            out["full_faces"] = np.asarray(norm["full_lmk_faces_idx"]).reshape(-1).astype(np.int64)
            out["full_bary"] = np.asarray(norm["full_lmk_bary_coords"], dtype=np.float64).reshape(-1, 3)
        if "lmk_face_idx" in norm and "static_faces" not in out:
            out["static_faces"] = np.asarray(norm["lmk_face_idx"]).reshape(-1).astype(np.int64)
            out["static_bary"] = np.asarray(norm["lmk_b_coords"], dtype=np.float64).reshape(-1, 3)
    elif path.suffix == ".pkl":
        with open(path, "rb") as f:
            d = _StubUnpickler(f, encoding="latin1").load()
        d = {k: _to_numpy(v) for k, v in d.items()}
        out["static_faces"] = np.asarray(d["lmk_face_idx"]).reshape(-1).astype(np.int64)
        out["static_bary"] = np.asarray(d["lmk_b_coords"], dtype=np.float64).reshape(-1, 3)
    else:
        raise ValueError(f"[flame_model FATAL] unsupported embedding file: {path}")

    if "static_faces" not in out and "full_faces" in out:
        # full is iBUG 0..67; static = rows 17..67
        out["static_faces"] = out["full_faces"][17:]
        out["static_bary"] = out["full_bary"][17:]
    if "static_faces" not in out:
        raise ValueError(
            f"[flame_model FATAL] could not extract a static 51-landmark embedding "
            f"from {path}. Keys seen: {sorted(out.keys())}."
        )
    ns = out["static_faces"].shape[0]
    if ns not in (51, 68, 70):  # 70 = with-eyes variant (68 + 2 eye centers)
        print(f"[flame_model WARN] static embedding has {ns} points (expected 51/68/70); "
              "using the first 51-compatible rows requires manual review -- STOPPING.")
        raise ValueError(f"unexpected static embedding size {ns}")
    if ns in (68, 70):
        # some releases pack the full 68(+eyes) as 'static'; iBUG 17..67 = rows 17..67
        out["full_faces"] = out["static_faces"][:68].copy()
        out["full_bary"] = out["static_bary"][:68].copy()
        out["static_faces"] = out["static_faces"][17:68]
        out["static_bary"] = out["static_bary"][17:68]
    print(f"[flame_model] landmark embedding: static={out['static_faces'].shape[0]} "
          f"contour={'yes (full-68)' if 'full_faces' in out else 'no'}")
    return out
