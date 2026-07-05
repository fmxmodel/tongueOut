"""Build the 52 ARKit blendshape delta meshes on the FLAME 2023 Open rig.

POD-ONLY (recon.pod_guard: requires CUDA torch). Consumes out/recon/
(face-reconstructor's artifacts) and writes out/shapes/ (see rig.config).

METHOD (rig/arkit_spec.py is the correspondence's single source of truth):
  pose shapes   jawOpen/jawLeft/jawRight + 8x eyeLook* drive FLAME's jaw/eye
                JOINTS through the reconstructor's exact LBS math (functions
                imported from recon.flame_model). Axes, signs and eye-joint
                laterality are CALIBRATED BY MEASUREMENT here, never assumed.
  jawForward    LBS-weighted +Z translation of the jaw joint (FLAME's jaw is
                rotation-only; protrusion cannot rotate) -- documented approx.
  pca shapes    per-shape sparse displacement targets on the iBUG-51 landmark
                set (FLAME's own CC-BY embedding), solved as ridge-regularized
                least squares over the 100 expression axes; mouthClose is
                linearized at the jawOpen pose (secant Jacobian).
  unsupported   tongueOut / cheekPuff / cheekSquint* are declared, not faked.

Every attempted shape passes MEASURED gates (direction, amplitude, leakage,
locality, non-triviality); failures are demoted to unsupported with the
measured numbers in arkit_manifest.json. Every emitted PLY is re-read and
byte-compared against out/recon/faces.npy -- topology drift is a hard STOP.

Run:  python -m rig.build_arkit_shapes     (after the recon stage, same venv)
"""

import json
import shutil
import sys
import time
from datetime import datetime, timezone

import numpy as np

from recon import config as RC
from recon.pod_guard import require_cuda_torch
from . import arkit_spec as S
from . import config as C

# Bound in main() AFTER the pod guard passes (recon.flame_model imports torch
# at module top; deferring keeps this module importable on the authoring box
# for py_compile-style checks without any compute stack).
torch = None
batch_rodrigues = None
_rigid_transform_chain = None

MM = 1e3  # meters -> mm for log readability


def die(msg: str) -> None:
    sys.exit(f"[rig FATAL] {msg}")


# --------------------------------------------------------------------------
# PLY helpers (same discipline as recon.fit_flame / recon.verify_outputs)
# --------------------------------------------------------------------------
def parse_ply_header(path):
    header = b""
    with open(path, "rb") as f:
        while b"end_header" not in header:
            chunk = f.read(4096)
            if not chunk:
                break
            header += chunk
    text = header.split(b"end_header")[0].decode("ascii", errors="replace")
    n_v = n_f = 0
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) == 3 and parts[0] == "element":
            if parts[1] == "vertex":
                n_v = int(parts[2])
            elif parts[1] == "face":
                n_f = int(parts[2])
    return n_v, n_f


def export_and_verify_ply(path, verts, faces_contract_i32):
    """Write a faced PLY and prove (by re-reading) that its topology is
    byte-identical to the contract. Any drift is a hard STOP."""
    import trimesh

    mesh = trimesh.Trimesh(vertices=np.asarray(verts, dtype=np.float64),
                           faces=faces_contract_i32, process=False)
    mesh.export(path)
    n_v, n_f = parse_ply_header(path)
    if (n_v, n_f) != (verts.shape[0], faces_contract_i32.shape[0]):
        die(f"{path}: header says V={n_v} F={n_f}, expected "
            f"V={verts.shape[0]} F={faces_contract_i32.shape[0]}.")
    back = trimesh.load(path, process=False)
    back_faces = np.asarray(back.faces, dtype=np.int32)
    if back_faces.tobytes() != faces_contract_i32.tobytes():
        die(f"{path}: re-read faces are NOT byte-identical to faces.npy. "
            "Topology drifted -- the entire pipeline forbids this. STOP.")
    if not np.isfinite(np.asarray(back.vertices)).all():
        die(f"{path}: non-finite vertices.")


# --------------------------------------------------------------------------
# Decoder around the fitted identity (expression_basis.npz), reusing the
# reconstructor's exact rotation/LBS functions => identical fit-time math.
# --------------------------------------------------------------------------
class BasisDecoder:
    def __init__(self, eb, device):
        t = lambda a, dt=None: torch.as_tensor(
            np.asarray(a), dtype=dt or torch.float32, device=device)
        self.device = device
        self.v_neutral = t(eb["v_neutral"])                 # (V,3) fitted id
        self.expr_dirs = t(eb["expr_dirs"])                 # (V,3,E)
        V = self.v_neutral.shape[0]
        pd = np.asarray(eb["posedirs"], dtype=np.float32)   # (V,3,9*(J-1))
        self.posedirs = t(pd.reshape(V * 3, -1).T)          # (P, V*3) as recon
        self.j_regressor = t(eb["j_regressor"])             # (J,V)
        self.lbs_weights = t(eb["lbs_weights"])             # (V,J)
        self.parents = np.asarray(eb["parents"], dtype=np.int64)
        self.n_verts = V
        self.n_expr = int(self.expr_dirs.shape[2])
        self.n_joints = int(self.j_regressor.shape[0])
        if self.n_joints != 5:
            die(f"expected 5 FLAME joints, basis has {self.n_joints}.")

    def decode(self, expression=None, global_orient=None, neck=None,
               jaw=None, eye_a=None, eye_b=None, transl=None):
        """All pose args axis-angle (B,3) tensors; expression (B,E).
        Joint order [global, neck, jaw, eye_a(3), eye_b(4)] as recon."""
        B = 1
        for x in (expression, global_orient, jaw, eye_a, eye_b, transl):
            if x is not None:
                B = x.shape[0]
                break
        dev, dt = self.device, torch.float32
        z3 = lambda x: x if x is not None else torch.zeros(B, 3, dtype=dt, device=dev)

        v = self.v_neutral.unsqueeze(0).expand(B, -1, -1)
        if expression is not None:
            v = v + torch.einsum("be,vce->bvc",
                                 expression, self.expr_dirs[:, :, :expression.shape[1]])
        joints = torch.einsum("jv,bvc->bjc", self.j_regressor, v)

        pose = torch.stack([z3(global_orient), z3(neck), z3(jaw),
                            z3(eye_a), z3(eye_b)], dim=1)       # (B,5,3)
        rot = batch_rodrigues(pose.reshape(-1, 3)).view(B, self.n_joints, 3, 3)
        eye3 = torch.eye(3, dtype=dt, device=dev)
        pose_feature = (rot[:, 1:] - eye3).reshape(B, -1)
        v_posed = v + torch.matmul(pose_feature, self.posedirs).view(B, self.n_verts, 3)

        _, rel_tf = _rigid_transform_chain(rot, joints, self.parents)
        T = torch.einsum("vj,bjmn->bvmn", self.lbs_weights, rel_tf)
        v_h = torch.cat([v_posed, torch.ones(B, self.n_verts, 1,
                                             dtype=dt, device=dev)], dim=2)
        verts = torch.matmul(T, v_h.unsqueeze(-1))[:, :, :3, 0]
        if transl is not None:
            verts = verts + transl.unsqueeze(1)
        return verts


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> None:  # noqa: C901  (one linear, auditable pipeline)
    global torch, batch_rodrigues, _rigid_transform_chain
    torch = require_cuda_torch()
    from recon import flame_model as FM
    batch_rodrigues = FM.batch_rodrigues
    _rigid_transform_chain = FM._rigid_transform_chain
    device = torch.device(C.DEVICE)
    t0 = time.time()
    C.ensure_out_dirs()
    aa = lambda v: torch.tensor([v], dtype=torch.float32, device=device)

    # ---- 0. name contract ---------------------------------------------------
    if not C.NAME_CONTRACT_JSON.is_file():
        die(f"name contract missing: {C.NAME_CONTRACT_JSON}")
    with open(C.NAME_CONTRACT_JSON, encoding="utf-8") as f:
        contract = json.load(f)
    arkit52 = list(contract["apple_canonical_52"])
    if len(arkit52) != 52 or len(set(arkit52)) != 52:
        die(f"name contract does not hold 52 unique names ({len(arkit52)}).")
    if sorted(S.SPEC.keys()) != sorted(arkit52):
        only_spec = sorted(set(S.SPEC) - set(arkit52))
        only_contract = sorted(set(arkit52) - set(S.SPEC))
        die(f"spec/contract name mismatch. spec-only={only_spec} "
            f"contract-only={only_contract}. Spelling is load-bearing -- STOP.")
    print("[rig] name contract OK: 52 exact, case-sensitive names")

    # ---- 1. inputs + topology contract ---------------------------------------
    for p in (C.RECON_NEUTRAL_PLY, C.RECON_FACES_NPY, C.RECON_ID_PARAMS_NPZ,
              C.RECON_EXPR_BASIS_NPZ, C.RECON_LANDMARKS_NPZ):
        if not p.is_file():
            die(f"recon artifact missing: {p} -- run scripts/run_recon_b1.sh first.")

    faces = np.load(C.RECON_FACES_NPY)
    if faces.dtype != np.int32 or faces.ndim != 2 or faces.shape[1] != 3:
        die(f"faces.npy malformed: shape={faces.shape} dtype={faces.dtype}")

    eb = np.load(C.RECON_EXPR_BASIS_NPZ)
    eb_faces = np.asarray(eb["faces"], dtype=np.int32)
    if eb_faces.tobytes() != faces.tobytes():
        die("expression_basis.npz faces != faces.npy (byte compare). STOP.")

    import trimesh
    neutral_mesh = trimesh.load(C.RECON_NEUTRAL_PLY, process=False)
    neutral_verts = np.asarray(neutral_mesh.vertices, dtype=np.float64)
    if np.asarray(neutral_mesh.faces, dtype=np.int32).tobytes() != faces.tobytes():
        die("neutral.ply faces != faces.npy (byte compare). STOP.")
    n_verts, n_faces = neutral_verts.shape[0], faces.shape[0]
    print(f"[rig] topology contract OK: V={n_verts} F={n_faces} (faces.npy byte-locked)")

    dec = BasisDecoder(eb, device)
    if dec.n_verts != n_verts:
        die(f"basis V={dec.n_verts} != neutral.ply V={n_verts}")

    # decode(0) must reproduce the reconstructor's neutral -- proves the math
    with torch.no_grad():
        v0 = dec.decode()[0].cpu().numpy().astype(np.float64)
    repro = float(np.abs(v0 - neutral_verts).max())
    if repro > C.NEUTRAL_REPRO_TOL_M:
        die(f"decode(0) differs from neutral.ply by {repro * MM:.3f} mm "
            f"(tol {C.NEUTRAL_REPRO_TOL_M * MM:.3f} mm). Basis desync -- STOP.")
    print(f"[rig] neutral reproduction OK: max|diff| = {repro * MM:.4f} mm")

    # ---- 2. landmark machinery (static iBUG 17..67 on the FLAME surface) -----
    # The FLAME 2023 Open release ships NO embedding file (pkl-only, measured
    # on the pod). recon.fit_flame persists whatever anchors it used (staged
    # release file OR the self-authored clean-room derivation) to
    # out/recon/lmk_embedding_static51.npz -- consuming THAT file guarantees
    # the rig solves against the exact anchors the fit optimized.
    from recon.flame_landmarks import load_persisted_embedding
    emb, emb_source = load_persisted_embedding()
    print(f"[rig] landmark anchors: {emb_source} "
          f"(from {RC.LMK_EMBEDDING_NPZ.name}; identical to the fit by construction)")
    st_faces, st_bary = emb["static_faces"], emb["static_bary"]
    if st_faces.shape[0] != 51:
        die(f"static embedding has {st_faces.shape[0]} points, expected 51.")
    tri = torch.as_tensor(faces.astype(np.int64)[st_faces], device=device)  # (51,3)
    bary = torch.as_tensor(st_bary, dtype=torch.float32, device=device)     # (51,3)

    def lmk(verts):
        """verts (B,V,3) -> (B,51,3) static-iBUG surface points."""
        return torch.einsum("blkc,lk->blc", verts[:, tri], bary)

    row = lambda ibug: ibug - 17                      # iBUG index -> static row
    rows = lambda ibugs: [i - 17 for i in ibugs]

    with torch.no_grad():
        v0_t = dec.decode()
        L0 = lmk(v0_t)[0]                             # (51,3) neutral landmarks
    L0np = L0.cpu().numpy().astype(np.float64)

    # ---- 3. laterality: measured, then photo-cross-checked --------------------
    # Axiom under test: FLAME/SMPL canonical +X = subject-left.
    lat_checks = {
        "eye_L_all(+x)": float(np.mean(L0np[rows(S.HANDLES["eye_L_all"])][:, 0])),
        "eye_R_all(-x)": -float(np.mean(L0np[rows(S.HANDLES["eye_R_all"])][:, 0])),
        "brow_L(+x)": float(np.mean(L0np[rows(S.HANDLES["brow_L"])][:, 0])),
        "brow_R(-x)": -float(np.mean(L0np[rows(S.HANDLES["brow_R"])][:, 0])),
        "corner_L54(+x)": float(L0np[row(54), 0]),
        "corner_R48(-x)": -float(L0np[row(48), 0]),
        "nostril_L35(+x)": float(L0np[row(35), 0]),
        "nostril_R31(-x)": -float(L0np[row(31), 0]),
    }
    bad = {k: v for k, v in lat_checks.items() if v <= 0}
    if bad:
        die(f"laterality inconsistency on the neutral mesh: {bad}. The iBUG "
            "semantics or the +X=subject-left axiom is broken -- STOP.")
    print("[rig] mesh laterality OK: all 8 left/right handle checks consistent")

    # photo cross-check: project both eye groups at the PHOTO state and compare
    # left/right ordering with MediaPipe's pixels (closes the loop end-to-end).
    idp = np.load(C.RECON_ID_PARAMS_NPZ, allow_pickle=True)
    lmn = np.load(C.RECON_LANDMARKS_NPZ, allow_pickle=True)
    fx, fy, cx, cy = [float(x) for x in idp["camera_fx_fy_cx_cy"]]
    with torch.no_grad():
        vp = dec.decode(
            expression=torch.as_tensor(idp["photo_expression"], dtype=torch.float32,
                                       device=device).unsqueeze(0),
            global_orient=aa(list(idp["photo_global_orient"])),
            jaw=aa(list(idp["photo_jaw_pose"])),
            transl=aa(list(idp["photo_transl"])),
        )
        Lp = lmk(vp)[0].cpu().numpy()
    proj_u = fx * Lp[:, 0] / np.clip(Lp[:, 2], 1e-6, None) + cx
    uL_f = float(np.mean(proj_u[rows(S.HANDLES["eye_L_all"])]))
    uR_f = float(np.mean(proj_u[rows(S.HANDLES["eye_R_all"])]))
    ibug68_px = lmn["ibug68_px"]
    uL_m = float(np.mean(ibug68_px[42:48, 0]))
    uR_m = float(np.mean(ibug68_px[36:42, 0]))
    if (uL_f - uR_f) * (uL_m - uR_m) <= 0:
        die("photo laterality cross-check FAILED: projected FLAME left/right "
            f"eye ordering (uL={uL_f:.1f}, uR={uR_f:.1f}) disagrees with "
            f"MediaPipe ({uL_m:.1f}, {uR_m:.1f}). Left/Right names would be "
            "mirrored -- STOP.")
    print(f"[rig] photo laterality OK: FLAME uL-uR={uL_f - uR_f:+.1f}px, "
          f"MediaPipe {uL_m - uR_m:+.1f}px (same sign)")

    # eye JOINTS: +x joint = subject-left. decode() slots: eye_a=joint3, eye_b=joint4.
    joints_neutral = np.asarray(eb["joints_neutral"], dtype=np.float64)  # (5,3)
    x3, x4 = joints_neutral[3, 0], joints_neutral[4, 0]
    if not (x3 * x4 < 0):
        die(f"eye joints 3/4 not on opposite x sides (x3={x3:+.4f}, x4={x4:+.4f}).")
    eye_slot = {"L": "a" if x3 > 0 else "b", "R": "b" if x3 > 0 else "a"}
    eye_joint_idx = {"L": 3 if x3 > 0 else 4, "R": 4 if x3 > 0 else 3}
    print(f"[rig] eye joints measured: subject-left = joint {eye_joint_idx['L']}, "
          f"subject-right = joint {eye_joint_idx['R']}")

    laterality_record = {
        "axiom": "FLAME/SMPL canonical +X = subject-left (verified 3 ways below)",
        "neutral_mesh_checks_m": lat_checks,
        "photo_cross_check_px": {"flame_uL_minus_uR": uL_f - uR_f,
                                 "mediapipe_uL_minus_uR": uL_m - uR_m},
        "eye_joint_left": eye_joint_idx["L"], "eye_joint_right": eye_joint_idx["R"],
    }

    # ---- 4. measured units (meters, subject-specific) -------------------------
    def _mean_pt(handle):
        return L0np[rows(S.HANDLES[handle])].mean(axis=0)

    units = {
        "MW": float(np.linalg.norm(L0np[row(54)] - L0np[row(48)])),
        "BH_L": float(np.linalg.norm(_mean_pt("brow_L") - _mean_pt("eye_L_all"))),
        "BH_R": float(np.linalg.norm(_mean_pt("brow_R") - _mean_pt("eye_R_all"))),
        "ABS": 1.0,
    }
    units["BH_MEAN"] = 0.5 * (units["BH_L"] + units["BH_R"])
    gaps = {i: float(np.linalg.norm(L0np[row(i)] - L0np[row(S.LID_COUNTERPART[i])]))
            for i in (37, 38, 43, 44)}
    units["EH_L"] = 0.5 * (gaps[43] + gaps[44])
    units["EH_R"] = 0.5 * (gaps[37] + gaps[38])
    print("[rig] units (mm): " + ", ".join(f"{k}={v * MM:.1f}" for k, v in units.items()
                                           if k != "ABS"))
    if units["MW"] < 0.02 or units["EH_L"] < 5e-4 or units["EH_R"] < 5e-4:
        die(f"implausible units {units} -- fit or embedding broken, STOP. "
            "(Near-zero eye aperture: photo eyes closed? Blink targets undefined.)")

    mouth_center = 0.5 * (L0np[row(48)] + L0np[row(54)])

    def direction_vec(ibug, token):
        """Resolve a spec token to a displacement DIRECTION (or, for 'gap',
        the full unnormalized gap vector) at neutral. Returns (vec, is_gap)."""
        basis = {"up": [0, 1, 0], "down": [0, -1, 0], "fwd": [0, 0, 1],
                 "back": [0, 0, -1], "left": [1, 0, 0], "right": [-1, 0, 0]}
        if token in basis:
            return np.asarray(basis[token], dtype=np.float64), False
        if token in ("to_center", "from_center"):
            d = mouth_center - L0np[row(ibug)]
            d[2] = 0.0                       # horizontal (XY-plane) only
            n = np.linalg.norm(d)
            if n < 1e-9:
                return np.zeros(3), False
            d /= n
            return (d if token == "to_center" else -d), False
        if token == "gap":
            cp = S.LID_COUNTERPART.get(ibug)
            if cp is None:
                die(f"'gap' target on non-lid iBUG {ibug}")
            return L0np[row(cp)] - L0np[row(ibug)], True
        die(f"unknown direction token '{token}'")

    # ---- 5. expression Jacobians on the landmarks ----------------------------
    E = dec.n_expr
    # At rest pose LBS is identity and expression is linear pre-LBS, so the
    # landmark Jacobian is EXACT: M[l,c,e] = sum_k bary[l,k]*expr_dirs[tri,c,e].
    with torch.no_grad():
        M_rest = torch.einsum("lk,lkce->lce", bary, dec.expr_dirs[tri]) \
            .cpu().numpy().astype(np.float64)                     # (51,3,E)

    def secant_jacobian(jaw_aa):
        """Landmark Jacobian at a posed state via batched secant (B=E decodes).
        Exact up to the (mild) expression->joint-location nonlinearity."""
        eps = C.SECANT_EPS
        with torch.no_grad():
            base = lmk(dec.decode(jaw=jaw_aa))                    # (1,51,3)
            eye_e = torch.eye(E, dtype=torch.float32, device=device) * eps
            pert = lmk(dec.decode(expression=eye_e,
                                  jaw=jaw_aa.expand(E, -1)))      # (E,51,3)
            J = (pert - base) / eps                               # (E,51,3)
        return J.permute(1, 2, 0).cpu().numpy().astype(np.float64), \
            base[0].cpu().numpy().astype(np.float64)

    def ridge_solve(M, d, w):
        """min ||W(Me - d)||^2 + lam||e||^2 with a coefficient-norm cap.
        M (51,3,E), d (51,3) desired, w (51,) row weights. Returns e, info."""
        Mf = M.reshape(-1, E)                                     # (153,E)
        df = d.reshape(-1)
        wf = np.repeat(w, 3)
        A = Mf.T @ (wf[:, None] ** 2 * Mf)
        b = Mf.T @ (wf ** 2 * df)
        lam = C.RIDGE_LAMBDA_REL * float(np.trace(A)) / E
        tries = 0
        while True:
            e = np.linalg.solve(A + lam * np.eye(E), b)
            if np.linalg.norm(e) <= C.E_NORM_CAP or tries >= C.LAMBDA_MAX_TRIES:
                break
            lam *= C.LAMBDA_ESCALATION
            tries += 1
        clipped = False
        nrm = float(np.linalg.norm(e))
        if nrm > C.E_NORM_CAP:
            e *= C.E_NORM_CAP / nrm
            clipped = True
        return e, {"lambda": lam, "lambda_escalations": tries,
                   "e_norm": float(np.linalg.norm(e)), "norm_clipped": clipped}

    # ---- 6. per-shape build ----------------------------------------------------
    manifest_shapes = {}
    params_store = {}
    demoted = []

    def record(name, supported, method, reason=None, max_d=None, mean_d=None,
               gates=None, params=None, notes=None):
        entry = {
            "supported": bool(supported),
            "status": "measured",
            "method": method,
            "intended": S.SPEC[name].get("intended"),
            "max_delta_m": None if max_d is None else float(max_d),
            "mean_delta_m": None if mean_d is None else float(mean_d),
            "gates": gates or {},
            "reason": reason,
            "notes": notes or S.SPEC[name].get("notes"),
            "ply": f"out/shapes/expr_{name}.ply" if supported else None,
        }
        if params:
            entry["params"] = params
        manifest_shapes[name] = entry
        tag = "SUPPORTED " if supported else "unsupported"
        stat = "" if max_d is None else f" max|d|={max_d * MM:.2f}mm"
        print(f"[rig] {tag} {name:24s} [{method}]{stat}"
              + (f"  reason: {reason}" if reason else ""))

    def emit(name, delta):
        """Delta (V,3, meters) -> out/shapes/expr_<name>.ply on the contract."""
        d = np.asarray(delta, dtype=np.float64)
        if not np.isfinite(d).all():
            die(f"{name}: non-finite delta.")
        export_and_verify_ply(C.SHAPES_DIR / f"expr_{name}.ply",
                              neutral_verts + d, faces)
        return float(np.linalg.norm(d, axis=1).max()), float(np.linalg.norm(d, axis=1).mean())

    def decode_delta(**kw):
        with torch.no_grad():
            v = dec.decode(**kw)[0].cpu().numpy().astype(np.float64)
        return v - v0

    # ---- 6a. jaw pose calibration (measured signs/axes) ----
    lower_inner = rows([65, 66, 67])
    lower_mid = rows([57, 66])

    def lmk_delta(**kw):
        with torch.no_grad():
            return (lmk(dec.decode(**kw))[0] - L0).cpu().numpy().astype(np.float64)

    jaw_open_aa, drop = None, 0.0
    for sgn in (+1.0, -1.0):
        cand = aa([sgn * C.JAW_OPEN_RAD, 0.0, 0.0])
        dl = lmk_delta(jaw=cand)[lower_inner].mean(axis=0)
        if -dl[1] > drop and abs(dl[1]) > abs(dl[0]):
            jaw_open_aa, drop = cand, float(-dl[1])
    jaw_cal = {"open_sign_axis": None if jaw_open_aa is None
               else [float(x) for x in jaw_open_aa[0].cpu()],
               "open_lip_drop_m": float(drop)}

    lat_axis, lat_dx = None, 0.0
    for axis in (1, 2):                       # try yaw (Y) then roll (Z)
        for sgn in (+1.0, -1.0):
            v3 = [0.0, 0.0, 0.0]
            v3[axis] = sgn * C.JAW_LAT_RAD
            dl = lmk_delta(jaw=aa(v3))[lower_mid].mean(axis=0)
            if abs(dl[0]) > abs(lat_dx) and abs(dl[0]) > abs(dl[1]):
                lat_axis, lat_dx = (axis, sgn), float(dl[0])
    jaw_cal["lat_axis_index"] = None if lat_axis is None else lat_axis[0]
    jaw_cal["lat_dx_m"] = float(lat_dx)
    print(f"[rig] jaw calibration: {jaw_cal}")

    # ---- 6b. build every shape in contract order ----
    for name in arkit52:
        spec = S.SPEC[name]
        method = spec["method"]

        if method == "none":
            record(name, False, "none", reason=spec["reason"])
            continue

        if method == "pose_jaw_open":
            if jaw_open_aa is None or drop < C.GATE_JAW_DROP_MIN_M:
                record(name, False, method, gates={"lip_drop_m": drop},
                       reason=f"measured lip drop {drop * MM:.2f}mm < "
                              f"{C.GATE_JAW_DROP_MIN_M * MM:.1f}mm gate")
                continue
            delta = decode_delta(jaw=jaw_open_aa)
            max_d, mean_d = emit(name, delta)
            params_store[f"jaw__{name}"] = np.asarray(jaw_cal["open_sign_axis"])
            record(name, True, method, max_d=max_d, mean_d=mean_d,
                   gates={"lip_drop_m": drop},
                   params={"jaw_pose_aa": jaw_cal["open_sign_axis"]})
            continue

        if method == "pose_jaw_lat":
            want_left = spec["dir"] == "left"
            ok = lat_axis is not None and abs(lat_dx) >= C.GATE_JAW_LAT_MIN_M
            if not ok:
                record(name, False, method, gates={"lat_dx_m": lat_dx},
                       reason=f"measured lateral lip motion {abs(lat_dx) * MM:.2f}mm < "
                              f"{C.GATE_JAW_LAT_MIN_M * MM:.1f}mm gate")
                continue
            axis, sgn = lat_axis
            if (lat_dx > 0) != want_left:     # calibrated sign moved it right
                sgn = -sgn
            v3 = [0.0, 0.0, 0.0]
            v3[axis] = sgn * C.JAW_LAT_RAD
            delta = decode_delta(jaw=aa(v3))
            max_d, mean_d = emit(name, delta)
            params_store[f"jaw__{name}"] = np.asarray(v3)
            record(name, True, method, max_d=max_d, mean_d=mean_d,
                   gates={"lat_dx_m": lat_dx}, params={"jaw_pose_aa": v3})
            continue

        if method == "pose_jaw_fwd":
            w_jaw = dec.lbs_weights[:, 2].cpu().numpy().astype(np.float64)
            delta = np.zeros((n_verts, 3))
            delta[:, 2] = w_jaw * C.JAW_FWD_M
            tri_np = faces.astype(np.int64)[st_faces]                  # (51,3)
            d_lmk = (delta[tri_np] * st_bary[..., None]).sum(axis=1)   # (51,3)
            fwd = float(d_lmk[rows([57, 66]), 2].mean())               # lower lip
            if fwd < C.GATE_JAW_FWD_MIN_M:
                record(name, False, method, gates={"lip_fwd_m": fwd},
                       reason=f"jaw-weighted lower lip advances only "
                              f"{fwd * MM:.2f}mm < {C.GATE_JAW_FWD_MIN_M * MM:.1f}mm "
                              "gate (skinning weights too diffuse)")
                continue
            max_d, mean_d = emit(name, delta)
            params_store[f"transl__{name}"] = np.array([0.0, 0.0, C.JAW_FWD_M])
            record(name, True, method, max_d=max_d, mean_d=mean_d,
                   gates={"lip_fwd_m": fwd},
                   params={"jaw_joint_translation_m": [0.0, 0.0, C.JAW_FWD_M]})
            continue

        if method == "pose_eye":
            side, look = spec["side"], spec["dir"]
            j = eye_joint_idx[side]
            w_eye = dec.lbs_weights[:, j].cpu().numpy().astype(np.float64)
            mask = w_eye > 0.5
            if int(mask.sum()) < C.GATE_EYE_MIN_VERTS:
                record(name, False, method,
                       gates={"eye_verts": int(mask.sum())},
                       reason=f"only {int(mask.sum())} vertices skinned to eye "
                              f"joint {j} (< {C.GATE_EYE_MIN_VERTS}); no eyeball "
                              "geometry to rotate")
                continue
            jz = joints_neutral[j, 2]
            front = mask & (v0[:, 2] > jz)
            if front.sum() < C.GATE_EYE_MIN_VERTS:
                front = mask
            mag = {"up": C.EYE_PITCH_UP_RAD, "down": C.EYE_PITCH_DOWN_RAD,
                   "in": C.EYE_YAW_RAD, "out": C.EYE_YAW_RAD}[look]
            axis = 0 if look in ("up", "down") else 1
            best, best_move = None, 0.0
            for sgn in (+1.0, -1.0):
                v3 = [0.0, 0.0, 0.0]
                v3[axis] = sgn * mag
                kw = {("eye_a" if eye_slot[side] == "a" else "eye_b"): aa(v3)}
                d = decode_delta(**kw)
                mv = d[front].mean(axis=0)
                # up: front of eyeball rises (+y). down: falls. in: toward the
                # nose midline (x moves opposite the eye's side). out: away.
                score = {"up": mv[1], "down": -mv[1],
                         "in": -mv[0] if side == "L" else mv[0],
                         "out": mv[0] if side == "L" else -mv[0]}[look]
                if score > best_move:
                    best, best_move = (v3, d), float(score)
            if best is None or best_move < C.GATE_EYE_MOVE_MIN_M:
                record(name, False, method, gates={"front_move_m": best_move},
                       reason=f"measured eyeball motion {best_move * MM:.2f}mm < "
                              f"{C.GATE_EYE_MOVE_MIN_M * MM:.1f}mm gate")
                continue
            v3, delta = best
            max_d, mean_d = emit(name, delta)
            params_store[f"eye__{name}"] = np.asarray(
                [float(j)] + [float(x) for x in v3])
            record(name, True, method, max_d=max_d, mean_d=mean_d,
                   gates={"front_move_m": best_move, "eye_verts": int(mask.sum())},
                   params={"eye_joint": j, "eye_pose_aa": v3})
            continue

        # ---- PCA shapes -----------------------------------------------------
        if method == "pca_jawopen":            # mouthClose
            if jaw_open_aa is None:
                record(name, False, method,
                       reason="jawOpen calibration failed; cannot linearize")
                continue
            M, base_lmk = secant_jacobian(jaw_open_aa)
            d = np.zeros((51, 3))
            target_rows = []
            for iu, il in S.INNER_LIP_PAIRS:
                mid = 0.5 * (base_lmk[row(iu)] + base_lmk[row(il)])
                d[row(iu)] = mid - base_lmk[row(iu)]
                d[row(il)] = mid - base_lmk[row(il)]
                target_rows += [row(iu), row(il)]
            free_rows = set()
            for h in spec["free"]:
                free_rows.update(rows(S.HANDLES[h]))
            w = np.full(51, C.STABILIZER_WEIGHT)
            w[list(free_rows)] = 0.0
            w[target_rows] = 1.0
            e, info = ridge_solve(M, d, w)
            e_t = torch.as_tensor(e, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                v_open = dec.decode(jaw=jaw_open_aa)
                v_seal = dec.decode(expression=e_t, jaw=jaw_open_aa)
                a = (lmk(v_seal) - lmk(v_open))[0].cpu().numpy().astype(np.float64)
                delta = (v_seal - v_open)[0].cpu().numpy().astype(np.float64)
            base_for_gates = d
        else:                                   # plain rest-pose pca
            M, base_lmk = M_rest, L0np
            d = np.zeros((51, 3))
            target_rows = []
            for t in spec["targets"]:
                for ibug in S.HANDLES[t["handle"]]:
                    vec, is_gap = direction_vec(ibug, t["dir"])
                    scale = t["factor"] * (1.0 if is_gap else units[t["unit"]])
                    d[row(ibug)] += vec * scale
                    target_rows.append(row(ibug))
            target_rows = sorted(set(target_rows))
            free_rows = set()
            for h in spec.get("free", []):
                free_rows.update(rows(S.HANDLES[h]))
            free_rows -= set(target_rows)
            w = np.full(51, C.STABILIZER_WEIGHT)
            w[list(free_rows)] = 0.0
            w[target_rows] = 1.0
            e, info = ridge_solve(M, d, w)
            e_t = torch.as_tensor(e, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                v_k = dec.decode(expression=e_t)
                a = (lmk(v_k)[0] - L0).cpu().numpy().astype(np.float64)
                delta = v_k[0].cpu().numpy().astype(np.float64) - v0
            base_for_gates = d

        # ---- measured gates (shared by both pca paths) ----
        tr = sorted(set(target_rows))
        a_t, d_t = a[tr].reshape(-1), base_for_gates[tr].reshape(-1)
        denom = float(np.linalg.norm(a_t) * np.linalg.norm(d_t))
        cos = float(a_t @ d_t / denom) if denom > 0 else 0.0
        amp = float(np.linalg.norm(a_t) / max(np.linalg.norm(d_t), 1e-12))
        max_d = float(np.linalg.norm(delta, axis=1).max())
        mean_d = float(np.linalg.norm(delta, axis=1).mean())
        rms = lambda x: float(np.sqrt((x ** 2).sum(axis=1).mean())) if len(x) else 0.0
        rms_t = rms(a[tr])
        gates = {"target_cos": cos, "amp_ratio": amp, "max_delta_m": max_d,
                 "solver": info}
        fails = []
        if max_d < C.GATE_MIN_MAX_DELTA_M:
            fails.append(f"max|delta| {max_d * MM:.2f}mm < {C.GATE_MIN_MAX_DELTA_M * MM:.1f}mm")
        if cos < C.GATE_TARGET_COS:
            fails.append(f"target cos {cos:.2f} < {C.GATE_TARGET_COS}")
        if not (C.GATE_AMP_MIN <= amp <= C.GATE_AMP_MAX):
            fails.append(f"amplitude ratio {amp:.2f} outside "
                         f"[{C.GATE_AMP_MIN},{C.GATE_AMP_MAX}]")
        leak = spec.get("leak_pair")
        if leak:
            mir_rows = sorted({r for h in leak[1] for r in rows(S.HANDLES[h])})
            rms_m = rms(a[mir_rows])
            gates["leak_ratio"] = None if rms_t == 0 else rms_m / rms_t
            if rms_t > 0 and rms_m / rms_t > C.GATE_LEAK_RATIO:
                fails.append(f"mirror-side leakage {rms_m / rms_t:.2f} > "
                             f"{C.GATE_LEAK_RATIO} (not one-sided in FLAME PCA)")
        stab_rows = sorted(set(range(51)) - set(tr) - free_rows)
        rms_s = rms(a[stab_rows])
        gates["stabilizer_ratio"] = None if rms_t == 0 else rms_s / rms_t
        if rms_t > 0 and rms_s / rms_t > C.GATE_STAB_RATIO:
            fails.append(f"off-target motion {rms_s / rms_t:.2f}x target > "
                         f"{C.GATE_STAB_RATIO}x (solve is not local)")

        if fails:
            demoted.append(name)
            record(name, False, method, gates=gates,
                   reason="; ".join(fails))
            continue
        max_d, mean_d = emit(name, delta)
        params_store[f"expr__{name}"] = e.astype(np.float32)
        if method == "pca_jawopen":
            params_store[f"jaw__{name}"] = np.asarray(jaw_cal["open_sign_axis"])
        record(name, True, method, max_d=max_d, mean_d=mean_d, gates=gates,
               params={"e_norm": info["e_norm"], "lambda": info["lambda"]})

    # ---- 7. neutral pass-through + manifest + params ---------------------------
    shutil.copyfile(C.RECON_NEUTRAL_PLY, C.NEUTRAL_OUT_PLY)
    print(f"[rig] neutral passed through -> {C.NEUTRAL_OUT_PLY}")

    if sorted(manifest_shapes.keys()) != sorted(arkit52):
        die("manifest does not account for all 52 names -- internal bug, STOP.")
    n_sup = sum(1 for v in manifest_shapes.values() if v["supported"])
    import hashlib
    faces_sha = hashlib.sha256(np.load(C.RECON_FACES_NPY).tobytes()).hexdigest()
    manifest = {
        "schema": "b1-arkit-shape-manifest/1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generated_by": "arkit-rigger (Track B1) -- rig/build_arkit_shapes.py, MEASURED on the GPU pod",
        "run_state": "measured-on-pod",
        "name_contract": str(C.NAME_CONTRACT_JSON),
        "topology": {"n_vertices": n_verts, "n_faces": n_faces,
                     "faces_npy_sha256": faces_sha,
                     "contract": "out/recon/faces.npy (byte-locked; every PLY re-read and compared)"},
        "laterality": laterality_record,
        "units_m": units,
        "jaw_calibration": jaw_cal,
        "counts": {"total": 52, "supported": n_sup, "unsupported": 52 - n_sup},
        "demoted_by_gates": demoted,
        "shapes": {name: manifest_shapes[name] for name in arkit52},
    }
    with open(C.ARKIT_MANIFEST_JSON, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=float)  # numpy scalars -> json
    np.savez(C.SHAPE_PARAMS_NPZ, **params_store)
    print(f"[rig] manifest -> {C.ARKIT_MANIFEST_JSON} "
          f"(supported {n_sup}/52, demoted by gates: {demoted or 'none'})")
    print(f"[rig] solved parameters -> {C.SHAPE_PARAMS_NPZ}")
    print(f"[rig] DONE in {time.time() - t0:.1f}s. Next: python -m rig.verify_shapes")


if __name__ == "__main__":
    main()
