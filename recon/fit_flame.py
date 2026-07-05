"""Stage 2 -- optimization-based FLAME 2023 Open fit to MediaPipe landmarks.

POD-ONLY (pod_guard: requires CUDA torch). The METHOD IS FIXED by the
licensing gate (out/compliance_report.md B1-4): pure optimization against
MediaPipe FaceLandmarker landmarks. NO learned reconstruction weights
(no DECA/EMOCA/Arc2Avatar/InsightFace), NO nvdiffrast, NO 3DGS.

CAMERA / COORDINATE CONVENTION (single source of truth, also used by
bake_texture.py):
  - OpenCV pinhole: camera at the origin, +X right, +Y down, +Z forward;
    u = f*x/z + cx, v = f*y/z + cy; principal point = image center;
    single square-pixel focal f (optimized, tethered to its init because a
    single image cannot resolve the focal/depth ambiguity).
  - The fitted global_orient/transl live INSIDE the FLAME decode (root joint
    + translation), so decoded vertices are already in camera space.
  - FLAME's canonical frame is +Y up / face toward +Z, so global_orient is
    initialized to a pi rotation about X, which maps FLAME +Y -> image up and
    points the face at the camera. The optimizer refines from there.

THREE STAGES (classic coarse-to-fine; landmark data term only):
  A rigid+camera : global_orient, transl, log_focal    (stable-core subset)
  B + identity   : + betas (300)                        (all static-51 [+contour])
  C + expression : + expression (100), jaw_pose         (all, full regularizers)
Neck and eye poses stay ZERO: with a single photo, neck is redundant with
global_orient, and MediaPipe eyelid landmarks do not constrain gaze.

LANDMARK ANCHORS (pkl-only Open release; measured on the pod 2026-07-05):
FLAME 2023 Open ships ONLY flame2023.pkl -- no landmark embedding, no UV
template. If the operator staged an embedding file it is honored as an
override; otherwise the static-51 anchors are SELF-AUTHORED clean-room from
the pkl's own arrays (recon/flame_landmarks.py -- no NC source touched).
Whichever anchors were used are persisted to lmk_embedding_static51.npz so
the rigger consumes the identical definitions.

EXPORTS (out/recon/):
  neutral.ply            IDENTITY-ONLY mesh: betas = fitted, expression = 0,
                         all poses = 0, no global transform. FLAME topology.
                         Face-count-verified before this stage reports success.
  faces.npy              (F,3) int32 triangles -- THE topology contract,
                         extracted from the pkl (cross-checked against a staged
                         template obj only if one exists -- optional override).
  id_params.npz          fitted identity (betas) + everything needed to
                         reproduce the photo-state and camera (for the bake).
  expression_basis.npz   the FLAME expression + jaw-pose basis handle for
                         arkit-rigger (see expression_basis_notes.json).
  lmk_embedding_static51.npz  the landmark anchors ACTUALLY used (see above).
  fit_summary.json, fit_debug/*.png
  flame_landmarks_selfauthored.{json,png}  (self-authored path only) debug dump

Run:  python -m recon.fit_flame   (after recon.landmarks)
"""

import json
import sys
import time
from datetime import datetime, timezone

import numpy as np

from . import config as C
from .mp_flame_correspondence import (STABLE_CORE_IBUG, per_landmark_weights)
from .pod_guard import require_cuda_torch

torch = None  # bound in main() after the pod guard passes


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def project_pinhole(pts, focal, cx, cy):
    """OpenCV pinhole projection. pts (B,L,3) camera-space -> (B,L,2) px."""
    z = pts[..., 2].clamp(min=1e-6)
    u = focal * pts[..., 0] / z + cx
    v = focal * pts[..., 1] / z + cy
    return torch.stack([u, v], dim=-1)


def assert_ply_has_faces(path):
    """Parse the PLY header; die if faces are absent or zero (the classic
    vertex-only silent failure). Returns (n_vertices, n_faces)."""
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
    if n_v <= 0 or n_f <= 0:
        sys.exit(
            f"[fit FATAL] {path} is not a valid faced mesh "
            f"(element vertex {n_v}, element face {n_f}). A vertex-only PLY is "
            "the classic silent failure -- REJECTED."
        )
    print(f"[fit] PLY verified: element vertex {n_v}, element face {n_f}")
    return n_v, n_f


def maybe_check_template_topology(flame_faces_np):
    """The FLAME 2023 Open release ships ONLY the pkl (measured on the pod),
    so the pkl faces ARE the topology contract. If the operator staged a
    template obj (optional override), cross-check it here; any mismatch is a
    hard STOP. Returns the template path, or None in the expected pkl-only case."""
    template_path = C.find_optional_flame_file(C.FLAME_TEMPLATE_CANDIDATES)
    if template_path is None:
        print("[fit] no template obj staged -- EXPECTED for the pkl-only FLAME "
              "2023 Open release. Topology contract comes from the pkl alone; "
              "the UV layout is generated clean-room at bake time "
              "(recon/uv_unwrap.py).")
        return None

    from pytorch3d.io import load_obj

    _verts, faces, _aux = load_obj(str(template_path), load_textures=False)
    tpl_faces = faces.verts_idx.cpu().numpy().astype(np.int64)
    if tpl_faces.shape != flame_faces_np.shape or not np.array_equal(tpl_faces, flame_faces_np):
        sys.exit(
            "[fit FATAL] TOPOLOGY MISMATCH between the FLAME pkl and the staged "
            f"{template_path.name}: pkl faces {flame_faces_np.shape}, template faces "
            f"{tpl_faces.shape}, equal={np.array_equal(tpl_faces, flame_faces_np)}. "
            "The one-topology guarantee is the entire reason this pipeline works -- "
            "STOPPING. Reconcile the FLAME download before rerunning."
        )
    print(f"[fit] topology cross-check OK: staged template faces == pkl faces "
          f"{tpl_faces.shape}")
    return template_path


def draw_fit_overlay(image_rgb, targets_px, projected_px, out_path, title):
    import cv2

    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR).copy()
    for (tx, ty), (px, py) in zip(targets_px, projected_px):
        t = (int(round(tx)), int(round(ty)))
        p = (int(round(px)), int(round(py)))
        cv2.line(bgr, t, p, (0, 200, 200), 1, cv2.LINE_AA)
        cv2.circle(bgr, t, 2, (0, 255, 0), -1)   # green = MediaPipe target
        cv2.circle(bgr, p, 2, (0, 0, 255), -1)   # red = projected FLAME landmark
    cv2.putText(bgr, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.imwrite(str(out_path), bgr)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> None:
    global torch
    torch = require_cuda_torch()
    import torch.nn.functional as F
    from PIL import Image

    C.ensure_out_dirs()
    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)
    device = torch.device(C.DEVICE)
    t0 = time.time()

    # ---- inputs from stage 1 ------------------------------------------------
    if not C.LANDMARKS_NPZ.is_file():
        sys.exit(f"[fit FATAL] {C.LANDMARKS_NPZ} missing -- run `python -m recon.landmarks` first.")
    lm = np.load(C.LANDMARKS_NPZ, allow_pickle=True)
    ibug68_px = lm["ibug68_px"]                       # (68,2)
    h, w = [int(x) for x in lm["image_hw"]]
    diag = float(np.hypot(h, w))
    image_rgb = np.asarray(Image.open(C.CANONICAL_IMAGE).convert("RGB"))
    assert image_rgb.shape[:2] == (h, w), "canonical image / landmarks desync"

    # ---- FLAME model + landmark anchors + topology contract -------------------
    from .flame_landmarks import (SELF_AUTHORED_SOURCE, build_static51_embedding,
                                  persist_embedding)
    from .flame_model import FlameModel, load_landmark_embedding

    model_path = C.find_flame_file(C.FLAME_MODEL_CANDIDATES, "FLAME shape model pkl")
    flame = FlameModel(model_path, n_shape=C.N_SHAPE, n_expr=C.N_EXPR, device=C.DEVICE)

    # Landmark anchors: staged release file if present (optional override),
    # else SELF-AUTHORED clean-room from the pkl arrays (the expected case for
    # the pkl-only FLAME 2023 Open release -- NC embeddings are barred).
    emb_path = C.find_optional_flame_file(C.FLAME_LMK_EMBEDDING_CANDIDATES)
    if emb_path is not None:
        emb = load_landmark_embedding(emb_path)
        emb_source = f"release/staged file: {emb_path}"
        print(f"[fit] landmark embedding from staged file (override): {emb_path}")
    else:
        print("[fit] no landmark-embedding file staged -- EXPECTED for the "
              "pkl-only FLAME 2023 Open release. Deriving the SELF-AUTHORED "
              "static-51 anchors geometrically from the pkl "
              "(recon/flame_landmarks.py; clean-room, no NC source).")
        rest_joints = (flame.np_j_regressor.astype(np.float64)
                       @ flame.np_v_template.astype(np.float64))
        emb = build_static51_embedding(
            v_template=flame.np_v_template.astype(np.float64),
            faces=flame.faces,
            lbs_weights=flame.np_lbs_weights.astype(np.float64),
            rest_joints=rest_joints,
            debug_json=C.FLAME_LMK_SELF_JSON,
            debug_png=C.FLAME_LMK_SELF_PNG,
        )
        emb_source = SELF_AUTHORED_SOURCE
    # Persist whichever anchors are in use: rig/build_arkit_shapes.py consumes
    # THIS file, so fit and rig can never diverge on anchor definitions.
    persist_embedding(emb, emb_source)

    n_verts, n_faces = flame.n_verts, flame.faces.shape[0]
    if (n_verts, n_faces) != (C.EXPECTED_N_VERTS, C.EXPECTED_N_FACES):
        print(
            f"[fit WARN] measured topology V={n_verts} F={n_faces} differs from the "
            f"documented FLAME expectation V={C.EXPECTED_N_VERTS} F={C.EXPECTED_N_FACES}. "
            "Proceeding with MEASURED values -- but reconcile recon_report.md and make "
            "sure downstream asserts use faces.npy, not the doc numbers."
        )
    template_path = maybe_check_template_topology(flame.faces)

    # ---- targets --------------------------------------------------------------
    w68 = per_landmark_weights()
    static_ibug = np.arange(17, 68)
    tgt_static = torch.tensor(ibug68_px[static_ibug], dtype=torch.float32, device=device)
    w_static = torch.tensor(w68[static_ibug], dtype=torch.float32, device=device)
    static_faces, static_bary = emb["static_faces"], emb["static_bary"]
    if static_faces.shape[0] != 51:
        sys.exit(f"[fit FATAL] static embedding has {static_faces.shape[0]} points, expected 51.")

    use_contour = "full_faces" in emb
    if use_contour:
        contour_ibug = np.arange(0, 17)
        tgt_contour = torch.tensor(ibug68_px[contour_ibug], dtype=torch.float32, device=device)
        w_contour = torch.tensor(w68[contour_ibug], dtype=torch.float32, device=device)
        contour_faces = emb["full_faces"][:17]
        contour_bary = emb["full_bary"][:17]
        print("[fit] contour (iBUG 0-16) enabled at low weight (full-68 embedding present)")
    else:
        print("[fit] contour disabled (no full-68 embedding in this FLAME release); "
              "fitting the static 51 only")

    core_rows = torch.tensor([i - 17 for i in STABLE_CORE_IBUG], dtype=torch.long, device=device)

    # ---- parameters + init -----------------------------------------------------
    cx, cy = w / 2.0, h / 2.0
    f_init = C.FOCAL_INIT_FACTOR * max(h, w)
    # depth init from outer-canthus pixel distance vs a nominal 3D interocular
    eye_px = float(np.linalg.norm(ibug68_px[36] - ibug68_px[45]))
    z0 = f_init * C.INTEROCULAR_3D_M / max(eye_px, 1.0)
    face_center = ibug68_px[[30, 33, 36, 45, 48, 54]].mean(axis=0)
    x0 = (face_center[0] - cx) * z0 / f_init
    y0 = (face_center[1] - cy) * z0 / f_init
    print(f"[fit] init: f={f_init:.1f}px z0={z0:.3f}m xy0=({x0:+.3f},{y0:+.3f})m")

    p = {
        "global_orient": torch.tensor([[np.pi, 0.0, 0.0]], dtype=torch.float32,
                                      device=device, requires_grad=True),
        "transl": torch.tensor([[x0, y0, z0]], dtype=torch.float32,
                               device=device, requires_grad=True),
        "log_focal": torch.zeros(1, dtype=torch.float32, device=device, requires_grad=True),
        "betas": torch.zeros(1, C.N_SHAPE_FIT, dtype=torch.float32,
                             device=device, requires_grad=True),
        "expression": torch.zeros(1, C.N_EXPR_FIT, dtype=torch.float32,
                                  device=device, requires_grad=True),
        "jaw_pose": torch.zeros(1, 3, dtype=torch.float32, device=device, requires_grad=True),
    }

    def decode_photo_state():
        return flame.decode(
            betas=p["betas"], expression=p["expression"],
            global_orient=p["global_orient"], jaw_pose=p["jaw_pose"],
            transl=p["transl"],
        )

    def projected_landmarks(verts, faces_idx, bary):
        pts = flame.surface_points(verts, faces_idx, bary)
        return project_pinhole(pts, f_init * torch.exp(p["log_focal"]), cx, cy)

    def data_term(proj, tgt, wts):
        resid = (proj[0] - tgt) * (100.0 / diag)      # scale-free units
        per = F.smooth_l1_loss(resid, torch.zeros_like(resid),
                               beta=1.0, reduction="none").sum(dim=1)
        return (per * wts).sum() / wts.sum()

    def total_loss(stage):
        verts = decode_photo_state()
        proj_static = projected_landmarks(verts, static_faces, static_bary)
        if stage == "A":
            loss = data_term(proj_static[:, core_rows], tgt_static[core_rows],
                             w_static[core_rows])
        else:
            loss = data_term(proj_static, tgt_static, w_static)
            if use_contour:
                proj_c = projected_landmarks(verts, contour_faces, contour_bary)
                loss = loss + data_term(proj_c, tgt_contour, w_contour)
        loss = loss + C.FOCAL_REG_W * (p["log_focal"] ** 2).sum()
        if stage in ("B", "C"):
            loss = loss + C.SHAPE_REG_W * (p["betas"] ** 2).mean()
        if stage == "C":
            loss = loss + C.EXPR_REG_W * (p["expression"] ** 2).mean()
            loss = loss + C.JAW_REG_W * (p["jaw_pose"] ** 2).sum()
        return loss

    # ---- optimize --------------------------------------------------------------
    log = {"stages": []}
    schedule = [
        ("A", [p["global_orient"], p["transl"], p["log_focal"]], C.STAGE_A_LR, C.STAGE_A_ITERS),
        ("B", [p["global_orient"], p["transl"], p["log_focal"], p["betas"]],
         C.STAGE_B_LR, C.STAGE_B_ITERS),
        ("C", list(p.values()), C.STAGE_C_LR, C.STAGE_C_ITERS),
    ]
    for stage, params, lr, iters in schedule:
        opt = torch.optim.Adam(params, lr=lr)
        losses = []
        for it in range(iters):
            opt.zero_grad()
            loss = total_loss(stage)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
            if it % 50 == 0 or it == iters - 1:
                print(f"[fit] stage {stage} it {it:4d}/{iters}  loss {losses[-1]:.5f}")
        log["stages"].append({"stage": stage, "iters": iters, "lr": lr,
                              "loss_first": losses[0], "loss_last": losses[-1]})
        with torch.no_grad():
            verts = decode_photo_state()
            proj = projected_landmarks(verts, static_faces, static_bary)[0].cpu().numpy()
        draw_fit_overlay(image_rgb, ibug68_px[static_ibug], proj,
                         C.FIT_DEBUG_DIR / f"stage_{stage}_overlay.png",
                         f"stage {stage}: green=MediaPipe target, red=FLAME projection")

    # ---- final metrics -----------------------------------------------------------
    with torch.no_grad():
        verts_photo = decode_photo_state()
        proj = projected_landmarks(verts_photo, static_faces, static_bary)[0].cpu().numpy()
        err_px = np.linalg.norm(proj - ibug68_px[static_ibug], axis=1)
    rmse_px = float(np.sqrt((err_px ** 2).mean()))
    focal_final = float(f_init * np.exp(float(p["log_focal"].detach())))
    print(f"[fit] static-51 landmark RMSE = {rmse_px:.2f}px  (image diag {diag:.0f}px)  "
          f"focal = {focal_final:.1f}px")

    # ---- exports -------------------------------------------------------------------
    import trimesh

    with torch.no_grad():
        v_neutral = flame.decode(betas=p["betas"])[0].cpu().numpy().astype(np.float64)
        joints_neutral = torch.einsum(
            "jv,bvc->bjc", flame.j_regressor,
            flame.shaped_vertices(p["betas"], None)
        )[0].cpu().numpy().astype(np.float32)

    # neutral.ply -- identity fixed, expression zero, no pose, no global transform
    mesh = trimesh.Trimesh(vertices=v_neutral, faces=flame.faces, process=False)
    mesh.export(C.NEUTRAL_PLY)
    ply_v, ply_f = assert_ply_has_faces(C.NEUTRAL_PLY)
    if (ply_v, ply_f) != (n_verts, n_faces):
        sys.exit(f"[fit FATAL] exported PLY counts ({ply_v},{ply_f}) != model ({n_verts},{n_faces}).")

    # faces.npy -- THE topology contract (int32; every downstream mesh must match)
    np.save(C.FACES_NPY, flame.faces.astype(np.int32))
    print(f"[fit] topology contract -> {C.FACES_NPY} {flame.faces.shape} int32")

    betas_np = p["betas"].detach()[0].cpu().numpy().astype(np.float64)
    np.savez(
        C.ID_PARAMS_NPZ,
        # ---- IDENTITY (the only thing downstream may treat as the subject) ----
        betas=betas_np,
        n_shape=np.int64(C.N_SHAPE_FIT),
        # ---- photo-state (NOT identity; needed by the bake + reproducibility) --
        photo_expression=p["expression"].detach()[0].cpu().numpy().astype(np.float64),
        photo_jaw_pose=p["jaw_pose"].detach()[0].cpu().numpy().astype(np.float64),
        photo_global_orient=p["global_orient"].detach()[0].cpu().numpy().astype(np.float64),
        photo_transl=p["transl"].detach()[0].cpu().numpy().astype(np.float64),
        photo_neck_pose=np.zeros(3), photo_eye_pose_a=np.zeros(3), photo_eye_pose_b=np.zeros(3),
        # ---- camera (OpenCV pinhole; convention documented in this module) ------
        camera_fx_fy_cx_cy=np.array([focal_final, focal_final, cx, cy]),
        image_hw=np.array([h, w], dtype=np.int64),
        # ---- provenance / quality ----------------------------------------------
        landmark_rmse_px=np.float64(rmse_px),
        n_expr=np.int64(C.N_EXPR_FIT),
        flame_model_file=str(model_path),
        flame_template_file=(str(template_path) if template_path is not None else
                             "ABSENT (pkl-only FLAME 2023 Open release; UV is "
                             "generated clean-room at bake -- recon/uv_unwrap.py)"),
        landmark_embedding_source=emb_source,
        created_utc=datetime.now(timezone.utc).isoformat(),
    )
    print(f"[fit] identity + photo-state params -> {C.ID_PARAMS_NPZ}")

    # expression-basis handle for arkit-rigger (see expression_basis_notes.json)
    np.savez(
        C.EXPR_BASIS_NPZ,
        expr_dirs=flame.np_expr_dirs,            # (V,3,100) PCA expression dirs
        posedirs=flame.np_posedirs,              # (V,3,36) pose correctives
        j_regressor=flame.np_j_regressor,        # (5,V)
        lbs_weights=flame.np_lbs_weights,        # (V,5)
        parents=flame.parents.astype(np.int64),  # (5,)
        faces=flame.faces.astype(np.int32),      # (F,3) == faces.npy
        v_neutral=v_neutral.astype(np.float32),  # fitted identity, expr=0, pose=0
        joints_neutral=joints_neutral,           # (5,3) joints of the fitted identity
        betas=betas_np.astype(np.float32),
    )
    _write_basis_notes(model_path, n_verts, n_faces)
    print(f"[fit] expression basis handle -> {C.EXPR_BASIS_NPZ} + {C.EXPR_BASIS_NOTES}")

    log.update({
        "landmark_rmse_px_static51": rmse_px,
        "per_landmark_error_px": {str(i): float(e) for i, e in zip(static_ibug, err_px)},
        "focal_px": focal_final,
        "contour_used": bool(use_contour),
        "landmark_embedding_source": emb_source,
        "template_obj": str(template_path) if template_path is not None else
                        "ABSENT (pkl-only Open release)",
        "n_vertices": n_verts, "n_faces": n_faces,
        "expected_vertices": C.EXPECTED_N_VERTS, "expected_faces": C.EXPECTED_N_FACES,
        "runtime_s": time.time() - t0,
    })
    with open(C.FIT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print(f"[fit] DONE in {time.time() - t0:.1f}s -> {C.FIT_SUMMARY_JSON}")


def _write_basis_notes(model_path, n_verts, n_faces) -> None:
    """The rigger-facing contract note. FLAME's expression space is NOT
    ARKit-named; mapping is the arkit-rigger's stage (plan section 3.2)."""
    notes = {
        "schema": "b1-flame-expression-basis-handle/1.0",
        "consumer": "arkit-rigger",
        "topology": {
            "n_vertices": n_verts,
            "n_faces": n_faces,
            "contract": "out/recon/faces.npy -- every synthesized mesh MUST reuse it verbatim",
        },
        "how_to_synthesize_a_mesh": {
            "formula": (
                "verts(expr, jaw) = LBS( v_neutral_rest + expr_dirs @ expr  "
                "+ posedirs @ posefeat(jaw), joints, lbs_weights ) ; identity (betas) "
                "stays FIXED at the fitted value. Recommended: import recon.flame_model."
                "FlameModel and call decode(betas=<fitted>, expression=..., jaw_pose=...) "
                "with the same flame2023 pkl -- that reproduces the exact fit-time math."
            ),
            "delta_definition": (
                "ARKit delta for shape k = verts(expr_k, jaw_k) - verts(0, 0); both terms "
                "decoded with the SAME fitted betas, global_orient=0, transl=0."
            ),
        },
        "NOT_arkit_named": (
            "FLAME's 100 expression components are PCA axes learned from 4D scans. "
            "They are NOT ARKit-named and no single axis corresponds to a single ARKit "
            "shape. jawOpen-class shapes come from jaw_pose (an articulated joint through "
            "LBS), not from the linear basis. Solving FLAME (expression, jaw_pose) "
            "coefficient vectors for each of the 52 ARKit names is the arkit-rigger's "
            "stage (plan section 3.2). This file only hands over the basis."
        ),
        "arkit_name_contract": "out/arkit_51_52_map.json (exact 52 spellings, case-sensitive)",
        "candidate_unsupported_for_rigger_to_adjudicate": {
            "tongueOut": "FLAME has NO tongue geometry -- cannot be expressed; expect UNSUPPORTED.",
            "cheekPuff": "cheek inflation is poorly spanned by FLAME's scan-based expression PCA -- likely weak/unsupported.",
            "cheekSquintLeft": "weak candidate in FLAME expression space -- verify visually.",
            "cheekSquintRight": "weak candidate in FLAME expression space -- verify visually.",
            "noseSneerLeft": "nose wrinkle/sneer is weakly represented -- verify visually.",
            "noseSneerRight": "nose wrinkle/sneer is weakly represented -- verify visually.",
        },
        "eye_shapes_note": (
            "eyeLookUp/Down/In/Out (x2) map to FLAME EYE JOINT rotations (joints 3 and 4), "
            "not to the expression PCA. Eye-joint LATERALITY IS UNVERIFIED: before using, "
            "rotate joint 3 alone, render, and record which eye moved. eyeBlink/eyeSquint/"
            "eyeWide live (to varying degree) in the expression PCA -- solve and verify."
        ),
        "do_not_fabricate": (
            "If a shape cannot be expressed, mark it unsupported in the rig manifest. "
            "Never rename or fake a delta to hide a gap (CLAUDE.md invariant 2/5)."
        ),
        "flame_model_file": str(model_path),
        "license": "FLAME 2023 Open, CC-BY-4.0 -- attribution obligations in out/compliance_report.md B1-1",
    }
    with open(C.EXPR_BASIS_NOTES, "w", encoding="utf-8") as f:
        json.dump(notes, f, indent=2)


if __name__ == "__main__":
    main()
