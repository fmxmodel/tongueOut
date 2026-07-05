"""Stage 4 -- measured verification of every out/recon/ artifact.

POD-ONLY (pod_guard). Proves success by MEASUREMENT, not by claim
(CLAUDE.md invariant 4): re-parses the PLY header, re-loads every array,
cross-checks the topology contract, hashes it, and writes
out/recon/recon_run_manifest.json for qa-verifier and arkit-rigger.

Exit code != 0 on ANY failed check. Run:  python -m recon.verify_outputs
"""

import hashlib
import json
import sys
import time

import numpy as np

from . import config as C
from .pod_guard import require_pod

FAILURES = []


def check(name: str, ok: bool, detail: str) -> None:
    tag = "PASS" if ok else "FAIL"
    print(f"[verify] {tag}  {name}: {detail}")
    if not ok:
        FAILURES.append(f"{name}: {detail}")


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


def sha256_of(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    require_pod()
    manifest = {"schema": "b1-recon-run-manifest/1.0",
                "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    # ---- neutral.ply: faces MUST exist (vertex-only PLY = classic silent failure)
    if C.NEUTRAL_PLY.is_file():
        n_v, n_f = parse_ply_header(C.NEUTRAL_PLY)
        check("neutral.ply faces", n_v > 0 and n_f > 0,
              f"element vertex {n_v}, element face {n_f}")
        manifest["neutral_ply"] = {"n_vertices": n_v, "n_faces": n_f,
                                   "sha256": sha256_of(C.NEUTRAL_PLY)}
        import trimesh
        mesh = trimesh.load(C.NEUTRAL_PLY, process=False)
        check("neutral.ply trimesh reload",
              mesh.vertices.shape[0] == n_v and mesh.faces.shape[0] == n_f,
              f"trimesh sees V={mesh.vertices.shape[0]} F={mesh.faces.shape[0]}")
        finite = bool(np.isfinite(mesh.vertices).all())
        check("neutral.ply finite vertices", finite, f"all finite = {finite}")
    else:
        check("neutral.ply exists", False, str(C.NEUTRAL_PLY))
        n_v = n_f = 0

    # ---- faces.npy: THE topology contract
    if C.FACES_NPY.is_file():
        faces = np.load(C.FACES_NPY)
        ok_shape = faces.ndim == 2 and faces.shape[1] == 3 and faces.shape[0] > 0
        check("faces.npy shape", ok_shape, f"{faces.shape} dtype={faces.dtype}")
        check("faces.npy indices in range", ok_shape and n_v > 0 and int(faces.max()) < n_v,
              f"max index {int(faces.max()) if ok_shape else '?'} < V={n_v}")
        check("faces.npy count == PLY faces", faces.shape[0] == n_f,
              f"{faces.shape[0]} vs {n_f}")
        exp = (C.EXPECTED_N_VERTS, C.EXPECTED_N_FACES)
        meas = (n_v, faces.shape[0])
        note = ("matches documented FLAME expectation" if meas == exp else
                f"DIFFERS from documented expectation {exp} -- reconcile recon_report.md")
        print(f"[verify] INFO topology measured (V,F)={meas}; {note}")
        manifest["topology_contract"] = {
            "n_vertices_measured": n_v, "n_faces_measured": int(faces.shape[0]),
            "expected_doc": {"n_vertices": exp[0], "n_faces": exp[1]},
            "matches_expected_doc": meas == exp,
            "faces_npy_sha256": sha256_of(C.FACES_NPY), "dtype": str(faces.dtype),
        }
    else:
        check("faces.npy exists", False, str(C.FACES_NPY))
        faces = None

    # ---- id_params.npz
    if C.ID_PARAMS_NPZ.is_file():
        idp = np.load(C.ID_PARAMS_NPZ, allow_pickle=True)
        need = ["betas", "photo_expression", "photo_jaw_pose", "photo_global_orient",
                "photo_transl", "camera_fx_fy_cx_cy", "image_hw", "landmark_rmse_px"]
        missing = [k for k in need if k not in idp]
        check("id_params.npz keys", not missing, f"missing={missing or 'none'}")
        if not missing:
            rmse = float(idp["landmark_rmse_px"])
            check("fit landmark RMSE sane", 0.0 < rmse < 0.05 * float(np.hypot(*idp["image_hw"])),
                  f"{rmse:.2f}px (soft gate: <5% of image diagonal; qa-verifier owns accept)")
            manifest["id_params"] = {"n_betas": int(idp["betas"].shape[0]),
                                     "landmark_rmse_px": rmse}
    else:
        check("id_params.npz exists", False, str(C.ID_PARAMS_NPZ))

    # ---- expression basis handle
    if C.EXPR_BASIS_NPZ.is_file():
        eb = np.load(C.EXPR_BASIS_NPZ)
        need = ["expr_dirs", "posedirs", "j_regressor", "lbs_weights", "parents",
                "faces", "v_neutral", "joints_neutral", "betas"]
        missing = [k for k in need if k not in eb]
        check("expression_basis.npz keys", not missing, f"missing={missing or 'none'}")
        if not missing and faces is not None:
            check("expression_basis faces == faces.npy",
                  np.array_equal(eb["faces"], faces),
                  "identical" if np.array_equal(eb["faces"], faces) else "MISMATCH -- STOP")
            ed = eb["expr_dirs"]
            check("expr_dirs shape", ed.ndim == 3 and ed.shape[0] == n_v and ed.shape[1] == 3,
                  f"{ed.shape} (V,3,n_expr)")
            manifest["expression_basis"] = {"n_expr": int(ed.shape[2]),
                                            "n_pose_correctives": int(eb["posedirs"].shape[2])}
        check("expression_basis_notes.json exists", C.EXPR_BASIS_NOTES.is_file(),
              str(C.EXPR_BASIS_NOTES))
    else:
        check("expression_basis.npz exists", False, str(C.EXPR_BASIS_NPZ))

    # ---- landmark anchors the fit actually used (release file OR self-authored)
    if C.LMK_EMBEDDING_NPZ.is_file():
        z = np.load(C.LMK_EMBEDDING_NPZ, allow_pickle=True)
        sf = np.asarray(z["static_faces"]) if "static_faces" in z else None
        sb = np.asarray(z["static_bary"]) if "static_bary" in z else None
        ok_shape = sf is not None and sb is not None and sf.shape == (51,) and sb.shape == (51, 3)
        check("lmk_embedding_static51.npz shapes", ok_shape,
              f"static_faces={None if sf is None else sf.shape} "
              f"static_bary={None if sb is None else sb.shape}")
        if ok_shape and faces is not None:
            check("lmk embedding face indices in range",
                  int(sf.min()) >= 0 and int(sf.max()) < faces.shape[0],
                  f"range [{int(sf.min())},{int(sf.max())}] < F={faces.shape[0]}")
            bary_ok = bool(np.allclose(sb.sum(axis=1), 1.0, atol=1e-6)
                           and (sb >= -1e-6).all())
            check("lmk embedding barycentrics valid", bary_ok,
                  "rows sum to 1, non-negative" if bary_ok else "INVALID barycentrics")
        src = str(z["source"]) if "source" in z else "UNRECORDED"
        print(f"[verify] INFO landmark-anchor source: {src}")
        manifest["landmark_embedding"] = {
            "source": src,
            "self_authored": src.startswith("self-authored"),
            "sha256": sha256_of(C.LMK_EMBEDDING_NPZ),
        }
        if src.startswith("self-authored"):
            check("self-authored landmark debug json exists (human-check material)",
                  C.FLAME_LMK_SELF_JSON.is_file(), str(C.FLAME_LMK_SELF_JSON))
    else:
        check("lmk_embedding_static51.npz exists", False, str(C.LMK_EMBEDDING_NPZ))

    # ---- albedo
    if C.ALBEDO_PNG.is_file():
        import cv2
        alb = cv2.imread(str(C.ALBEDO_PNG), cv2.IMREAD_COLOR)
        ok = alb is not None and alb.shape[0] == alb.shape[1] and float(alb.std()) > 1.0
        check("albedo.png readable + non-constant", ok,
              f"shape={None if alb is None else alb.shape} std={None if alb is None else round(float(alb.std()), 2)}")
        if C.BAKE_SUMMARY_JSON.is_file():
            with open(C.BAKE_SUMMARY_JSON, encoding="utf-8") as f:
                bake = json.load(f)
            direct = bake.get("texels_direct", 0)
            covered = max(bake.get("texels_covered", 1), 1)
            check("albedo direct-sample fraction", direct / covered > 0.10,
                  f"{direct}/{covered} = {direct / covered:.2%} sampled straight from the photo")
            manifest["albedo"] = bake
        check("albedo_mask.png exists", C.ALBEDO_MASK_PNG.is_file(), str(C.ALBEDO_MASK_PNG))
        check("uv_coords.npz exists", C.UV_COORDS_NPZ.is_file(), str(C.UV_COORDS_NPZ))
        if C.UV_COORDS_NPZ.is_file():
            uvz = np.load(C.UV_COORDS_NPZ, allow_pickle=True)
            if faces is not None and "faces_verts_idx" in uvz:
                fv = np.asarray(uvz["faces_verts_idx"], dtype=np.int64)
                same = np.array_equal(fv, faces.astype(np.int64))
                check("uv_coords faces_verts_idx == faces.npy", same,
                      "identical" if same else "MISMATCH -- UV layout indexes a "
                      "different topology. STOP.")
            uv_src = str(uvz["uv_source"]) if "uv_source" in uvz else "UNRECORDED"
            print(f"[verify] INFO UV source: {uv_src}")
            manifest["uv"] = {
                "source": uv_src,
                "generated_clean_room": uv_src.startswith("generated-clean-room"),
                "n_uv_verts": int(np.asarray(uvz["verts_uvs"]).shape[0]),
            }
    else:
        check("albedo.png exists", False, str(C.ALBEDO_PNG))

    # ---- landmark-stage artifacts (correspondence verification material)
    check("landmarks.npz exists", C.LANDMARKS_NPZ.is_file(), str(C.LANDMARKS_NPZ))
    check("landmarks_debug.png exists (correspondence overlay for human check)",
          C.LANDMARKS_DEBUG_PNG.is_file(), str(C.LANDMARKS_DEBUG_PNG))

    manifest["failures"] = FAILURES
    manifest["verdict"] = "PASS" if not FAILURES else "FAIL"
    with open(C.RUN_MANIFEST_JSON, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[verify] manifest -> {C.RUN_MANIFEST_JSON}")

    if FAILURES:
        sys.exit("[verify FATAL] " + f"{len(FAILURES)} check(s) FAILED:\n  - "
                 + "\n  - ".join(FAILURES))
    print("[verify] ALL CHECKS PASSED (measured, not claimed).")


if __name__ == "__main__":
    main()
