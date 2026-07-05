"""Measured verification of every out/shapes/ artifact (arkit-rigger stage 2).

POD-ONLY (recon.pod_guard). Proves success by MEASUREMENT, not by claim
(CLAUDE.md invariant 4). Independent of build_arkit_shapes.py: it re-reads
every file from disk and re-derives every verdict. Checks:

  1. arkit_manifest.json exists, run_state == "measured-on-pod", and its
     shapes account for ALL 52 canonical names -- exact, case-sensitive,
     compared against out/arkit_51_52_map.json (the authority).
  2. out/shapes/neutral.ply is byte-identical (sha256) to out/recon/neutral.ply.
  3. Every SUPPORTED shape has expr_<name>.ply whose vertex count matches
     neutral and whose faces are BYTE-IDENTICAL to out/recon/faces.npy, whose
     delta vs neutral is finite + non-trivial (an all-zero delta means the
     solve was wrong) and matches the manifest's recorded max_delta_m.
  4. Every UNSUPPORTED shape has NO expr_<name>.ply (a stale/fabricated mesh
     is as much a failure as a missing one) and carries a reason.
  5. tongueOut is unsupported (FLAME has no tongue; fabrication tripwire).

Writes out/shapes/shapes_run_manifest.json; exit != 0 on ANY failed check.
Run:  python -m rig.verify_shapes
"""

import hashlib
import json
import sys
import time

import numpy as np

from recon.pod_guard import require_pod
from . import config as C

FAILURES = []


def check(name: str, ok: bool, detail: str) -> None:
    tag = "PASS" if ok else "FAIL"
    print(f"[verify-shapes] {tag}  {name}: {detail}")
    if not ok:
        FAILURES.append(f"{name}: {detail}")


def sha256_of(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    require_pod()
    import trimesh

    t0 = time.time()
    out = {"schema": "b1-shapes-run-manifest/1.0",
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    # ---- name authority -------------------------------------------------------
    with open(C.NAME_CONTRACT_JSON, encoding="utf-8") as f:
        arkit52 = json.load(f)["apple_canonical_52"]
    check("name contract", len(arkit52) == 52 and len(set(arkit52)) == 52,
          f"{len(arkit52)} canonical names")

    # ---- manifest --------------------------------------------------------------
    if not C.ARKIT_MANIFEST_JSON.is_file():
        check("arkit_manifest.json exists", False, str(C.ARKIT_MANIFEST_JSON))
        _finish(out, t0)
        return
    with open(C.ARKIT_MANIFEST_JSON, encoding="utf-8") as f:
        man = json.load(f)
    check("manifest run_state measured", man.get("run_state") == "measured-on-pod",
          f"run_state={man.get('run_state')!r} (DEFERRED placeholder must be "
          "overwritten by the pod build)")
    shapes = man.get("shapes", {})
    missing = [n for n in arkit52 if n not in shapes]
    extra = [n for n in shapes if n not in arkit52]
    check("all 52 names, exact case", not missing and not extra,
          f"missing={missing or 'none'} extra={extra or 'none'}")

    # ---- topology contract -------------------------------------------------------
    faces = np.load(C.RECON_FACES_NPY)
    faces_bytes = faces.astype(np.int32).tobytes()
    check("faces.npy loaded", faces.ndim == 2 and faces.shape[1] == 3,
          f"{faces.shape} {faces.dtype}")

    # ---- neutral pass-through ------------------------------------------------------
    if C.NEUTRAL_OUT_PLY.is_file() and C.RECON_NEUTRAL_PLY.is_file():
        same = sha256_of(C.NEUTRAL_OUT_PLY) == sha256_of(C.RECON_NEUTRAL_PLY)
        check("neutral.ply byte-identical to recon", same, "sha256 equal" if same
              else "DIFFERS -- the base mesh was modified, STOP")
        neutral = trimesh.load(C.NEUTRAL_OUT_PLY, process=False)
        nv = np.asarray(neutral.vertices, dtype=np.float64)
    else:
        check("neutral.ply present (both)", False,
              f"{C.NEUTRAL_OUT_PLY} / {C.RECON_NEUTRAL_PLY}")
        nv = None

    # ---- per-shape ---------------------------------------------------------------
    n_sup = n_unsup = 0
    per_shape = {}
    for name in arkit52:
        entry = shapes.get(name)
        if entry is None:
            continue
        ply = C.SHAPES_DIR / f"expr_{name}.ply"
        if entry.get("supported"):
            n_sup += 1
            if not ply.is_file():
                check(f"{name} ply exists", False, str(ply))
                continue
            mesh = trimesh.load(ply, process=False)
            mf = np.asarray(mesh.faces, dtype=np.int32)
            ok_topo = mf.tobytes() == faces_bytes
            check(f"{name} topology", ok_topo,
                  "faces byte-identical to faces.npy" if ok_topo
                  else f"TOPOLOGY DRIFT {mf.shape} -- STOP")
            if nv is not None:
                mv = np.asarray(mesh.vertices, dtype=np.float64)
                ok_v = mv.shape == nv.shape
                check(f"{name} vertex count", ok_v, f"{mv.shape} vs {nv.shape}")
                if ok_v:
                    delta = np.linalg.norm(mv - nv, axis=1)
                    max_d = float(delta.max())
                    finite = bool(np.isfinite(mv).all())
                    check(f"{name} delta non-trivial+finite",
                          finite and max_d >= C.GATE_MIN_MAX_DELTA_M,
                          f"max|delta|={max_d * 1e3:.3f}mm finite={finite}")
                    rec = entry.get("max_delta_m")
                    if rec is not None:
                        agree = abs(max_d - rec) <= max(1e-6, 0.05 * rec)
                        check(f"{name} matches manifest", agree,
                              f"measured {max_d:.6f} vs recorded {rec:.6f} m")
                    per_shape[name] = {"max_delta_m": max_d,
                                       "sha256": sha256_of(ply)}
        else:
            n_unsup += 1
            check(f"{name} honestly absent", not ply.is_file(),
                  "no mesh (declared unsupported)" if not ply.is_file()
                  else f"{ply} EXISTS for an unsupported shape -- fabricated/stale")
            check(f"{name} has reason", bool(entry.get("reason")),
                  str(entry.get("reason"))[:100])

    check("counts consistent",
          man.get("counts", {}).get("supported") == n_sup
          and man.get("counts", {}).get("unsupported") == n_unsup,
          f"manifest says {man.get('counts')}, measured sup={n_sup} unsup={n_unsup}")
    tongue = shapes.get("tongueOut", {})
    check("tongueOut unsupported (fabrication tripwire)",
          tongue.get("supported") is False, str(tongue.get("supported")))

    out["counts"] = {"supported": n_sup, "unsupported": n_unsup}
    out["per_shape_measured"] = per_shape
    _finish(out, t0)


def _finish(out, t0) -> None:
    out["failures"] = FAILURES
    out["verdict"] = "PASS" if not FAILURES else "FAIL"
    out["runtime_s"] = time.time() - t0
    C.ensure_out_dirs()
    with open(C.SHAPES_RUN_MANIFEST_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[verify-shapes] manifest -> {C.SHAPES_RUN_MANIFEST_JSON}")
    if FAILURES:
        sys.exit(f"[verify-shapes FATAL] {len(FAILURES)} check(s) FAILED:\n  - "
                 + "\n  - ".join(FAILURES))
    print("[verify-shapes] ALL CHECKS PASSED (measured, not claimed).")


if __name__ == "__main__":
    main()
