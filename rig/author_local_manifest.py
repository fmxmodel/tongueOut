"""Author the DEFERRED arkit_manifest.json on the authoring box (stdlib ONLY).

This script performs NO numeric/model compute -- it only cross-checks the
self-authored spec (rig/arkit_spec.py) against the name contract
(out/arkit_51_52_map.json) and writes a placeholder manifest in the exact
schema the pod build will overwrite. Every measured field is null and every
shape is status "DEFERRED-pod-run", so no downstream stage can mistake the
placeholder for a result (they must gate on run_state == "measured-on-pod").

Run locally:  python3 -m rig.author_local_manifest
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import arkit_spec as S

_REPO_ROOT = Path(__file__).resolve().parent.parent
NAME_CONTRACT = _REPO_ROOT / "out" / "arkit_51_52_map.json"
SHAPES_DIR = _REPO_ROOT / "out" / "shapes"
MANIFEST = SHAPES_DIR / "arkit_manifest.json"


def main() -> None:
    with open(NAME_CONTRACT, encoding="utf-8") as f:
        contract = json.load(f)
    arkit52 = list(contract["apple_canonical_52"])
    if len(arkit52) != 52 or len(set(arkit52)) != 52:
        sys.exit(f"[author FATAL] contract has {len(arkit52)} names, expected 52 unique.")
    if sorted(S.SPEC.keys()) != sorted(arkit52):
        sys.exit("[author FATAL] spec/contract name mismatch: "
                 f"spec-only={sorted(set(S.SPEC) - set(arkit52))} "
                 f"contract-only={sorted(set(arkit52) - set(S.SPEC))}")

    shapes = {}
    for name in arkit52:
        spec = S.SPEC[name]
        unsupported_a_priori = spec["method"] == "none"
        shapes[name] = {
            "supported": False if unsupported_a_priori else None,
            "status": "DEFERRED-pod-run",
            "method": spec["method"],
            "intended": spec["intended"],
            "max_delta_m": None,
            "mean_delta_m": None,
            "gates": {},
            "reason": spec.get("reason") if unsupported_a_priori else None,
            "notes": spec.get("notes"),
            "ply": None,
        }

    manifest = {
        "schema": "b1-arkit-shape-manifest/1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generated_by": ("arkit-rigger (Track B1) -- rig/author_local_manifest.py, "
                         "AUTHORED LOCALLY (stdlib only, no compute, no meshes)"),
        "run_state": "DEFERRED-pod-run",
        "run_state_note": ("NOTHING here is measured. The 52 delta meshes do not "
                           "exist yet; they are produced ON THE GPU POD by "
                           "scripts/run_rig_b1.sh, which overwrites this file with "
                           "measured values. Downstream stages MUST gate on "
                           "run_state == 'measured-on-pod'."),
        "name_contract": "out/arkit_51_52_map.json",
        "topology": {"n_vertices": None, "n_faces": None, "faces_npy_sha256": None,
                     "contract": "out/recon/faces.npy (byte-locked at pod runtime)"},
        "laterality": {"status": "DEFERRED -- measured on pod (mesh x-signs + "
                                 "photo projection cross-check + eye-joint x-signs)"},
        "units_m": None,
        "jaw_calibration": None,
        "counts": {"total": 52,
                   "supported": None,
                   "unsupported": None,
                   "intended_strong": S.INTENDED_COUNTS["strong"],
                   "intended_weak_attempt": S.INTENDED_COUNTS["weak"],
                   "unsupported_a_priori": S.INTENDED_COUNTS["unsupported"]},
        "demoted_by_gates": None,
        "shapes": shapes,
    }
    SHAPES_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[author] DEFERRED manifest -> {MANIFEST} "
          f"(52 names; intended {S.INTENDED_COUNTS})")


if __name__ == "__main__":
    main()
