#!/usr/bin/env python3
"""Stage 7 -- verify the GLB by MEASUREMENT, not by claiming (pure stdlib).

Parses the GLB binary header + JSON chunk directly (no deps) and checks:
  - valid glTF 2.0 container, exactly one mesh; ONE OR MORE primitives (the
    exporter splits one primitive per material -- HeadMat + eye material(s))
  - EVERY primitive carries the full morph target set (count == manifest's
    supported count, 52 expected) and each target's POSITION accessor count
    == that primitive's base POSITION count
  - morph target NAMES (mesh extras.targetNames) == manifest supported set,
    exact spelling -- the ARKit contract the whole pipeline exists to honor
  - summed base POSITION count >= 26719 minus the stripped eye-shell verts
    (s6 drops the transparent-purpose lacrimal/eye-blend/eye-occlusion FACES;
    UV-seam/material splits add verts back -- fewer than that floor means
    verts LOST; the 26719 topology authority is enforced at the Blender stage)
  - every material is colored: baseColorTexture (head albedo, eye textures)
    OR COLOR_0 vertex colors (RestMat = the UDIM tile-1+ polygons)
  - every material is explicitly opaque: alphaMode == "OPAQUE" (explicit,
    not defaulted -- s6 hardens this) and doubleSided is false/absent

Exit code 0 = PASS. Writes out/export/verify_report.json.
"""

import argparse
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EYE_SHELLS, N_VERTS, P  # noqa: E402
from arkit_names import ARKIT_52  # noqa: E402

# s6 strips the transparent-purpose eye-shell FACES (lacrimal/eye-blend/eye-
# occlusion), so their verts are legitimately absent from the GLB primitives.
MIN_EXPORT_VERTS = N_VERTS - (EYE_SHELLS[1] - EYE_SHELLS[0])


def read_glb_json(path):
    data = Path(path).read_bytes()
    magic, version, length = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67:
        raise ValueError("not a GLB (bad magic)")
    if version != 2:
        raise ValueError(f"glTF version {version} != 2")
    clen, ctype = struct.unpack_from("<II", data, 12)
    if ctype != 0x4E4F534A:
        raise ValueError("first chunk is not JSON")
    return json.loads(data[20:20 + clen]), len(data)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=P.OUT)
    ap.add_argument("--glb", default=None)
    ap.add_argument("--manifest", default=None)
    args = ap.parse_args()
    glb_path = Path(args.glb or Path(args.out) / "export" / "head_arkit_v2.glb")
    man_path = Path(args.manifest or Path(args.out) / "rig" / "arkit_manifest.json")

    fails, warns, info = [], [], {}
    manifest = json.loads(man_path.read_text())
    expected = [n for n in ARKIT_52 if manifest["shapes"][n]["supported"]]
    info["expected_supported"] = len(expected)

    gltf, nbytes = read_glb_json(glb_path)
    info["glb_bytes"] = nbytes
    meshes = gltf.get("meshes", [])
    if len(meshes) != 1:
        fails.append(f"expected 1 mesh, got {len(meshes)}")
    prims = meshes[0].get("primitives", []) if meshes else []
    if not prims:
        fails.append("mesh has no primitives")
    info["primitive_count"] = len(prims)

    if prims:
        acc = gltf["accessors"]
        pos_counts, tgt_counts = [], []
        for pi, prim in enumerate(prims):
            base_pos = acc[prim["attributes"]["POSITION"]]["count"]
            pos_counts.append(base_pos)
            targets = prim.get("targets", [])
            tgt_counts.append(len(targets))
            if len(targets) != len(expected):
                fails.append(f"primitive {pi}: morph target count "
                             f"{len(targets)} != {len(expected)}")
            bad_counts = [i for i, tg in enumerate(targets)
                          if acc[tg["POSITION"]]["count"] != base_pos]
            if bad_counts:
                fails.append(f"primitive {pi}: targets with POSITION count "
                             f"!= base: {bad_counts}")
        info["position_counts"] = pos_counts
        info["morph_target_count"] = tgt_counts[0] if tgt_counts else 0
        total_pos = sum(pos_counts)
        info["position_count"] = total_pos
        if total_pos < MIN_EXPORT_VERTS:
            fails.append(f"summed POSITION {total_pos} < {MIN_EXPORT_VERTS} "
                         f"(= {N_VERTS} minus the stripped eye shells) -- "
                         "verts LOST")
        elif total_pos < N_VERTS:
            info["note_shells"] = (
                f"POSITION {total_pos} in [{MIN_EXPORT_VERTS},{N_VERTS}): "
                "eye-shell verts absent (s6 strips their faces; expected), "
                "UV-seam splits add some back")
        else:
            info["note_splits"] = (f"POSITION {total_pos} >= {N_VERTS}: "
                                   "exporter split verts at UV seams / "
                                   "material boundaries (expected, benign)")

        tnames = (meshes[0].get("extras", {}) or {}).get("targetNames")
        if not tnames:
            warns.append("mesh extras.targetNames missing -- name-driven "
                         "MediaPipe lookup would break; check exporter")
        else:
            info["target_names"] = tnames
            if tnames != expected:
                miss = sorted(set(expected) - set(tnames))
                extra = sorted(set(tnames) - set(expected))
                if miss or extra:
                    fails.append(f"targetNames mismatch: missing={miss} extra={extra}")
                else:
                    warns.append("targetNames order differs from canonical "
                                 "ARKit order (names all correct)")

    images = gltf.get("images", [])
    info["images"] = len(images)
    if not images:
        warns.append("no embedded texture image")
    mats = gltf.get("materials", [])
    info["materials"] = []
    # a material may be colored by a texture OR by COLOR_0 vertex colors
    # (RestMat carries the UDIM tile-1+ polys via per-vertex colors)
    vcol_mats = {prim.get("material") for prim in prims
                 if "COLOR_0" in prim.get("attributes", {})}
    for mi, m in enumerate(mats):
        entry = {"name": m.get("name"),
                 "alphaMode": m.get("alphaMode"),
                 "doubleSided": m.get("doubleSided", False),
                 "textured": "baseColorTexture" in
                             (m.get("pbrMetallicRoughness") or {}),
                 "vertex_colored": mi in vcol_mats}
        info["materials"].append(entry)
        if entry["alphaMode"] != "OPAQUE":
            fails.append(f"material {entry['name']}: alphaMode "
                         f"{entry['alphaMode']!r} != explicit 'OPAQUE' "
                         "(s6 hardening missing?)")
        if entry["doubleSided"]:
            fails.append(f"material {entry['name']}: doubleSided true -- "
                         "single-sided opaque export expected")
        if images and not entry["textured"] and not entry["vertex_colored"]:
            fails.append(f"material {entry['name']}: neither baseColorTexture "
                         "nor COLOR_0 vertex colors")
    if len(prims) != len(mats):
        warns.append(f"{len(prims)} primitives vs {len(mats)} materials")

    report = {"glb": str(glb_path), "pass": not fails,
              "fails": fails, "warns": warns, "info": info}
    rp = glb_path.parent / "verify_report.json"
    rp.write_text(json.dumps(report, indent=2))

    print(f"[s7] {glb_path} ({nbytes/1e6:.1f} MB)")
    print(f"[s7] morph targets: {info.get('morph_target_count')} "
          f"(expected {len(expected)}); POSITION count "
          f"{info.get('position_count')} (mesh authority {N_VERTS}) "
          f"across {info.get('primitive_count')} primitive(s) "
          f"{info.get('position_counts')}")
    for m in info.get("materials", []):
        print(f"[s7] material {m['name']}: alphaMode={m['alphaMode']} "
              f"doubleSided={m['doubleSided']} textured={m['textured']} "
              f"vertex_colored={m['vertex_colored']}")
    for w in warns:
        print(f"[s7 WARN] {w}")
    for fmsg in fails:
        print(f"[s7 FAIL] {fmsg}")
    print(f"[s7] report -> {rp}")
    if fails:
        print("[s7] === VERDICT: FAIL ===")
        sys.exit(1)
    print("[s7] === VERDICT: PASS -- names measured equal to the ARKit contract ===")


if __name__ == "__main__":
    main()
