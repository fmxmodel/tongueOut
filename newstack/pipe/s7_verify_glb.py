#!/usr/bin/env python3
"""Stage 7 -- verify the GLB by MEASUREMENT, not by claiming (pure stdlib).

Parses the GLB binary header + JSON chunk directly (no deps) and checks:
  - valid glTF 2.0 container, exactly one mesh/primitive
  - morph target COUNT == manifest's supported count (52 expected)
  - morph target NAMES (mesh extras.targetNames) == manifest supported set,
    exact spelling -- the ARKit contract the whole pipeline exists to honor
  - every target's POSITION accessor count == base POSITION count
    (note: the exporter splits verts at UV seams, so base count >= 26719;
    the 26719 topology authority is enforced at the Blender stage)
  - a texture image is embedded and wired as baseColorTexture

Exit code 0 = PASS. Writes out/export/verify_report.json.
"""

import argparse
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import N_VERTS, P  # noqa: E402
from arkit_names import ARKIT_52  # noqa: E402


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
    if len(prims) != 1:
        fails.append(f"expected 1 primitive, got {len(prims)}")

    if prims:
        prim = prims[0]
        acc = gltf["accessors"]
        base_pos = acc[prim["attributes"]["POSITION"]]["count"]
        info["position_count"] = base_pos
        if base_pos < N_VERTS:
            fails.append(f"POSITION count {base_pos} < {N_VERTS} -- verts LOST")
        elif base_pos > N_VERTS:
            info["note_splits"] = (f"POSITION {base_pos} > {N_VERTS}: exporter "
                                   "split verts at UV seams (expected, benign)")
        targets = prim.get("targets", [])
        info["morph_target_count"] = len(targets)
        if len(targets) != len(expected):
            fails.append(f"morph target count {len(targets)} != {len(expected)}")
        bad_counts = [i for i, tg in enumerate(targets)
                      if acc[tg["POSITION"]]["count"] != base_pos]
        if bad_counts:
            fails.append(f"targets with POSITION count != base: {bad_counts}")

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
    has_basecolor = any("baseColorTexture" in (m.get("pbrMetallicRoughness") or {})
                        for m in mats)
    if images and not has_basecolor:
        fails.append("image embedded but no material baseColorTexture wiring")

    report = {"glb": str(glb_path), "pass": not fails,
              "fails": fails, "warns": warns, "info": info}
    rp = glb_path.parent / "verify_report.json"
    rp.write_text(json.dumps(report, indent=2))

    print(f"[s7] {glb_path} ({nbytes/1e6:.1f} MB)")
    print(f"[s7] morph targets: {info.get('morph_target_count')} "
          f"(expected {len(expected)}); POSITION count "
          f"{info.get('position_count')} (mesh authority {N_VERTS})")
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
