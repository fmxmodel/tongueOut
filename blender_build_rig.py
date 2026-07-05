"""Track B1 — Blender assembly + GLB export (headless).

Transcribes the arkit-rigger's topology-locked delta meshes into a single GLB
whose morph targets are the exact, case-sensitive ARKit strings the MediaPipe
driver looks up. Blender's shape-key system IS the glTF morph-target system, so
this stage is a mechanical transcription — *valid only because* every mesh
shares one topology (out/recon/faces.npy). If that guarantee is broken this
script STOPS rather than forcing it (topology is arkit-rigger's contract).

    RUN (POD ONLY, after recon + rig produced the real PLYs):
        blender --background --python blender_build_rig.py
    or via the turnkey runner:
        bash scripts/run_glb_b1.sh

WHY THIS IS DEFERRED TO THE POD
  The real out/shapes/neutral.ply + out/shapes/expr_*.ply + out/recon/albedo.png
  do not exist until recon (run_recon_b1.sh) and rig (run_rig_b1.sh) have run on
  the GPU pod. This script REFUSES to run without those real inputs and without a
  MEASURED manifest (run_state == "measured-on-pod") so it can never emit an
  empty / fabricated GLB. Nothing was ever run on the authoring box.

INPUTS (paths mirror recon.config / rig.config; env-overridable via B1_GLB_*):
  out/shapes/arkit_manifest.json   which of the 52 ARKit names are supported
                                   (gate: run_state == "measured-on-pod")
  out/shapes/neutral.ply           base mesh (fitted identity, faces.npy order)
  out/shapes/expr_<arkitName>.ply  one per SUPPORTED shape, identical topology
  out/recon/faces.npy              THE topology contract, int32 (F,3)
  out/recon/albedo.png             baked per-subject albedo (sRGB lit appearance)
  out/recon/uv_coords.npz          FLAME UV layout (verts_uvs + per-corner idx)

OUTPUTS (out/):
  head_arkit.glb                   GLB: mesh + morph targets + baseColor texture
  head_rigged.blend                inspectable Blender intermediate
  glb_report.md                    morph-target names, texture survival, sizes

DISCIPLINE (invariant #2/#3 of CLAUDE.md):
  * Shape-key names are the EXACT ARKit strings — never renamed to fake coverage
    or to paper over a driver mismatch. A silent rename is a contract break.
  * Unsupported shapes (tongueOut, cheekPuff, cheekSquintL/R, + any pod-demoted)
    get NO morph target. The viewer no-ops them by name. We never fabricate one.
  * Every vertex copy is index-aligned, which is ONLY valid on identical
    topology; vertex counts are asserted equal and faces byte-compared to
    faces.npy (and its sha256 to the manifest) before anything is built.
"""

import hashlib
import json
import os
import struct
import sys
from pathlib import Path

# NB: numpy and bpy are imported lazily inside main() so that this module stays
# importable / `python -m py_compile`-able on the CPU authoring box (where bpy
# cannot and must not be installed). py_compile never executes these imports.


# --------------------------------------------------------------------------
# Paths (repo layout; env-overridable). REPO_ROOT is the dir holding this file.
# --------------------------------------------------------------------------
REPO_ROOT = Path(os.environ.get("B1_GLB_REPO_ROOT", str(Path(__file__).resolve().parent)))
OUT_DIR = Path(os.environ.get("B1_GLB_OUT_DIR", str(REPO_ROOT / "out")))
SHAPES_DIR = Path(os.environ.get("B1_GLB_SHAPES_DIR", str(OUT_DIR / "shapes")))
RECON_DIR = Path(os.environ.get("B1_GLB_RECON_DIR", str(OUT_DIR / "recon")))

MANIFEST_JSON = SHAPES_DIR / "arkit_manifest.json"
NEUTRAL_PLY = SHAPES_DIR / "neutral.ply"
FACES_NPY = RECON_DIR / "faces.npy"
ALBEDO_PNG = RECON_DIR / "albedo.png"
UV_COORDS_NPZ = RECON_DIR / "uv_coords.npz"
NAME_CONTRACT_JSON = OUT_DIR / "arkit_51_52_map.json"          # optional cross-check

GLB_OUT = Path(os.environ.get("B1_GLB_OUT", str(OUT_DIR / "head_arkit.glb")))
BLEND_OUT = Path(os.environ.get("B1_GLB_BLEND", str(OUT_DIR / "head_rigged.blend")))
REPORT_MD = OUT_DIR / "glb_report.md"

# Draco is OFF by default: the three.js viewer (out/viewer/) consumes the GLB
# without a DRACO decoder configured, so an uncompressed GLB is the compatible
# default. Enable smaller files with B1_GLB_DRACO=1 (documented in glb_report).
DRACO = os.environ.get("B1_GLB_DRACO", "0") == "1"

POD_CMD = "blender --background --python blender_build_rig.py"

# The 4 a-priori unsupported shapes (arkit-rigger's declaration). Recorded so the
# report can distinguish a-priori-unsupported from pod-gate-demoted shapes.
APRIORI_UNSUPPORTED = ("tongueOut", "cheekPuff", "cheekSquintLeft", "cheekSquintRight")


def log(msg):
    print(f"[glb] {msg}", flush=True)


def die(msg):
    sys.exit(f"[glb FATAL] {msg}")


# --------------------------------------------------------------------------
# Self-contained PLY vertex reader (ascii + binary_little_endian), numpy-based.
# Deliberately independent of Blender's PLY importer: reading the file directly
# guarantees vertex order == file order == the faces.npy-aligned order the rig
# wrote (trimesh export, process=False), which is what makes the index-aligned
# shape-key copy valid. The vertex element is always the first PLY element, so
# we never need to parse the (list-typed) face block.
# --------------------------------------------------------------------------
_PLY_NP = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def read_ply_vertices(path):
    """Return the (N,3) float64 vertex XYZ of a PLY, in file order."""
    import numpy as np

    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            die(f"{path}: not a PLY file (missing 'ply' magic).")
        fmt = None
        elements = []                       # [name, count, [(pname, ptype), ...]]
        cur = None
        while True:
            raw = f.readline()
            if not raw:
                die(f"{path}: unexpected EOF in PLY header.")
            s = raw.strip()
            if s == b"end_header":
                break
            parts = s.split()
            if not parts:
                continue
            key = parts[0]
            if key == b"format":
                fmt = parts[1].decode("ascii")
            elif key == b"element":
                cur = [parts[1].decode("ascii"), int(parts[2]), []]
                elements.append(cur)
            elif key == b"property" and cur is not None:
                if parts[1] == b"list":
                    cur[2].append(("__list__", None))       # e.g. face vertex_indices
                else:
                    cur[2].append((parts[2].decode("ascii"), parts[1].decode("ascii")))

        if not elements or elements[0][0] != "vertex":
            die(f"{path}: expected 'vertex' as the first PLY element, got "
                f"{elements[0][0] if elements else 'none'}.")
        vname, vcount, vprops = elements[0]
        if any(p == "__list__" for p, _ in vprops):
            die(f"{path}: vertex element has a list property (unexpected).")
        names = [p for p, _ in vprops]
        for axis in ("x", "y", "z"):
            if axis not in names:
                die(f"{path}: vertex element lacks '{axis}' property.")

        if fmt == "ascii":
            verts = np.empty((vcount, 3), dtype=np.float64)
            ix, iy, iz = names.index("x"), names.index("y"), names.index("z")
            for i in range(vcount):
                toks = f.readline().split()
                verts[i, 0] = float(toks[ix])
                verts[i, 1] = float(toks[iy])
                verts[i, 2] = float(toks[iz])
            return verts
        if fmt == "binary_little_endian":
            for _, ptype in vprops:
                if ptype not in _PLY_NP:
                    die(f"{path}: unsupported PLY scalar type '{ptype}'.")
            dt = np.dtype([(p, "<" + _PLY_NP[t]) for p, t in vprops])
            buf = f.read(vcount * dt.itemsize)
            if len(buf) < vcount * dt.itemsize:
                die(f"{path}: truncated binary vertex block.")
            rec = np.frombuffer(buf, dtype=dt, count=vcount)
            return np.stack([rec["x"], rec["y"], rec["z"]], axis=1).astype(np.float64)
        die(f"{path}: unsupported PLY format '{fmt}' (need ascii or "
            "binary_little_endian).")


# --------------------------------------------------------------------------
# GLB re-read verification (stdlib only): parse the JSON chunk and confirm the
# morph-target names three.js will expose via morphTargetDictionary equal our
# supported set, and that a base-color texture survived. Owned finally by
# qa-verifier; this is the build-stage self-check handed to it.
# --------------------------------------------------------------------------
def read_glb_json(path):
    with open(path, "rb") as f:
        head = f.read(12)
        if len(head) < 12:
            die(f"{path}: too small to be a GLB.")
        magic, version, _length = struct.unpack("<III", head)
        if magic != 0x46546C67:             # 'glTF'
            die(f"{path}: bad GLB magic (not a glTF binary).")
        clen, ctype = struct.unpack("<II", f.read(8))
        if ctype != 0x4E4F534A:             # 'JSON'
            die(f"{path}: first GLB chunk is not JSON.")
        return json.loads(f.read(clen).decode("utf-8"))


def verify_glb(path, expected_names):
    """Return a dict of measured facts; STOP on any contract violation."""
    doc = read_glb_json(path)
    meshes = doc.get("meshes", [])
    if not meshes:
        die(f"{path}: contains no meshes.")

    target_names, n_targets_geom = [], 0
    for m in meshes:
        tn = (m.get("extras") or {}).get("targetNames") or []
        target_names.extend(tn)
        for prim in m.get("primitives", []):
            n_targets_geom = max(n_targets_geom, len(prim.get("targets", [])))

    got = list(target_names)
    exp = list(expected_names)
    # Names must match the supported set EXACTLY (case-sensitive), no extras.
    if sorted(got) != sorted(exp):
        missing = sorted(set(exp) - set(got))
        extra = sorted(set(got) - set(exp))
        die(f"{path}: morph-target names != supported set. "
            f"missing={missing} extra={extra}. Names are the driver contract — STOP.")
    if n_targets_geom != len(exp):
        die(f"{path}: primitive has {n_targets_geom} morph targets but "
            f"{len(exp)} supported names — geometry/name desync. STOP.")

    images = doc.get("images", [])
    materials = doc.get("materials", [])
    has_basecolor_tex = any(
        (mat.get("pbrMetallicRoughness") or {}).get("baseColorTexture") is not None
        for mat in materials
    )
    return {
        "target_names": got,
        "n_targets": n_targets_geom,
        "n_images": len(images),
        "n_materials": len(materials),
        "has_basecolor_texture": has_basecolor_tex,
        "draco": "KHR_draco_mesh_compression" in doc.get("extensionsUsed", []),
    }


# --------------------------------------------------------------------------
# Report (emitted by the pod run with MEASURED facts)
# --------------------------------------------------------------------------
def write_report(supported, unsupported, demoted, topo, glb_facts, sizes):
    apriori = [n for n in unsupported if n in APRIORI_UNSUPPORTED]
    gate_demoted = [n for n in unsupported if n not in APRIORI_UNSUPPORTED]
    lines = []
    lines.append("# GLB Report — Track B1 (Blender assembly → head_arkit.glb)\n")
    lines.append("> Owner: `blender-glb-builder` (Opus 4.8) · **MEASURED on the GPU pod** "
                 "by `blender_build_rig.py`.\n")
    lines.append(f"> Pod-run command: `{POD_CMD}`\n")
    lines.append("")
    lines.append("## Topology (byte-locked to the contract)\n")
    lines.append(f"- vertices: **{topo['n_vertices']}**, faces: **{topo['n_faces']}**")
    lines.append(f"- faces.npy sha256: `{topo['faces_sha256']}`")
    lines.append(f"- manifest cross-check: {topo['manifest_check']}")
    lines.append("")
    lines.append("## Morph targets (== manifest supported set, exact ARKit spelling)\n")
    lines.append(f"- morph-target count in GLB: **{glb_facts['n_targets']}** "
                 f"(supported names: {len(supported)})")
    lines.append(f"- GLB `extras.targetNames` resolve for three.js "
                 f"`morphTargetDictionary`: **{'YES' if glb_facts['target_names'] else 'NO'}**")
    lines.append("")
    lines.append("| # | ARKit morph-target name |")
    lines.append("|---|---|")
    for i, n in enumerate(supported):
        lines.append(f"| {i} | `{n}` |")
    lines.append("")
    lines.append("## Unsupported — deliberately NO morph target (viewer no-ops by name)\n")
    lines.append(f"- a-priori unsupported ({len(apriori)}): "
                 + (", ".join(f"`{n}`" for n in apriori) or "none"))
    lines.append(f"- pod-gate-demoted ({len(gate_demoted)}): "
                 + (", ".join(f"`{n}`" for n in gate_demoted) or "none"))
    lines.append(f"- (manifest `demoted_by_gates`: {demoted or 'none'})")
    lines.append("")
    lines.append("## Texture survival\n")
    lines.append(f"- images in GLB: {glb_facts['n_images']}, materials: {glb_facts['n_materials']}")
    lines.append(f"- baseColor texture present: **{glb_facts['has_basecolor_texture']}** "
                 "(baked albedo via FLAME UV, sRGB)")
    lines.append("")
    lines.append("## File / export settings\n")
    lines.append(f"- `head_arkit.glb`: **{sizes['glb']:,} bytes**")
    lines.append(f"- `head_rigged.blend`: {sizes['blend']:,} bytes")
    lines.append(f"- Draco mesh compression: **{'ON' if DRACO else 'OFF'}** "
                 f"(GLB reports KHR_draco: {glb_facts['draco']}; toggle with B1_GLB_DRACO=1)")
    lines.append("- exporter flags: `export_morph=True`, `export_morph_normal=True`, "
                 "`export_format='GLB'`")
    lines.append("")
    lines.append("## Handoff\n")
    lines.append("- `viewer-driver`: load `out/head_arkit.glb`; drive `morphTargetInfluences` "
                 "by ARKit name — every supported name above resolves 1:1.")
    lines.append("- `qa-verifier`: reconcile these morph-target names against "
                 "`out/shapes/arkit_manifest.json` (supported set) and "
                 "`out/shapes/shapes_run_manifest.json` (measured verdict). qa-verifier owns "
                 "the final ACCEPT gate.")
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"report -> {REPORT_MD}")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    import numpy as np

    log(f"repo root: {REPO_ROOT}")

    # ---- 0. refuse without the real inputs (never emit a garbage GLB) --------
    required = {
        "manifest": MANIFEST_JSON, "neutral.ply": NEUTRAL_PLY, "faces.npy": FACES_NPY,
        "albedo.png": ALBEDO_PNG, "uv_coords.npz": UV_COORDS_NPZ,
    }
    missing = [f"{k} ({p})" for k, p in required.items() if not p.is_file() or p.stat().st_size == 0]
    if missing:
        die("required inputs missing/empty:\n  - " + "\n  - ".join(missing) +
            "\nThese are produced ON THE GPU POD by recon (scripts/run_recon_b1.sh) "
            "and rig (scripts/run_rig_b1.sh). This stage is DEFERRED until they exist "
            "— refusing to build an empty/garbage GLB.")

    # ---- 1. gate on a MEASURED manifest -------------------------------------
    with open(MANIFEST_JSON, encoding="utf-8") as f:
        manifest = json.load(f)
    run_state = manifest.get("run_state")
    if run_state != "measured-on-pod":
        die(f"arkit_manifest.json run_state == {run_state!r}, not 'measured-on-pod'. "
            "The supported set is decided by pod MEASUREMENT; refusing to build from a "
            "DEFERRED/placeholder manifest. Run scripts/run_rig_b1.sh on the pod first.")
    shapes = manifest.get("shapes", {})
    if len(shapes) != 52:
        die(f"manifest has {len(shapes)} shapes, expected the full 52 ARKit names.")

    supported = [n for n, v in shapes.items() if v.get("supported") is True]
    unsupported = [n for n, v in shapes.items() if v.get("supported") is not True]
    demoted = manifest.get("demoted_by_gates") or []
    if not supported:
        die("manifest reports ZERO supported shapes — nothing to rig. STOP.")
    log(f"manifest OK (measured-on-pod): {len(supported)} supported, "
        f"{len(unsupported)} unsupported (gate-demoted: {demoted or 'none'})")

    # optional: cross-check names against the 52-name contract, if present
    if NAME_CONTRACT_JSON.is_file():
        with open(NAME_CONTRACT_JSON, encoding="utf-8") as f:
            contract52 = list(json.load(f).get("apple_canonical_52", []))
        if contract52 and sorted(shapes.keys()) != sorted(contract52):
            die("manifest shape names != arkit_51_52_map.json apple_canonical_52. "
                "The name contract is load-bearing — STOP.")

    # ---- 2. topology contract (byte-lock faces; sha256 vs manifest) ---------
    faces = np.load(FACES_NPY)
    if faces.dtype != np.int32 or faces.ndim != 2 or faces.shape[1] != 3:
        die(f"faces.npy malformed: shape={faces.shape} dtype={faces.dtype}")
    faces_sha = hashlib.sha256(faces.tobytes()).hexdigest()
    man_sha = (manifest.get("topology") or {}).get("faces_npy_sha256")
    manifest_check = "n/a (manifest carried no sha256)"
    if man_sha:
        if man_sha != faces_sha:
            die(f"faces.npy sha256 {faces_sha} != manifest {man_sha}. Building on a "
                "DIFFERENT topology than the rig measured — STOP (invariant #3).")
        manifest_check = "faces.npy sha256 matches manifest"

    neutral_verts = read_ply_vertices(NEUTRAL_PLY)
    n_verts, n_faces = int(neutral_verts.shape[0]), int(faces.shape[0])
    man_nv = (manifest.get("topology") or {}).get("n_vertices")
    if man_nv is not None and int(man_nv) != n_verts:
        die(f"neutral.ply has {n_verts} vertices but manifest says {man_nv} — STOP.")
    if int(faces.max()) >= n_verts:
        die(f"faces.npy indexes vertex {int(faces.max())} but neutral has {n_verts}.")
    log(f"topology OK: V={n_verts} F={n_faces} ({manifest_check})")

    # ---- 3. FLAME UV layout (separate per-corner UV indexing) ---------------
    uv = np.load(UV_COORDS_NPZ)
    verts_uvs = np.asarray(uv["verts_uvs"], dtype=np.float64)          # (T,2)
    faces_uv = np.asarray(uv["faces_uv_idx"], dtype=np.int64)          # (F,3)
    faces_v = np.asarray(uv["faces_verts_idx"], dtype=np.int32)        # (F,3)
    if faces_v.tobytes() != faces.tobytes():
        die("uv_coords.npz faces_verts_idx != faces.npy (byte compare). The UV layout "
            "and the geometry disagree on topology — STOP.")
    if faces_uv.shape != faces.shape:
        die(f"faces_uv_idx shape {faces_uv.shape} != faces {faces.shape}.")
    log(f"UV layout OK: {verts_uvs.shape[0]} UV verts, per-corner indexing")

    # ---- 4. Blender: build base mesh from faces.npy (deterministic loops) ----
    try:
        import bpy
    except ImportError:
        die("`import bpy` failed — run this via Blender headless:\n    " + POD_CMD)

    bpy.ops.wm.read_factory_settings(use_empty=True)   # empty scene (no default cube)

    mesh = bpy.data.meshes.new("HeadMesh")
    mesh.from_pydata([tuple(v) for v in neutral_verts.tolist()], [],
                     [tuple(f) for f in faces.tolist()])
    mesh.update()
    if len(mesh.vertices) != n_verts or len(mesh.polygons) != n_faces:
        die(f"Blender mesh built {len(mesh.vertices)}V/{len(mesh.polygons)}F, "
            f"expected {n_verts}V/{n_faces}F — from_pydata altered topology. STOP.")
    obj = bpy.data.objects.new("Head", mesh)
    bpy.context.scene.collection.objects.link(obj)

    # ---- 5. UVs: assign per-loop from the FLAME layout, matched by vertex ----
    # from_pydata preserves polygon order (poly i == face i); we map each loop's
    # vertex_index to its UV index via the face's (vertex -> uv) correspondence,
    # so any loop rotation is irrelevant. Robust + deterministic.
    uv_layer = mesh.uv_layers.new(name="UVMap")
    loops = mesh.loops
    uv_flat = np.empty(len(loops) * 2, dtype=np.float32)
    for i, poly in enumerate(mesh.polygons):
        v2u = {int(faces_v[i, k]): int(faces_uv[i, k]) for k in range(3)}
        for L in poly.loop_indices:
            coord = verts_uvs[v2u[int(loops[L].vertex_index)]]
            uv_flat[2 * L] = coord[0]
            uv_flat[2 * L + 1] = coord[1]
    uv_layer.data.foreach_set("uv", uv_flat)
    log("UVs assigned to mesh loops")

    # ---- 6. baked albedo as the base-color texture (survives glTF export) ----
    mat = bpy.data.materials.new("HeadAlbedo")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    if bsdf is None:
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        out = nt.nodes.get("Material Output") or nt.nodes.new("ShaderNodeOutputMaterial")
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    tex = nt.nodes.new("ShaderNodeTexImage")
    img = bpy.data.images.load(str(ALBEDO_PNG))
    img.colorspace_settings.name = "sRGB"   # baked map is the sRGB lit appearance
    tex.image = img
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    mesh.materials.append(mat)
    log(f"albedo applied: {ALBEDO_PNG.name} -> Base Color (sRGB)")

    # ---- 7. shape keys: Basis + one per SUPPORTED ARKit name (exact spelling)-
    obj.shape_key_add(name="Basis", from_mix=False)
    built = []
    for name in supported:
        expr_ply = SHAPES_DIR / f"expr_{name}.ply"
        if not expr_ply.is_file() or expr_ply.stat().st_size == 0:
            die(f"supported shape '{name}' has no mesh at {expr_ply}. The manifest "
                "claims it, but the rig did not emit it — escalate to arkit-rigger. STOP.")
        verts = read_ply_vertices(expr_ply)
        if verts.shape[0] != n_verts:
            die(f"{expr_ply.name}: {verts.shape[0]} vertices != neutral {n_verts}. "
                "TOPOLOGY GUARANTEE BROKEN — index-aligned copy is invalid. This is "
                "arkit-rigger's contract failing; escalate, do not patch here. STOP.")
        kb = obj.shape_key_add(name=name, from_mix=False)
        if kb.name != name:      # Blender silently renamed (collision/length) => contract break
            die(f"shape key for '{name}' was stored as '{kb.name}'. A morph-target "
                "rename breaks the driver contract — STOP.")
        kb.data.foreach_set("co", verts.reshape(-1).astype(np.float32))
        kb.value = 0.0
        built.append(name)

    key_names = [kb.name for kb in obj.data.shape_keys.key_blocks]
    if key_names[0] != "Basis":
        die(f"first shape key is '{key_names[0]}', expected 'Basis'.")
    if sorted(key_names[1:]) != sorted(supported):
        die(f"shape-key set {sorted(key_names[1:])} != supported set — STOP.")
    obj.active_shape_key_index = 0
    log(f"shape keys built: Basis + {len(built)} ARKit morph targets "
        f"(unsupported {len(unsupported)} deliberately absent)")

    # ---- 8. export GLB + save .blend ----------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.export_scene.gltf(
        filepath=str(GLB_OUT),
        export_format="GLB",
        use_selection=False,
        export_apply=False,                 # applying modifiers would DROP shape keys
        export_morph=True,                  # shape keys -> glTF morph targets
        export_morph_normal=True,           # per-morph normals => smoother deformation
        export_draco_mesh_compression_enable=DRACO,
    )
    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_OUT))
    log(f"exported {GLB_OUT}  (Draco {'ON' if DRACO else 'OFF'})")
    log(f"saved     {BLEND_OUT}")

    # ---- 9. re-read the GLB and prove the morph-target names resolve ---------
    facts = verify_glb(GLB_OUT, supported)
    log(f"GLB verified: {facts['n_targets']} morph targets, names == supported set, "
        f"baseColor texture={facts['has_basecolor_texture']}")

    topo = {"n_vertices": n_verts, "n_faces": n_faces, "faces_sha256": faces_sha,
            "manifest_check": manifest_check}
    sizes = {"glb": GLB_OUT.stat().st_size, "blend": BLEND_OUT.stat().st_size}
    if sizes["glb"] < 1024:
        die(f"{GLB_OUT} is only {sizes['glb']} bytes — implausibly small. STOP.")
    write_report(supported, unsupported, demoted, topo, facts, sizes)
    log("DONE.")


if __name__ == "__main__":
    main()
