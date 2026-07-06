"""Stage 6 -- assemble + export head_arkit_v2.glb (RUNS INSIDE BLENDER):

  xvfb-run -a blender --background --factory-startup \
      --python pipe/s6_export_blender.py -- [args]

Deterministic assembly, NO mesh importers (so vertex ORDER can never drift):
the mesh is built with from_pydata straight from out/rig/arkit_deltas.npz
(refined neutral + polygons + ICT UVs), shape keys are set per-vertex from the
additive ARKit deltas, the baked albedo is wired to a Principled BSDF, and the
scene exports as GLB with morph targets named EXACTLY per the ARKit contract.

MATERIALS: the eyeball polygons (verts in ICT_REGIONS["eyeballs"]) get their
own eye material(s) bound to s5's dedicated eye texture(s) -- eye_left.png
for the x<0 eyeball, eye_right.png for x>=0 when present, otherwise one
shared EyeMat (the file contract: eye_right.png existing == separate irises).
Polys whose UVs live beyond UDIM tile 0 (MEASURED: ICT's atlas is multi-tile,
u up to 7 -- skull back, mouth socket, teeth, lashes) cannot use the tile-0
albedo (a wrapping sampler would paint them with the FACE image: the
stretched-face back of head), so they get RestMat driven by s5's per-vertex
colors (photo where visible, clay hair, honest interior defaults).
Everything else stays on HeadMat. Multiple materials => the exporter splits
the mesh into multiple primitives; morph targets are exported on ALL of them
and extras.targetNames stays mesh-level, so the ARKit name contract holds.

EYE SHELLS: ICT's lacrimal-fluid/eye-blend/eye-occlusion meshes (see
common.EYE_SHELLS) exist for Unreal-style TRANSLUCENT shaders; measured, they
cover the eyeball forward pole, so in an opaque export they'd hide the irises
behind skin-textured lids. Their FACES are stripped (all 26719 verts stay;
the exporter simply never references the loose ones). Eyelashes are kept.

OPAQUE HARDENING (kills the see-through-head viewer artifact): every material
is single-sided (use_backface_culling=True -> glTF doubleSided=false; the head
and eyeballs are closed meshes) and the exported GLB's JSON is post-processed
to carry an EXPLICIT "alphaMode": "OPAQUE" on every material. After export the
GLB is re-imported into a fresh scene and each material is MEASURED for
functional opacity (alpha socket unlinked & 1.0, surface_render_method
DITHERED / blend_method not BLEND, backface culling on). NOTE (measured on
Blender 4.2.3): Material.blend_method is a deprecated alias there -- even a
factory-new material reads 'HASHED' and assigning 'OPAQUE' does not stick, so
the check treats 'HASHED'+alpha==1.0+DITHERED as opaque (which it renders as)
and FAILS on 'BLEND'/'BLENDED'/linked-alpha.

Axes/units: Blender coords = (x, -z, y) of ICT * 0.01. The glTF exporter's
Z-up -> Y-up conversion is (x, z, -y), so GLB coords == ICT coords in meters:
+Y up, +Z front, exactly what three.js expects.

Outputs under out/export/: head_arkit_v2.glb, head_arkit_v2.blend,
export_info.json.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (EYE_SHELLS, ICT_REGIONS, N_VERTS, P,  # noqa: E402
                    faces_as_lists, out_dir)

import bpy  # noqa: E402

CM_TO_M = 0.01


def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def harden_opaque_props(mat):
    mat.use_backface_culling = True  # -> glTF doubleSided=false
    for attr, val in (("blend_method", "OPAQUE"),
                      ("surface_render_method", "DITHERED"),
                      ("show_transparent_back", False)):
        try:
            setattr(mat, attr, val)
        except (AttributeError, TypeError):
            pass  # property renamed/removed across Blender versions


def make_opaque_vcol_material(name, attr_name, roughness):
    """Vertex-colored Principled material (for the UDIM tiles the single
    baked albedo cannot carry -- see the s5 vertex_colors.npy rationale)."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Alpha"].default_value = 1.0
    vc = mat.node_tree.nodes.new("ShaderNodeVertexColor")
    vc.layer_name = attr_name
    mat.node_tree.links.new(vc.outputs["Color"], bsdf.inputs["Base Color"])
    harden_opaque_props(mat)
    return mat


def make_opaque_tex_material(name, img_path, roughness):
    """Textured Principled material hardened for opaque single-sided export."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Alpha"].default_value = 1.0
    if img_path is not None and Path(img_path).is_file():
        img = bpy.data.images.load(str(img_path))
        img.alpha_mode = "NONE"  # never wire texture alpha into the shader
        img.pack()
        tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
        tex.image = img
        mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        print(f"[s6 WARN] {img_path} missing -- {name} exports untextured")
    harden_opaque_props(mat)
    return mat


def harden_glb_opaque(glb_path):
    """Rewrite the GLB's JSON chunk in place: every material gets an EXPLICIT
    alphaMode OPAQUE and doubleSided false. stdlib-only, 4-byte aligned."""
    import struct
    data = Path(glb_path).read_bytes()
    magic, version, _ = struct.unpack_from("<III", data, 0)
    assert magic == 0x46546C67 and version == 2, "not a GLB v2"
    clen, ctype = struct.unpack_from("<II", data, 12)
    assert ctype == 0x4E4F534A, "first chunk is not JSON"
    gltf = json.loads(data[20:20 + clen])
    changed = []
    for m in gltf.get("materials", []):
        if m.get("alphaMode") != "OPAQUE":
            m["alphaMode"] = "OPAQUE"
            changed.append(m.get("name", "?"))
        if m.get("doubleSided"):
            m["doubleSided"] = False
            changed.append(m.get("name", "?") + ":doubleSided")
    payload = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    payload += b" " * (-len(payload) % 4)  # spaces pad JSON chunks (spec)
    rest = data[20 + clen:]  # BIN chunk(s), already aligned
    out = bytearray()
    out += struct.pack("<III", magic, version, 12 + 8 + len(payload) + len(rest))
    out += struct.pack("<II", len(payload), ctype) + payload + rest
    Path(glb_path).write_bytes(bytes(out))
    print(f"[s6] GLB hardened opaque (explicit alphaMode) -- touched: "
          f"{changed or 'nothing (already explicit)'}")
    return gltf


def reimport_opacity_check(glb_path):
    """Round-trip proof: import the GLB into a FRESH scene and measure that
    every material lands functionally opaque. Returns (ok, per-material list).

    Functional opacity = Principled Alpha unlinked and == 1.0, render method
    not BLENDED/BLEND, backface culling on (doubleSided=false round-tripped).
    On Blender 4.2 blend_method is a deprecated alias that reads 'HASHED' for
    ALL materials (measured: even factory-new ones; setting 'OPAQUE' does not
    stick), so 'HASHED' with alpha==1.0 is opaque there; 'BLEND' is a FAIL.
    """
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(glb_path))
    results, ok = [], True
    for m in bpy.data.materials:
        alpha_ok = True
        if m.use_nodes:
            for n in m.node_tree.nodes:
                if n.type == "BSDF_PRINCIPLED":
                    a = n.inputs["Alpha"]
                    alpha_ok &= (not a.is_linked
                                 and abs(a.default_value - 1.0) < 1e-6)
        entry = {
            "material": m.name,
            "blend_method": getattr(m, "blend_method", None),
            "surface_render_method": getattr(m, "surface_render_method", None),
            "backface_culling": bool(m.use_backface_culling),
            "alpha_unlinked_and_1": bool(alpha_ok),
            "show_transparent_back": getattr(m, "show_transparent_back", None),
        }
        entry["opaque"] = bool(
            alpha_ok
            and entry["blend_method"] != "BLEND"
            and entry["surface_render_method"] in (None, "DITHERED")
            and entry["backface_culling"])
        ok &= entry["opaque"]
        results.append(entry)
        print(f"[s6] reimport-check {m.name}: blend_method="
              f"{entry['blend_method']} surface_render_method="
              f"{entry['surface_render_method']} alpha_1={alpha_ok} "
              f"backface_culling={entry['backface_culling']} -> "
              f"{'OPAQUE' if entry['opaque'] else 'NOT OPAQUE'}")
    return ok, results


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(description="s6 GLB export")
    ap.add_argument("--out", default=P.OUT)
    ap.add_argument("--name", default="head_arkit_v2")
    ap.add_argument("--draco", action="store_true")
    ap.add_argument("--morph-normals", action="store_true",
                    help="export per-target normals (bigger GLB)")
    return ap.parse_args(argv)


def ict_to_blender(v_cm):
    """(N,3) ICT cm (+Y up,+Z front) -> Blender m (Z up). See module docstring."""
    v = np.asarray(v_cm, dtype=np.float64)
    return np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1) * CM_TO_M


def main():
    args = parse_args()
    t0 = time.time()
    od = out_dir(args.out, "export")
    rig_dir = Path(args.out) / "rig"

    z = np.load(rig_dir / "arkit_deltas.npz")
    names = [str(n) for n in z["names"]]
    deltas = z["deltas"].astype(np.float64)
    neutral = z["refined_neutral"].astype(np.float64)
    faces_flat, faces_off = z["faces_flat"], z["faces_off"]
    corner_vt, vt = z["corner_vt"], z["vt"].astype(np.float64)
    manifest = json.loads((rig_dir / "arkit_manifest.json").read_text())
    assert len(neutral) == N_VERTS, f"neutral {len(neutral)} != {N_VERTS} -- STOP"
    assert deltas.shape[1] == N_VERTS, "delta topology drift -- STOP"
    sup = [n for n in manifest["shapes"] if manifest["shapes"][n]["supported"]]
    assert set(names) == set(sup), "npz names != manifest supported set -- STOP"
    print(f"[s6] {len(names)} morph targets on {N_VERTS} verts "
          f"/ {len(faces_off)-1} polys")

    # ---- strip the transparent-purpose eye shells (lacrimal fluid + eye
    # blend + eye occlusion). MEASURED: they cover the eyeball forward pole,
    # so an all-OPAQUE export would render them as skin-textured lids hiding
    # the irises. FACES only -- all 26719 verts stay (loose verts are simply
    # not referenced by any primitive). Eyelashes are kept.
    sh0, sh1 = EYE_SHELLS
    in_sh = ((faces_flat >= sh0) & (faces_flat < sh1)).astype(np.uint8)
    seg_all = faces_off[:-1]
    sh_all = np.minimum.reduceat(in_sh, seg_all).astype(bool)
    sh_any = np.maximum.reduceat(in_sh, seg_all).astype(bool)
    assert (sh_all == sh_any).all(), \
        "eye-shell polys weld into other regions -- region drift, STOP"
    counts_all = np.diff(faces_off)
    keep = ~sh_all
    corner_keep = np.repeat(keep, counts_all)
    faces_flat = faces_flat[corner_keep]
    corner_vt = corner_vt[corner_keep]
    faces_off = np.concatenate(
        [[0], np.cumsum(counts_all[keep])]).astype(np.int64)
    n_stripped = int((~keep).sum())
    print(f"[s6] stripped {n_stripped} transparent-purpose eye-shell polys "
          f"(verts [{sh0},{sh1})); {len(faces_off)-1} polys remain")

    bpy.ops.wm.read_factory_settings(use_empty=True)

    mesh = bpy.data.meshes.new("HeadARKit")
    mesh.from_pydata(ict_to_blender(neutral).tolist(), [],
                     faces_as_lists(faces_flat, faces_off))
    mesh.update()
    assert len(mesh.vertices) == N_VERTS, "from_pydata vertex-count drift -- STOP"
    assert len(mesh.loops) == len(faces_flat), "loop-count drift -- STOP"

    # UVs: loop order after from_pydata follows the polygon corner order exactly
    uvl = mesh.uv_layers.new(name="UVMap")
    uv_flat = vt[corner_vt].astype(np.float32).ravel()
    uvl.data.foreach_set("uv", uv_flat)
    mesh.polygons.foreach_set("use_smooth",
                              np.ones(len(mesh.polygons), dtype=bool))
    mesh.update()

    obj = bpy.data.objects.new("HeadARKit", mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # ---- shape keys: exact ARKit names, absolute coords = neutral + delta
    obj.shape_key_add(name="Basis", from_mix=False)
    for i, name in enumerate(names):
        sk = obj.shape_key_add(name=name, from_mix=False)
        co = ict_to_blender(neutral + deltas[i]).astype(np.float32).ravel()
        sk.data.foreach_set("co", co)
        sk.slider_min, sk.slider_max = 0.0, 1.0
    kb = obj.data.shape_keys.key_blocks
    got = [k.name for k in kb if k.name != "Basis"]
    assert got == names, f"shape key name drift: {set(names) ^ set(got)}"
    print(f"[s6] shape keys created: {len(got)} (+Basis)")

    # ---- materials: HeadMat (baked albedo) + dedicated eye material(s)
    tex_dir = Path(args.out) / "tex"
    albedo = tex_dir / "albedo.png"
    obj.data.materials.append(make_opaque_tex_material("HeadMat", albedo, 0.6))

    eye_l_png = tex_dir / "eye_left.png"
    eye_r_png = tex_dir / "eye_right.png"
    vcol_npy = tex_dir / "vertex_colors.npy"
    mat_names = ["HeadMat"]
    seg = faces_off[:-1]
    counts = np.diff(faces_off)
    mat_idx = np.zeros(len(counts), dtype=np.int32)

    # per-polygon eyeball membership from the region vertex range
    eb0, eb1 = ICT_REGIONS["eyeballs"]
    inr = ((faces_flat >= eb0) & (faces_flat < eb1)).astype(np.uint8)
    all_in = np.minimum.reduceat(inr, seg).astype(bool)
    any_in = np.maximum.reduceat(inr, seg).astype(bool)
    assert (all_in == any_in).all(), \
        "polygons straddling the eyeball vertex range -- region drift, STOP"

    if eye_l_png.is_file():
        meanx = np.add.reduceat(neutral[faces_flat, 0], seg) / counts
        shared_eye = not eye_r_png.is_file()  # s5's file contract
        if shared_eye:
            obj.data.materials.append(
                make_opaque_tex_material("EyeMat", eye_l_png, 0.35))
            mat_names.append("EyeMat")
            mat_idx[all_in] = 1
        else:
            obj.data.materials.append(
                make_opaque_tex_material("EyeL", eye_l_png, 0.35))
            obj.data.materials.append(
                make_opaque_tex_material("EyeR", eye_r_png, 0.35))
            mat_names += ["EyeL", "EyeR"]
            mat_idx[all_in & (meanx < 0)] = 1
            mat_idx[all_in & (meanx >= 0)] = 2
        n_eye = int(all_in.sum())
        print(f"[s6] eye material(s) {mat_names[1:]} on {n_eye} polys "
              f"(x<0: {int((all_in & (meanx < 0)).sum())}, "
              f"x>=0: {int((all_in & (meanx >= 0)).sum())})")
        assert n_eye > 0, "no eyeball polygons found -- region drift, STOP"
    else:
        print("[s6 WARN] out/tex/eye_left.png missing -- eyes stay on HeadMat "
              "(flat sclera); run s5 first for real irises")

    # UDIM tile-1+ polys: the baked albedo only carries tile 0 (MEASURED:
    # ICT UVs span u up to 7); with a wrapping sampler these polys would show
    # the FACE image (stretched-face back of head, face-textured teeth).
    # They get RestMat driven by s5's per-vertex colors instead.
    if vcol_npy.is_file():
        umax = np.maximum.reduceat(vt[corner_vt, 0], seg)
        rest = (umax > 1.0) & ~all_in
        if rest.any():
            vcol = np.load(vcol_npy).astype(np.float64)
            assert len(vcol) == N_VERTS, "vertex_colors.npy topology drift"
            attr = mesh.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
            rgba = np.concatenate(
                [srgb_to_linear(vcol), np.ones((N_VERTS, 1))], axis=1)
            attr.data.foreach_set("color", rgba.astype(np.float32).ravel())
            obj.data.materials.append(
                make_opaque_vcol_material("RestMat", "Col", 0.6))
            mat_idx[rest] = len(mat_names)
            mat_names.append("RestMat")
            print(f"[s6] RestMat (vertex colors) on {int(rest.sum())} "
                  f"UDIM tile-1+ polys; HeadMat keeps "
                  f"{int((mat_idx == 0).sum())}")
    else:
        print("[s6 WARN] out/tex/vertex_colors.npy missing -- tile-1+ polys "
              "stay on HeadMat (wrapped face texture); rerun s5")

    mesh.polygons.foreach_set("material_index", mat_idx)
    mesh.update()

    glb_path = od / f"{args.name}.glb"
    export_kwargs = dict(
        filepath=str(glb_path),
        export_format="GLB",
        export_morph=True,
        export_morph_normal=bool(args.morph_normals),
        export_yup=True,
        export_image_format="AUTO",
        export_draco_mesh_compression_enable=bool(args.draco),
    )
    try:
        bpy.ops.export_scene.gltf(**export_kwargs)
    except TypeError as e:  # exporter kwarg drift across Blender versions
        print(f"[s6 WARN] full-kwarg export failed ({e}); minimal retry")
        bpy.ops.export_scene.gltf(filepath=str(glb_path), export_format="GLB",
                                  export_morph=True)
    print(f"[s6] GLB -> {glb_path} ({glb_path.stat().st_size/1e6:.1f} MB)")

    # explicit alphaMode OPAQUE + doubleSided false, measured in the file
    gltf_json = harden_glb_opaque(glb_path)
    for m in gltf_json.get("materials", []):
        assert m.get("alphaMode") == "OPAQUE" and not m.get("doubleSided"), \
            f"material {m.get('name')} not hardened opaque -- STOP"

    blend_path = od / f"{args.name}.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path), compress=True)
    print(f"[s6] assembly blend -> {blend_path}")

    # round-trip proof (AFTER the .blend is saved -- this wipes the scene)
    reimport_ok, reimport_results = reimport_opacity_check(glb_path)

    with open(od / "export_info.json", "w", encoding="utf-8") as f:
        json.dump({"glb": str(glb_path), "blend": str(blend_path),
                   "n_verts": N_VERTS, "n_polys": int(len(faces_off) - 1),
                   "stripped_eye_shell_polys": n_stripped,
                   "morph_targets": names, "draco": bool(args.draco),
                   "textured": albedo.is_file(),
                   "materials": mat_names,
                   "reimport_opaque": {"pass": reimport_ok,
                                       "materials": reimport_results},
                   "units": "meters, +Y up, +Z front (glTF standard)"},
                  f, indent=2)
    print(f"[s6] DONE in {time.time()-t0:.1f}s")
    if not reimport_ok:
        print("[s6 FATAL] reimported GLB is NOT opaque -- see export_info.json")
        sys.exit(1)


main()
