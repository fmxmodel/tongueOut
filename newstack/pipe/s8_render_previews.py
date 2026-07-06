"""Stage 8 -- render PROOF images from the exported GLB (RUNS INSIDE BLENDER):

  xvfb-run -a blender --background --factory-startup \
      --python pipe/s8_render_previews.py -- [--out OUT] [--glb GLB]

Imports out/export/head_arkit_v2.glb into a FRESH scene (so the renders show
exactly what a downstream importer gets -- materials, textures, opacity) and
renders to out/renders/:

  glb_front_full.png    full head, front (front = Blender -Y after import)
  glb_back_full.png     full head, back -- must be SOLID (opaque + culling ok)
  glb_eyes_front.png    tight eye close-up: iris + pupil + sclera, forward gaze
  glb_eyes_look.png     same framing with eye-look morphs at 1.0 -- the irises
                        must MOVE (proves the eye texture rides the eyeball
                        geometry through the ARKit morphs)

EEVEE with Standard view transform (honest texture colors); Workbench TEXTURE
fallback if EEVEE cannot initialize on a headless GPU.
"""

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import P  # noqa: E402

import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402

LOOK_SHAPES = ("eyeLookOutLeft", "eyeLookInRight")  # both eyes -> subject left


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(description="s8 render previews")
    ap.add_argument("--out", default=P.OUT)
    ap.add_argument("--glb", default=None)
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--look-value", type=float, default=1.0)
    return ap.parse_args(argv)


def aim(obj, position, target):
    obj.location = position
    obj.rotation_euler = (Vector(target) - Vector(position)
                          ).to_track_quat("-Z", "Y").to_euler()


def main():
    args = parse_args()
    t0 = time.time()
    glb = Path(args.glb or Path(args.out) / "export" / "head_arkit_v2.glb")
    od = Path(args.out) / "renders"
    od.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(glb))
    obj = next(o for o in bpy.data.objects if o.type == "MESH")
    print(f"[s8] imported {glb.name}: mesh {obj.name}, "
          f"{len(obj.data.vertices)} verts, "
          f"{len(obj.material_slots)} material slot(s) "
          f"{[s.material.name for s in obj.material_slots]}")

    # ---- measured framing targets
    bb = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    bb_min = Vector((min(v[i] for v in bb) for i in range(3)))
    bb_max = Vector((max(v[i] for v in bb) for i in range(3)))
    center = (bb_min + bb_max) / 2
    size = max(bb_max - bb_min)
    # eye centroid from the polygons bound to the Eye material(s)
    eye_slots = [i for i, s in enumerate(obj.material_slots)
                 if s.material and s.material.name.startswith("Eye")]
    eye_vids = {v for p in obj.data.polygons if p.material_index in eye_slots
                for v in p.vertices}
    if eye_vids:
        eye_c = obj.matrix_world @ (
            sum((obj.data.vertices[v].co for v in eye_vids), Vector()) /
            len(eye_vids))
        print(f"[s8] eye centroid (measured from Eye material polys): "
              f"{tuple(round(c, 4) for c in eye_c)}")
    else:
        eye_c = Vector((0.0, -0.0844, 0.0361))  # measured ICT fallback
        print("[s8 WARN] no Eye material polys -- using ICT fallback centroid")

    # ---- scene: camera, sun, neutral gray ambient, honest colors
    cam = bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam"))
    sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN"))
    sun.data.energy = 3.0
    scn = bpy.context.scene
    for o in (cam, sun):
        scn.collection.objects.link(o)
    scn.camera = cam
    world = bpy.data.worlds.new("W")
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.18, 0.18, 0.18, 1.0)
    bg.inputs[1].default_value = 1.0
    scn.world = world
    scn.render.resolution_x = scn.render.resolution_y = args.res
    scn.render.image_settings.file_format = "PNG"
    scn.view_settings.view_transform = "Standard"  # judge the real colors
    for eng in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "BLENDER_WORKBENCH"):
        try:
            scn.render.engine = eng
            break
        except TypeError:
            continue
    print(f"[s8] engine {scn.render.engine}")

    def shot(name, position, target, lens_mm):
        cam.data.lens = lens_mm
        aim(cam, position, target)
        # key light from just above the camera, aimed at the target
        aim(sun, Vector(position) + Vector((0, 0, 0.3)), target)
        scn.render.filepath = str(od / name)
        try:
            bpy.ops.render.render(write_still=True)
        except Exception as e:  # headless-GPU EEVEE failure -> workbench
            print(f"[s8 WARN] {scn.render.engine} failed ({e}); workbench")
            scn.render.engine = "BLENDER_WORKBENCH"
            scn.display.shading.light = "STUDIO"
            scn.display.shading.color_type = "TEXTURE"
            bpy.ops.render.render(write_still=True)
        print(f"[s8] render -> {od / name}")

    def frame_dist(width, lens_mm, margin=1.25):
        half_angle = math.atan(18.0 / lens_mm)  # 36mm sensor, square render
        return margin * (width / 2) / math.tan(half_angle)

    d_full = frame_dist(size, 50)
    shot("glb_front_full.png", (center.x, center.y - d_full, center.z),
         center, 50)
    shot("glb_back_full.png", (center.x, center.y + d_full, center.z),
         center, 50)

    d_eye = frame_dist(0.13, 85)
    eye_cam = (eye_c.x, eye_c.y - d_eye, eye_c.z)
    shot("glb_eyes_front.png", eye_cam, eye_c, 85)

    kb = obj.data.shape_keys.key_blocks if obj.data.shape_keys else {}
    missing = [n for n in LOOK_SHAPES if n not in kb]
    if missing:
        print(f"[s8 WARN] look shapes missing from GLB: {missing} -- "
              "skipping the gaze render")
    else:
        for n in LOOK_SHAPES:
            kb[n].value = args.look_value
        print(f"[s8] set {', '.join(LOOK_SHAPES)} = {args.look_value}")
        shot("glb_eyes_look.png", eye_cam, eye_c, 85)
        for n in LOOK_SHAPES:
            kb[n].value = 0.0
    print(f"[s8] DONE in {time.time()-t0:.1f}s")


main()
