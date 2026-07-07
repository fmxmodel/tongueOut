"""Render a neutral head mesh grey/opaque from canonical views (RUNS INSIDE
BLENDER, Workbench engine + cavity shading -- geometry inspection only):

  xvfb-run -a blender --background --factory-startup \
      --python pipe/render_neutral.py -- --out OUT --tag triposg \
      --mesh OUT/refine/refined_neutral.npy [--views front,right,left,back]

Mesh formats: .npy (verts; polygons from out/fit/topology.npz), .npz
(verts+faces keys, e.g. clay_sg_aligned.npz), .obj. Coordinates are ICT model
space (cm, +Y up, +Z front); cameras are placed in that same frame, so no
axis-conversion import quirks. Writes OUT/renders/neutral_{tag}_{view}.png.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import faces_as_lists, read_obj  # noqa: E402

import bpy  # noqa: E402

VIEWS = {  # name -> (location offset dir, rotation_euler)
    "front": ((0, 0, 1), (0.0, 0.0, 0.0)),
    "back": ((0, 0, -1), (0.0, math.pi, 0.0)),
    "right": ((1, 0, 0), (0.0, math.pi / 2, 0.0)),
    "left": ((-1, 0, 0), (0.0, -math.pi / 2, 0.0)),
}


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(description="grey geometry proof renders")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mesh", required=True, help=".npy | .npz | .obj")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--views", default="front,right,left,back")
    ap.add_argument("--res", type=int, default=1000)
    return ap.parse_args(argv)


def load_mesh(path, out):
    p = Path(path)
    if p.suffix == ".npy":
        verts = np.load(p)
        topo = np.load(Path(out) / "fit" / "topology.npz")
        faces = faces_as_lists(topo["faces_flat"], topo["faces_off"])
    elif p.suffix == ".npz":
        z = np.load(p)
        verts = z["verts"].astype(np.float64)
        faces = [tuple(int(i) for i in f) for f in z["faces"]]
    elif p.suffix == ".obj":
        o = read_obj(p)
        verts = o["v"]
        faces = faces_as_lists(o["faces_flat"], o["faces_off"])
    else:
        raise SystemExit(f"unsupported mesh format: {p}")
    return np.asarray(verts, dtype=np.float64), faces


def main():
    args = parse_args()
    verts, faces = load_mesh(args.mesh, args.out)
    rd = Path(args.out) / "renders"
    rd.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    mesh = bpy.data.meshes.new("head")
    mesh.from_pydata(verts.tolist(), [], faces)
    mesh.update()
    obj = bpy.data.objects.new("head", mesh)
    scene.collection.objects.link(obj)

    scene.render.engine = "BLENDER_WORKBENCH"
    sh = scene.display.shading
    sh.light = "STUDIO"
    sh.color_type = "SINGLE"
    sh.single_color = (0.78, 0.78, 0.78)
    sh.show_cavity = True
    scene.display.render_aa = "8"
    scene.render.resolution_x = args.res
    scene.render.resolution_y = int(args.res * 1.25)
    scene.render.image_settings.file_format = "PNG"

    lo, hi = verts.min(0), verts.max(0)
    c = (lo + hi) / 2.0
    extent = float((hi - lo).max())
    dist = extent * 3.0

    cam_data = bpy.data.cameras.new("cam")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = extent * 1.25
    cam_data.clip_start = 0.1
    cam_data.clip_end = dist * 4.0
    cam = bpy.data.objects.new("cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam

    for view in [v.strip() for v in args.views.split(",") if v.strip()]:
        if view not in VIEWS:
            raise SystemExit(f"unknown view {view} (have {list(VIEWS)})")
        d, rot = VIEWS[view]
        cam.location = (c[0] + d[0] * dist, c[1] + d[1] * dist, c[2] + d[2] * dist)
        cam.rotation_euler = rot
        out_png = rd / f"neutral_{args.tag}_{view}.png"
        scene.render.filepath = str(out_png)
        bpy.ops.render.render(write_still=True)
        print(f"[render] {out_png}")


main()
