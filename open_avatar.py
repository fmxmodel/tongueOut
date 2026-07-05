import bpy

# clean scene (drop default cube/camera/light)
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)

# import the exported avatar
bpy.ops.import_scene.gltf(filepath="/home/darpa/Desktop/newARC/out/head_arkit.glb")

# select + frame the head, and show the baked texture (Material Preview)
mesh = next((o for o in bpy.data.objects if o.type == 'MESH'), None)
if mesh:
    bpy.ops.object.select_all(action='DESELECT')
    mesh.select_set(True)
    bpy.context.view_layer.objects.active = mesh

for win in bpy.context.window_manager.windows:
    for area in win.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = 'MATERIAL'

print("[open_avatar] imported head_arkit.glb (20 ARKit morph targets)")
