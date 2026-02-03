import bpy
import json
import os

# Get camera
camera = bpy.context.scene.camera
if not camera:
    # Find first camera
    cameras = [obj for obj in bpy.data.objects if obj.type == 'CAMERA']
    if cameras:
        camera = cameras[0]
        bpy.context.scene.camera = camera
    else:
        raise ValueError("No camera found in scene")

# Set camera location and rotation
camera.location = [7.0, -7.0, 3.0]
camera.rotation_euler = [1.2, 0.0, 0.78]

# Render after setting camera
render_dir = os.environ.get("RENDER_DIR", "/tmp")
render_engine = os.environ.get("RENDER_ENGINE", "BLENDER_EEVEE").upper()
engine_map = {'EEVEE': 'BLENDER_EEVEE', 'BLENDER_EEVEE_NEXT': 'BLENDER_EEVEE', 'CYCLES': 'CYCLES', 'WORKBENCH': 'BLENDER_WORKBENCH'}
bpy.context.scene.render.engine = engine_map.get(render_engine, render_engine)
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.resolution_x = 512
bpy.context.scene.render.resolution_y = 512
bpy.context.scene.render.filepath = os.path.join(render_dir, "output.png")
bpy.ops.render.render("EXEC_DEFAULT", write_still=True)

camera_info = [{
    "location": list(camera.location),
    "rotation": list(camera.rotation_euler)
}]

with open(f"#output/static_scene/20260203_155726/test2 (leave empty for new run)/investigator/tmp/camera_info.json", "w") as f:
    json.dump(camera_info, f)

print("Camera set to location and rotation and rendered")
