import bpy
import json
import math
import os
from mathutils import Vector

object_names = ['Floor', 'BackWall', 'LeftWall']
objects = []

# Find objects to observe
if not object_names:
    # If empty list, observe all mesh objects
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and obj.name not in ['Ground', 'Plane']:
            objects.append(obj)
else:
    # Find specified objects by name
    for obj_name in object_names:
        obj = bpy.data.objects.get(obj_name)
        if obj:
            objects.append(obj)

if not objects:
    raise ValueError("No valid objects found")

# Calculate bounding box
min_x = min_y = min_z = float('inf')
max_x = max_y = max_z = float('-inf')

for obj in objects:
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    for corner in bbox_corners:
        min_x = min(min_x, corner.x)
        min_y = min(min_y, corner.y)
        min_z = min(min_z, corner.z)
        max_x = max(max_x, corner.x)
        max_y = max(max_y, corner.y)
        max_z = max(max_z, corner.z)

center_x = (min_x + max_x) / 2
center_y = (min_y + max_y) / 2
center_z = (min_z + max_z) / 2

size_x = max_x - min_x
size_y = max_y - min_y
size_z = max_z - min_z
max_size = max(size_x, size_y, size_z)
margin = max_size * 0.5

camera_positions = [
    (center_x - margin, center_y - margin, center_z + margin),
    (center_x + margin, center_y - margin, center_z + margin),
    (center_x - margin, center_y + margin, center_z + margin),
    (center_x + margin, center_y + margin, center_z + margin)
]

# Get camera
camera = bpy.context.scene.camera
if not camera:
    cameras = [obj for obj in bpy.data.objects if obj.type == 'CAMERA']
    if cameras:
        camera = cameras[0]
        bpy.context.scene.camera = camera

# Store original position
original_location = camera.location.copy()
original_rotation = camera.rotation_euler.copy()
camera_infos = []

# Set up viewpoints and render each
render_dir = os.environ.get("RENDER_DIR", "/tmp")
render_engine = os.environ.get("RENDER_ENGINE", "BLENDER_EEVEE").upper()
engine_map = {'EEVEE': 'BLENDER_EEVEE', 'BLENDER_EEVEE_NEXT': 'BLENDER_EEVEE', 'CYCLES': 'CYCLES', 'WORKBENCH': 'BLENDER_WORKBENCH'}
render_engine = engine_map.get(render_engine, render_engine)

for i, pos in enumerate(camera_positions):
    camera.location = pos
    camera.rotation_euler = (math.radians(60), 0, math.radians(45))
    
    # Look at center
    direction = Vector((center_x, center_y, center_z)) - camera.location
    camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    # Render per viewpoint
    bpy.context.scene.render.engine = render_engine
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = 512
    bpy.context.scene.render.filepath = os.path.join(render_dir, str(i+1)+".png")
    bpy.ops.render.render("EXEC_DEFAULT", write_still=True)
    
    camera_infos.append({
        "location": list(camera.location),
        "rotation": list(camera.rotation_euler)
    })
    
with open(f"#output/static_scene/20260203_155726/test2 (leave empty for new run)/investigator/tmp/camera_info.json", "w") as f:
    json.dump(camera_infos, f)
    
# Restore original position
camera.location = original_location
camera.rotation_euler = original_rotation

print("Viewpoints initialized and rendered for", len(objects), "objects")
