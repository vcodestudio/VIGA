import bpy

# Clear existing objects
bpy.ops.wm.read_factory_settings(use_empty=True)

# Create Floor
bpy.ops.mesh.primitive_plane_add(size=20, enter_editmode=False, align='WORLD', location=(0, 0, 0))
floor = bpy.context.object
floor.name = "Floor"

# Create Back Wall
bpy.ops.mesh.primitive_plane_add(size=20, enter_editmode=False, align='WORLD', location=(0, 10, 5), rotation=(1.5708, 0, 0))
back_wall = bpy.context.object
back_wall.name = "BackWall"

# Create Left Wall
bpy.ops.mesh.primitive_plane_add(size=20, enter_editmode=False, align='WORLD', location=(-10, 0, 5), rotation=(0, 1.5708, 0))
left_wall = bpy.context.object
left_wall.name = "LeftWall"

# Create Camera
bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=(5.5, -5.5, 2.5), rotation=(1.5, 0, 0.78))
camera = bpy.context.object
camera.name = 'Camera'
bpy.context.scene.camera = camera

# Create Sun Light
bpy.ops.object.light_add(type='SUN', radius=1, align='WORLD', location=(0, 0, 0))
sun_light = bpy.context.object
sun_light.name = "SunLight"
sun_light.data.energy = 2.0
sun_light.rotation_euler = (0.7, -0.6, 0.5)