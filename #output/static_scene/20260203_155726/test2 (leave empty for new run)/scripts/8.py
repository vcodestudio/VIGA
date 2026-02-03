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
bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=(7.0, -7.0, 3.0), rotation=(1.2, 0, 0.78))
camera = bpy.context.object
camera.name = 'Camera'
bpy.context.scene.camera = camera

# Create Sun Light
bpy.ops.object.light_add(type='SUN', radius=1, align='WORLD', location=(0, 0, 0))
sun_light = bpy.context.object
sun_light.name = "SunLight"
sun_light.data.energy = 2.0
sun_light.rotation_euler = (0.7, -0.6, 0.5)

# Create L-shaped counter - base cabinets (green)
# Long part of the L (along -X axis, against left wall)
bpy.ops.mesh.primitive_cube_add(size=1, enter_editmode=False, location=(-2.0, 0.5, 0.4))
counter_base_long = bpy.context.object
counter_base_long.name = "CounterBaseLong"
counter_base_long.scale = (4.0, 0.6, 0.4)
bpy.ops.object.modifier_add(type='BEVEL')
counter_base_long.modifiers["Bevel"].width = 0.1
counter_base_long.modifiers["Bevel"].segments = 4

# Short part of the L (along +Y axis, against back wall)
bpy.ops.mesh.primitive_cube_add(size=1, enter_editmode=False, location=(-4.5, 2.5, 0.4))
counter_base_short = bpy.context.object
counter_base_short.name = "CounterBaseShort"
counter_base_short.scale = (0.6, 2.0, 0.4)
bpy.ops.object.modifier_add(type='BEVEL')
counter_base_short.modifiers["Bevel"].width = 0.1
counter_base_short.modifiers["Bevel"].segments = 4

# Create L-shaped counter - countertop (white)
# Long part of the L (along -X axis)
bpy.ops.mesh.primitive_cube_add(size=1, enter_editmode=False, location=(-2.0, 0.5, 0.85))
counter_top_long = bpy.context.object
counter_top_long.name = "CounterTopLong"
counter_top_long.scale = (4.1, 0.7, 0.05)
bpy.ops.object.modifier_add(type='BEVEL')
counter_top_long.modifiers["Bevel"].width = 0.05
counter_top_long.modifiers["Bevel"].segments = 4

# Short part of the L (along +Y axis)
bpy.ops.mesh.primitive_cube_add(size=1, enter_editmode=False, location=(-4.5, 2.5, 0.85))
counter_top_short = bpy.context.object
counter_top_short.name = "CounterTopShort"
counter_top_short.scale = (0.7, 2.1, 0.05)
bpy.ops.object.modifier_add(type='BEVEL')
counter_top_short.modifiers["Bevel"].width = 0.05
counter_top_short.modifiers["Bevel"].segments = 4

# Create illuminated base strip
bpy.ops.mesh.primitive_cube_add(size=1, enter_editmode=False, location=(-2.0, 0.5, 0.05))
light_strip_long = bpy.context.object
light_strip_long.name = "LightStripLong"
light_strip_long.scale = (3.9, 0.5, 0.05)

bpy.ops.mesh.primitive_cube_add(size=1, enter_editmode=False, location=(-4.5, 2.5, 0.05))
light_strip_short = bpy.context.object
light_strip_short.name = "LightStripShort"
light_strip_short.scale = (0.5, 1.9, 0.05)