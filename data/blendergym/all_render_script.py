"""Blender script to render all camera views (Camera1-5) after executing scene code."""
import bpy
import os
import sys

# Disable audio sync/playback at scene level
for scene in bpy.data.scenes:
    scene.use_audio = False
    scene.sync_mode = 'NONE'

# Disable audio device in preferences
try:
    bpy.context.preferences.system.audio_device = 'NONE'
except Exception:
    pass

if __name__ == "__main__":

    # Parse arguments after '--' separator to handle variable number of Blender flags
    try:
        separator_idx = sys.argv.index('--')
        args_after_separator = sys.argv[separator_idx + 1:]
        code_fpath = args_after_separator[0]  # Path to the code file
        rendering_dir = args_after_separator[1]  # Path to save the rendering from camera1
    except (ValueError, IndexError):
        raise ValueError("Usage: blender --background [flags] -- code.py render_dir")

    # Enable GPU rendering
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'  # or 'OPTIX' if your GPU supports it
    bpy.context.preferences.addons['cycles'].preferences.get_devices()

    # Check and select the GPUs
    for device in bpy.context.preferences.addons['cycles'].preferences.devices:
        if device.type == 'GPU' and not device.use:
            device.use = True

    # Set the rendering device to GPU
    bpy.context.scene.cycles.device = 'GPU'

    # Setting up rendering resolution
    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = 512

    # Set max samples to 1024
    bpy.context.scene.cycles.samples = 512

    # Set color mode to RGB
    bpy.context.scene.render.image_settings.color_mode = 'RGB'

    # Read and execute the code from the specified file
    with open(code_fpath, "r") as f:
        code = f.read()
    try:
        exec(code)
    except:
        raise ValueError

    # Render from camera1
    if 'Camera1' in bpy.data.objects:
        # Convert rendering_dir to absolute path to ensure correct file location
        rendering_dir = os.path.abspath(rendering_dir)
        os.makedirs(rendering_dir, exist_ok=True)
        bpy.context.scene.camera = bpy.data.objects['Camera1']
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.filepath = os.path.join(rendering_dir, 'render1.png')
        # Use EXEC_DEFAULT explicitly for Blender 5 headless rendering compatibility
        bpy.ops.render.render("EXEC_DEFAULT", write_still=True)

    # Render from camera2
    if 'Camera2' in bpy.data.objects:
        bpy.context.scene.camera = bpy.data.objects['Camera2']
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.filepath = os.path.join(rendering_dir, 'render2.png')
        # Use EXEC_DEFAULT explicitly for Blender 5 headless rendering compatibility
        bpy.ops.render.render("EXEC_DEFAULT", write_still=True)

    # Render from camera3
    if 'Camera3' in bpy.data.objects:
        bpy.context.scene.camera = bpy.data.objects['Camera3']
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.filepath = os.path.join(rendering_dir, 'render3.png')
        # Use EXEC_DEFAULT explicitly for Blender 5 headless rendering compatibility
        bpy.ops.render.render("EXEC_DEFAULT", write_still=True)

    # Render from camera4
    if 'Camera4' in bpy.data.objects:
        bpy.context.scene.camera = bpy.data.objects['Camera4']
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.filepath = os.path.join(rendering_dir, 'render4.png')
        # Use EXEC_DEFAULT explicitly for Blender 5 headless rendering compatibility
        bpy.ops.render.render("EXEC_DEFAULT", write_still=True)

    # Render from camera5
    if 'Camera5' in bpy.data.objects:
        bpy.context.scene.camera = bpy.data.objects['Camera5']
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.filepath = os.path.join(rendering_dir, 'render5.png')
        # Use EXEC_DEFAULT explicitly for Blender 5 headless rendering compatibility
        bpy.ops.render.render("EXEC_DEFAULT", write_still=True)




