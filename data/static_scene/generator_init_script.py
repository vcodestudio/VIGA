"""Blender script for static scene initialization with Camera1 rendering."""
import bpy
import os
import sys

if __name__ == "__main__":

    # Parse arguments after '--' separator to handle variable number of Blender flags
    try:
        separator_idx = sys.argv.index('--')
        args_after_separator = sys.argv[separator_idx + 1:]
        code_fpath = args_after_separator[0]  # Path to the code file
        rendering_dir = args_after_separator[1] if len(args_after_separator) > 1 else None  # Path to save the rendering
        save_blend = args_after_separator[2] if len(args_after_separator) > 2 else None  # Path to save the blend file
    except (ValueError, IndexError):
        raise ValueError("Usage: blender --background [flags] -- code.py [render_dir] [save_blend]")
    
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
    
    # Remove non-printable characters before execution to prevent SyntaxError
    import re
    code = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\uAC00-\uD7A3]', '', code)
    
    try:
        exec(code)
    except:
        raise ValueError

    # Render from camera1
    if 'Camera1' in bpy.data.objects and rendering_dir:
        # Convert rendering_dir to absolute path to ensure correct file location
        rendering_dir = os.path.abspath(rendering_dir)
        os.makedirs(rendering_dir, exist_ok=True)
        bpy.context.scene.camera = bpy.data.objects['Camera1']
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.filepath = os.path.join(rendering_dir, 'render1.png')
        # Use EXEC_DEFAULT explicitly for Blender 5 headless rendering compatibility
        bpy.ops.render.render("EXEC_DEFAULT", write_still=True)

    # Save the blend file
    if save_blend:
        # Convert save_blend to absolute path to ensure correct file location
        save_blend = os.path.abspath(save_blend)
        # Set the save version to 0
        bpy.context.preferences.filepaths.save_version = 0
        # Save the blend file
        bpy.ops.wm.save_as_mainfile(filepath=save_blend)


