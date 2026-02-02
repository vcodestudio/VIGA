"""Blender script for static scene generator with all-camera rendering."""
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

    # Read and execute the code from the specified file
    with open(code_fpath, "r") as f:
        code = f.read()
    
    # Remove non-printable characters before execution to prevent SyntaxError
    # Allow: \x09 (tab), \x0A (newline), \x0D (carriage return), \x20-\x7E (printable ASCII), \uAC00-\uD7A3 (Hangul)
    import re
    code = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\uAC00-\uD7A3]', '', code)
    
    try:
        exec(code)
    except Exception as e:
        import traceback
        print(f"[ERROR] Code execution failed: {e}")
        traceback.print_exc()
        raise ValueError(f"Code execution failed: {e}")

    if not rendering_dir:
        print("[INFO] No rendering directory provided, skipping rendering.")
        exit(0)
    
    # Convert rendering_dir to absolute path to ensure correct file location
    rendering_dir = os.path.abspath(rendering_dir)
    os.makedirs(rendering_dir, exist_ok=True)
    
    # Get render engine from environment variable (default: EEVEE)
    render_engine = os.environ.get('RENDER_ENGINE', 'BLENDER_EEVEE').upper()
    
    # Map common names to Blender engine names
    # Note: Blender 5.0 uses BLENDER_EEVEE, older versions may use BLENDER_EEVEE_NEXT
    engine_map = {
        'EEVEE': 'BLENDER_EEVEE',
        'BLENDER_EEVEE': 'BLENDER_EEVEE',
        'BLENDER_EEVEE_NEXT': 'BLENDER_EEVEE',  # Map to BLENDER_EEVEE for Blender 5.0
        'CYCLES': 'CYCLES',
        'WORKBENCH': 'BLENDER_WORKBENCH',
    }
    render_engine = engine_map.get(render_engine, render_engine)
    
    print(f"[INFO] Using render engine: {render_engine}")
    bpy.context.scene.render.engine = render_engine
    
    if render_engine == 'CYCLES':
        # Enable GPU rendering for Cycles
        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
        bpy.context.preferences.addons['cycles'].preferences.get_devices()
        
        # Check and select the GPUs
        for device in bpy.context.preferences.addons['cycles'].preferences.devices:
            if device.type == 'GPU' and not device.use:
                device.use = True
        
        # Set the rendering device to GPU
        bpy.context.scene.cycles.device = 'GPU'
        # Set samples for Cycles
        bpy.context.scene.cycles.samples = 512
    elif 'EEVEE' in render_engine:
        # EEVEE settings for quality/speed balance (Blender 5.0+)
        # Enable Ambient Occlusion for edge shadows
        try:
            bpy.context.view_layer.use_ao = True  # Enable AO in view layer
        except:
            pass
        try:
            # Create world if it doesn't exist (needed for AO and proper lighting)
            if not bpy.context.scene.world:
                world = bpy.data.worlds.new("World")
                bpy.context.scene.world = world
            # Set AO factor (strength) - higher = more visible shadows at edges
            bpy.context.scene.world.light_settings.ao_factor = 1.0
            bpy.context.scene.world.light_settings.use_ambient_occlusion = True
        except:
            pass
        # Legacy settings for older Blender versions
        try:
            bpy.context.scene.eevee.taa_render_samples = 64  # Anti-aliasing samples
        except:
            pass
        try:
            bpy.context.scene.eevee.use_gtao = True  # Legacy AO setting
        except:
            pass

    # Setting up rendering resolution
    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = 512

    # Set color mode to RGB
    bpy.context.scene.render.image_settings.color_mode = 'RGB'

    # Ensure view layer is updated before rendering
    bpy.context.view_layer.update()
    
    # render from all the camera, save the rendering to the rendering_dir
    for camera in bpy.data.objects:
        if camera.type == 'CAMERA':
            bpy.context.scene.camera = camera
            bpy.context.scene.render.image_settings.file_format = 'PNG'
            bpy.context.scene.render.filepath = os.path.join(rendering_dir, f'{camera.name}.png')
            # Use EXEC_DEFAULT explicitly for Blender 5 headless rendering compatibility
            # This prevents hangs/freezes in background mode
            bpy.ops.render.render("EXEC_DEFAULT", write_still=True)

    # Save the blend file
    if save_blend:
        # Convert save_blend to absolute path to ensure correct file location
        save_blend = os.path.abspath(save_blend)
        # Set the save version to 0
        bpy.context.preferences.filepaths.save_version = 0
        # Save the blend file
        bpy.ops.wm.save_as_mainfile(filepath=save_blend)


