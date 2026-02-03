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
    
    import math
    
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
    
    # Get render engine and effect from environment variables
    render_engine = os.environ.get('RENDER_ENGINE', 'BLENDER_EEVEE').upper()
    render_effect = os.environ.get('RENDER_EFFECT', 'none').lower()
    
    # Map common names to Blender engine names
    # Blender 5.0.1: EEVEE is 'BLENDER_EEVEE' (EEVEE Next is now the default EEVEE)
    engine_map = {
        'EEVEE': 'BLENDER_EEVEE',
        'BLENDER_EEVEE': 'BLENDER_EEVEE',
        'BLENDER_EEVEE_NEXT': 'BLENDER_EEVEE',  # Legacy name, maps to BLENDER_EEVEE
        'CYCLES': 'CYCLES',
        'WORKBENCH': 'BLENDER_WORKBENCH',
        'SOLID': 'BLENDER_WORKBENCH',
        'OUTLINE': 'BLENDER_WORKBENCH', # Add OUTLINE as alias for Workbench with outline
    }
    render_engine = engine_map.get(render_engine, render_engine)
    
    print(f"[INFO] Using render engine: {render_engine}")
    print(f"[INFO] Using render effect: {render_effect}")
    # print(f"[DEBUG] Freestyle condition: effect={render_effect}, engine={render_engine}")
    bpy.context.scene.render.engine = render_engine
    
    if render_engine == 'BLENDER_WORKBENCH':
        # Solid View (Workbench) settings
        bpy.context.scene.display.shading.light = 'STUDIO'
        bpy.context.scene.display.shading.color_type = 'MATERIAL'
        bpy.context.scene.display.shading.show_shadows = True
        bpy.context.scene.display.shading.show_cavity = True
        # Set to Flat/Solid-like look
        bpy.context.scene.display.shading.render_pass = 'COMBINED'
        
        # Enable Outline for Workbench
        bpy.context.scene.display.shading.show_object_outline = True
        bpy.context.scene.display.shading.object_outline_color = (0, 0, 0)
        print("[INFO] Workbench outline enabled")
    
    if render_engine == 'CYCLES':
        # Force CPU rendering for Cycles
        bpy.context.scene.cycles.device = 'CPU'
        print("[INFO] Using CPU for Cycles as requested")

        # Minimal settings for Cycles (Fastest)
        bpy.context.scene.cycles.samples = 32  # Very low samples
        bpy.context.scene.cycles.use_denoising = True
        bpy.context.scene.cycles.max_bounces = 2
        bpy.context.scene.cycles.diffuse_bounces = 1
        bpy.context.scene.cycles.glossy_bounces = 1
        bpy.context.scene.cycles.transparent_max_bounces = 2
        bpy.context.scene.cycles.transmission_bounces = 1
        
        # Set world background to light gray for visibility (prevents black renders)
        world = bpy.context.scene.world
        if world is None:
            world = bpy.data.worlds.new("World")
            bpy.context.scene.world = world
        world.use_nodes = True
        bg_node = world.node_tree.nodes.get("Background")
        if bg_node:
            bg_node.inputs[0].default_value = (0.8, 0.8, 0.8, 1)  # Light gray
            bg_node.inputs[1].default_value = 1.0  # Strength
        print("[INFO] Set world background to light gray for Cycles")
        
    # --- Freestyle Effect Setup (Render Pass + Compositor) ---
    if render_effect == 'freestyle' and render_engine in ['CYCLES', 'BLENDER_EEVEE']:
        # print("[DEBUG] Enabling Freestyle setup...")
        # Enable Freestyle
        bpy.context.scene.render.use_freestyle = True
        bpy.context.view_layer.use_freestyle = True
        
        # Set as separate render pass
        freestyle_settings = bpy.context.view_layer.freestyle_settings
        freestyle_settings.as_render_pass = True
        
        # Configure Line Set
        if len(freestyle_settings.linesets) == 0:
            lineset = freestyle_settings.linesets.new("OutlineSet")
        else:
            lineset = freestyle_settings.linesets[0]
            
        lineset.select_silhouette = True
        lineset.select_border = True
        lineset.select_crease = True
        lineset.select_contour = True
        lineset.select_edge_mark = False
        
        # Configure Line Style (create if not exists)
        if lineset.linestyle is None:
            # Create a new linestyle
            new_linestyle = bpy.data.linestyles.new("OutlineStyle")
            lineset.linestyle = new_linestyle
        
        lineset.linestyle.thickness = 2.0
        lineset.linestyle.color = (0, 0, 0) # Black
        
        # --- Workbench Outline Alternative ---
        if render_engine == 'BLENDER_WORKBENCH':
            bpy.context.scene.display.shading.show_object_outline = True
            bpy.context.scene.display.shading.object_outline_color = (0, 0, 0)
            print("[INFO] Workbench outline enabled")
        # -------------------------------------
        
        # Setup Compositor Nodes
        bpy.context.scene.use_nodes = True
        
        # In Blender 5.0+, compositor nodes are in scene.node_tree or scene.compositing_node_group
        # We need to ensure the tree is initialized
        tree = None
        
        # Check if it's a world or scene node tree
        if hasattr(bpy.context.scene, 'node_tree'):
            tree = bpy.context.scene.node_tree
        
        if not tree:
            # Try to force initialization
            try:
                bpy.ops.node.new_node_tree(type='CompositorNodeTree', name="Compositing")
            except:
                pass
            
            if hasattr(bpy.context.scene, 'node_tree'):
                tree = bpy.context.scene.node_tree
            elif hasattr(bpy.context.scene, 'compositing_node_group'):
                tree = bpy.context.scene.compositing_node_group
        
        if not tree:
            # Try to find any compositor node tree in bpy.data
            for nt in bpy.data.node_groups:
                if nt.type == 'COMPOSITOR':
                    tree = nt
                    break
        
        if not tree:
            # Create one manually in bpy.data
            tree = bpy.data.node_groups.new("Compositing", "CompositorNodeTree")
            # Try to assign it if possible
            if hasattr(bpy.context.scene, 'node_tree'):
                bpy.context.scene.node_tree = tree
            elif hasattr(bpy.context.scene, 'compositing_node_group'):
                bpy.context.scene.compositing_node_group = tree
            
        if tree:
            # Clear existing nodes
            for node in tree.nodes:
                tree.nodes.remove(node)
                
            # Create nodes
            render_layers = tree.nodes.new('CompositorNodeRLayers')
            render_layers.location = (0, 0)
            
            alpha_over = tree.nodes.new('CompositorNodeAlphaOver')
            alpha_over.location = (300, 0)
            
            # In Blender 5.0+, CompositorNodeComposite might be missing or renamed.
            try:
                composite = tree.nodes.new('CompositorNodeComposite')
            except:
                # If it fails, try to find any output node
                print("[DEBUG] CompositorNodeComposite failed, trying OutputFile...")
                composite = tree.nodes.new('CompositorNodeOutputFile')
                composite.file_name = "Camera_Freestyle"
                # In some versions, composite.format might be restricted
                try:
                    composite.format.file_format = 'PNG'
                except:
                    pass
                
                if hasattr(composite, 'file_slots') and len(composite.file_slots) > 0:
                    composite.file_slots[0].path = "Camera_Freestyle"
                elif hasattr(composite, 'file_output_items'):
                    if len(composite.file_output_items) == 0:
                        try:
                            composite.file_output_items.new("RGBA", name="Image")
                        except:
                            composite.file_output_items.new("IMAGE", name="Image")
                    
                    item = composite.file_output_items[0]
                    if hasattr(item, 'path'):
                        item.path = "Camera_Freestyle"
                    elif hasattr(item, 'name'):
                        item.name = "Camera_Freestyle"
                    
                    try:
                        item.format.file_format = 'PNG'
                    except:
                        pass
            
            composite.location = (600, 0)
            
            # Connect nodes
            # Combined Image -> Alpha Over (Bottom)
            tree.links.new(render_layers.outputs['Image'], alpha_over.inputs[1])
            # Freestyle Pass -> Alpha Over (Top)
            
            # Re-fetch render layers to update outputs
            if 'Freestyle' in render_layers.outputs:
                tree.links.new(render_layers.outputs['Freestyle'], alpha_over.inputs[2])
            else:
                print("[DEBUG] 'Freestyle' output still not found in Render Layers. Available outputs:", [o.name for o in render_layers.outputs])
                # Try to find by name case-insensitive
                found = False
                for out in render_layers.outputs:
                    if 'freestyle' in out.name.lower():
                        tree.links.new(out, alpha_over.inputs[2])
                        found = True
                        print(f"[DEBUG] Found freestyle pass by name: {out.name}")
                        break
                if not found:
                    print("[DEBUG] Could not find any freestyle-like output.")
            
            # Alpha Over -> Composite
            if composite.type == 'COMPOSITE':
                tree.links.new(alpha_over.outputs['Image'], composite.inputs['Image'])
            else:
                # Fallback for OutputFile, Viewer or other nodes
                tree.links.new(alpha_over.outputs['Image'], composite.inputs[0])
            
            print("[INFO] Freestyle render pass + compositor enabled")
        else:
            print("[ERROR] Could not initialize compositor node tree")
    # ---------------------------------------------------------

    elif 'EEVEE' in render_engine:
        # Legacy settings for older Blender versions
        try:
            bpy.context.scene.eevee.taa_render_samples = 64  # Anti-aliasing samples
        except:
            pass

    # Setting up rendering resolution
    # Match aspect ratio to target image if available
    target_image_path = os.environ.get('TARGET_IMAGE_PATH')
    base_resolution = 1024  # Base resolution for the longer side
    
    if target_image_path and os.path.exists(target_image_path):
        try:
            # Use Blender's built-in image loading instead of PIL
            # This works in Blender's Python environment
            tmp_img = bpy.data.images.load(target_image_path)
            target_width, target_height = tmp_img.size[0], tmp_img.size[1]
            bpy.data.images.remove(tmp_img)  # Clean up
            
            aspect_ratio = target_width / target_height
            
            if aspect_ratio >= 1:  # Landscape or square
                render_width = base_resolution
                render_height = int(base_resolution / aspect_ratio)
            else:  # Portrait
                render_height = base_resolution
                render_width = int(base_resolution * aspect_ratio)
            
            print(f"[INFO] Target image: {target_width}x{target_height}, aspect ratio: {aspect_ratio:.3f}")
            print(f"[INFO] Render resolution: {render_width}x{render_height}")
        except Exception as e:
            print(f"[WARN] Failed to read target image for aspect ratio: {e}")
            render_width = base_resolution
            render_height = base_resolution
    else:
        render_width = base_resolution
        render_height = base_resolution
        print(f"[INFO] No target image, using default {render_width}x{render_height}")
    
    bpy.context.scene.render.resolution_x = render_width
    bpy.context.scene.render.resolution_y = render_height

    # Set color mode to RGB
    bpy.context.scene.render.image_settings.color_mode = 'RGB'

    # Ensure view layer is updated before rendering
    bpy.context.view_layer.update()
    
    # render from all the camera, save the rendering to the rendering_dir
    cameras = [obj for obj in bpy.data.objects if obj.type == 'CAMERA']
    print(f"[INFO] Found {len(cameras)} camera(s): {[c.name for c in cameras]}")
    
    for camera in cameras:
        bpy.context.scene.camera = camera
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.filepath = os.path.join(rendering_dir, f'{camera.name}.png')
        print(f"[INFO] Rendering from camera {camera.name} to {bpy.context.scene.render.filepath}")
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
