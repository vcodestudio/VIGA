# blender_server.py
import math
import os
import sys
from pathlib import Path
import logging
from mcp.server.fastmcp import FastMCP
import json
import traceback
from typing import Optional, Dict, Any

# Import Executor from exec_blender
try:
    from .exec_blender import Executor
except ImportError:
    # If running as standalone
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from exec_blender import Executor

# tool config for agent (only the function w/ @mcp.tools)
tool_configs = [
    {
        "type": "function",
        "function": {
            "name": "initialize_viewpoint",
            "description": "Adds a viewpoint to observe the listed objects. The viewpoints are added to the four corners of the bounding box of the listed objects. This tool returns the positions and rotations of the four viewpoint cameras, as well as the rendered images of the four cameras. You can use these information to set the camera to a good initial position and orientation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "object_names": {
                        "type": "array", 
                        "description": "The names of the objects to observe. Objects must exist in the scene (you can check the scene information to see if they exist). If you want to observe the whole scene, you can pass an empty list.",
                        "items": {
                            "type": "string",
                            "description": "The name of the object to observe."
                        }
                    }
                },
                "required": ["object_names"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_camera",
            "description": "Set the current active camera to the given location and rotation",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "array", 
                        "description": "The location of the camera (in world coordinates)",
                        "items": {
                            "type": "number",
                            "description": "The location of the camera (in world coordinates)"
                        }
                    },
                    "rotation_euler": {
                        "type": "array", 
                        "description": "The rotation of the camera (in euler angles)",
                        "items": {
                            "type": "number",
                            "description": "The rotation of the camera (in euler angles)"
                        }
                    }
                },
                "required": ["location", "rotation_euler"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "investigate",
            "description": "Investigate the scene by the current camera. You can zoom, move, and focus on the object you want to investigate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "choices": ["zoom", "move", "focus"], "description": "The operation to perform."},
                    "object_name": {"type": "string", "description": "If the operation is focus, you need to provide the name of the object to focus on. The object must exist in the scene."},
                    "direction": {"type": "string", "choices": ["up", "down", "left", "right", "in", "out"], "description": "If the operation is move or zoom, you need to provide the direction to move or zoom."}
                },
                "required": ["operation"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_object_visibility",
            "description": "Set the visibility of the objects in the scene. You can decide to show or hide the objects. You do not need to mention all the objects here, the objects you do not metioned will keep their original visibility.",
            "parameters": {
                "type": "object",
                "properties": {
                    "show_object_list": {
                        "type": "array", 
                        "description": "The names of the objects to show. Objects must exist in the scene.",
                        "items": {
                            "type": "string",
                            "description": "The name of the object to show."
                        }
                    },
                    "hide_object_list": {
                        "type": "array", 
                        "description": "The names of the objects to hide. Objects must exist in the scene.",
                        "items": {
                            "type": "string",
                            "description": "The name of the object to hide."
                        }
                    }
                },
                "required": ["show_object_list", "hide_object_list"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_keyframe",
            "description": "Set the scene to a specific frame number for observation",
            "parameters": {
                "type": "object",
                "properties": {
                    "frame_number": {"type": "integer", "description": "The specific frame number to set the scene to."}
                },
                "required": ["frame_number"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_scene_info",
            "description": "Get the scene information",
        }
    }
]

# Create global MCP instance
mcp = FastMCP("scene-server")

# Global tool instance
_investigator = None

# ======================
# Camera investigator (Fixed: save path first then load)
# ======================

class Investigator3D:
    def __init__(self, save_dir: str, blender_path: str, blender_command: str):
        self.blender_background = blender_path  # Initial blender file path
        self.blender_command = blender_command
        self.base = Path(save_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        
        # Create executor
        self.executor = Executor(
            blender_command=blender_command,
            blender_file=blender_path,
            blender_script="data/dynamic_scene/pipeline_render_script.py",  # Default script
            script_save=str(self.base / "scripts"),
            render_save=str(self.base / "renders"),
            blender_save=str(self.base / "current_scene.blend")
        )
        
        # State variables
        self.target = None
        self.radius = 5.0
        self.theta = 0.0
        self.phi = 0.0
        self.count = 0
        self.scene_info_cache = None

    def _generate_scene_info_script(self) -> str:
        """Generate script to get scene information"""
        return '''import bpy
import json
import sys

# Get scene information
scene_info = {"objects": [], "materials": [], "lights": [], "cameras": []}

for obj in bpy.data.objects:
    if obj.type == 'CAMERA' or obj.type == 'LIGHT':
        continue
    scene_info["objects"].append({
        "name": obj.name, 
        "type": obj.type,
        "location": list(obj.matrix_world.translation),
        "rotation": list(obj.rotation_euler),
        "scale": list(obj.scale),
        "visible": not (obj.hide_viewport or obj.hide_render)
    })

for mat in bpy.data.materials:
    scene_info["materials"].append({
        "name": mat.name,
        "use_nodes": mat.use_nodes,
        "diffuse_color": list(mat.diffuse_color),
    })

for light in [o for o in bpy.data.objects if o.type == 'LIGHT']:
    scene_info["lights"].append({
        "name": light.name,
        "type": light.data.type,
        "energy": light.data.energy,
        "color": list(light.data.color),
        "location": list(light.matrix_world.translation),
        "rotation": list(light.rotation_euler)
    })

for cam in [o for o in bpy.data.objects if o.type == 'CAMERA']:
    scene = bpy.context.scene
    scene_info["cameras"].append({
        "name": cam.name,
        "lens": cam.data.lens,
        "location": list(cam.matrix_world.translation),
        "rotation": list(cam.rotation_euler),
        "is_active": cam == scene.camera,
    })

# Save to file for retrieval
with open("/tmp/scene_info.json", "w") as f:
    json.dump(scene_info, f)

print("Scene info extracted successfully")
'''

    def _generate_render_script(self, additional_operations: str = "") -> str:
        """Generate script to render current scene"""
        return f'''import bpy
import math

{additional_operations}

# Set render engine to cycles
bpy.context.scene.render.engine = 'CYCLES'

# Render the scene (the pipeline script will handle the actual rendering)
print("Render script executed")
'''

    def _generate_camera_focus_script(self, object_name: str) -> str:
        """Generate script to focus camera on object"""
        return f'''import bpy
import math

# Get target object
target_obj = bpy.data.objects.get('{object_name}')
if not target_obj:
    raise ValueError(f"Object '{object_name}' not found")

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

# Calculate camera position relative to target
target_pos = target_obj.matrix_world.translation
camera_pos = camera.matrix_world.translation
distance = (camera_pos - target_pos).length

# Set up track-to constraint
constraint = None
for c in camera.constraints:
    if c.type == 'TRACK_TO':
        constraint = c
        break

if not constraint:
    constraint = camera.constraints.new('TRACK_TO')

constraint.target = target_obj
constraint.track_axis = 'TRACK_NEGATIVE_Z'
constraint.up_axis = 'UP_Y'

print(f"Camera focused on object '{object_name}'")
'''

    def _generate_camera_set_script(self, location: list, rotation_euler: list) -> str:
        """Generate script to set camera position and rotation"""
        return f'''import bpy

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
camera.location = {location}
camera.rotation_euler = {rotation_euler}

print(f"Camera set to location {location} and rotation {rotation_euler}")
'''

    def _generate_visibility_script(self, show_objects: list, hide_objects: list) -> str:
        """Generate script to set object visibility"""
        return f'''import bpy

show_list = {show_objects}
hide_list = {hide_objects}

# Apply visibility changes
for obj in bpy.data.objects:
    if obj.name in hide_list:
        obj.hide_viewport = True
        obj.hide_render = True
    if obj.name in show_list:
        obj.hide_viewport = False
        obj.hide_render = False

print("Visibility updated: show", show_list, ", hide", hide_list)
'''

    def _generate_camera_move_script(self, target_obj_name: str, radius: float, theta: float, phi: float) -> str:
        """Generate script to move camera around target object"""
        return f'''import bpy
import math

# Get target object
target_obj = bpy.data.objects.get('{target_obj_name}')
if not target_obj:
    raise ValueError(f"Target object '{target_obj_name}' not found")

# Get camera
camera = bpy.context.scene.camera
if not camera:
    cameras = [obj for obj in bpy.data.objects if obj.type == 'CAMERA']
    if cameras:
        camera = cameras[0]
        bpy.context.scene.camera = camera

# Calculate new camera position
target_pos = target_obj.matrix_world.translation
x = {radius} * math.cos({phi}) * math.cos({theta})
y = {radius} * math.cos({phi}) * math.sin({theta})
z = {radius} * math.sin({phi})

new_pos = (target_pos.x + x, target_pos.y + y, target_pos.z + z)
camera.matrix_world.translation = new_pos

print("Camera moved to position:", new_pos)
'''

    def _generate_keyframe_script(self, frame_number: int) -> str:
        """Generate script to set frame number"""
        return f'''import bpy

scene = bpy.context.scene
current_frame = scene.frame_current

# Ensure frame number is within valid range
target_frame = max(scene.frame_start, min(scene.frame_end, {frame_number}))
scene.frame_set(target_frame)

print("Changed to frame", target_frame, "(was", current_frame, ")")
'''

    def _generate_viewpoint_script(self, object_names: list) -> str:
        """Generate script to initialize viewpoints around objects"""
        return f'''import bpy
import math
from mathutils import Vector

object_names = {object_names}
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

# Set up viewpoints (the actual rendering will be handled by the pipeline)
for i, pos in enumerate(camera_positions):
    camera.location = pos
    camera.rotation_euler = (math.radians(60), 0, math.radians(45))
    
    # Look at center
    direction = Vector((center_x, center_y, center_z)) - camera.location
    camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

# Restore original position
camera.location = original_location
camera.rotation_euler = original_rotation

print("Viewpoints initialized for", len(objects), "objects")
'''

    def _execute_script(self, script_code: str, description: str = "") -> dict:
        """Execute a blender script and return results"""
        try:
            result = self.executor.execute(
                thought=description,
                code_edition="",
                full_code=script_code
            )
            
            # Update blender_background to the saved blend file
            if result.get("status") == "success":
                self.blender_background = str(self.base / "current_scene.blend")
                self.executor.blender_file = self.blender_background
                
            return result
        except Exception as e:
            logging.error(f"Script execution failed: {e}")
            return {"status": "error", "output": {"text": [str(e)]}}

    def _render(self):
        """Render current scene and return image path and camera parameters"""
        render_script = self._generate_render_script()
        result = self._execute_script(render_script, "Render current scene")
        
        if result.get("status") == "success":
            self.count += 1
            # The executor handles the actual rendering and returns image paths
            images = result.get("output", {}).get("image", [])
            if images:
                return {
                    "image_path": images[0] if isinstance(images, list) else images,
                    "camera_parameters": "Camera parameters extracted from render"
                }
        return {"image_path": None, "camera_parameters": "Render failed"}
    
    def get_info(self) -> dict:
        """Get scene information by executing a script"""
        try:
            # Use cached info if available
            if self.scene_info_cache:
                return self.scene_info_cache
                
            script = self._generate_scene_info_script()
            result = self._execute_script(script, "Extract scene information")
            
            # Try to read the scene info from the temporary file
            try:
                import json
                if os.path.exists("/tmp/scene_info.json"):
                    with open("/tmp/scene_info.json", "r") as f:
                        scene_info = json.load(f)
                        self.scene_info_cache = scene_info
                        return scene_info
            except Exception:
                pass
                
            # Fallback: return empty info if file reading fails
            return {"objects": [], "materials": [], "lights": [], "cameras": []}
        except Exception as e:
            logging.error(f"scene info error: {e}")
            return {}

    def focus_on_object(self, object_name: str) -> dict:
        """Focus camera on a specific object"""
        self.target = object_name  # Store object name instead of object reference
        
        # Generate and execute focus script
        focus_script = self._generate_camera_focus_script(object_name)
        result = self._execute_script(focus_script, f"Focus camera on object {object_name}")
        
        if result.get("status") == "success":
            # For now, set default values for radius, theta, phi
            # In a more complete implementation, we'd extract these from the script execution
            self.radius = 5.0
            self.theta = 0.0
            self.phi = 0.0
            return self._render()
        else:
            raise ValueError(f"Failed to focus on object {object_name}")

    def zoom(self, direction: str) -> dict:
        """Zoom camera in or out"""
        if direction == 'in':
            self.radius = max(1, self.radius-3)
        elif direction == 'out':
            self.radius += 3
        return self._update_and_render()

    def move_camera(self, direction: str) -> dict:
        """Move camera around target object"""
        if not self.target:
            raise ValueError("No target object set. Call focus first.")
            
        step = self.radius
        theta_step = step / (self.radius*math.cos(self.phi)) if math.cos(self.phi) != 0 else 0.1
        phi_step = step / self.radius
        
        if direction=='up': 
            self.phi = min(math.pi/2-0.1, self.phi+phi_step)
        elif direction=='down': 
            self.phi = max(-math.pi/2+0.1, self.phi-phi_step)
        elif direction=='left': 
            self.theta -= theta_step
        elif direction=='right': 
            self.theta += theta_step
            
        return self._update_and_render()

    def _update_and_render(self) -> dict:
        """Update camera position and render"""
        if not self.target:
            return self._render()
            
        # Generate script to move camera
        move_script = self._generate_camera_move_script(self.target, self.radius, self.theta, self.phi)
        result = self._execute_script(move_script, f"Move camera around {self.target}")
        
        if result.get("status") == "success":
            return self._render()
        else:
            return {"image_path": None, "camera_parameters": "Camera move failed"}

    def set_camera(self, location: list, rotation_euler: list) -> dict:
        """Set camera position and rotation"""
        script = self._generate_camera_set_script(location, rotation_euler)
        return self._execute_script(script, f"Set camera to location {location} and rotation {rotation_euler}")

    def initialize_viewpoint(self, object_names: list) -> dict:
        """Initialize viewpoints around specified objects"""
        try:
            script = self._generate_viewpoint_script(object_names)
            result = self._execute_script(script, f"Initialize viewpoints for objects: {object_names}")
            
            if result.get("status") == "success":
                # For now, return a simplified response
                # The actual viewpoint rendering would need to be handled differently
                return {
                    'status': 'success',
                    'output': {
                        'image': [],
                        'text': [f"Viewpoints initialized for objects: {object_names}"]
                    }
                }
            else:
                return result
        except Exception as e:
            return {'status': 'error', 'output': {'text': [str(e)]}}

    def set_keyframe(self, frame_number: int) -> dict:
        """Set scene to a specific frame"""
        try:
            script = self._generate_keyframe_script(frame_number)
            result = self._execute_script(script, f"Set frame to {frame_number}")
            
            if result.get("status") == "success":
                render_result = self._render()
                return {
                    'status': 'success',
                    'output': {'image': [render_result['image_path']], 'text': [f"Camera parameters: {render_result['camera_parameters']}"]}
                }
            else:
                return result
        except Exception as e:
            return {'status': 'error', 'output': {'text': [str(e)]}}

@mcp.tool()
def initialize(args: dict) -> dict:
    """
    Initialize 3D scene investigation tool.
    """
    global _investigator
    try:
        save_dir = args.get("output_dir") + "/investigator/"
        _investigator = Investigator3D(save_dir, str(args.get("blender_file")), str(args.get("blender_command")))
        return {"status": "success", "output": {"text": ["Investigator3D initialized successfully"], "tool_configs": tool_configs}}
    except Exception as e:
        return {"status": "error", "output": {"text": [str(e)]}}

def focus(object_name: str) -> dict:
    global _investigator
    if _investigator is None:
        return {"status": "error", "output": {"text": ["Investigator3D not initialized. Call initialize_investigator first."]}}

    try:
        # Object existence check is now handled in the script execution
        result = _investigator.focus_on_object(object_name)
        return {
            "status": "success", 
            "output": {"image": [result["image_path"]], "text": [f"Camera parameters: {result['camera_parameters']}"]}
        }
    except Exception as e:
        logging.error(f"Focus failed: {e}")
        return {"status": "error", "output": {"text": [str(e)]}}

def zoom(direction: str) -> dict:
    global _investigator
    if _investigator is None:
        return {"status": "error", "output": {"text": ["Investigator3D not initialized. Call initialize_investigator first."]}}

    try:
        # Check if there is a target object
        if _investigator.target is None:
            return {"status": "error", "output": {"text": ["No target object set. Call focus first."]}}

        result = _investigator.zoom(direction)
        return {
            "status": "success", 
            "output": {'image': [result["image_path"]], 'text': [f"Camera parameters: {result['camera_parameters']}"]}
        }
    except Exception as e:
        logging.error(f"Zoom failed: {e}")
        return {"status": "error", "output": {"text": [str(e)]}}

def move(direction: str) -> dict:
    global _investigator
    if _investigator is None:
        return {"status": "error", "output": {"text": ["Investigator3D not initialized. Call initialize_investigator first."]}}

    try:
        # Check if there is a target object
        if _investigator.target is None:
            return {"status": "error", "output": {"text": ["No target object set. Call focus first."]}}

        result = _investigator.move_camera(direction)
        return {
            "status": "success", 
            "output": {'image': [result["image_path"]], 'text': [f"Camera parameters: {result['camera_parameters']}"]}
        }
    except Exception as e:
        logging.error(f"Move failed: {e}")
        return {"status": "error", "output": {"text": [str(e)]}}

@mcp.tool()
def get_scene_info() -> dict:
    try:
        global _investigator
        if _investigator is None:
            return {"status": "error", "output": {"text": ["SceneInfo not initialized. Call initialize first."]}}
        info = _investigator.get_info()
        return {"status": "success", "output": {"text": [str(info)]}}
    except Exception as e:
        logging.error(f"Failed to get scene info: {e}")
        return {"status": "error", "output": {"text": [str(e)]}}

@mcp.tool()
def initialize_viewpoint(object_names: list) -> dict:
    global _investigator
    if _investigator is None:
        return {"status": "error", "output": {"text": ["Investigator3D not initialized. Call initialize_investigator first."]}}
    try:
        result = _investigator.initialize_viewpoint(object_names)
        return result
    except Exception as e:
        logging.error(f"Add viewpoint failed: {e}")
        return {"status": "error", "output": {"text": [str(e)]}}

@mcp.tool()
def investigate(operation: str, object_name: str = None, direction: str = None) -> dict:
    if operation == "focus":
        if not object_name:
            return {"status": "error", "output": {"text": ["object_name is required for focus"]}}
        return focus(object_name=object_name)
    elif operation == "zoom":
        if direction not in ("in", "out"):
            return {"status": "error", "output": {"text": ["direction must be 'in' or 'out' for zoom"]}}
        return zoom(direction=direction)
    elif operation == "move":
        if direction not in ("up", "down", "left", "right"):
            return {"status": "error", "output": {"text": ["direction must be one of up/down/left/right for move"]}}
        return move(direction=direction)
    else:
        return {"status": "error", "output": {"text": [f"Unknown operation: {operation}"]}}

@mcp.tool()
def set_object_visibility(show_object_list: list = None, hide_object_list: list = None) -> dict:
    global _investigator
    if _investigator is None:
        return {"status": "error", "output": {"text": ["Investigator3D not initialized. Call initialize_investigator first."]}}
    try:
        show_object_list = show_object_list or []
        hide_object_list = hide_object_list or []
        
        # Generate and execute visibility script
        script = _investigator._generate_visibility_script(show_object_list, hide_object_list)
        result = _investigator._execute_script(script, f"Set visibility: show {show_object_list}, hide {hide_object_list}")
        
        if result.get("status") == "success":
            render_result = _investigator._render()
            return {"status": "success", "output": {'image': [render_result["image_path"]], 'text': [f"Camera parameters: {render_result['camera_parameters']}"]}}
        else:
            return result
    except Exception as e:
        logging.error(f"set_object_visibility failed: {e}")
        return {"status": "error", "output": {"text": [str(e)]}}

@mcp.tool()
def set_keyframe(frame_number: int) -> dict:
    global _investigator
    if _investigator is None:
        return {"status": "error", "output": {"text": ["Investigator3D not initialized. Call initialize_investigator first."]}}
    try:
        return _investigator.set_keyframe(frame_number)
    except Exception as e:
        logging.error(f"Set keyframe failed: {e}")
        return {"status": "error", "output": {"text": [str(e)]}, "status": "error"}
    
@mcp.tool()
def set_camera(location: list, rotation_euler: list) -> dict:
    global _investigator
    if _investigator is None:
        return {"status": "error", "output": {"text": ["Investigator3D not initialized. Call initialize_investigator first."]}}
    try:
        _investigator.set_camera(location, rotation_euler)
        return {"status": "success", "output": {"text": ["Successfully set the camera with the given location and rotation"]}}
    except Exception as e:
        logging.error(f"set_camera failed: {e}")
        return {"status": "error", "output": {"text": [str(e)]}}
    
@mcp.tool()
def reload_scene(script_path: str) -> dict:
    global _investigator
    if _investigator is None:
        return {"status": "error", "output": {"text": ["Investigator3D not initialized. Call initialize first."]}}

    if script_path:
        if not os.path.exists(script_path):
            return {"status": "error", "output": {"text": [f"script not found: {script_path}"]}}
        with open(script_path, "r", encoding="utf-8") as f:
            script_code = f.read()
    else:
        return {"status": "error", "output": {"text": ["script_path is required"]}}

    try:
        # Reload the blender_background file
        _investigator.blender_background = str(_investigator.executor.blender_file)
        _investigator.executor.blender_file = _investigator.blender_background
        
        # Execute the script using the executor
        result = _investigator.executor.execute(
            thought="Reload scene with new script",
            code_edition="",
            full_code=script_code
        )
        
        if result.get("status") == "success":
            # Update blender_background to the new save file
            _investigator.blender_background = str(_investigator.base / "current_scene.blend")
            _investigator.executor.blender_file = _investigator.blender_background
            return {"status": "success", "output": {"text": ["Scene reloaded successfully"]}}
        else:
            return result
            
    except Exception as e:
        tb = traceback.format_exc(limit=12)
        return {"status": "error", "output": {"text": [repr(e), tb]}}

# ======================
# Entry point and testing
# ======================

def main():
    # Check if script is run directly (for testing)
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("Running investigator tools test...")
        test_tools()
    else:
        # Run MCP server normally
        mcp.run()

def test_tools():
    """Test all investigator tool functions (read environment variable configuration)"""
    print("=" * 50)
    print("Testing Scene Tools")
    print("=" * 50)

    # Set test paths (read from environment variables)
    blender_file = os.getenv("BLENDER_FILE", "output/static_scene/20251018_012341/christmas1/blender_file.blend")
    test_save_dir = os.getenv("THOUGHT_SAVE", "output/test/investigator/")
    script_path = os.getenv("SCRIPT_PATH", "output/static_scene/20251018_012341/christmas1/scripts/34.py")

    # Check if blender file exists
    if not os.path.exists(blender_file):
        print(f"⚠ Blender file not found: {blender_file}")
        print("Skipping all tests.")
        return

    print(f"✓ Using blender file: {blender_file}")

    # Test 1: Initialize investigation tool
    print("\n1. Testing initialize...")
    args = {"output_dir": test_save_dir, "blender_file": blender_file}
    result = initialize(args)
    print(f"Result: {result}")
        
    # Test 2: Get scene info
    print("\n2. Testing get_scene_info...")
    scene_info_result = get_scene_info()
    print(f"Result: {scene_info_result}")
    
    # Extract object list from scene info for later tests
    object_names = []
    if scene_info_result.get("status") == "success":
        scene_info_text = scene_info_result.get("output", {}).get("text", [])
        if scene_info_text:
            import json
            try:
                scene_info = json.loads(scene_info_text[0])
                object_names = [obj["name"] for obj in scene_info.get("objects", [])]
                print(f"Found {len(object_names)} objects: {object_names}")
            except (json.JSONDecodeError, KeyError, TypeError):
                print("Could not parse scene info, using empty object list")
    
    # Test 3: Reload scene
    print("\n3. Testing reload_scene...")
    reload_result = reload_scene(script_path=script_path)
    print(f"Result: {reload_result}")
        
    # Test 4: Initialize viewpoint
    print("\n4. Testing initialize_viewpoint...")
    viewpoint_result = initialize_viewpoint(object_names=object_names)
    print(f"Result: {viewpoint_result}")

    # Test 5: Focus, zoom, move, set_keyframe if objects exist
    if object_names:
        first_object = object_names[0]
        print(f"\n5. Testing camera operations with object: {first_object}")
        
        # Test focus
        print("\n5.1. Testing focus...")
        focus_result = focus(object_name=first_object)
        print(f"Result: {focus_result}")

        # Test zoom
        print("\n5.2. Testing zoom...")
        zoom_result = zoom(direction="in")
        print(f"Result: {zoom_result}")

        # Test move
        print("\n5.3. Testing move...")
        move_result = move(direction="up")
        print(f"Result: {move_result}")

        # Test set_keyframe
        print("\n5.4. Testing set_keyframe...")
        keyframe_result = set_keyframe(frame_number=1)
        print(f"Result: {keyframe_result}")
    else:
        print("\n5. Skipping camera operations - no objects found in scene")

    print("\n" + "=" * 50)
    print("Test completed!")
    print("=" * 50)
    print(f"\nTest files saved to: {test_save_dir}")
    print("\nTo run the MCP server normally:")
    print("python tools/investigator.py")
    print("\nTo run tests:")
    print("BLENDER_FILE=/path/to.blend THOUGHT_SAVE=output/test/scene_test python tools/investigator.py --test")


if __name__ == "__main__":
    main()