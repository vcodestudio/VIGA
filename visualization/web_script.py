#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
web_script.py
Create a web interface to display trajectory steps with thought, code, and rendered images.
Features:
- Character-by-character typing for thought (like ChatGPT)
- Scrollable code area with syntax highlighting
- Rendered image display (with potential for 3D Blender file interaction)
- Continue button to proceed to next step
- Real-time appearance even though loading existing trajectory

Note on 3D Blender File Interaction:
To enable interactive 3D viewing of Blender files (move mouse to change camera):
1. Convert .blend files to glTF/glb format (requires Blender Python API)
2. Serve the converted files via the /api/blend endpoint
3. Load them in the frontend using three.js GLTFLoader
4. Use OrbitControls for camera interaction

For now, the rendered images are displayed. The structure is in place for future 3D support.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from flask import Flask, render_template, jsonify, send_from_directory, request

# Set template folder explicitly
template_dir = os.path.join(os.path.dirname(__file__), 'templates')
app = Flask(__name__, template_folder=template_dir)

# Optional CORS support
try:
    from flask_cors import CORS
    CORS(app)
except ImportError:
    # CORS not available, but not critical for local development
    pass

# Global state
STEPS_DATA: List[Dict] = []  # Current active trajectory
BASE_PATH: str = ""
RENDERS_DIR: Path = None
IMAGE_PATH: str = ""
VIDEO_PATH: str = ""

# Scene to trajectory path mapping
SCENE_TRAJECTORY_MAP = {
    "goldengate8": "output/static_scene/demo/20251030_033307/goldengate8",
    "christmas1": "output/static_scene/demo/20251028_133713/christmas1",
    "restroom5": "output/static_scene/demo/20251017_133317/restroom5",
    "whitehouse9": "output/static_scene/demo/20251030_121357/whitehouse9",
    "house11": "output/static_scene/demo/20251030_121641/house11",
    "cake15": "output/static_scene/demo/20251031_084509/cake15",
    "bathroom20": "output/static_scene/demo/20251030_121643/bathroom20",
    "glass24": "output/static_scene/demo/20251030_121642/glass24",
    "blueroom26": "output/static_scene/demo/20251205_133154/blueroom",
    "bedroom32": "output/static_scene/demo/20251214_043022/bedroom32"
}

# Preloaded trajectories for all scenes
PRELOADED_TRAJECTORIES: Dict[str, Dict] = {}


def parse_trajectory(traj_path: str, animation: bool = False, fix_camera: bool = False, return_data: bool = False):
    """Parse trajectory file and extract steps, same logic as video_script.py
    
    Args:
        traj_path: Path to trajectory file
        animation: Whether this is an animation trajectory
        fix_camera: Whether to use fixed camera renders
        return_data: If True, return parsed data instead of modifying global STEPS_DATA
    
    Returns:
        If return_data is True, returns dict with steps_data, renders_dir, image_path, video_path
        Otherwise, modifies global STEPS_DATA and returns None
    """
    global STEPS_DATA, RENDERS_DIR, IMAGE_PATH, VIDEO_PATH
    
    traj = json.loads(Path(traj_path).read_text(encoding="utf-8"))
    renders_dir = Path(traj_path).parent / "renders"
    
    if return_data:
        steps_data = []
        image_path = ""
        video_path = ""
    else:
        STEPS_DATA = []
        RENDERS_DIR = renders_dir
        IMAGE_PATH = ""
        VIDEO_PATH = ""
        steps_data = STEPS_DATA
    
    if fix_camera:
        image_path_val = os.path.dirname(traj_path) + '/video/renders'
        if return_data:
            image_path = image_path_val
        else:
            IMAGE_PATH = image_path_val
    if animation:
        video_path_val = os.path.dirname(traj_path) + '/video/renders'
        if return_data:
            video_path = video_path_val
        else:
            VIDEO_PATH = video_path_val
    
    code_count = 0
    count = 0
    
    for i, complete_step in enumerate(traj, start=1):
        if 'tool_calls' not in complete_step:
            continue
        tool_call = complete_step['tool_calls'][0]
        if tool_call['function']['name'] == "execute_and_evaluate" or tool_call['function']['name'] == "get_scene_info":
            code_count += 1
        if tool_call['function']['name'] != "execute_and_evaluate":
            continue
        
        step = json.loads(tool_call['function']['arguments'])
        thought = step.get("thought", "").strip()
        if "code" in step:
            code = step.get("code", "").strip()
        else:
            code = step.get("full_code", "").strip()
        
        if i+1 >= len(traj):
            continue
        
        step_data = {
            "step_index": len(steps_data),
            "code_count": code_count,
            "thought": thought,
            "code": code,
            "image_path": None,
            "blend_path": None,
            "video_path": None,
            "is_animation": animation,
            "is_fix_camera": fix_camera
        }
        
        if animation:
            # Check for video file
            video_dir = os.path.join(video_path_val if return_data else VIDEO_PATH, f'{code_count}')
            video_file = os.path.join(video_dir, 'Camera_Main.mp4')
            if os.path.exists(video_file):
                step_data["video_path"] = video_file
                step_data["image_path"] = None  # Use video instead
        elif fix_camera:
            # Use fixed camera renders
            right_img_path = os.path.join(image_path_val if return_data else IMAGE_PATH, f'{code_count}.png')
            if os.path.exists(right_img_path):
                step_data["image_path"] = right_img_path
        else:
            # Use trajectory-based image paths
            user_message = traj[i+1]
            if user_message['role'] != 'user':
                continue
            if len(user_message['content']) < 3:
                continue
            if 'Image loaded from local path: ' not in user_message['content'][2]['text']:
                continue
            img_path = user_message['content'][2]['text'].split("Image loaded from local path: ")[1]
            image_name = img_path.split("/renders/")[-1]
            right_img_path = os.path.join(str(renders_dir), image_name)
            right_img_path = os.path.abspath(right_img_path)  # 转换为绝对路径并规范化
            if os.path.exists(right_img_path):
                step_data["image_path"] = right_img_path
                count += 1
        
        # Check for .blend file (saved in render directories)
        if fix_camera or animation:
            # For fix_camera/animation, blend file might be in video/renders/{code_count}/state.blend
            blend_path = os.path.join(os.path.dirname(traj_path), 'video', 'renders', f'{code_count}', 'state.blend')
            if not os.path.exists(blend_path):
                # Try alternative location
                blend_path = os.path.join(os.path.dirname(traj_path), 'renders', f'{code_count}', 'state.blend')
        else:
            # For normal trajectory, blend file is in renders/{code_count}/state.blend
            blend_path = os.path.join(str(renders_dir), f'{code_count}', 'state.blend')
            if not os.path.exists(blend_path):
                # Try parent renders directory
                blend_path = os.path.join(renders_dir.parent, 'renders', f'{code_count}', 'state.blend')
        
        if os.path.exists(blend_path):
            step_data["blend_path"] = blend_path
        
        # Only add step if it has image or video
        if step_data["image_path"] or step_data["video_path"]:
            steps_data.append(step_data)
    
    if return_data:
        return {
            "steps_data": steps_data,
            "renders_dir": renders_dir,
            "image_path": image_path if fix_camera else "",
            "video_path": video_path if animation else ""
        }
    else:
        print(f"Parsed {len(STEPS_DATA)} steps from trajectory")


@app.route('/')
def index():
    """Main page"""
    return render_template('trajectory_viewer.html')


@app.route('/api/preview-images')
def get_preview_images():
    """Get preview images for entry page (10 target images)"""
    preview_images = []
    
    # List of target images to display (scene name, file extension)
    target_images = [
        ("goldengate8", "png"),
        ("christmas1", "png"),
        ("restroom5", "png"),
        ("whitehouse9", "png"),
        ("house11", "png"),
        ("cake15", "png"),
        ("bathroom20", "png"),
        ("glass24", "png"),
        ("blueroom26", "jpeg"),
        ("bedroom32", "png")
    ]
    
    # Add all target images
    for idx, (scene_name, ext) in enumerate(target_images):
        target_path = os.path.abspath(f'data/static_scene/{scene_name}/target.{ext}')
        if os.path.exists(target_path):
            preview_images.append({
                "step_index": -1 - idx,  # Special negative index for target images
                "image_url": f"/api/target-image/{scene_name}",
                "is_target": True,
                "scene_name": scene_name
            })
    
    return jsonify({"images": preview_images})


@app.route('/api/target-image/<scene_name>')
def get_target_image(scene_name):
    """Serve target image for a specific scene"""
    # Try different file extensions
    possible_extensions = ['png', 'jpeg', 'jpg']
    target_path = None
    
    for ext in possible_extensions:
        path = os.path.abspath(f'data/static_scene/{scene_name}/target.{ext}')
        if os.path.exists(path):
            target_path = path
            break
    
    if not target_path or not os.path.exists(target_path):
        print(f"[DEBUG] Target image not found for scene: {scene_name}")
        return jsonify({"error": f"Target image not found for scene: {scene_name}"}), 404
    
    # 确保路径是字符串格式
    target_path = str(target_path)
    image_dir = os.path.dirname(target_path)
    image_file = os.path.basename(target_path)
    return send_from_directory(image_dir, image_file)


def find_trajectory_file(base_path):
    """Find generator_memory.json in base_path or its subdirectories"""
    if not os.path.isdir(base_path):
        return None
    
    # Check if generator_memory.json is directly in base_path
    traj_path = os.path.join(base_path, 'generator_memory.json')
    if os.path.exists(traj_path):
        return traj_path
    
    # Look in subdirectories
    for task in os.listdir(base_path):
        task_path = os.path.join(base_path, task)
        if os.path.isdir(task_path):
            traj_path = os.path.join(task_path, 'generator_memory.json')
            if os.path.exists(traj_path):
                return traj_path
    
    return None


def preload_all_trajectories():
    """Preload all trajectories for all scenes"""
    global PRELOADED_TRAJECTORIES
    
    print("Preloading all trajectories...")
    for scene_name, trajectory_path in SCENE_TRAJECTORY_MAP.items():
        if trajectory_path is None:
            continue
        
        base_path = trajectory_path
        traj_path = find_trajectory_file(base_path)
        
        if traj_path and os.path.exists(traj_path):
            try:
                print(f"  Loading {scene_name} from {traj_path}")
                parsed_data = parse_trajectory(traj_path, animation=False, fix_camera=False, return_data=True)
                PRELOADED_TRAJECTORIES[scene_name] = {
                    "steps_data": parsed_data["steps_data"],
                    "base_path": base_path,
                    "renders_dir": parsed_data["renders_dir"],
                    "image_path": parsed_data["image_path"],
                    "video_path": parsed_data["video_path"]
                }
                print(f"  Loaded {len(parsed_data['steps_data'])} steps for {scene_name}")
            except Exception as e:
                print(f"  Error loading {scene_name}: {e}")
        else:
            print(f"  Warning: Could not find trajectory for {scene_name} at {base_path}")
    
    print(f"Preloaded {len(PRELOADED_TRAJECTORIES)} trajectories")


@app.route('/api/load-trajectory/<scene_name>')
def load_trajectory(scene_name):
    """Load trajectory for a specific scene from preloaded data"""
    global STEPS_DATA, BASE_PATH, RENDERS_DIR, IMAGE_PATH, VIDEO_PATH
    
    # Check if trajectory is preloaded
    if scene_name in PRELOADED_TRAJECTORIES:
        preloaded = PRELOADED_TRAJECTORIES[scene_name]
        STEPS_DATA = preloaded["steps_data"]
        BASE_PATH = preloaded["base_path"]
        RENDERS_DIR = preloaded["renders_dir"]
        IMAGE_PATH = preloaded["image_path"]
        VIDEO_PATH = preloaded["video_path"]
        
        return jsonify({
            "success": True,
            "total_steps": len(STEPS_DATA),
            "scene_name": scene_name
        })
    else:
        return jsonify({"error": f"Trajectory not found for scene: {scene_name}"}), 404


@app.route('/api/steps')
def get_steps():
    """Get all steps metadata"""
    return jsonify({
        "total_steps": len(STEPS_DATA),
        "steps": [{
            "step_index": s["step_index"],
            "code_count": s["code_count"]
        } for s in STEPS_DATA]
    })


@app.route('/api/step/<int:step_index>')
def get_step(step_index):
    """Get step data"""
    if step_index < 0 or step_index >= len(STEPS_DATA):
        return jsonify({"error": "Step not found"}), 404
    
    step = STEPS_DATA[step_index]
    response = {
        "step_index": step["step_index"],
        "code_count": step["code_count"],
        "thought": step["thought"],
        "code": step["code"],
        "has_image": step["image_path"] is not None,
        "has_video": step["video_path"] is not None,
        "has_blend": step["blend_path"] is not None
    }
    
    # Add relative paths for serving
    if step["image_path"]:
        # Make path relative to base
        rel_path = os.path.relpath(step["image_path"], BASE_PATH)
        response["image_url"] = f"/api/image/{step_index}"
    
    if step["video_path"]:
        rel_path = os.path.relpath(step["video_path"], BASE_PATH)
        response["video_url"] = f"/api/video/{step_index}"
    
    if step["blend_path"]:
        response["blend_url"] = f"/api/blend/{step_index}"
    
    return jsonify(response)


@app.route('/api/image/<int:step_index>')
def get_image(step_index):
    """Serve rendered image"""
    if step_index < 0 or step_index >= len(STEPS_DATA):
        return jsonify({"error": "Step not found"}), 404
    
    step = STEPS_DATA[step_index]
    image_path = step.get("image_path")
    
    # 确保路径是字符串格式
    if image_path:
        image_path = str(image_path)
    
    if not image_path or not os.path.exists(image_path):
        # 添加调试信息
        print(f"[DEBUG] Image not found for step {step_index}: {image_path}")
        return jsonify({"error": f"Image not found: {image_path}"}), 404
    
    image_dir = os.path.dirname(image_path)
    image_file = os.path.basename(image_path)
    return send_from_directory(image_dir, image_file)


@app.route('/api/video/<int:step_index>')
def get_video(step_index):
    """Serve video file"""
    if step_index < 0 or step_index >= len(STEPS_DATA):
        return jsonify({"error": "Step not found"}), 404
    
    step = STEPS_DATA[step_index]
    if not step["video_path"] or not os.path.exists(step["video_path"]):
        return jsonify({"error": "Video not found"}), 404
    
    video_dir = os.path.dirname(step["video_path"])
    video_file = os.path.basename(step["video_path"])
    return send_from_directory(video_dir, video_file)


@app.route('/api/blend/<int:step_index>')
def get_blend(step_index):
    """Serve Blender file (for potential 3D interaction)"""
    if step_index < 0 or step_index >= len(STEPS_DATA):
        return jsonify({"error": "Step not found"}), 404
    
    step = STEPS_DATA[step_index]
    if not step["blend_path"] or not os.path.exists(step["blend_path"]):
        return jsonify({"error": "Blend file not found"}), 404
    
    blend_dir = os.path.dirname(step["blend_path"])
    blend_file = os.path.basename(step["blend_path"])
    return send_from_directory(blend_dir, blend_file, as_attachment=True)


def main():
    global BASE_PATH
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", type=str, default="20251028_133713")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--fix_camera", action="store_true", help="固定相机位置和方向")
    ap.add_argument("--animation", action="store_true", help="从video_path加载MP4文件并拼接（与fix_camera相同逻辑）")
    args = ap.parse_args()
    
    # Determine base path (same logic as video_script.py)
    if args.animation:
        if os.path.exists(f'output/dynamic_scene/demo/{args.name}'):
            base_path = f'output/dynamic_scene/demo/{args.name}'
        else:
            base_path = f'output/dynamic_scene/{args.name}'
    else:
        if os.path.exists(f'output/static_scene/demo/{args.name}'):
            base_path = f'output/static_scene/demo/{args.name}'
        else:
            base_path = f'output/static_scene/{args.name}'
    
    BASE_PATH = base_path
    
    # Find trajectory file
    traj_path = ''
    for task in os.listdir(base_path):
        task_path = os.path.join(base_path, task)
        if os.path.isdir(task_path) and os.path.exists(os.path.join(task_path, 'generator_memory.json')):
            traj_path = os.path.join(task_path, 'generator_memory.json')
            break
    
    # Preload all trajectories first
    preload_all_trajectories()
    
    # Load default trajectory (goldengate8) as initial active trajectory
    if "goldengate8" in PRELOADED_TRAJECTORIES:
        preloaded = PRELOADED_TRAJECTORIES["goldengate8"]
        STEPS_DATA = preloaded["steps_data"]
        BASE_PATH = preloaded["base_path"]
        RENDERS_DIR = preloaded["renders_dir"]
        IMAGE_PATH = preloaded["image_path"]
        VIDEO_PATH = preloaded["video_path"]
        print(f"Loaded default trajectory (goldengate8) with {len(STEPS_DATA)} steps")
    else:
        # Fallback to old method if goldengate8 not preloaded
        if not traj_path or not os.path.exists(traj_path):
            print(f"Error: Could not find generator_memory.json in {base_path}")
            return
        
        print(f"Loading trajectory from: {traj_path}")
        parse_trajectory(traj_path, animation=args.animation, fix_camera=args.fix_camera)
        
        if len(STEPS_DATA) == 0:
            print("Warning: No steps found in trajectory")
            return
    
    print(f"Starting web server on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()

# python visualization/web_script.py --name 20251030_033307