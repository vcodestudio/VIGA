import os
import bpy
import json
import time
from pathlib import Path
import logging

def initialize_3d_scene_from_image(image_path: str, output_dir: str = "output/demo/scenes") -> dict:
    """
    从输入图片初始化一个3D场景
    
    Args:
        image_path: 输入图片路径
        output_dir: 输出目录
        
    Returns:
        dict: 包含场景信息的字典
    """
    try:
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建新的Blender场景
        bpy.ops.wm.read_homefile(app_template="")
        
        # 生成场景文件名
        timestamp = int(time.time())
        image_name = Path(image_path).stem
        scene_name = f"scene_{image_name}_{timestamp}"
        blender_file_path = os.path.join(output_dir, f"{scene_name}.blend")
        
        # 设置基本场景参数
        scene = bpy.context.scene
        scene.render.resolution_x = 1024
        scene.render.resolution_y = 1024
        scene.render.engine = 'CYCLES'
        
        # 创建基础相机
        bpy.ops.object.camera_add(location=(5, -5, 3))
        camera = bpy.context.active_object
        camera.name = "Camera1"
        camera.rotation_euler = (1.1, 0, 0.785)
        scene.camera = camera
        
        # 创建基础灯光
        bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
        sun_light = bpy.context.active_object
        sun_light.name = "Sun"
        sun_light.data.energy = 3.0
        
        # 添加环境光
        bpy.ops.object.light_add(type='AREA', location=(-5, -5, 5))
        area_light = bpy.context.active_object
        area_light.name = "AreaLight"
        area_light.data.energy = 2.0
        area_light.data.size = 10
        
        # 创建一个简单的平面作为地面
        bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, 0))
        ground = bpy.context.active_object
        ground.name = "Ground"
        
        # 保存初始场景
        bpy.ops.wm.save_mainfile(filepath=blender_file_path)
        
        # 创建场景信息文件
        scene_info = {
            "scene_name": scene_name,
            "blender_file_path": blender_file_path,
            "source_image": image_path,
            "created_at": timestamp,
            "objects": [],
            "target_objects": []  # 将在后续循环中填充
        }
        
        # 保存场景信息
        info_file_path = os.path.join(output_dir, f"{scene_name}_info.json")
        with open(info_file_path, 'w', encoding='utf-8') as f:
            json.dump(scene_info, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 3D场景初始化完成: {scene_name}")
        print(f"  - Blender文件: {blender_file_path}")
        print(f"  - 场景信息: {info_file_path}")
        
        return {
            "status": "success",
            "message": f"3D场景 '{scene_name}' 初始化成功",
            "scene_name": scene_name,
            "blender_file_path": blender_file_path,
            "scene_info_path": info_file_path,
            "scene_info": scene_info
        }
        
    except Exception as e:
        logging.error(f"Failed to initialize 3D scene: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

def load_scene_info(scene_info_path: str) -> dict:
    """
    加载场景信息
    
    Args:
        scene_info_path: 场景信息文件路径
        
    Returns:
        dict: 场景信息
    """
    try:
        with open(scene_info_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load scene info: {e}")
        return {}

def update_scene_info(scene_info_path: str, updates: dict) -> dict:
    """
    更新场景信息
    
    Args:
        scene_info_path: 场景信息文件路径
        updates: 要更新的信息
        
    Returns:
        dict: 更新结果
    """
    try:
        # 加载现有信息
        scene_info = load_scene_info(scene_info_path)
        if not scene_info:
            return {"status": "error", "error": "Failed to load scene info"}
        
        # 更新信息
        scene_info.update(updates)
        
        # 保存更新后的信息
        with open(scene_info_path, 'w', encoding='utf-8') as f:
            json.dump(scene_info, f, indent=2, ensure_ascii=False)
        
        return {
            "status": "success",
            "message": "Scene info updated successfully",
            "scene_info": scene_info
        }
        
    except Exception as e:
        logging.error(f"Failed to update scene info: {e}")
        return {"status": "error", "error": str(e)}
