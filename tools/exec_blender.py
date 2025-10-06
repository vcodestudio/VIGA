# blender_executor_server.py
import os
import subprocess
import base64
import io
from typing import Optional
from pathlib import Path
from PIL import Image
import logging
from typing import Tuple, Dict
from mcp.server.fastmcp import FastMCP
import json
import requests
import tempfile
import zipfile
import shutil
import math
import cv2
import numpy as np
import time
from openai import OpenAI
import re
import PIL
from .meshy import MeshyAPI, ImageCropper, download_meshy_asset, download_meshy_asset_from_image

mcp = FastMCP("blender-executor")

# Global executor instance
_executor = None

# Global meshy API instance
_meshy_api = None

# Global image cropper instance
_image_cropper = None

# Global save_dir
_save_dir = None

class Executor:
    def __init__(self,
                 blender_command: str,
                 blender_file: str,
                 blender_script: str,
                 script_save: str,
                 render_save: str,
                 blender_save: Optional[str] = None,
                 gpu_devices: Optional[str] = None):
        self.blender_command = blender_command
        self.blender_file = blender_file
        self.blender_script = blender_script
        self.script_path = Path(script_save)
        self.render_path = Path(render_save)
        self.blend_path = blender_save
        self.gpu_devices = gpu_devices  # 例如: "0,1" 或 "0"

        self.script_path.mkdir(parents=True, exist_ok=True)
        self.render_path.mkdir(parents=True, exist_ok=True)

    def _execute_blender(self, script_path: str, render_path: str) -> Tuple[bool, str, str]:
        cmd = [
            self.blender_command,
            "--background", self.blender_file,
            "--python", self.blender_script,
            "--", script_path, render_path
        ]
        if self.blend_path:
            cmd.append(self.blend_path)
        cmd_str = " ".join(cmd)
        
        # 设置环境变量以控制GPU设备
        env = os.environ.copy()
        if self.gpu_devices:
            env['CUDA_VISIBLE_DEVICES'] = self.gpu_devices
            logging.info(f"Setting CUDA_VISIBLE_DEVICES to: {self.gpu_devices}")
        
        try:
            proc = subprocess.run(cmd_str, shell=True, check=True, capture_output=True, text=True, env=env)
            out = proc.stdout
            err = proc.stderr
            # We do not consider intermediate errors that do not affect the result.
            # if 'Error:' in out:
            #     logging.error(f"Error in Blender stdout: {out}")
            #     return False, err, out
            # find rendered image(s)
            if os.path.isdir(render_path):
                imgs = sorted([str(p) for p in Path(render_path).glob("*") if p.suffix in ['.png','.jpg']])
                if not imgs:
                    return False, "No images", out
                paths = imgs[:2]
                return True, "PATH:" + ",".join(paths), out
            return True, out, err
        except subprocess.CalledProcessError as e:
            logging.error(f"Blender failed: {e}")
            return False, e.stderr, e.stdout

    def _encode_image(self, img_path: str) -> str:
        img = Image.open(img_path)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def execute(self, code: str, round: int) -> Dict:
        script_file = self.script_path / f"{round}.py"
        render_file = self.render_path / f"{round}"
        with open(script_file, "w") as f:
            f.write(code)
        success, stdout, stderr = self._execute_blender(str(script_file), str(render_file))
        if not success or not os.path.exists(render_file):
            return {"status": "failure", "output": stderr or stdout}
        return {"status": "success", "output": str(render_file), "stdout": stdout, "stderr": stderr}



@mcp.tool()
def generate_and_download_3d_asset(
    object_name: str,
    reference_type: str,
    object_description: str = None,
) -> dict:
    # First check if local asset exists in task assets directory
    local_asset_path = _check_local_asset(object_name)
    if local_asset_path:
        return {
            "status": "success",
            "message": f"Local asset found: {local_asset_path}",
            "object_name": object_name,
            "local_path": local_asset_path,
            "save_dir": _save_dir
        }
    
    # If no local asset found, proceed with Meshy generation
    if reference_type == "text":
        generate_result = download_meshy_asset(
            object_name=object_name, 
            description=object_description, 
            save_dir=_save_dir,
            meshy_api=_meshy_api
        )
    elif reference_type == "image":
        cropped_bbox = _image_cropper.crop_image_by_text(object_name=object_name)
        cropped_bbox = cropped_bbox['data'][0][0]['bounding_box']
        cropped_image = PIL.Image.open(_image_cropper.target_image_path).crop(cropped_bbox)
        save_path = os.path.join(_save_dir, f"cropped_{object_name}.png")
        os.makedirs(_save_dir, exist_ok=True)
        cropped_image.save(save_path)
        generate_result = download_meshy_asset_from_image(
            image_path=save_path, 
            object_name=object_name, 
            save_dir=_save_dir,
            meshy_api=_meshy_api
        )
    
    return generate_result

def _check_local_asset(object_name: str) -> str:
    """Check if a local .glb asset exists for the given object name."""
    if not hasattr(_check_local_asset, '_task_assets_dir'):
        # Try to determine task assets directory from current context
        # This will be set during initialization
        return None
    
    assets_dir = _check_local_asset._task_assets_dir
    if not assets_dir or not os.path.exists(assets_dir):
        return None
    
    # Look for exact match first
    exact_path = os.path.join(assets_dir, f"{object_name}.glb")
    if os.path.exists(exact_path):
        return exact_path
    
    # Look for fuzzy match (case-insensitive, space-removed)
    object_name_clean = object_name.replace(" ", "").lower()
    for asset_file in os.listdir(assets_dir):
        if asset_file.endswith('.glb'):
            asset_name_clean = os.path.splitext(asset_file)[0].replace(" ", "").lower()
            if object_name_clean in asset_name_clean or asset_name_clean in object_name_clean:
                return os.path.join(assets_dir, asset_file)
    
    return None
    
@mcp.tool()
def initialize_executor(args: dict) -> dict:
    """
    初始化 Blender 执行器，设置所有必要的参数。
    
    Args:
        args: 包含以下键的字典:
            - blender_command: Blender可执行文件路径
            - blender_file: Blender文件路径
            - blender_script: Blender脚本路径
            - script_save: 脚本保存目录
            - render_save: 渲染结果保存目录
            - blender_save: Blender文件保存路径（可选）
            - gpu_devices: GPU设备ID，如"0"或"0,1"（可选）
            - meshy_api_key: Meshy API密钥（可选）
            - va_api_key: VA API密钥（可选）
            - target_image_path: 目标图片路径（可选）
    """
    global _executor
    global _meshy_api
    global _image_cropper
    global _save_dir
    try:
        _executor = Executor(
            blender_command=args.get("blender_command"),
            blender_file=args.get("blender_file"),
            blender_script=args.get("blender_script"),
            script_save=args.get("script_save"),
            render_save=args.get("render_save"),
            blender_save=args.get("blender_save"),
            gpu_devices=args.get("gpu_devices")
        )
        _save_dir = os.path.dirname(args.get("script_save")) + '/assets'
        
        # Set up task assets directory for local asset checking
        task_assets_dir = args.get("task_assets_dir")
        if task_assets_dir:
            _check_local_asset._task_assets_dir = task_assets_dir
        
        if args.get("meshy_api_key"):
            _meshy_api = MeshyAPI(args.get("meshy_api_key"))
        if args.get("va_api_key") and args.get("target_image_path"):
            _image_cropper = ImageCropper(args.get("va_api_key"), args.get("target_image_path"))
        return {"status": "success", "message": "Executor initialized successfully"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@mcp.tool()
def exec_script(code: str, round: int) -> dict:
    """
    执行传入的 Blender Python 脚本 code，并返回 base64 编码后的渲染图像。
    需要先调用 initialize_executor 进行初始化。
    """
    global _executor
    if _executor is None:
        return {"status": "error", "error": "Executor not initialized. Call initialize_executor first."}
    
    try:
        result = _executor.execute(code, round)
        return {"status": "success", "result": result}
    except Exception as e:
        return {"status": "error", "error": str(e)}
    

def main():
    # 如果直接运行此脚本，执行测试
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # 测试：先初始化执行器，再调用生成与导入资产接口
        meshy_api_key = os.getenv("MESHY_API_KEY")
        va_api_key = os.getenv("VA_API_KEY")
        if not meshy_api_key or not va_api_key:
            print("[TEST] Missing MESHY_API_KEY or VA_API_KEY in environment. Skipping online API test.")
            print("Set both environment variables to run this test.")
            sys.exit(0)

        blender_command = "utils/blender/infinigen/blender/blender"
        blender_file = "data/blendergym_hard/level4/christmas1/blender_file_empty.blend"
        blender_script = "data/blendergym_hard/level4/christmas1/pipeline_render_script.py"
        script_save = "output/test/christmas1/scripts"
        render_save = "output/test/christmas1/renders"
        target_image_path = "data/blendergym_hard/level4/christmas1/renders/goal/visprompt1.png"
        save_dir = os.path.dirname(script_save) + '/assets'
        os.makedirs(script_save, exist_ok=True)
        os.makedirs(render_save, exist_ok=True)

        print("[TEST] Initializing executor...")
        init_resp = initialize_executor(
            blender_command=blender_command,
            blender_file=blender_file,
            blender_script=blender_script,
            script_save=script_save,
            render_save=render_save,
            meshy_api_key=meshy_api_key,
            va_api_key=va_api_key,
            target_image_path=target_image_path,
        )
        print(f"[TEST] initialize_executor response: {init_resp}")
        if init_resp.get("status") != "success":
            print("[TEST] initialize_executor failed. Abort.")
            sys.exit(1)

        print("[TEST] Calling generate_and_download_3d_asset (text reference)...")
        # 使用文本参考分支
        gen_resp = generate_and_download_3d_asset(
            object_name="snowman",
            reference_type="text",
            object_description="A white snowman with a black hat and a red scarf.",
            save_dir=_save_dir
        )
        print(f"[TEST] generate_and_download_3d_asset response: {gen_resp}")
        sys.exit(0)
    else:
        # 正常运行 MCP 服务
        mcp.run(transport="stdio")

if __name__ == "__main__":
    main()


# # 首先检查本地assets目录是否有匹配的文件
# if os.path.exists(assets_dir):
#     for asset_file in os.listdir(assets_dir):
#         # 模糊匹配：将object_name和asset_file中的空格移除，转换为小写，判断是否互相包含
#         new_object_name = object_name.replace(" ", "")
#         new_asset_file = asset_file.replace(" ", "")
#         new_asset_file = new_asset_file.split(".")[0]
#         if new_object_name.lower() in new_asset_file.lower() or new_asset_file.lower() in new_object_name.lower():
#             if asset_file.endswith('.glb') or asset_file.endswith('.obj'):
#                 generate_result = {
#                     'status': 'success',
#                     'message': 'Local asset found',
#                     'object_name': object_name,
#                     'local_path': os.path.join(assets_dir, asset_file),
#                     'save_dir': save_dir
#                 }
#                 break
#         elif os.path.isdir(os.path.join(assets_dir, asset_file)):
#             for asset_file_ in os.listdir(os.path.join(assets_dir, asset_file)):
#                 if object_name.lower() in asset_file_.lower() or asset_file_.lower() in object_name.lower():
#                     if asset_file_.endswith('.glb') or asset_file_.endswith('.obj'):
#                         generate_result = {
#                             'status': 'success',
#                             'message': 'Local asset found',
#                             'object_name': object_name,
#                             'local_path': os.path.join(assets_dir, asset_file, asset_file_),
#                             'save_dir': save_dir
#                         }
#                         break
#             if generate_result is not None:
#                 break