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

# tool config for agent
tool_configs = [
    {
        "type": "function",
        "function": {
            "name": "execute_and_evaluate",
            "description": "Execute code modifications and trigger verifier evaluation. This tool combines code execution with automatic verification. Always use this tool when you want to execute your code changes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": "Analyze the current state and provide a clear plan for the required changes. Consider scene representation consistency and infinigen optimization opportunities."
                    },
                    "code_edition": {
                        "type": "string", 
                        "description": "Provide your code modifications in the following format:\n-: [lines to remove]\n+: [lines to add]\nFocus on scene consistency and use infinigen functions when appropriate."
                    },
                    "full_code": {
                        "type": "string",
                        "description": "Merge your code changes into the full code with proper formatting. Ensure consistent scene representation."
                    }
                },
                "required": ["thought", "code_edition", "full_code"]
            }
        }
    }
]

mcp = FastMCP("blender-executor")

# Global executor instance
_executor = None

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
            if os.path.isdir(render_path):
                imgs = sorted([str(p) for p in Path(render_path).glob("*") if p.suffix in ['.png','.jpg']])
                if not imgs:
                    return False, "No images", out
                return True, imgs, out
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
        for img in os.listdir(render_file):
            os.remove(os.path.join(render_file, img))
        with open(script_file, "w") as f:
            f.write(code)
        success, stdout, stderr = self._execute_blender(str(script_file), str(render_file))
        if not success or not os.path.exists(render_file):
            return {"status": "failure", "output": stderr or stdout}
        return {"status": "success", "output": stdout}



@mcp.tool()
def initialize(args: dict) -> dict:
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
        
        return {"status": "success", "output": "Executor initialized successfully"}
    except Exception as e:
        return {"status": "error", "output": str(e)}

@mcp.tool()
def exec_script(code: str, round: int) -> dict:
    """
    执行传入的 Blender Python 脚本 code，并返回 base64 编码后的渲染图像。
    需要先调用 initialize_executor 进行初始化。
    """
    global _executor
    if _executor is None:
        return {"status": "error", "output": "Executor not initialized. Call initialize_executor first."}
    try:
        result = _executor.execute(code, round)
        return result
    except Exception as e:
        return {"status": "error", "output": str(e)}
    
def main():
    # 如果直接运行此脚本，执行测试
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("Running blender-executor tools test...")
        # Read args from environment for convenience
        args = {
            "blender_command": os.getenv("BLENDER_COMMAND", "utils/blender/infinigen/blender/blender"),
            "blender_file": os.getenv("BLENDER_FILE", "output/test/test.blend"),
            "blender_script": os.getenv("BLENDER_SCRIPT", "data/static_scene/pipeline_render_script.py"),
            "script_save": os.getenv("SCRIPT_SAVE", "output/test/scripts"),
            "render_save": os.getenv("RENDER_SAVE", "output/test/renders"),
            "blender_save": os.getenv("BLENDER_SAVE", None),
            "gpu_devices": os.getenv("GPU_DEVICES", None),
        }
        if not os.path.exists(args["blender_file"]):
            import bpy
            bpy.ops.wm.save_as_mainfile(filepath=args["blender_file"])
            print(f"Created blender file: {args['blender_file']}")
        print("[test] initialize(...) with:", json.dumps({k:v for k,v in args.items() if k!="gpu_devices"}, ensure_ascii=False))
        init_res = initialize(args)
        print("[test:init]", json.dumps(init_res, ensure_ascii=False))

        # exec_script: minimal code; actual execution requires Blender + driver script to read script file; create a NEW camera, setup its position and render it
        sample_code = """import bpy
import math

# 清空
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# 场景物体
bpy.ops.mesh.primitive_plane_add(size=4, location=(0,0,0))
bpy.ops.mesh.primitive_cube_add(size=1, location=(0,0,1))

# 相机
bpy.ops.object.camera_add()
camera = bpy.context.active_object
camera.location = (3, 3, 3)
camera.rotation_mode = 'XYZ'
camera.rotation_euler = (0.9553166, 0.0, -2.3561945)  # 指向(0,0,0)
camera.data.clip_start = 0.01
camera.data.clip_end = 1000

# **关键：把它设为场景的渲染相机**
bpy.context.scene.camera = camera

# **加一盏灯**（否则很黑）
bpy.ops.object.light_add(type='SUN', location=(5,5,10))
sun = bpy.context.active_object
sun.data.energy = 3.0

# 也可以加一点环境光（可选）
bpy.context.scene.world.use_nodes = True
bg = bpy.context.scene.world.node_tree.nodes['Background']
bg.inputs[1].default_value = 1.0   # 强度"""
        exec_res = exec_script(sample_code, round=1)
        print("[test:exec_script]", json.dumps(exec_res, ensure_ascii=False))
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