# meshy.py
import os
import base64
import json
import requests
import time
import logging
import PIL
from typing import Optional


class MeshyAPI:
    """Meshy API 客户端：Text-to-3D 生成 + 轮询 + 下载"""
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("MESHY_API_KEY")
        if not self.api_key:
            raise ValueError("Meshy API key is required. Set MESHY_API_KEY environment variable or pass api_key parameter.")
        self.base_url = "https://api.meshy.ai"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
        }

    def create_text_to_3d_preview(self, prompt: str, **kwargs) -> str:
        """
        创建 Text-to-3D 预览任务（无贴图）
        Returns: task_id (str)
        """
        url = f"{self.base_url}/openapi/v2/text-to-3d"
        payload = {
            "mode": "preview",
            "prompt": prompt[:600],
        }
        payload.update(kwargs or {})
        resp = requests.post(url, headers=self.headers, data=json.dumps(payload))
        resp.raise_for_status()
        data = resp.json()
        # 有的环境返回 {"result": "<id>"}，有的返回 {"id": "<id>"}
        return data.get("result") or data.get("id")

    def poll_text_to_3d(self, task_id: str, interval_sec: float = 5.0, timeout_sec: int = 1800) -> dict:
        """
        轮询 Text-to-3D 任务直到结束
        Returns: 任务 JSON（包含 status / model_urls 等）
        """
        url = f"{self.base_url}/openapi/v2/text-to-3d/{task_id}"
        deadline = time.time() + timeout_sec
        while True:
            r = requests.get(url, headers=self.headers)
            r.raise_for_status()
            js = r.json()
            status = js.get("status")
            if status in ("SUCCEEDED", "FAILED", "CANCELED"):
                return js
            if time.time() > deadline:
                raise TimeoutError(f"Meshy task {task_id} polling timeout")
            time.sleep(interval_sec)

    def create_text_to_3d_refine(self, preview_task_id: str, **kwargs) -> str:
        """
        基于 preview 发起 refine 贴图任务
        Returns: refine_task_id (str)
        """
        url = f"{self.base_url}/openapi/v2/text-to-3d"
        payload = {
            "mode": "refine",
            "preview_task_id": preview_task_id,
        }
        payload.update(kwargs or {})
        resp = requests.post(url, headers=self.headers, data=json.dumps(payload))
        resp.raise_for_status()
        data = resp.json()
        return data.get("result") or data.get("id")

    def download_model_url(self, file_url: str, output_path: str) -> None:
        """
        从 model_urls 的直链下载文件到本地
        """
        r = requests.get(file_url, stream=True)
        r.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    def create_image_to_3d_preview(self, image_path: str, prompt: str = None, **kwargs) -> str:
        """
        创建 Image-to-3D 预览任务（无贴图）
        
        Args:
            image_path: 输入图片路径
            prompt: 可选的文本提示
            **kwargs: 其他参数
            
        Returns: task_id (str)
        """
        url = f"{self.base_url}/openapi/v1/image-to-3d"
        
        # 准备文件上传
        with open(image_path, 'rb') as f:
            # 将image转为base64格式
            image_base64 = base64.b64encode(f.read()).decode('utf-8')
            files = {
                'image_url': f"data:image/png;base64,{image_base64}",
                'enable_pbr': True,
            }
            
            # 发送请求（注意：这里不使用JSON headers，因为要上传文件）
            headers = {
                "Authorization": f"Bearer {self.api_key}"
            }
            
            resp = requests.post(url, headers=headers, json=files)
            resp.raise_for_status()
            data = resp.json()
            return data.get("result") or data.get("id")

    def poll_image_to_3d(self, task_id: str, interval_sec: float = 5.0, timeout_sec: int = 1800) -> dict:
        """
        轮询 Image-to-3D 任务直到结束
        Returns: 任务 JSON（包含 status / model_urls 等）
        """
        url = f"{self.base_url}/openapi/v1/image-to-3d/{task_id}"
        deadline = time.time() + timeout_sec
        while True:
            r = requests.get(url, headers=self.headers)
            r.raise_for_status()
            js = r.json()
            status = js.get("status")
            if status in ("SUCCEEDED", "FAILED", "CANCELED"):
                return js
            if time.time() > deadline:
                raise TimeoutError(f"Meshy Image-to-3D task {task_id} polling timeout")
            time.sleep(interval_sec)


class ImageCropper:
    """图片截取工具，支持基于文本描述的智能截取（Grounding DINO + SAM / YOLO / OpenAI 兜底）"""

    def __init__(self, api_key: str, target_image_path: str):
        self.url = "https://api.va.landing.ai/v1/tools/agentic-object-detection"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
        }
        self.target_image_path = target_image_path

    # ---------------------------
    # 对外：裁剪（签名与返回结构保留）
    # ---------------------------
    def crop_image_by_text(self, object_name: str) -> dict:
        files = {
            "image": open(self.target_image_path, "rb")
        }
        data = {
            "prompts": object_name,
            "model": "agentic"
        }
        response = requests.post(self.url, files=files, data=data, headers=self.headers)
        return response.json()


def download_meshy_asset(
    object_name: str,
    description: str,
    save_dir: str = "assets",
    refine: bool = True,
    meshy_api: MeshyAPI = None,
) -> dict:
    """
    下载 Meshy Text-to-3D 资产
    
    Args:
        object_name: 对象名称
        description: 对象描述
        save_dir: 保存目录
        refine: 是否进行贴图细化
        meshy_api: MeshyAPI 实例，如果为 None 则创建新实例
    """
    try:
        # 初始化 Meshy API
        if meshy_api is None:
            meshy_api = MeshyAPI()

        # 1) 创建 preview 任务
        print(f"[Meshy] Creating preview task for: {description}")
        preview_id = meshy_api.create_text_to_3d_preview(description)

        # 2) 轮询 preview
        preview_task = meshy_api.poll_text_to_3d(preview_id, interval_sec=5, timeout_sec=900)
        if preview_task.get("status") != "SUCCEEDED":
            return {"status": "error", "error": f"Preview failed: {preview_task.get('status')}"}
        final_task = preview_task

        # 3) 可选 refine（贴图）
        if refine:
            print(f"[Meshy] Starting refine for preview task: {preview_id}")
            refine_id = meshy_api.create_text_to_3d_refine(preview_id)
            refine_task = meshy_api.poll_text_to_3d(refine_id, interval_sec=5, timeout_sec=1800)
            if refine_task.get("status") != "SUCCEEDED":
                return {"status": "error", "error": f"Refine failed: {refine_task.get('status')}"}
            final_task = refine_task

        # 4) 从 model_urls 取下载链接
        model_urls = (final_task or {}).get("model_urls", {}) or {}
        candidate_keys = ["glb", "fbx", "obj", "zip"]
        file_url = None
        for k in candidate_keys:
            if model_urls.get(k):
                file_url = model_urls[k]
                break
        if not file_url:
            return {"status": "error", "error": "No downloadable model_urls found"}
        
        # 5) 下载模型到本地持久目录
        os.makedirs(save_dir, exist_ok=True)
        # 处理无扩展名直链：默认 .glb
        guessed_ext = os.path.splitext(file_url.split("?")[0])[1].lower()
        if guessed_ext not in [".glb", ".gltf", ".fbx", ".obj", ".zip"]:
            guessed_ext = ".glb"
        local_path = os.path.join(save_dir, f"{object_name}{guessed_ext}")
        print(f"[Meshy] Downloading model to: {local_path}")
        meshy_api.download_model_url(file_url, local_path)
        
        return {
            'status': 'success',
            'message': f'Meshy Text-to-3D asset downloaded to {local_path}',
            'object_name': object_name,
            'local_path': local_path,
            'save_dir': save_dir
        }

    except Exception as e:
        logging.error(f"Failed to download Meshy asset: {e}")
        return {"status": "error", "error": str(e)}


def download_meshy_asset_from_image(
    object_name: str,
    image_path: str,
    save_dir: str = "assets",
    prompt: str = None,
    meshy_api: MeshyAPI = None,
) -> dict:
    """
    使用 Meshy Image-to-3D 根据输入图片生成资产并下载到本地（生成→轮询→下载）

    Args:
        object_name: 对象名称
        image_path: 输入图片路径
        save_dir: 保存目录
        prompt: 可选的文本提示，用于指导生成
        meshy_api: MeshyAPI 实例，如果为 None 则创建新实例
    """
    try:
        # 检查图片文件是否存在
        if not os.path.exists(image_path):
            return {"status": "error", "error": f"Image file not found: {image_path}"}

        # 初始化 Meshy API
        if meshy_api is None:
            meshy_api = MeshyAPI()

        # 1) 创建 Image-to-3D preview 任务
        print(f"[Meshy] Creating Image-to-3D preview task for: {image_path}")
        if prompt:
            print(f"[Meshy] Using prompt: {prompt}")
        
        preview_id = meshy_api.create_image_to_3d_preview(image_path, prompt)

        # 2) 轮询 preview
        preview_task = meshy_api.poll_image_to_3d(preview_id, interval_sec=5, timeout_sec=900)
        if preview_task.get("status") != "SUCCEEDED":
            return {"status": "error", "error": f"Image-to-3D preview failed: {preview_task.get('status')}"}
        final_task = preview_task

        # 3) 从 model_urls 取下载链接
        model_urls = (final_task or {}).get("model_urls", {}) or {}
        candidate_keys = ["glb", "fbx", "obj", "zip"]
        file_url = None
        for k in candidate_keys:
            if model_urls.get(k):
                file_url = model_urls[k]
                break
        if not file_url:
            return {"status": "error", "error": "No downloadable model_urls found"}

        # 4) 下载模型到本地持久目录
        os.makedirs(save_dir, exist_ok=True)
        # 处理无扩展名直链：默认 .glb
        guessed_ext = os.path.splitext(file_url.split("?")[0])[1].lower()
        if guessed_ext not in [".glb", ".gltf", ".fbx", ".obj", ".zip"]:
            guessed_ext = ".glb"
        local_path = os.path.join(save_dir, f"{object_name}{guessed_ext}")
        print(f"[Meshy] Downloading Image-to-3D model to: {local_path}")
        meshy_api.download_model_url(file_url, local_path)

        return {
            'status': 'success',
            'message': f'Meshy Image-to-3D asset downloaded to {local_path}',
            'object_name': object_name,
            'local_path': local_path,
            'save_dir': save_dir
        }
        
    except Exception as e:
        logging.error(f"Failed to download Meshy asset from image: {e}")
        return {"status": "error", "error": str(e)}
