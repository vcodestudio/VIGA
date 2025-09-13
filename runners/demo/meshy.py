# ======================
# Meshy API（从scene.py迁移）
# ======================

import os
import json
import time
import base64
import requests
import logging
import re
import tempfile
import shutil
import zipfile
import cv2
import numpy as np
from typing import Optional, Tuple
from pathlib import Path

# 尝试导入Blender
try:
    import bpy
except ImportError:
    bpy = None

# 尝试导入OpenAI
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

_HAS_GDINOSAM = True
try:
    import torch
    import torchvision
    from PIL import Image
    from groundingdino.util.inference import load_model, predict
    from groundingdino.util import box_ops
    # SAM v1
    from segment_anything import sam_model_registry, SamPredictor
except Exception as _e:
    _HAS_GDINOSAM = False
    # logging.warning(f"Grounded-SAM backend not available: {_e}")
    
_HAS_YOLO = True
try:
    from ultralytics import YOLO
except Exception as _e:
    _HAS_YOLO = False
    # logging.warning(f"YOLO backend not available: {_e}")

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
        import time
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
                'image_url': f"data:image/jpeg;base64,{image_base64}",
                'enable_pbr': True
            }
            
            # 发送请求（注意：这里不使用JSON headers，因为要上传文件）
            headers = {
                "Authorization": f"Bearer {self.api_key}"
            }
            
            resp = requests.post(url, headers=headers, json=files)
            resp.raise_for_status()
            data = resp.json()
            return data.get("result") or data.get("id")

    def create_image_to_3d_refine(self, preview_task_id: str, **kwargs) -> str:
        """
        基于 preview 发起 refine 贴图任务（Image-to-3D）
        Returns: refine_task_id (str)
        """
        url = f"{self.base_url}/openapi/v1/image-to-3d"
        payload = {
            "mode": "refine",
            "preview_task_id": preview_task_id,
        }
        payload.update(kwargs or {})
        resp = requests.post(url, headers=self.headers, data=json.dumps(payload))
        resp.raise_for_status()
        data = resp.json()
        return data.get("result") or data.get("id")

    def poll_image_to_3d(self, task_id: str, interval_sec: float = 5.0, timeout_sec: int = 1800) -> dict:
        """
        轮询 Image-to-3D 任务直到结束
        Returns: 任务 JSON（包含 status / model_urls 等）
        """
        import time
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


# ======================
# 图片截取工具
# ======================

class ImageCropper:
    """图片截取工具，支持基于文本描述的智能截取（Grounding DINO + SAM / YOLO / OpenAI 兜底）"""

    def __init__(self):
        self.temp_dir = None
        # OpenAI client（仅在 openai 兜底时用）
        self._client = None

        # 后端选择：grounded_sam / yolo / openai
        self._backend = os.getenv("DETECT_BACKEND", "grounded_sam").lower()

        # Grounded-SAM 相关句柄（惰性加载）
        self._gdino_model = None
        self._sam_predictor = None
        self._gdino_device = os.getenv("DET_DEVICE", "cuda" if self._torch_has_cuda() else "cpu")
        self._gdino_repo = os.getenv("GDINO_REPO", "models/IDEA-Research/grounding-dino-base")
        self._sam_type = os.getenv("SAM_TYPE", "vit_h")  # vit_h/vit_l/vit_b
        self._sam_repo = os.getenv("SAM_REPO", "models/facebook/sam-vit-huge")

        # YOLO 相关（惰性加载）
        self._yolo = None
        self._yolo_model = os.getenv("YOLO_MODEL", "models/yolov8x.pt")  # 本地或auto-download

    # ---------------------------
    # OpenAI（兜底）客户端（接口保留）
    # ---------------------------
    def _get_openai_client(self) -> OpenAI:
        """Initialize and cache the OpenAI client using environment variables."""
        if self._client is None:
            if OpenAI is None:
                raise RuntimeError("openai package not installed")
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not set; required for VLM-based cropping")
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            self._client = OpenAI(api_key=api_key, base_url=base_url)
        return self._client

    # ---------------------------
    # 工具函数
    # ---------------------------
    def _torch_has_cuda(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def _encode_image_as_data_url(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        # 用 jpg 更通用；若需严格 png，可改回 png
        return f"data:image/jpeg;base64,{b64}"

    # ---------------------------
    # 后端：Grounded-SAM
    # ---------------------------
    def _lazy_init_grounded_sam(self):
        if self._gdino_model is None or self._sam_predictor is None:
            if not _HAS_GDINOSAM:
                raise RuntimeError("Grounding DINO + SAM backend not available")
            # 加载 GroundingDINO
            self._gdino_model = load_model(model_checkpoint_path=self._gdino_repo, device=self._gdino_device)
            # 加载 SAM
            if self._sam_type == "vit_h":
                sam_type_key = "vit_h"
            elif self._sam_type == "vit_l":
                sam_type_key = "vit_l"
                self._sam_repo = "models/facebook/sam-vit-large"
            else:
                sam_type_key = "vit_b"
                self._sam_repo = "models/facebook/sam-vit-base"

            sam = sam_model_registry[sam_type_key](checkpoint=None)  # 会从 repo 自动拉取权重
            # 手动加载权重（segment-anything 库会自动处理下载缓存）
            # 这里通过 from_pretrained 路径加载更省心，但官方API是注册表+checkpoint，本实现采用纯注册表+自动权重
            # 若你本地已有 ckpt，可用 sam.load_state_dict(torch.load('/path/to/sam_vit_h.pth'))
            self._sam_predictor = SamPredictor(sam.to(self._gdino_device))

    def _grounded_sam_bbox(self, image_path: str, description: str) -> Optional[Tuple[int, int, int, int]]:
        """
        用 GroundingDINO 根据文本找框，再用 SAM 细化掩膜 -> 取最小外接矩形为 bbox
        返回 (x, y, w, h)（像素）
        """
        self._lazy_init_grounded_sam()

        image_pil = Image.open(image_path).convert("RGB")
        image_np = np.array(image_pil)

        # 1) DINO 文本检索检测：返回 xyxy + logits
        #   box_threshold 控制候选框，text_threshold 控制文本匹配
        #   若你希望更“严格”，可以把 box_threshold 调大（如 0.35~0.5）
        boxes, logits, phrases = predict(
            model=self._gdino_model,
            image=image_pil,
            caption=description,
            box_threshold=float(os.getenv("GDINO_BOX_THRESH", "0.25")),
            text_threshold=float(os.getenv("GDINO_TEXT_THRESH", "0.25")),
            device=self._gdino_device
        )
        if boxes is None or len(boxes) == 0:
            return None

        # 选择得分最高的一个框
        best_idx = int(np.argmax(logits))
        # GroundingDINO 返回的是相对坐标（0~1），转成像素 xyxy
        H, W = image_np.shape[:2]
        box_xyxy = boxes[best_idx]  # tensor [x_center, y_center, w, h] or xyxy? 依API而定
        # groundingdino.util.inference.predict 返回的是 xyxy 的 0~1 归一化坐标（当前版本）
        # 若遇到老版本返回 cxcywh，可改用 box_ops.box_cxcywh_to_xyxy
        if box_xyxy.shape[-1] == 4 and float(box_xyxy[2]) <= 1.0 and float(box_xyxy[3]) <= 1.0:
            # 归一化 xyxy -> 像素
            x1 = int(box_xyxy[0] * W)
            y1 = int(box_xyxy[1] * H)
            x2 = int(box_xyxy[2] * W)
            y2 = int(box_xyxy[3] * H)
        else:
            # 兼容：如果拿到 cxcywh，则先转 xyxy
            if hasattr(box_ops, "box_cxcywh_to_xyxy"):
                xyxy = box_ops.box_cxcywh_to_xyxy(box_xyxy.unsqueeze(0)).squeeze(0)
                x1 = int(xyxy[0].item())
                y1 = int(xyxy[1].item())
                x2 = int(xyxy[2].item())
                y2 = int(xyxy[3].item())
            else:
                # 退而求其次：直接当 xyxy 像素
                x1, y1, x2, y2 = [int(v) for v in box_xyxy.tolist()]

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W - 1, x2), min(H - 1, y2)
        if x2 <= x1 or y2 <= y1:
            return None

        # 2) SAM 细化掩膜，然后再从掩膜取更紧的 bbox（更贴边）
        self._sam_predictor.set_image(image_np)
        # SAM 支持以 bbox 作为 prompt
        mask, _, _ = self._sam_predictor.predict(
            box=np.array([x1, y1, x2, y2]),
            multimask_output=False
        )
        if mask is None or mask.shape[0] == 0:
            # 没掩膜就返回 dino 原框
            return (x1, y1, x2 - x1, y2 - y1)

        mask_bin = (mask[0] > 0) if mask.ndim == 3 else (mask > 0)
        ys, xs = np.where(mask_bin)
        if len(xs) == 0 or len(ys) == 0:
            return (x1, y1, x2 - x1, y2 - y1)

        tight_x1, tight_y1 = int(xs.min()), int(ys.min())
        tight_x2, tight_y2 = int(xs.max()), int(ys.max())
        return (tight_x1, tight_y1, max(1, tight_x2 - tight_x1 + 1), max(1, tight_y2 - tight_y1 + 1))

    # ---------------------------
    # 后端：YOLO（类名匹配）
    # ---------------------------
    def _lazy_init_yolo(self):
        if self._yolo is None:
            if not _HAS_YOLO:
                raise RuntimeError("YOLO backend not available")
            self._yolo = YOLO(self._yolo_model)

    def _yolo_bbox(self, image_path: str, description: str) -> Optional[Tuple[int, int, int, int]]:
        """
        用 YOLO 做类别检测；通过简单的同义词表 matching 选出最相关类别的最高分框。
        返回 (x, y, w, h)
        """
        self._lazy_init_yolo()
        res = self._yolo(image_path)[0]
        if res is None or res.boxes is None or len(res.boxes) == 0:
            return None

        # 构造同义词映射（可根据你的任务扩展）
        synonyms = {
            'person': ['human', 'people', 'man', 'woman', 'girl', 'boy', 'child'],
            'car': ['vehicle', 'automobile', 'auto', 'sedan', 'coupe', 'suv'],
            'dog': ['puppy', 'canine'],
            'cat': ['kitten', 'feline'],
            'chair': ['seat', 'stool', 'armchair'],
            'table': ['desk', 'surface'],
            'bottle': ['drink', 'water bottle'],
        }
        desc = description.lower()

        def _match(cls_name: str) -> bool:
            cls = cls_name.lower()
            if cls in desc or desc in cls:
                return True
            for k, vs in synonyms.items():
                if cls == k and any(v in desc for v in vs):
                    return True
                if any(v == cls for v in vs) and k in desc:
                    return True
            return False

        best = None
        for b in res.boxes:
            cls_idx = int(b.cls.item())
            cls_name = res.names.get(cls_idx, str(cls_idx))
            conf = float(b.conf.item())
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            if _match(cls_name):
                if (best is None) or (conf > best[0]):
                    best = (conf, x1, y1, x2, y2)

        # 若没有匹配上类别，则拿最高分框兜底
        if best is None:
            if len(res.boxes) == 0:
                return None
            b = max(res.boxes, key=lambda bb: float(bb.conf.item()))
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            return (x1, y1, max(1, x2 - x1), max(1, y2 - y1))

        _, x1, y1, x2, y2 = best
        return (x1, y1, max(1, x2 - x1), max(1, y2 - y1))

    # ---------------------------
    # 后端：OpenAI（保持你原逻辑作为最后兜底）
    # ---------------------------
    def _openai_bbox(self, image_path: str, description: str, model: str = None, temperature: float = 0.0):
        """保留你原有的 OpenAI 查询（仅当其它后端不可用时）"""
        client = self._get_openai_client()
        model_name = model or os.getenv("VISION_MODEL", "gpt-4o")
        data_url = self._encode_image_as_data_url(image_path)

        system_prompt = (
            "Given an input image and a target description, respond ONLY with a JSON object "
            "with keys x, y, w, h (integers, pixel coordinates, origin at top-left). "
            "If the target is not present, respond with {\"x\": -1, \"y\": -1, \"w\": 0, \"h\": 0}."
        )
        user_text = f"Target description: {description}\nReturn strictly JSON with keys x,y,w,h."

        resp = client.chat.completions.create(
            model=model_name,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
        )

        content = resp.choices[0].message.content if resp.choices else ""
        if not content:
            return None

        def _extract_json(text: str) -> Optional[dict]:
            try:
                return json.loads(text)
            except Exception:
                pass
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except Exception:
                    return None
            return None

        js = _extract_json(content)
        if not js:
            return None
        x = int(js.get("x", -1))
        y = int(js.get("y", -1))
        w = int(js.get("w", 0))
        h = int(js.get("h", 0))
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            return None
        return (x, y, w, h)

    # ---------------------------
    # 统一接口：询问 bbox（签名保留）
    # ---------------------------
    def _ask_vlm_for_bbox(self, image_path: str, description: str, model: str = None, temperature: float = 0.0) -> Optional[tuple]:
        """
        返回 (x, y, w, h) 或 None（找不到）
        优先 grounded_sam -> 失败用 yolo -> 失败再用 openai
        """
        backend_order = []
        # 用户明确指定 backend？
        b = self._backend
        if b == "grounded_sam":
            backend_order = ["grounded_sam", "yolo", "openai"]
        elif b == "yolo":
            backend_order = ["yolo", "grounded_sam", "openai"]
        else:
            backend_order = ["grounded_sam", "yolo", "openai"]

        last_err = None
        for be in backend_order:
            try:
                if be == "grounded_sam" and _HAS_GDINOSAM:
                    bbox = self._grounded_sam_bbox(image_path, description)
                    if bbox:
                        return bbox
                elif be == "yolo" and _HAS_YOLO:
                    bbox = self._yolo_bbox(image_path, description)
                    if bbox:
                        return bbox
                elif be == "openai":
                    bbox = self._openai_bbox(image_path, description, model=model, temperature=temperature)
                    if bbox:
                        return bbox
            except Exception as e:
                last_err = e
                logging.warning(f"[{be}] backend failed: {e}")
                continue

        if last_err:
            logging.error(f"All backends failed; last error: {last_err}")
        return None

    # ---------------------------
    # 对外：裁剪（签名与返回结构保留）
    # ---------------------------
    def crop_image_by_text(self, image_path: str, description: str, output_path: str = None,
                           confidence_threshold: float = 0.5, padding: int = 20) -> dict:
        """
        根据文本描述从图片中截取相关区域（通过 Grounded-SAM/YOLO/OpenAI 获取 bbox 后裁剪）
        返回结构与原版本一致
        """
        try:
            if not os.path.exists(image_path):
                return {"status": "error", "error": f"Image file not found: {image_path}"}

            image = cv2.imread(image_path)
            if image is None:
                return {"status": "error", "error": f"Failed to load image: {image_path}"}

            bbox = self._ask_vlm_for_bbox(image_path, description)
            if not bbox:
                return {"status": "error", "error": f"No valid bbox for '{description}' from any backend"}

            x, y, w, h = bbox
            height, width = image.shape[:2]
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(width, x + w + padding)
            y2 = min(height, y + h + padding)

            if x2 <= x1 or y2 <= y1:
                return {"status": "error", "error": "Computed bbox is invalid after padding"}

            cropped_image = image[y1:y2, x1:x2]

            if output_path is None:
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                output_dir = os.path.dirname(image_path)
                safe_desc = re.sub(r"[^a-zA-Z0-9_\-]+", "_", description.strip())[:40]
                output_path = os.path.join(output_dir, f"{base_name}_cropped_{safe_desc}.jpg")

            ok = cv2.imwrite(output_path, cropped_image)
            if not ok:
                return {"status": "error", "error": f"Failed to write output to: {output_path}"}

            return {
                "status": "success",
                "message": f"Successfully cropped image based on '{description}'",
                "input_image": image_path,
                "output_image": output_path,
                "detected_object": {
                    "description": description,
                    "confidence": None,          # 本实现不统一返回置信度，若需要可在各后端填充
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "original_bbox": [x, y, w, h]
                },
                "crop_info": {
                    "original_size": [image.shape[1], image.shape[0]],
                    "cropped_size": [x2 - x1, y2 - y1],
                    "padding": padding,
                    "backend": self._backend
                }
            }

        except Exception as e:
            logging.error(f"Failed to crop image: {e}")
            return {"status": "error", "error": str(e)}

    # ---------------------------
    # 下面两个保持原样（兼容）
    # ---------------------------
    def _detect_objects(self, image, description: str, confidence_threshold: float) -> list:
        logging.warning("_detect_objects is deprecated; use unified backends instead")
        return []

    def _is_description_match(self, class_name: str, description: str) -> bool:
        description_lower = description.lower()
        class_name_lower = class_name.lower()
        if class_name_lower in description_lower or description_lower in class_name_lower:
            return True
        synonyms = {
            'person': ['human', 'people', 'man', 'woman', 'child'],
            'car': ['vehicle', 'automobile', 'auto'],
            'dog': ['puppy', 'canine'],
            'cat': ['kitten', 'feline'],
            'bird': ['flying', 'winged'],
            'tree': ['plant', 'vegetation'],
            'building': ['house', 'structure', 'architecture'],
            'chair': ['seat', 'furniture'],
            'table': ['desk', 'surface'],
            'book': ['text', 'reading', 'literature']
        }
        for key, values in synonyms.items():
            if class_name_lower == key and any(v in description_lower for v in values):
                return True
            if any(v == class_name_lower for v in values) and key in description_lower:
                return True
        return False

    def _fallback_detection(self, image, description: str) -> list:
        height, width = image.shape[:2]
        mock_detections = []
        if 'person' in description.lower() or 'human' in description.lower():
            mock_detections.append({'class': 'person', 'confidence': 0.8, 'bbox': [width//4, height//4, width//2, height//2]})
        elif 'car' in description.lower() or 'vehicle' in description.lower():
            mock_detections.append({'class': 'car', 'confidence': 0.7, 'bbox': [width//6, height//3, width//3, height//3]})
        elif any(k in description.lower() for k in ['animal', 'dog', 'cat']):
            mock_detections.append({'class': 'animal', 'confidence': 0.6, 'bbox': [width//3, height//3, width//4, height//4]})
        return mock_detections


class AssetImporter:
    """3D资产导入器，支持多种格式"""
    def __init__(self, blender_path: str):
        self.blender_path = blender_path

    def import_asset(self, asset_path: str, location: tuple = (0, 0, 0), scale: float = 1.0) -> str:
        """导入3D资产到Blender场景"""
        try:
            # 确保文件存在
            if not os.path.exists(asset_path):
                raise FileNotFoundError(f"Asset file not found: {asset_path}")

            # 根据文件扩展名选择导入方法
            ext = os.path.splitext(asset_path)[1].lower()

            if ext in ['.fbx', '.obj', '.gltf', '.glb', '.dae', '.3ds', '.blend']:
                # 使用Blender的导入操作符
                if ext == '.fbx':
                    bpy.ops.import_scene.fbx(filepath=asset_path)
                elif ext == '.obj':
                    bpy.ops.import_scene.obj(filepath=asset_path)
                elif ext in ['.gltf', '.glb']:
                    bpy.ops.import_scene.gltf(filepath=asset_path)
                elif ext == '.dae':
                    bpy.ops.wm.collada_import(filepath=asset_path)
                elif ext == '.3ds':
                    bpy.ops.import_scene.autodesk_3ds(filepath=asset_path)
                elif ext == '.blend':
                    # 附注：append 需要 directory + filename（指向 .blend 内部路径）
                    # 这里保留占位，以防未来确实需要 .blend 的 append
                    bpy.ops.wm.append(filepath=asset_path)

                # 获取导入的对象
                imported_objects = [obj for obj in bpy.context.selected_objects]
                if not imported_objects:
                    raise RuntimeError("No objects were imported")

                # 设置位置和缩放
                for obj in imported_objects:
                    obj.location = location
                    obj.scale = (scale, scale, scale)

                # 返回导入的对象名称
                return imported_objects[0].name
            else:
                raise ValueError(f"Unsupported file format: {ext}")

        except Exception as e:
            logging.error(f"Failed to import asset: {e}")
            raise

    def extract_zip_asset(self, zip_path: str, extract_dir: str) -> str:
        """从ZIP文件中提取3D资产"""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # 查找3D文件
                asset_files = []
                for file_info in zip_ref.filelist:
                    filename = file_info.filename
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in ['.fbx', '.obj', '.gltf', '.glb', '.dae', '.3ds', '.blend']:
                        asset_files.append(filename)

                if not asset_files:
                    raise ValueError("No supported 3D files found in ZIP")

                # 提取第一个找到的3D文件
                asset_file = asset_files[0]
                zip_ref.extract(asset_file, extract_dir)

                return os.path.join(extract_dir, asset_file)

        except Exception as e:
            logging.error(f"Failed to extract ZIP asset: {e}")
            raise

def add_meshy_asset(
    description: str,
    location: str = "0,0,0",
    scale: float = 1.0,
    api_key: str = None,
    refine: bool = True,
    save_dir: str = "output/meshy_assets",
    filename: str = None,
    blender_path: str = None
) -> dict:
    """
    使用 Meshy Text-to-3D 生成资产并导入到当前场景（生成→轮询→下载→导入）

    Args:
        description: 文本描述（prompt）
        blender_path: Blender 文件路径
        location: 资产位置 "x,y,z"
        scale: 缩放比例
        api_key: Meshy API 密钥（可选，默认读 MESHY_API_KEY）
        refine: 是否在 preview 后进行 refine（含贴图）
    """
    try:
        # 解析位置参数
        try:
            loc_parts = [float(x.strip()) for x in location.split(",")]
            if len(loc_parts) != 3:
                return {"status": "error", "error": "Location must be in format 'x,y,z'"}
            asset_location = tuple(loc_parts)
        except Exception:
            return {"status": "error", "error": "Invalid location format. Use 'x,y,z'"}

        # 初始化 Meshy API
        meshy = MeshyAPI(api_key)

        # 1) 创建 preview 任务
        print(f"[Meshy] Creating preview task for: {description}")
        preview_id = meshy.create_text_to_3d_preview(description)

        # 2) 轮询 preview
        preview_task = meshy.poll_text_to_3d(preview_id, interval_sec=5, timeout_sec=900)
        if preview_task.get("status") != "SUCCEEDED":
            return {"status": "error", "error": f"Preview failed: {preview_task.get('status')}"}
        final_task = preview_task

        # 3) 可选 refine（贴图）
        if refine:
            print(f"[Meshy] Starting refine for preview task: {preview_id}")
            refine_id = meshy.create_text_to_3d_refine(preview_id)
            refine_task = meshy.poll_text_to_3d(refine_id, interval_sec=5, timeout_sec=1800)
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
        safe_desc = re.sub(r"[^a-zA-Z0-9_-]+", "_", description)[:60] or "asset"
        base_name = filename or f"text_{safe_desc}_{int(time.time())}"
        local_path = os.path.join(save_dir, f"{base_name}{guessed_ext}")
        print(f"[Meshy] Downloading model to: {local_path}")
        meshy.download_model_url(file_url, local_path)

        # 6) 若为 ZIP，解压出 3D 文件到保存目录下的同名子目录
        if blender_path:
            importer = AssetImporter(blender_path)
            if local_path.endswith(".zip"):
                extract_subdir = os.path.join(save_dir, base_name)
                os.makedirs(extract_subdir, exist_ok=True)
                extracted = importer.extract_zip_asset(local_path, extract_subdir)
                import_path = extracted
            else:
                import_path = local_path

            # 7) 导入 Blender
            imported_object_name = importer.import_asset(import_path, location=asset_location, scale=scale)
            print(f"[Meshy] Imported object: {imported_object_name}")

            # 8) 保存 Blender 文件
            try:
                bpy.ops.wm.save_mainfile(filepath=blender_path)
                print(f"Blender file saved to: {blender_path}")
                
                # 清理备份文件以避免生成 .blend1 文件
                backup_file = blender_path + "1"
                if os.path.exists(backup_file):
                    os.remove(backup_file)
                    print(f"Removed backup file: {backup_file}")
                    
            except Exception as save_error:
                print(f"Warning: Failed to save blender file: {save_error}")

        return {
            "status": "success",
            "message": "Meshy Text-to-3D asset generated and imported",
            "asset_name": description,
            "object_name": imported_object_name,
            "location": asset_location,
            "scale": scale,
            "saved_model_path": import_path
        }

    except Exception as e:
        logging.error(f"Failed to add Meshy asset: {e}")
        return {"status": "error", "error": str(e)}

def add_meshy_asset_from_image(
    image_path: str,
    location: str = "0,0,0",
    scale: float = 1.0,
    prompt: str = None,
    api_key: str = None,
    refine: bool = True,
    save_dir: str = "output/meshy_assets",
    filename: str = None,
    blender_path: str = None,
) -> dict:
    """
    使用 Meshy Image-to-3D 根据输入图片生成资产并导入到当前场景（生成→轮询→下载→导入）

    Args:
        image_path: 输入图片路径
        blender_path: Blender 文件路径
        location: 资产位置 "x,y,z"
        scale: 缩放比例
        prompt: 可选的文本提示，用于指导生成
        api_key: Meshy API 密钥（可选，默认读 MESHY_API_KEY）
        refine: 是否在 preview 后进行 refine（含贴图）
    """
    try:
        # 检查图片文件是否存在
        if not os.path.exists(image_path):
            return {"status": "error", "error": f"Image file not found: {image_path}"}
        
        # 解析位置参数
        try:
            loc_parts = [float(x.strip()) for x in location.split(",")]
            if len(loc_parts) != 3:
                return {"status": "error", "error": "Location must be in format 'x,y,z'"}
            asset_location = tuple(loc_parts)
        except Exception:
            return {"status": "error", "error": "Invalid location format. Use 'x,y,z'"}

        # 初始化 Meshy API
        meshy = MeshyAPI(api_key)

        # 1) 创建 Image-to-3D preview 任务
        print(f"[Meshy] Creating Image-to-3D preview task for: {image_path}")
        if prompt:
            print(f"[Meshy] Using prompt: {prompt}")
        
        preview_id = meshy.create_image_to_3d_preview(image_path, prompt)

        # 2) 轮询 preview
        preview_task = meshy.poll_image_to_3d(preview_id, interval_sec=5, timeout_sec=900)
        if preview_task.get("status") != "SUCCEEDED":
            return {"status": "error", "error": f"Image-to-3D preview failed: {preview_task.get('status')}"}
        final_task = preview_task

        # 3) 可选 refine（贴图）
        if refine:
            print(f"[Meshy] Starting refine for Image-to-3D preview task: {preview_id}")
            refine_id = meshy.create_image_to_3d_refine(preview_id)
            refine_task = meshy.poll_image_to_3d(refine_id, interval_sec=5, timeout_sec=1800)
            if refine_task.get("status") != "SUCCEEDED":
                return {"status": "error", "error": f"Image-to-3D refine failed: {refine_task.get('status')}"}
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
        safe_source = re.sub(r"[^a-zA-Z0-9_-]+", "_", os.path.splitext(os.path.basename(image_path))[0])[:60] or "image"
        base_name = filename or f"image_{safe_source}_{int(time.time())}"
        local_path = os.path.join(save_dir, f"{base_name}{guessed_ext}")
        print(f"[Meshy] Downloading Image-to-3D model to: {local_path}")
        meshy.download_model_url(file_url, local_path)

        # 6) 若为 ZIP，解压出 3D 文件到保存目录下的同名子目录
        if blender_path:
            importer = AssetImporter(blender_path)
            if local_path.endswith(".zip"):
                extract_subdir = os.path.join(save_dir, base_name)
                os.makedirs(extract_subdir, exist_ok=True)
                extracted = importer.extract_zip_asset(local_path, extract_subdir)
                import_path = extracted
            else:
                import_path = local_path

            # 7) 导入 Blender
            imported_object_name = importer.import_asset(import_path, location=asset_location, scale=scale)
            print(f"[Meshy] Imported Image-to-3D object: {imported_object_name}")

            # 8) 保存 Blender 文件
            try:
                bpy.ops.wm.save_mainfile(filepath=blender_path)
                print(f"Blender file saved to: {blender_path}")
                
                # 清理备份文件以避免生成 .blend1 文件
                backup_file = blender_path + "1"
                if os.path.exists(backup_file):
                    os.remove(backup_file)
                    print(f"Removed backup file: {backup_file}")
                    
            except Exception as save_error:
                print(f"Warning: Failed to save blender file: {save_error}")

        return {
            "status": "success",
            "message": "Meshy Image-to-3D asset generated and imported",
            "image_path": image_path,
            "prompt": prompt,
            "object_name": imported_object_name,
            "location": asset_location,
            "scale": scale,
            "saved_model_path": local_path
        }
        
    except Exception as e:
        logging.error(f"Failed to add Meshy asset from image: {e}")
        return {"status": "error", "error": str(e)}

def crop_image_by_text(
    image_path: str,
    description: str,
    output_path: str = None,
    confidence_threshold: float = 0.5,
    padding: int = 20
) -> dict:
    """
    根据文本描述从图片中截取相关区域（类似物体检测）
    
    Args:
        image_path: 输入图片路径
        description: 文本描述，描述要截取的对象（如："person", "car", "dog", "building"等）
        output_path: 输出图片路径（可选，默认自动生成）
        confidence_threshold: 置信度阈值（0.0-1.0），默认0.5
        padding: 截取区域周围的填充像素，默认20像素
        
    Returns:
        dict: 包含截取结果的字典，格式为：
        {
            "status": "success/error",
            "message": "操作结果描述",
            "input_image": "输入图片路径",
            "output_image": "输出图片路径",
            "detected_object": {
                "description": "检测到的对象类别",
                "confidence": 置信度,
                "bbox": [x, y, width, height],
                "original_bbox": [原始边界框]
            },
            "crop_info": {
                "original_size": [原始图片尺寸],
                "cropped_size": [截取后尺寸],
                "padding": 填充像素
            }
        }
    """
    try:
        # 创建图片截取器实例
        cropper = ImageCropper()
        
        # 执行截取操作
        result = cropper.crop_image_by_text(
            image_path=image_path,
            description=description,
            output_path=output_path,
            confidence_threshold=confidence_threshold,
            padding=padding
        )
        
        return result
        
    except Exception as e:
        logging.error(f"Failed to crop image by text: {e}")
        return {"status": "error", "error": str(e)}

def crop_and_generate_3d_asset(
    image_path: str,
    description: str,
    blender_path: str,
    location: str = "0,0,0",
    scale: float = 1.0,
    prompt: str = None,
    api_key: str = None,
    refine: bool = True,
    confidence_threshold: float = 0.5,
    padding: int = 20
) -> dict:
    """
    结合图片截取和3D资产生成的工具：
    1. 根据文本描述从图片中截取相关区域
    2. 将截取的图片送入Meshy生成3D资产
    3. 导入到Blender场景中
    
    Args:
        image_path: 输入图片路径
        description: 文本描述，描述要截取的对象（如："person", "car", "dog", "building"等）
        blender_path: Blender文件路径
        location: 资产位置 "x,y,z"，默认为 "0,0,0"
        scale: 缩放比例，默认为 1.0
        prompt: 可选的文本提示，用于指导3D生成
        api_key: Meshy API密钥（可选，默认读MESHY_API_KEY环境变量）
        refine: 是否进行refine处理（含贴图），默认为True
        confidence_threshold: 截取时的置信度阈值（0.0-1.0），默认0.5
        padding: 截取区域周围的填充像素，默认20像素
        
    Returns:
        dict: 包含完整操作结果的字典，格式为：
        {
            "status": "success/error",
            "message": "操作结果描述",
            "crop_result": {
                "input_image": "输入图片路径",
                "cropped_image": "截取的图片路径",
                "detected_object": {...}
            },
            "generation_result": {
                "object_name": "导入的对象名称",
                "location": [x, y, z],
                "scale": 缩放比例
            }
        }
    """
    try:
        print(f"[Crop&Generate] Starting combined crop and 3D generation process...")
        print(f"[Crop&Generate] Input image: {image_path}")
        print(f"[Crop&Generate] Description: {description}")
        
        # 步骤1: 图片截取
        print(f"[Crop&Generate] Step 1: Cropping image based on '{description}'...")
        cropper = ImageCropper()
        
        # 生成截取图片的临时路径
        temp_dir = tempfile.mkdtemp(prefix="crop_generate_")
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        cropped_image_path = os.path.join(temp_dir, f"{base_name}_cropped_{description.replace(' ', '_')}.jpg")
        
        crop_result = cropper.crop_image_by_text(
            image_path=image_path,
            description=description,
            output_path=cropped_image_path,
            confidence_threshold=confidence_threshold,
            padding=padding
        )
        
        if crop_result.get("status") != "success":
            # 清理临时目录
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except:
                pass
            return {
                "status": "error",
                "error": f"Image cropping failed: {crop_result.get('error')}",
                "crop_result": crop_result
            }
        
        print(f"[Crop&Generate] ✓ Image cropped successfully: {cropped_image_path}")
        
        # 步骤2: 3D资产生成
        print(f"[Crop&Generate] Step 2: Generating 3D asset from cropped image...")
        
        # 如果没有提供prompt，使用description作为默认prompt
        if not prompt:
            prompt = f"A 3D model of {description}"
        
        generation_result = add_meshy_asset_from_image(
            image_path=cropped_image_path,
            blender_path=blender_path,
            location=location,
            scale=scale,
            prompt=prompt,
            api_key=api_key,
            refine=refine
        )
        
        if generation_result.get("status") != "success":
            # 清理临时目录
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except:
                pass
            return {
                "status": "error",
                "error": f"3D asset generation failed: {generation_result.get('error')}",
                "crop_result": crop_result,
                "generation_result": generation_result
            }
        
        print(f"[Crop&Generate] ✓ 3D asset generated and imported successfully")
        
        # 步骤3: 清理临时文件
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"[Crop&Generate] ✓ Cleaned up temporary files")
        except Exception as cleanup_error:
            print(f"[Crop&Generate] ⚠ Warning: Failed to cleanup temp files: {cleanup_error}")
        
        # 返回完整结果
        return {
            "status": "success",
            "message": f"Successfully cropped image and generated 3D asset for '{description}'",
            "crop_result": {
                "input_image": image_path,
                "cropped_image": crop_result.get("output_image"),
                "detected_object": crop_result.get("detected_object"),
                "crop_info": crop_result.get("crop_info")
            },
            "generation_result": {
                "object_name": generation_result.get("object_name"),
                "location": generation_result.get("location"),
                "scale": generation_result.get("scale"),
                "prompt": prompt,
                "refine": refine
            },
            "summary": {
                "description": description,
                "original_image": image_path,
                "cropped_image": crop_result.get("output_image"),
                "generated_object": generation_result.get("object_name"),
                "final_location": location,
                "final_scale": scale
            }
        }
        
    except Exception as e:
        logging.error(f"Failed to crop and generate 3D asset: {e}")
        return {"status": "error", "error": str(e)}

def render_scene_for_test(blender_path: str, test_name: str, output_dir: str = "output/test/demo/renders") -> dict:
    """
    为测试渲染当前场景
    
    Args:
        blender_path: Blender文件路径
        test_name: 测试名称，用于生成输出文件名
        output_dir: 输出目录
        
    Returns:
        dict: 渲染结果
    """
    try:
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成输出文件名
        # timestamp = int(time.time())
        output_filename = blender_path.split("/")[-1].split(".")[0] + ".png"
        output_path = os.path.join(output_dir, output_filename)
        
        # 设置渲染参数
        scene = bpy.context.scene
        scene.render.resolution_x = 1024
        scene.render.resolution_y = 1024
        # scene.render.resolution_percentage = 50  # 50%分辨率以加快渲染速度
        scene.render.filepath = output_path
        
        # 设置渲染引擎为Cycles（如果可用）或Eevee
        if 'CYCLES' in bpy.context.scene.render.engine:
            scene.render.engine = 'CYCLES'
            scene.cycles.samples = 32  # 减少采样数以加快渲染
        else:
            scene.render.engine = 'BLENDER_EEVEE'
        
        # 确保有相机
        if not any(obj.type == 'CAMERA' for obj in scene.objects):
            # 如果没有相机，创建一个
            bpy.ops.object.camera_add(location=(5, -5, 3))
            camera = bpy.context.active_object
            camera.rotation_euler = (1.1, 0, 0.785)  # 设置相机角度
            scene.camera = camera
            print(f"[Render] Created camera for {test_name}")
            
        # 否则，设置Camera1为渲染相机
        else:
            scene.camera = bpy.data.objects['Camera1']
        
        # 渲染场景
        print(f"[Render] Rendering scene for {test_name}...")
        bpy.ops.render.render(write_still=True)
        
        print(f"[Render] ✓ Scene rendered successfully: {output_path}")
        
        return {
            "status": "success",
            "message": f"Scene rendered for {test_name}",
            "output_path": output_path,
            "test_name": test_name
        }
        
    except Exception as e:
        print(f"[Render] ❌ Failed to render scene for {test_name}: {e}")
        return {
            "status": "error",
            "error": str(e),
            "test_name": test_name
        }

def test_meshy_assets() -> dict:
    """
    测试 Meshy 资产生成功能：
    1. 测试 Text-to-3D 资产生成
    2. 测试 Image-to-3D 资产生成
    """
    print("🧪 Testing Meshy Asset Generation Functions...")
    
    # 测试配置
    test_blender_path = "output/test/demo/old_blender_file.blend"
    test_image_path = "output/test/demo/visprompt1.png"
    
    # 确保测试目录存在
    os.makedirs(os.path.dirname(test_blender_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_image_path), exist_ok=True)
    
    # 创建测试用的Blender文件
    try:
        # 打开现有的blender文件
        bpy.ops.wm.open_mainfile(filepath=test_blender_path)
        print(f"✓ Opened test Blender file: {test_blender_path}")
        
        # 渲染初始场景
        print("\n📸 Rendering initial scene...")
        initial_render = render_scene_for_test(test_blender_path, "initial_scene")
        if initial_render.get("status") == "success":
            print(f"✓ Initial scene rendered: {initial_render.get('output_path')}")
        
    except Exception as e:
        print(f"⚠ Warning: Could not open test Blender file: {e}")
        return {"status": "error", "error": f"Failed to open test Blender file: {e}"}
    
    # 创建测试图片（如果不存在）
    if not os.path.exists(test_image_path):
        try:
            from PIL import Image, ImageDraw
            # 创建一个简单的测试图片
            img = Image.new('RGB', (400, 300), color='lightblue')
            draw = ImageDraw.Draw(img)
            # 画一个简单的房子
            draw.rectangle([150, 150, 250, 250], fill='brown', outline='black')
            draw.polygon([(150, 150), (200, 100), (250, 150)], fill='red', outline='black')
            draw.rectangle([180, 180, 220, 220], fill='blue', outline='black')
            img.save(test_image_path)
            print(f"✓ Created test image: {test_image_path}")
        except Exception as e:
            print(f"⚠ Warning: Could not create test image: {e}")
            return {"status": "error", "error": f"Failed to create test image: {e}"}
    
    test_results = {
        "text_to_3d": {"status": "skipped", "message": "API key not provided"},
        "image_to_3d": {"status": "skipped", "message": "API key not provided"},
        "crop_image": {"status": "skipped", "message": "Test image not available"},
        "crop_and_generate": {"status": "skipped", "message": "API key not provided"}
    }
    
    # # 测试1: Text-to-3D 资产生成
    # print("\n📝 Testing Text-to-3D Asset Generation...")
    # try:
    #     # 检查是否有API密钥
    #     api_key = os.getenv("MESHY_API_KEY")
    #     if not api_key:
    #         print("⚠ Skipping Text-to-3D test: MESHY_API_KEY not set")
    #         test_results["text_to_3d"]["message"] = "MESHY_API_KEY environment variable not set"
    #     else:
    #         print("✓ API key found, testing Text-to-3D generation...")
    #         result = add_meshy_asset(
    #             description="A beautiful flower",
    #             blender_path=test_blender_path,
    #             location="2,0,0",
    #             scale=1.0,
    #             api_key=api_key,
    #             refine=True # 跳过refine以节省时间
    #         )
            
    #         if result.get("status") == "success":
    #             print(f"✓ Text-to-3D test successful: {result.get('message')}")
    #             test_results["text_to_3d"] = {
    #                 "status": "success",
    #                 "message": result.get("message"),
    #                 "object_name": result.get("object_name")
    #             }
                
    #             # 渲染场景以查看添加的物体
    #             render_result = render_scene_for_test(test_blender_path, "text_to_3d")
    #             if render_result.get("status") == "success":
    #                 test_results["text_to_3d"]["render_path"] = render_result.get("output_path")
    #                 print(f"✓ Rendered scene after Text-to-3D: {render_result.get('output_path')}")
    #         else:
    #             print(f"❌ Text-to-3D test failed: {result.get('error')}")
    #             test_results["text_to_3d"] = {
    #                 "status": "failed",
    #                 "message": result.get("error")
    #             }
    # except Exception as e:
    #     print(f"❌ Text-to-3D test error: {e}")
    #     test_results["text_to_3d"] = {
    #         "status": "error",
    #         "message": str(e)
    #     }
    
    # # 测试2: Image-to-3D 资产生成
    # print("\n🖼️ Testing Image-to-3D Asset Generation...")
    # try:
    #     api_key = os.getenv("MESHY_API_KEY")
    #     if not api_key:
    #         print("⚠ Skipping Image-to-3D test: MESHY_API_KEY not set")
    #         test_results["image_to_3d"]["message"] = "MESHY_API_KEY environment variable not set"
    #     else:
    #         print("✓ API key found, testing Image-to-3D generation...")
    #         result = add_meshy_asset_from_image(
    #             image_path=test_image_path,
    #             blender_path=test_blender_path,
    #             location="-2,0,0",
    #             scale=1.0,
    #             prompt="A 3D model of a house",
    #             api_key=api_key,
    #             refine=False  # 跳过refine以节省时间
    #         )
            
    #         if result.get("status") == "success":
    #             print(f"✓ Image-to-3D test successful: {result.get('message')}")
    #             test_results["image_to_3d"] = {
    #                 "status": "success",
    #                 "message": result.get("message"),
    #                 "object_name": result.get("object_name")
    #             }
                
    #             # 渲染场景以查看添加的物体
    #             render_result = render_scene_for_test(test_blender_path, "image_to_3d")
    #             if render_result.get("status") == "success":
    #                 test_results["image_to_3d"]["render_path"] = render_result.get("output_path")
    #                 print(f"✓ Rendered scene after Image-to-3D: {render_result.get('output_path')}")
    #         else:
    #             print(f"❌ Image-to-3D test failed: {result.get('error')}")
    #             test_results["image_to_3d"] = {
    #                 "status": "failed",
    #                 "message": result.get("error")
    #             }
    # except Exception as e:
    #     print(f"❌ Image-to-3D test error: {e}")
    #     test_results["image_to_3d"] = {
    #         "status": "error",
    #         "message": str(e)
    #     }
    
    # 测试3: 图片截取功能
    print("\n✂️ Testing Image Cropping...")
    try:
        if not os.path.exists(test_image_path):
            print("⚠ Skipping crop test: Test image not available")
            test_results["crop_image"]["message"] = "Test image not available"
        else:
            print("✓ Testing image cropping...")
            result = crop_image_by_text(
                image_path=test_image_path,
                description="coffee cup",
                output_path="test_output/cropped_building.jpg",
                confidence_threshold=0.3,
                padding=10
            )
            
            if result.get("status") == "success":
                print(f"✓ Image cropping test successful: {result.get('message')}")
                test_results["crop_image"] = {
                    "status": "success",
                    "message": result.get("message"),
                    "output_image": result.get("output_image")
                }
            else:
                print(f"❌ Image cropping test failed: {result.get('error')}")
                test_results["crop_image"] = {
                    "status": "failed",
                    "message": result.get("error")
                }
    except Exception as e:
        print(f"❌ Image cropping test error: {e}")
        test_results["crop_image"] = {
            "status": "error",
            "message": str(e)
        }
    
    # 测试4: 组合工具 - 图片截取 + 3D资产生成
    print("\n🔄 Testing Combined Crop and Generate Tool...")
    try:
        api_key = os.getenv("MESHY_API_KEY")
        if not api_key:
            print("⚠ Skipping combined test: MESHY_API_KEY not set")
            test_results["crop_and_generate"]["message"] = "MESHY_API_KEY environment variable not set"
        elif not os.path.exists(test_image_path):
            print("⚠ Skipping combined test: Test image not available")
            test_results["crop_and_generate"]["message"] = "Test image not available"
        else:
            print("✓ Testing combined crop and generate...")
            result = crop_and_generate_3d_asset(
                image_path=test_image_path,
                description="coffee cup",
                blender_path=test_blender_path,
                location="4,0,0",
                scale=1.0,
                prompt="A detailed 3D model of a house with realistic textures",
                api_key=api_key,
                refine=False,  # 跳过refine以节省时间
                confidence_threshold=0.3,
                padding=15
            )
            
            if result.get("status") == "success":
                print(f"✓ Combined test successful: {result.get('message')}")
                test_results["crop_and_generate"] = {
            "status": "success",
                    "message": result.get("message"),
                    "crop_result": result.get("crop_result"),
                    "generation_result": result.get("generation_result")
                }
                
                # 渲染场景以查看添加的物体
                render_result = render_scene_for_test(test_blender_path, "crop_and_generate")
                if render_result.get("status") == "success":
                    test_results["crop_and_generate"]["render_path"] = render_result.get("output_path")
                    print(f"✓ Rendered scene after Combined test: {render_result.get('output_path')}")
            else:
                print(f"❌ Combined test failed: {result.get('error')}")
                test_results["crop_and_generate"] = {
                    "status": "failed",
                    "message": result.get("error")
                }
    except Exception as e:
        print(f"❌ Combined test error: {e}")
        test_results["crop_and_generate"] = {
            "status": "error",
            "message": str(e)
        }
    
    # 总结测试结果
    print("\n📊 Test Results Summary:")
    print("=" * 50)
    
    success_count = 0
    total_tests = 0
    
    for test_name, result in test_results.items():
        total_tests += 1
        status = result["status"]
        message = result["message"]
        
        if status == "success":
            print(f"✅ {test_name}: SUCCESS - {message}")
            if "render_path" in result:
                print(f"   📸 Render saved: {result['render_path']}")
            success_count += 1
        elif status == "skipped":
            print(f"⏭️ {test_name}: SKIPPED - {message}")
        elif status == "failed":
            print(f"❌ {test_name}: FAILED - {message}")
        else:
            print(f"💥 {test_name}: ERROR - {message}")
    
    print("=" * 50)
    print(f"Tests completed: {success_count}/{total_tests} successful")
    
    # 返回测试结果
    overall_success = success_count > 0 or all(r["status"] == "skipped" for r in test_results.values())
        
    return {
        "status": "success" if overall_success else "failed",
        "message": f"Meshy asset generation tests completed: {success_count}/{total_tests} successful",
        "test_results": test_results,
        "summary": {
            "total_tests": total_tests,
            "successful": success_count,
            "skipped": sum(1 for r in test_results.values() if r["status"] == "skipped"),
            "failed": sum(1 for r in test_results.values() if r["status"] in ["failed", "error"])
        }
    }
