import os
import json
import logging
from typing import Optional, Dict, Any
from .meshy import add_meshy_asset, add_meshy_asset_from_image

class AssetGenerator:
    """3D资产生成器，支持从文本和图片生成资产"""
    
    def __init__(self, blender_path: str, api_key: str = None):
        """
        初始化资产生成器
        
        Args:
            blender_path: Blender文件路径
            api_key: Meshy API密钥（可选，默认从环境变量读取）
        """
        self.blender_path = blender_path
        self.api_key = api_key or os.getenv("MESHY_API_KEY")
        if not self.api_key:
            raise ValueError("Meshy API key is required. Set MESHY_API_KEY environment variable or pass api_key parameter.")
    
    def ask_vlm_for_object_description(self, object_name: str, context_image_path: str = None) -> str:
        """
        询问VLM获取物体的详细文本描述
        
        Args:
            object_name: 物体名称
            context_image_path: 上下文图片路径（可选）
            
        Returns:
            str: 详细的物体描述
        """
        try:
            # 这里应该调用VLM API，现在先用简单的模板
            # 在实际实现中，你需要调用OpenAI或其他VLM服务
            
            # 简单的描述模板（实际应用中应该用VLM生成）
            description_templates = {
                "chair": "A comfortable wooden chair with a high backrest and armrests, suitable for dining or office use",
                "table": "A sturdy wooden dining table with four legs and a smooth surface, perfect for family meals",
                "lamp": "A modern table lamp with a metal base and fabric shade, providing warm ambient lighting",
                "sofa": "A plush three-seater sofa with soft cushions and elegant fabric upholstery",
                "bookshelf": "A tall wooden bookshelf with multiple shelves for storing books and decorative items",
                "coffee_table": "A low wooden coffee table with a glass top, perfect for placing drinks and magazines",
                "bed": "A comfortable double bed with a wooden headboard and soft mattress",
                "desk": "A spacious wooden desk with drawers and a smooth work surface for studying or working",
                "television": "A modern flat-screen television with a sleek black frame and remote control",
                "plant": "A healthy green houseplant in a ceramic pot, adding natural beauty to the room",
                "clock": "A classic wall clock with Roman numerals and brass hands",
                "vase": "An elegant ceramic vase with a smooth finish, perfect for displaying flowers",
                "painting": "A beautiful framed artwork with vibrant colors and artistic composition",
                "mirror": "A large rectangular mirror with a wooden frame, perfect for a bedroom or hallway",
                "rug": "A soft area rug with geometric patterns and warm colors",
                "curtain": "Heavy fabric curtains with elegant drapes and curtain rods",
                "pillow": "A soft decorative pillow with colorful fabric and comfortable filling",
                "candle": "An aromatic scented candle in a decorative holder with a warm flame",
                "book": "A hardcover book with an interesting cover and pages ready to be read",
                "cup": "A ceramic coffee cup with a handle, perfect for morning coffee or tea"
            }
            
            # 如果物体名称在模板中，使用模板
            if object_name.lower() in description_templates:
                return description_templates[object_name.lower()]
            
            # 否则生成通用描述
            return f"A detailed 3D model of a {object_name}, with realistic textures and proper proportions"
            
        except Exception as e:
            logging.error(f"Failed to get object description: {e}")
            return f"A 3D model of {object_name}"
    
    def generate_asset_from_text(self, object_name: str, location: str = "0,0,0", scale: float = 1.0) -> Dict[str, Any]:
        """
        从文本生成3D资产
        
        Args:
            object_name: 物体名称
            location: 资产位置 "x,y,z"
            scale: 缩放比例
            
        Returns:
            dict: 生成结果
        """
        try:
            # 获取详细描述
            description = self.ask_vlm_for_object_description(object_name)
            print(f"[AssetGenerator] Generating text-to-3D asset for '{object_name}': {description}")
            
            # 调用meshy.py中的函数
            result = add_meshy_asset(
                description=description,
                blender_path=self.blender_path,
                location=location,
                scale=scale,
                api_key=self.api_key,
                refine=True,
                save_dir="output/demo/assets/text",
                filename=f"text_{object_name}"
            )
            
            return {
                "type": "text_to_3d",
                "object_name": object_name,
                "description": description,
                "result": result
            }
            
        except Exception as e:
            logging.error(f"Failed to generate text asset: {e}")
            return {
                "type": "text_to_3d",
                "object_name": object_name,
                "status": "error",
                "error": str(e)
            }
    
    def generate_asset_from_image(self, object_name: str, image_path: str, location: str = "0,0,0", scale: float = 1.0) -> Dict[str, Any]:
        """
        从图片生成3D资产
        
        Args:
            object_name: 物体名称
            image_path: 输入图片路径
            location: 资产位置 "x,y,z"
            scale: 缩放比例
            
        Returns:
            dict: 生成结果
        """
        try:
            # 获取详细描述作为prompt
            description = self.ask_vlm_for_object_description(object_name)
            print(f"[AssetGenerator] Generating image-to-3D asset for '{object_name}' from image: {image_path}")
            
            # 调用meshy.py中的函数
            result = add_meshy_asset_from_image(
                image_path=image_path,
                blender_path=self.blender_path,
                location=location,
                scale=scale,
                prompt=description,
                api_key=self.api_key,
                refine=True,
                save_dir="output/demo/assets/image",
                filename=f"image_{object_name}"
            )
            
            return {
                "type": "image_to_3d",
                "object_name": object_name,
                "image_path": image_path,
                "description": description,
                "result": result
            }
            
        except Exception as e:
            logging.error(f"Failed to generate image asset: {e}")
            return {
                "type": "image_to_3d",
                "object_name": object_name,
                "status": "error",
                "error": str(e)
            }
    
    def generate_both_assets(self, object_name: str, image_path: str = None, location: str = "0,0,0", scale: float = 1.0) -> Dict[str, Any]:
        """
        同时生成文本和图片两种3D资产
        
        Args:
            object_name: 物体名称
            image_path: 输入图片路径（可选）
            location: 资产位置 "x,y,z"
            scale: 缩放比例
            
        Returns:
            dict: 包含两种资产生成结果
        """
        try:
            print(f"[AssetGenerator] Generating both text and image assets for '{object_name}'")
            
            # 生成文本资产
            text_result = self.generate_asset_from_text(object_name, location, scale)
            
            # 生成图片资产（如果提供了图片路径）
            image_result = None
            if image_path and os.path.exists(image_path):
                # 为图片资产调整位置（避免重叠）
                image_location = f"{float(location.split(',')[0]) + 2},{location.split(',')[1]},{location.split(',')[2]}"
                image_result = self.generate_asset_from_image(object_name, image_path, image_location, scale)
            else:
                print(f"[AssetGenerator] Warning: Image path not provided or file not found: {image_path}")
                image_result = {
                    "type": "image_to_3d",
                    "object_name": object_name,
                    "status": "skipped",
                    "message": "Image path not provided or file not found"
                }
            
            return {
                "object_name": object_name,
                "text_asset": text_result,
                "image_asset": image_result,
                "status": "success" if text_result.get("result", {}).get("status") == "success" else "partial"
            }
            
        except Exception as e:
            logging.error(f"Failed to generate both assets: {e}")
            return {
                "object_name": object_name,
                "status": "error",
                "error": str(e)
            }
    
    def get_asset_summary(self, generation_result: Dict[str, Any]) -> str:
        """
        获取资产生成结果的摘要
        
        Args:
            generation_result: 资产生成结果
            
        Returns:
            str: 结果摘要
        """
        try:
            object_name = generation_result.get("object_name", "Unknown")
            text_asset = generation_result.get("text_asset", {})
            image_asset = generation_result.get("image_asset", {})
            
            summary_parts = [f"Asset generation for '{object_name}':"]
            
            # 文本资产结果
            if text_asset.get("type") == "text_to_3d":
                text_result = text_asset.get("result", {})
                if text_result.get("status") == "success":
                    summary_parts.append(f"  ✓ Text-to-3D: {text_result.get('message', 'Success')}")
                else:
                    summary_parts.append(f"  ✗ Text-to-3D: {text_result.get('error', 'Failed')}")
            
            # 图片资产结果
            if image_asset.get("type") == "image_to_3d":
                if image_asset.get("status") == "skipped":
                    summary_parts.append(f"  ⏭️ Image-to-3D: {image_asset.get('message', 'Skipped')}")
                else:
                    image_result = image_asset.get("result", {})
                    if image_result.get("status") == "success":
                        summary_parts.append(f"  ✓ Image-to-3D: {image_result.get('message', 'Success')}")
                    else:
                        summary_parts.append(f"  ✗ Image-to-3D: {image_result.get('error', 'Failed')}")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            return f"Failed to generate summary: {e}"
