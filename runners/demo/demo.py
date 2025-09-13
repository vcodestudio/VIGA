import os
import json
import logging
import time
from typing import List, Dict, Any, Optional
from .init import initialize_3d_scene_from_image, load_scene_info, update_scene_info
from .asset import AssetGenerator

class SceneReconstructionDemo:
    """场景重建演示类"""
    
    def __init__(self, api_key: str = None):
        """
        初始化演示类
        
        Args:
            api_key: Meshy API密钥（可选，默认从环境变量读取）
        """
        self.api_key = api_key or os.getenv("MESHY_API_KEY")
        if not self.api_key:
            raise ValueError("Meshy API key is required. Set MESHY_API_KEY environment variable or pass api_key parameter.")
        
        self.current_scene = None
        self.asset_generator = None
        self.max_iterations = 10  # 最大循环次数
        self.completed_objects = []  # 已完成的物体列表
    
    def ask_vlm_for_missing_objects(self, current_scene_info: Dict[str, Any], target_image_path: str) -> List[str]:
        """
        询问VLM当前场景相比目标场景缺少哪些物体
        
        Args:
            current_scene_info: 当前场景信息
            target_image_path: 目标图片路径
            
        Returns:
            List[str]: 缺少的物体名称列表
        """
        try:
            # 这里应该调用VLM API来分析当前场景和目标图片
            # 现在先用简单的模拟逻辑
            
            print(f"[VLM Analysis] Analyzing scene vs target image...")
            print(f"  - Current scene objects: {len(current_scene_info.get('objects', []))}")
            print(f"  - Target image: {target_image_path}")
            
            # 模拟VLM分析结果
            # 在实际实现中，你需要：
            # 1. 使用VLM分析目标图片，识别其中的物体
            # 2. 使用VLM分析当前场景，识别已有的物体
            # 3. 比较两者，找出缺少的物体
            
            # 简单的模拟逻辑：假设目标图片中常见的物体
            target_objects = [
                "chair", "table", "lamp", "sofa", "bookshelf", 
                "coffee_table", "bed", "desk", "television", "plant"
            ]
            
            # 获取当前场景中已有的物体
            current_objects = [obj.get("name", "").lower() for obj in current_scene_info.get("objects", [])]
            current_objects.extend(self.completed_objects)
            
            # 找出缺少的物体
            missing_objects = []
            for obj in target_objects:
                if obj not in current_objects and obj not in self.completed_objects:
                    missing_objects.append(obj)
            
            # 限制每次最多返回3个物体，避免一次性生成太多
            missing_objects = missing_objects[:3]
            
            print(f"[VLM Analysis] Found {len(missing_objects)} missing objects: {missing_objects}")
            
            return missing_objects
            
        except Exception as e:
            logging.error(f"Failed to analyze missing objects: {e}")
            return []
    
    def run_reconstruction_loop(self, target_image_path: str, output_dir: str = "output/demo/reconstruction") -> Dict[str, Any]:
        """
        运行场景重建循环
        
        Args:
            target_image_path: 目标图片路径
            output_dir: 输出目录
            
        Returns:
            dict: 重建结果
        """
        try:
            print("=" * 60)
            print("🚀 Starting Scene Reconstruction Demo")
            print("=" * 60)
            print(f"Target image: {target_image_path}")
            print(f"Output directory: {output_dir}")
            
            # 步骤1: 初始化3D场景
            print("\n📋 Step 1: Initializing 3D scene...")
            scene_init_result = initialize_3d_scene_from_image(target_image_path, output_dir)
            
            if scene_init_result.get("status") != "success":
                return {
                    "status": "error",
                    "error": f"Failed to initialize scene: {scene_init_result.get('error')}"
                }
            
            self.current_scene = scene_init_result
            self.asset_generator = AssetGenerator(
                blender_path=scene_init_result["blender_file_path"],
                api_key=self.api_key
            )
            
            print(f"✓ Scene initialized: {scene_init_result['scene_name']}")
            
            # 步骤2: 进入重建循环
            print("\n🔄 Step 2: Starting reconstruction loop...")
            iteration = 0
            reconstruction_results = []
            
            while iteration < self.max_iterations:
                iteration += 1
                print(f"\n--- Iteration {iteration} ---")
                
                # 加载当前场景信息
                scene_info = load_scene_info(scene_init_result["scene_info_path"])
                if not scene_info:
                    print("❌ Failed to load scene info")
                    break
                
                # 询问VLM缺少的物体
                missing_objects = self.ask_vlm_for_missing_objects(scene_info, target_image_path)
                
                if not missing_objects:
                    print("✅ No missing objects found. Reconstruction complete!")
                    break
                
                print(f"🎯 Missing objects: {missing_objects}")
                
                # 为每个缺少的物体生成资产
                iteration_results = []
                for obj_name in missing_objects:
                    print(f"\n🔧 Generating assets for '{obj_name}'...")
                    
                    # 生成两种资产（文本和图片）
                    asset_result = self.asset_generator.generate_both_assets(
                        object_name=obj_name,
                        image_path=target_image_path,  # 使用目标图片作为参考
                        location=f"{len(self.completed_objects) * 2},0,0",  # 避免重叠
                        scale=1.0
                    )
                    
                    # 显示结果摘要
                    summary = self.asset_generator.get_asset_summary(asset_result)
                    print(summary)
                    
                    iteration_results.append(asset_result)
                    
                    # 标记为已完成
                    self.completed_objects.append(obj_name)
                
                reconstruction_results.append({
                    "iteration": iteration,
                    "missing_objects": missing_objects,
                    "results": iteration_results
                })
                
                # 更新场景信息
                scene_info["target_objects"].extend(missing_objects)
                update_scene_info(scene_init_result["scene_info_path"], scene_info)
                
                print(f"✓ Iteration {iteration} completed. Added {len(missing_objects)} objects.")
            
            # 步骤3: 启动场景编辑（这部分留空，等待后续实现）
            print(f"\n🎨 Step 3: Starting scene editing (placeholder)...")
            editing_result = self.start_scene_editing(scene_init_result["blender_file_path"])
            
            # 返回最终结果
            final_result = {
                "status": "success",
                "message": f"Scene reconstruction completed in {iteration} iterations",
                "scene_info": scene_init_result,
                "iterations": iteration,
                "completed_objects": self.completed_objects,
                "reconstruction_results": reconstruction_results,
                "editing_result": editing_result
            }
            
            print("\n" + "=" * 60)
            print("🎉 Scene Reconstruction Demo Completed!")
            print("=" * 60)
            print(f"Total iterations: {iteration}")
            print(f"Objects added: {len(self.completed_objects)}")
            print(f"Final objects: {self.completed_objects}")
            
            return final_result
            
        except Exception as e:
            logging.error(f"Failed to run reconstruction loop: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def start_scene_editing(self, blender_file_path: str) -> Dict[str, Any]:
        """
        启动场景编辑（占位符函数，等待后续实现）
        
        Args:
            blender_file_path: Blender文件路径
            
        Returns:
            dict: 编辑结果
        """
        try:
            print(f"[Scene Editing] Starting scene editing for: {blender_file_path}")
            print("[Scene Editing] This is a placeholder function - waiting for implementation")
            
            # 这里将来会调用main.py中的场景编辑功能
            # 现在先返回占位符结果
            
            return {
                "status": "placeholder",
                "message": "Scene editing functionality not yet implemented",
                "blender_file_path": blender_file_path,
                "note": "This will be implemented in main.py"
            }
            
        except Exception as e:
            logging.error(f"Failed to start scene editing: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

def run_demo(target_image_path: str, api_key: str = None, output_dir: str = "output/demo/reconstruction") -> Dict[str, Any]:
    """
    运行场景重建演示
    
    Args:
        target_image_path: 目标图片路径
        api_key: Meshy API密钥（可选）
        output_dir: 输出目录
        
    Returns:
        dict: 演示结果
    """
    try:
        # 检查输入图片是否存在
        if not os.path.exists(target_image_path):
            return {
                "status": "error",
                "error": f"Target image not found: {target_image_path}"
            }
        
        # 创建演示实例并运行
        demo = SceneReconstructionDemo(api_key=api_key)
        result = demo.run_reconstruction_loop(target_image_path, output_dir)
        
        return result
        
    except Exception as e:
        logging.error(f"Failed to run demo: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

def test_demo():
    """
    测试演示功能
    """
    print("🧪 Testing Scene Reconstruction Demo...")
    
    # 创建测试图片路径
    test_image_path = "output/demo/test_target.png"
    
    # 确保测试目录存在
    os.makedirs(os.path.dirname(test_image_path), exist_ok=True)
    
    # 创建简单的测试图片
    if not os.path.exists(test_image_path):
        try:
            from PIL import Image, ImageDraw
            # 创建一个简单的测试图片
            img = Image.new('RGB', (800, 600), color='lightblue')
            draw = ImageDraw.Draw(img)
            
            # 画一个简单的房间场景
            # 地面
            draw.rectangle([0, 400, 800, 600], fill='brown')
            
            # 墙壁
            draw.rectangle([0, 0, 800, 400], fill='white')
            
            # 一些家具
            # 椅子
            draw.rectangle([200, 300, 250, 380], fill='darkblue')
            draw.rectangle([180, 280, 270, 300], fill='darkblue')
            
            # 桌子
            draw.rectangle([300, 350, 500, 380], fill='brown')
            draw.rectangle([310, 330, 320, 350], fill='brown')
            draw.rectangle([480, 330, 490, 350], fill='brown')
            
            # 台灯
            draw.ellipse([550, 200, 600, 250], fill='yellow')
            draw.rectangle([570, 250, 580, 350], fill='brown')
            
            img.save(test_image_path)
            print(f"✓ Created test image: {test_image_path}")
            
        except Exception as e:
            print(f"⚠ Warning: Could not create test image: {e}")
            return {"status": "error", "error": f"Failed to create test image: {e}"}
    
    # 运行演示
    try:
        result = run_demo(test_image_path)
        print(f"\n📊 Demo Result: {result.get('status', 'unknown')}")
        
        if result.get("status") == "success":
            print(f"✓ Demo completed successfully")
            print(f"  - Iterations: {result.get('iterations', 0)}")
            print(f"  - Objects added: {len(result.get('completed_objects', []))}")
        else:
            print(f"❌ Demo failed: {result.get('error', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        print(f"❌ Demo test error: {e}")
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    # 运行测试
    test_demo()
