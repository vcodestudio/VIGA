# 场景重建演示 (Scene Reconstruction Demo)

这个演示展示了如何使用AI从一张图片重建3D场景的完整流程。

## 功能概述

1. **初始化3D场景** (`init.py`): 从输入图片创建一个基础的3D场景
2. **资产生成器** (`asset.py`): 使用Meshy API生成3D资产
3. **主演示逻辑** (`demo.py`): 协调整个重建流程
4. **Meshy API集成** (`meshy.py`): 提供3D资产生成功能

## 工作流程

```
输入图片 → 初始化3D场景 → 循环分析 → 生成资产 → 场景编辑
    ↓           ↓              ↓          ↓          ↓
  init.py   创建基础场景    VLM分析     asset.py   main.py
                              ↓
                         识别缺少物体
```

## 文件结构

```
runners/demo/
├── init.py              # 场景初始化
├── asset.py             # 资产生成器
├── demo.py              # 主演示逻辑
├── meshy.py             # Meshy API集成
├── example_usage.py     # 使用示例
└── README.md           # 说明文档
```

## 安装依赖

```bash
# 基础依赖
pip install requests pillow opencv-python numpy

# Meshy API (需要API密钥)
export MESHY_API_KEY="your_meshy_api_key_here"

# 可选：AI检测后端
pip install torch torchvision ultralytics
pip install groundingdino segment-anything
```

## 使用方法

### 1. 基本使用

```python
from demo import run_demo

# 运行场景重建演示
result = run_demo(
    target_image_path="path/to/your/image.jpg",
    api_key="your_meshy_api_key",  # 可选，默认从环境变量读取
    output_dir="output/demo/reconstruction"
)
```

### 2. 使用示例脚本

```bash
# 设置API密钥
export MESHY_API_KEY="your_api_key_here"

# 运行示例
python example_usage.py
```

### 3. 单独使用组件

```python
# 初始化场景
from init import initialize_3d_scene_from_image
scene_info = initialize_3d_scene_from_image("input.jpg")

# 生成资产
from asset import AssetGenerator
generator = AssetGenerator(scene_info["blender_file_path"])
result = generator.generate_both_assets("chair", "input.jpg")
```

## 配置选项

### 环境变量

- `MESHY_API_KEY`: Meshy API密钥（必需）
- `DETECT_BACKEND`: 检测后端选择 (`grounded_sam`, `yolo`, `openai`)
- `OPENAI_API_KEY`: OpenAI API密钥（用于VLM分析）
- `OPENAI_BASE_URL`: OpenAI API基础URL

### 参数配置

```python
# 场景重建参数
demo = SceneReconstructionDemo(api_key="your_key")
demo.max_iterations = 10  # 最大循环次数

# 资产生成参数
generator = AssetGenerator(blender_path, api_key)
result = generator.generate_both_assets(
    object_name="chair",
    image_path="input.jpg",
    location="0,0,0",
    scale=1.0
)
```

## 输出结构

```
output/demo/reconstruction/
├── scene_image_1234567890.blend          # Blender场景文件
├── scene_image_1234567890_info.json      # 场景信息
└── assets/
    ├── text/                             # 文本生成的资产
    └── image/                            # 图片生成的资产
```

## API参考

### init.py

- `initialize_3d_scene_from_image(image_path, output_dir)`: 初始化3D场景
- `load_scene_info(scene_info_path)`: 加载场景信息
- `update_scene_info(scene_info_path, updates)`: 更新场景信息

### asset.py

- `AssetGenerator(blender_path, api_key)`: 资产生成器
- `generate_asset_from_text(object_name, location, scale)`: 从文本生成资产
- `generate_asset_from_image(object_name, image_path, location, scale)`: 从图片生成资产
- `generate_both_assets(object_name, image_path, location, scale)`: 生成两种资产

### demo.py

- `SceneReconstructionDemo(api_key)`: 场景重建演示类
- `run_reconstruction_loop(target_image_path, output_dir)`: 运行重建循环
- `ask_vlm_for_missing_objects(scene_info, target_image_path)`: 分析缺少的物体
- `run_demo(target_image_path, api_key, output_dir)`: 运行完整演示

### meshy.py

- `add_meshy_asset(description, blender_path, location, scale, ...)`: 添加文本生成的资产
- `add_meshy_asset_from_image(image_path, blender_path, location, scale, ...)`: 添加图片生成的资产
- `crop_image_by_text(image_path, description, ...)`: 根据文本截取图片
- `crop_and_generate_3d_asset(...)`: 截取并生成3D资产

## 扩展开发

### 添加新的VLM分析

在 `demo.py` 的 `ask_vlm_for_missing_objects` 方法中集成你的VLM服务：

```python
def ask_vlm_for_missing_objects(self, current_scene_info, target_image_path):
    # 调用你的VLM API
    # 分析目标图片和当前场景
    # 返回缺少的物体列表
    pass
```

### 添加新的资产生成方法

在 `asset.py` 中扩展 `AssetGenerator` 类：

```python
def generate_asset_from_custom_source(self, object_name, custom_data):
    # 实现自定义资产生成逻辑
    pass
```

### 集成场景编辑功能

在 `demo.py` 的 `start_scene_editing` 方法中集成你的场景编辑逻辑：

```python
def start_scene_editing(self, blender_file_path):
    # 调用main.py中的场景编辑功能
    # 实现自动场景优化
    pass
```

## 故障排除

### 常见问题

1. **API密钥错误**: 确保 `MESHY_API_KEY` 环境变量设置正确
2. **Blender导入失败**: 确保在Blender环境中运行
3. **图片路径错误**: 检查输入图片路径是否存在
4. **网络连接问题**: 确保可以访问Meshy API

### 调试模式

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 运行演示时会显示详细日志
result = run_demo("input.jpg")
```

## 贡献

欢迎提交Issue和Pull Request来改进这个演示系统。

## 许可证

请参考项目根目录的许可证文件。