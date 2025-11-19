import bpy
import math
from mathutils import Vector, Matrix

# ==============================
# 可根据需求修改
# ==============================
CAM_START = Vector((0.0, -1.0, 1.5))
CAM_END   = Vector((0.0,  1.0, 1.5))
LOOK_AT   = Vector((0.0,  0.0, 1.0))   # 这里也会作为圆心使用
CAM_LENS = 22

TARGET_TOTAL_FRAMES = 45   # 最终渲染的总帧数
FPS = 3
OUTPUT_PATH = "//table.mp4"
# ==============================


def enable_gpu_for_cycles():
    """
    将 Cycles 渲染设备切换到 GPU。
    优先尝试 OPTIX，其次 CUDA；如果都不可用则退回 CPU。
    """
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"

    prefs_all = bpy.context.preferences

    # 确保 Cycles 插件已启用
    if "cycles" not in prefs_all.addons:
        try:
            bpy.ops.preferences.addon_enable(module="cycles")
        except Exception as e:
            print(f"[WARN] Cannot enable Cycles addon automatically: {e}")
            return

    try:
        cycles_prefs = prefs_all.addons["cycles"].preferences
    except KeyError:
        print("[WARN] Cycles addon not found in preferences; fallback to CPU.")
        return

    backend_chosen = None
    for backend in ("OPTIX", "CUDA"):
        try:
            cycles_prefs.compute_device_type = backend
            backend_chosen = backend
            break
        except TypeError:
            continue

    if backend_chosen is None:
        print("[WARN] No supported GPU backend (CUDA/OPTIX); fallback to CPU.")
        return

    # 刷新设备列表
    try:
        cycles_prefs.refresh_devices()
    except TypeError:
        cycles_prefs.refresh_devices(bpy.context)

    # 只启用 GPU 设备
    for dev in cycles_prefs.devices:
        dev.use = (dev.type == "GPU" or dev.type == backend_chosen)
        print(f"[GPU] {dev.type}: {dev.name}, use={dev.use}")

    scene.cycles.device = "GPU"
    print(f"[INFO] Cycles is now using GPU with backend = {backend_chosen}.")

def ensure_camera():
    """若场景没有摄像机，则创建一个新摄像机。"""
    for obj in bpy.data.objects:
        if obj.type == "CAMERA":
            bpy.context.scene.camera = obj
            return obj

    cam_data = bpy.data.cameras.new("AutoCamera")
    cam_data.lens = CAM_LENS
    cam = bpy.data.objects.new("AutoCamera", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    return cam


def point_camera_to_target(camera, target_vec: Vector):
    """让摄像机看向 target_vec 的方向."""
    direction = (target_vec - camera.location).normalized()
    quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = quat.to_euler()


def animate_camera(camera):
    """
    让摄像机沿着一个“上半圆弧”运动：
    - 起点为 CAM_START
    - 终点为 CAM_END
    - 圆心为 LOOK_AT（例如 (0,0,1)）
    - 半径由 CAM_START 到 LOOK_AT 的距离决定
      路径将经过圆的最高点（即“上半弧”，例如 (0,0,2)）
    """
    scene = bpy.context.scene

    # 圆心：这里直接使用 LOOK_AT
    center = LOOK_AT.copy()

    # 半径：起点到圆心的距离
    radius_vec = CAM_START - center
    radius = radius_vec.length

    if radius == 0:
        print("[WARN] CAM_START 恰好在 LOOK_AT 上，无法形成圆弧，放弃动画。")
        return

    # 假设圆在平面 x = center.x 上（即在 YZ 平面中画一个圆）
    # 对于你现在的设置：center = (0,0,1) → 半径 1 → 上半弧经过 (0,0,2)
    # 我们让角度 theta 从 -π/2 → +π/2：
    #   theta = -π/2 时：y = -1, z = 1   → CAM_START
    #   theta =  0     时：y =  0, z = 2   → 最高点
    #   theta = +π/2   时：y =  1, z = 1   → CAM_END

    for f in range(TARGET_TOTAL_FRAMES):
        t = f / (TARGET_TOTAL_FRAMES - 1) if TARGET_TOTAL_FRAMES > 1 else 0.0

        # 角度插值：-90° → +90°
        theta = -0.5 * math.pi + t * math.pi  # [-π/2, +π/2]

        # 在圆上取点：x 固定，y/z 按正弦/余弦变化
        x = center.x
        y = center.y + radius * math.sin(theta)
        z = center.z + radius * math.cos(theta)
        pos = Vector((x, y, z))

        scene.frame_set(f)
        camera.location = pos

        # 始终看向 LOOK_AT
        point_camera_to_target(camera, LOOK_AT)

        camera.keyframe_insert(data_path="location", frame=f)
        camera.keyframe_insert(data_path="rotation_euler", frame=f)

    print(f"[INFO] Arc camera animation created for {TARGET_TOTAL_FRAMES} frames.")


def retime_all_existing_animation():
    """
    把原始动画时间线从 (original_start → original_end)
    自动扩展/压缩到 (0 → TARGET_TOTAL_FRAMES - 1)。
    """
    scene = bpy.context.scene

    orig_start = scene.frame_start
    orig_end   = 30

    print(f"[INFO] Original animation range: {orig_start} → {orig_end}")
    if orig_end <= orig_start:
        print("[WARN] Original animation invalid. Skipping retime.")
        return

    factor = (TARGET_TOTAL_FRAMES - 1) / (orig_end - orig_start)

    print(f"[INFO] Retiming factor = {factor:.4f}")

    # 对所有 F-Curve 进行时间缩放
    for action in bpy.data.actions:
        for fcurve in action.fcurves:
            for key in fcurve.keyframe_points:
                old_frame = key.co.x
                new_frame = (old_frame - orig_start) * factor
                key.co.x = new_frame
                key.handle_left.x = new_frame
                key.handle_right.x = new_frame

    scene.frame_start = 0
    scene.frame_end = TARGET_TOTAL_FRAMES - 1

    print("[INFO] All original animation retimed.")


def setup_render():
    scene = bpy.context.scene
    scene.render.filepath = OUTPUT_PATH
    scene.render.engine = "CYCLES"

    scene.render.fps = FPS
    scene.frame_start = 0
    scene.frame_end = TARGET_TOTAL_FRAMES - 1

    scene.render.resolution_x = 1080
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100
    scene.render.use_motion_blur = False

    scene.render.image_settings.file_format = "FFMPEG"
    ffmpeg = scene.render.ffmpeg
    ffmpeg.format = "MPEG4"
    ffmpeg.codec = "H264"
    ffmpeg.constant_rate_factor = "MEDIUM"
    ffmpeg.ffmpeg_preset = "GOOD"

    print("[INFO] Render setup complete.")


def main():
    print("[INFO] Beginning processing...")

    enable_gpu_for_cycles()

    # 1. 扩展原有动画
    retime_all_existing_animation()

    # 2. 创建摄像机弧线动画
    cam = ensure_camera()
    animate_camera(cam)

    # 3. 设置渲染
    setup_render()

    # 4. 开始渲染
    print("[INFO] Rendering animation...")
    bpy.ops.render.render(animation=True)
    print("[INFO] Rendering finished.")


if __name__ == "__main__":
    main()
