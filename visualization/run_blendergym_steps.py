# run_blender_steps.py
# 在 Blender 中执行每一步 code，并用固定相机渲染图片：renders/step_001.png...
# 注意：请用 blender 的 -P 方式运行此脚本
import bpy
import sys
import json
import math
from math import radians
from pathlib import Path

def parse_args(argv):
    # Blender 启动参数里用 -- 分隔，此处解析其后的自定义参数
    if "--" in argv:
        argv = argv[argv.index("--")+1:]
    else:
        argv = []

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", required=True, type=str, help="steps.json")
    ap.add_argument("--out_dir", type=str, default="renders")
    ap.add_argument("--cam_name", type=str, default="AgentFixedCam")
    ap.add_argument("--cam_loc", type=str, default="0,-6,4", help="x,y,z")
    ap.add_argument("--cam_rot_deg", type=str, default="65,0,0", help="x,y,z (degrees, Euler XYZ)")
    ap.add_argument("--lens", type=float, default=35.0)
    ap.add_argument("--engine", type=str, default="BLENDER_EEVEE", choices=["BLENDER_EEVEE", "CYCLES"])
    ap.add_argument("--res", type=str, default="1920x1080")
    ap.add_argument("--device", type=str, default="GPU", choices=["GPU","CPU"], help="仅对 Cycles 有效")
    ap.add_argument("--samples", type=int, default=64, help="渲染采样")
    ap.add_argument("--color_mgt_view", type=str, default="Filmic")
    ap.add_argument("--color_mgt_look", type=str, default="None")
    ap.add_argument("--file_format", type=str, default="PNG", choices=["PNG","JPEG"])
    args = ap.parse_args(argv)
    return args

def ensure_camera(name, loc, rot_deg, lens):
    # 获取或新建相机
    cam_obj = bpy.data.objects.get(name)
    if cam_obj is None or cam_obj.type != 'CAMERA':
        cam_data = bpy.data.cameras.new(name)
        cam_obj = bpy.data.objects.new(name, cam_data)
        bpy.context.scene.collection.objects.link(cam_obj)
    # 设置参数
    cam_obj.location = loc
    cam_obj.rotation_mode = 'XYZ'
    cam_obj.rotation_euler = (radians(rot_deg[0]), radians(rot_deg[1]), radians(rot_deg[2]))
    cam_obj.data.lens = lens
    bpy.context.scene.camera = cam_obj
    return cam_obj

def setup_render(engine, res_xy, file_format, samples, device, view="Filmic", look="None"):
    scene = bpy.context.scene
    W, H = res_xy
    scene.render.resolution_x = W
    scene.render.resolution_y = H
    scene.render.image_settings.file_format = file_format
    scene.view_settings.view_transform = view
    scene.view_settings.look = look

    if engine == "CYCLES":
        scene.render.engine = "CYCLES"
        prefs = bpy.context.preferences
        cycles = scene.cycles
        # 设备
        cycles.device = "GPU" if device == "GPU" else "CPU"
        # GPU 打开
        try:
            prefs.addons["cycles"].preferences.compute_device_type = "CUDA"  # 或 "OPTIX" / "HIP" / "METAL"
        except Exception:
            pass
        cycles.samples = samples
        cycles.use_adaptive_sampling = True
    else:
        scene.render.engine = "BLENDER_EEVEE"
        scene.eevee.taa_render_samples = max(1, samples)

def exec_code_safely(code_str, step_idx):
    # 在 Blender 环境执行用户代码（建议你的代码是幂等/增量安全的）
    # 为避免污染内置命名空间，用一个隔离 dict；同时保留 bpy 可用
    glb = {"bpy": bpy, "math": math}
    lcl = {}
    try:
        exec(code_str, glb, lcl)
    except Exception as e:
        print(f"[Step {step_idx:03d}] 执行代码出错：{e}")
        import traceback; traceback.print_exc()

def main():
    args = parse_args(sys.argv)
    steps = json.loads(Path(args.steps).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 相机
    cam_loc = tuple(float(x) for x in args.cam_loc.split(","))
    cam_rot_deg = tuple(float(x) for x in args.cam_rot_deg.split(","))
    cam = ensure_camera(args.cam_name, cam_loc, cam_rot_deg, args.lens)

    # 渲染设置
    W, H = (int(s) for s in args.res.lower().split("x"))
    setup_render(args.engine, (W, H), args.file_format, args.samples, args.device,
                 view=args.color_mgt_view, look=args.color_mgt_look)

    # 顺序执行每一步 code，并渲染
    for i, step in enumerate(steps, start=1):
        code = step.get("code", "")
        print(f"[Step {i:03d}] 执行代码...")
        exec_code_safely(code, i)

        # 固定相机渲染
        bpy.context.scene.camera = cam
        out_path = out_dir / f"step_{i:03d}.png"
        bpy.context.scene.render.filepath = str(out_path)
        try:
            bpy.ops.render.render(write_still=True)
            print(f"[Step {i:03d}] 渲染完毕：{out_path}")
        except Exception as e:
            print(f"[Step {i:03d}] 渲染失败：{e}")

    print("[OK] 所有步骤执行并完成渲染。")

if __name__ == "__main__":
    main()
