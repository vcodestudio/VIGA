# run_blender_steps_anim.py
import bpy, sys, json, math
from math import radians, sin, cos
from pathlib import Path

def parse_args(argv):
    if "--" in argv: argv = argv[argv.index("--")+1:]
    else: argv = []
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", required=True, type=str)
    ap.add_argument("--out_dir", type=str, default="renders_anim")
    ap.add_argument("--cam_name", type=str, default="AgentFixedCam")
    ap.add_argument("--cam_loc", type=str, default="0,-6,4")    # x,y,z
    ap.add_argument("--cam_rot_deg", type=str, default="65,0,0")# x,y,z
    ap.add_argument("--lens", type=float, default=35.0)
    ap.add_argument("--engine", type=str, default="BLENDER_EEVEE", choices=["BLENDER_EEVEE","CYCLES"])
    ap.add_argument("--res", type=str, default="1920x1080")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--dur", type=float, default=3.0)
    ap.add_argument("--file_format", type=str, default="FFMPEG")
    ap.add_argument("--codec", type=str, default="H264")
    ap.add_argument("--bitrate", type=int, default=8000)
    ap.add_argument("--device", type=str, default="GPU", choices=["GPU","CPU"])
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--color_mgt_view", type=str, default="Filmic")
    ap.add_argument("--color_mgt_look", type=str, default="None")
    ap.add_argument("--anim_mode", type=str, default="scene", choices=["scene","orbit","dolly"])
    ap.add_argument("--orbit_deg", type=float, default=20.0)  # orbit 模式：左右摇头角度总幅度
    ap.add_argument("--dolly_amp", type=float, default=0.8)   # dolly 模式：推拉幅度（米）
    return ap.parse_args(argv)

def ensure_camera(name, loc, rot_deg, lens):
    cam = bpy.data.objects.get(name)
    if cam is None or cam.type != 'CAMERA':
        data = bpy.data.cameras.new(name)
        cam = bpy.data.objects.new(name, data)
        bpy.context.scene.collection.objects.link(cam)
    cam.location = loc
    cam.rotation_mode = 'XYZ'
    cam.rotation_euler = (radians(rot_deg[0]), radians(rot_deg[1]), radians(rot_deg[2]))
    cam.data.lens = lens
    bpy.context.scene.camera = cam
    return cam

def setup_render(engine, res_xy, fps, fmt, codec, bitrate, samples, device, view, look):
    sc = bpy.context.scene
    W, H = res_xy
    sc.render.resolution_x = W
    sc.render.resolution_y = H
    sc.render.fps = fps
    sc.view_settings.view_transform = view
    sc.view_settings.look = look
    if engine == "CYCLES":
        sc.render.engine = "CYCLES"
        sc.cycles.device = "GPU" if device == "GPU" else "CPU"
        try:
            bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
        except Exception:
            pass
        sc.cycles.samples = samples
        sc.cycles.use_adaptive_sampling = True
    else:
        sc.render.engine = "BLENDER_EEVEE"
        sc.eevee.taa_render_samples = max(1, samples)

    sc.render.image_settings.file_format = fmt  # "FFMPEG"
    sc.render.ffmpeg.format = 'MPEG4'
    sc.render.ffmpeg.codec = codec             # "H264"
    sc.render.ffmpeg.constant_rate_factor = 'MEDIUM'
    sc.render.ffmpeg.video_bitrate = bitrate
    sc.render.ffmpeg.minrate = 0
    sc.render.ffmpeg.maxrate = 0
    sc.render.ffmpeg.buffersize = 1792
    sc.render.ffmpeg.gopsize = int(fps)

def clear_cam_keyframes(cam):
    if cam.animation_data and cam.animation_data.action:
        cam.animation_data_clear()

def add_anim_orbit(cam, start, end, orbit_deg):
    # 绕 Z 轴做小幅 yaw
    cam.rotation_mode = 'XYZ'
    yaw0 = cam.rotation_euler[2]
    cam.keyframe_insert(data_path="rotation_euler", frame=start)
    cam.rotation_euler[2] = yaw0 + radians(orbit_deg)
    cam.keyframe_insert(data_path="rotation_euler", frame=(start+end)//2)
    cam.rotation_euler[2] = yaw0
    cam.keyframe_insert(data_path="rotation_euler", frame=end)

def add_anim_dolly(cam, start, end, amp):
    # 沿相机前向方向稍微推拉
    cam.keyframe_insert(data_path="location", frame=start)
    # 计算前向向量（基于相机朝向）
    import mathutils
    forward = mathutils.Vector((0,0,-1))
    forward.rotate(cam.rotation_euler)
    cam.location = cam.location + forward * amp
    cam.keyframe_insert(data_path="location", frame=(start+end)//2)
    # 回到初始
    cam.location = cam.location - forward * amp
    cam.keyframe_insert(data_path="location", frame=end)

def exec_code_safely(code_str, i):
    glb = {"bpy": bpy, "math": math}
    lcl = {}
    try:
        exec(code_str, glb, lcl)
    except Exception as e:
        print(f"[Step {i:03d}] 代码执行错误：{e}")
        import traceback; traceback.print_exc()

def main():
    args = parse_args(sys.argv)
    steps = json.loads(Path(args.steps).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    cam_loc = tuple(float(x) for x in args.cam_loc.split(","))
    cam_rot = tuple(float(x) for x in args.cam_rot_deg.split(","))
    cam = ensure_camera(args.cam_name, cam_loc, cam_rot, args.lens)

    W,H = (int(s) for s in args.res.lower().split("x"))
    setup_render(args.engine, (W,H), args.fps, args.file_format, args.codec, args.bitrate,
                 args.samples, args.device, args.color_mgt_view, args.color_mgt_look)

    sc = bpy.context.scene
    frames = max(1, int(args.dur * args.fps))
    start_f = 1
    end_f = start_f + frames - 1

    for i, step in enumerate(steps, start=1):
        print(f"[Step {i:03d}] 执行代码并准备动画...")
        exec_code_safely(step.get("code",""), i)

        # 为该步设置帧范围
        sc.frame_start, sc.frame_end = start_f, end_f

        # 如果选择自动相机动画，清除旧关键帧并添加新关键帧
        if args.anim_mode in ("orbit","dolly"):
            clear_cam_keyframes(cam)
            if args.anim_mode == "orbit":
                add_anim_orbit(cam, start_f, end_f, args.orbit_deg)
            else:
                add_anim_dolly(cam, start_f, end_f, args.dolly_amp)

        # 输出文件
        out_path = out_dir / f"step_{i:03d}.mp4"
        sc.render.filepath = str(out_path)  # 对 FFMPEG 会作为前缀使用
        try:
            bpy.ops.render.render(animation=True)
            print(f"[Step {i:03d}] 动画渲染完成：{out_path}")
        except Exception as e:
            print(f"[Step {i:03d}] 动画渲染失败：{e}")

    print("[OK] 所有步骤动画渲染完成。")

if __name__ == "__main__":
    main()
