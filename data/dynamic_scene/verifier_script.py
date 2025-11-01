import bpy
import random
import json
import os
import sys
from sys import platform

# 场景层面禁用音频同步/回放
for scene in bpy.data.scenes:
    scene.use_audio = False
    scene.sync_mode = 'NONE'  # 或 'AUDIO_SYNC' 以外的选项

# 偏好设置层面禁用音频设备（不同版本可能是 'NONE'/'NULL'/'OpenAL' 等）
try:
    bpy.context.preferences.system.audio_device = 'NONE'
except Exception:
    pass

if __name__ == "__main__":
    code_fpath = sys.argv[6]  # Path to the code file
    rendering_dir = sys.argv[7] # Path to save the rendering from camera1
    if len(sys.argv) > 8:
        save_blend = sys.argv[8] # Path to save the blend file
    else:
        save_blend = None

    with open(code_fpath, "r") as f:
        code = f.read()
    try:
        exec(code)
    except:
        raise ValueError

    # Save the blend file
    if save_blend:
        # Set the save version to 0
        bpy.context.preferences.filepaths.save_version = 0
        # Save the blend file
        bpy.ops.wm.save_as_mainfile(filepath=save_blend)