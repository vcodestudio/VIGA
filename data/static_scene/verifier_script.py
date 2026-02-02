"""Blender script for static scene verifier to execute code and save blend."""
import bpy
import os
import sys

if __name__ == "__main__":
    # Parse arguments after '--' separator to handle variable number of Blender flags
    try:
        separator_idx = sys.argv.index('--')
        args_after_separator = sys.argv[separator_idx + 1:]
        code_fpath = args_after_separator[0]  # Path to the code file
        rendering_dir = args_after_separator[1] if len(args_after_separator) > 1 else None  # Path to save the rendering
        save_blend = args_after_separator[2] if len(args_after_separator) > 2 else None  # Path to save the blend file
    except (ValueError, IndexError):
        raise ValueError("Usage: blender --background [flags] -- code.py [render_dir] [save_blend]")

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