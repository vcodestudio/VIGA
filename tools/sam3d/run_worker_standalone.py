"""Run SAM worker in a subprocess with the same command init.py uses. Use to verify the worker runs.

From repo root:
  python tools/sam3d/run_worker_standalone.py

Requires conda env "sam" (see utils/_path.py ENV_MAPPING). If you see ModuleNotFoundError: cv2,
create and install:
  conda create -n sam python=3.10 -y
  conda activate sam
  pip install -r requirements/requirement_sam.txt
Then ensure utils/_path.py CONDA_BASE points to your conda envs dir so .../envs/sam/python.exe exists.
"""
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from utils._path import path_to_cmd

def main():
    image = os.path.join(ROOT, "data", "static_scene", "artist", "target.png")
    out_dir = os.path.join(ROOT, "output", "test_sam_worker")
    out_npy = os.path.join(out_dir, "all_masks.npy")
    os.makedirs(out_dir, exist_ok=True)

    python_exe = path_to_cmd.get("tools/sam3d/sam_worker.py", sys.executable)
    worker_script = os.path.join(ROOT, "tools", "sam3d", "sam_worker.py")

    cmd = [python_exe, worker_script, "--image", image, "--out", out_npy]
    print("Command:", " ".join(cmd), flush=True)
    if "sam" not in python_exe.replace("\\", "/").lower():
        print("WARNING: Python path does not contain 'sam' — conda env 'sam' may be missing. Create it with:", flush=True)
        print("  conda create -n sam python=3.10 -y && conda activate sam && pip install -r requirements/requirement_sam.txt", flush=True)
    print("", flush=True)

    result = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=False,
        text=True,
    )
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
