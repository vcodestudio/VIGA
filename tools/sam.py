import os, sys, json, subprocess
from mcp.server.fastmcp import FastMCP
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils._path import path_to_cmd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAM3_WORKER = os.path.join(os.path.dirname(__file__), "sam3_worker.py")
SAM3D_WORKER = os.path.join(os.path.dirname(__file__), "sam3d_worker.py")

mcp = FastMCP("sam-bridge")
_target_image = _output_dir = _sam3_cfg = None
_sam3_env_bin = path_to_cmd["tools/sam3_worker.py"]
_sam3d_env_bin = path_to_cmd["tools/sam3d_worker.py"]


@mcp.tool()
def initialize(args: dict) -> dict:
    global _target_image, _output_dir, _sam3_cfg
    _target_image = args["target_image_path"]
    _output_dir = args.get("output_dir") or os.path.join(ROOT, "output", "sam_bridge")
    os.makedirs(_output_dir, exist_ok=True)
    _sam3_cfg = args.get("sam3d_config_path") or os.path.join(
        ROOT, "utils", "sam3d", "checkpoints", "hf", "pipeline.yaml"
    )
    return {"status": "success", "output": {"text": ["sam bridge init ok"]}}


@mcp.tool()
def extract_3d_object(object_name: str) -> dict:
    if not _target_image or not _output_dir:
        return {"status": "error", "output": {"text": ["call initialize first"]}}
    mask_path = os.path.join(_output_dir, f"{object_name}_mask.npy")
    glb_path = os.path.join(_output_dir, f"{object_name}.glb")

    subprocess.run(
        [
            _sam3_env_bin,
            SAM3_WORKER,
            "--image",
            _target_image,
            "--object",
            object_name,
            "--out",
            mask_path,
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    r2 = subprocess.run(
        [
            _sam3d_env_bin,
            SAM3D_WORKER,
            "--image",
            _target_image,
            "--mask",
            mask_path,
            "--config",
            _sam3_cfg,
            "--glb",
            glb_path,
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    info = json.loads(r2.stdout.strip().splitlines()[-1])
    info["glb_path"] = info.get("glb_path") or glb_path
    return {"status": "success", "output": {"text": [f"glb: {info['glb_path']}", f"data: {info}"]}}


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        initialize(
            {
                "target_image_path": "data/static_scene/christmas/target.png",
                "output_dir": os.path.join(ROOT, "output", "test", "sam3"),
            }
        )
        print(extract_3d_object("snowman"))
    else:
        mcp.run()


if __name__ == "__main__":
    main()


