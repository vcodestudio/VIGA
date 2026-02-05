"""SAM Scene Initialization MCP Server.

Segment-only flow: SAM segments the image, cropped images are exported
and optionally sent to ComfyUI API (no local 3D mesh/3DGS generation).
"""
# pyright: reportMissingModuleSource=false, reportMissingImports=false

import json
import os
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional

import requests
from mcp.server.fastmcp import FastMCP

# numpy is only needed for local mode (not when using remote API)
# Lazy import to avoid ModuleNotFoundError when running in remote API mode
np = None

def _ensure_numpy():
    """Lazy load numpy only when needed (local mode)."""
    global np
    if np is None:
        import numpy
        np = numpy
    return np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils._path import path_to_cmd

# Tool config for get_better_object (agent-facing)
tool_configs: List[Dict[str, object]] = [
    {
        "type": "function",
        "function": {
            "name": "get_better_object",
            "description": (
                "Generate high-quality 3D assets, download them locally, and provide "
                "their paths for later use. The textures, materials, and finishes of "
                "these objects are already high-quality with fine-grained detail; "
                "please do not repaint them. If you do, you will need to re-import "
                "the object."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "object_name": {
                        "type": "string",
                        "description": (
                            "The name of the object to download. For example, "
                            "'chair', 'table', 'lamp', etc."
                        ),
                    },
                },
                "required": ["object_name"],
            },
        },
    }
]

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SAM_WORKER = os.path.join(os.path.dirname(__file__), "sam_worker.py")
WORKFLOW_PATH = os.path.join(os.path.dirname(__file__), "workflows", "comfy_workflow.json")
LOAD_IMAGE_NODE_ID = "76"

mcp = FastMCP("sam-init")

# Global state (segment + export only; no 3D)
_target_image: Optional[str] = None
_output_dir: Optional[str] = None  # sam_init: segment/crop outputs
_assets_dir: Optional[str] = None  # assets: GLB save dir (same as Meshy)
_log_file: Optional[object] = None
_sam_env_bin: Optional[str] = None
_comfyui_url: Optional[str] = os.getenv("COMFYUI_API_URL")

try:
    _sam_env_bin = path_to_cmd.get("tools/sam3d/sam_worker.py")
except Exception as e:
    print(f"[SAM_INIT] Error initializing paths: {e}", file=sys.stderr)


def log(message: str) -> None:
    print(message, file=sys.stderr)
    if _log_file is not None:
        try:
            _log_file.write(message + "\n")
            _log_file.flush()
        except Exception:
            pass


def get_conda_prefix_from_python_path(python_path: str) -> Optional[str]:
    if not python_path:
        return None
    normalized_path = python_path.replace("\\", "/") if os.path.isabs(python_path) else os.path.abspath(python_path).replace("\\", "/")
    if normalized_path.endswith("/bin/python") or normalized_path.endswith("/bin/python3"):
        return os.path.dirname(os.path.dirname(normalized_path))
    if "/envs/" in normalized_path:
        parts = normalized_path.split("/envs/")
        if len(parts) == 2:
            return os.path.join(parts[0], "envs", parts[1].split("/")[0])
    if "/bin/python" in normalized_path:
        idx = normalized_path.rfind("/bin/python")
        return normalized_path[:idx]
    return None


def prepare_env_with_conda_prefix(python_path: str) -> Dict[str, str]:
    env = os.environ.copy()
    conda_prefix = get_conda_prefix_from_python_path(python_path)
    if conda_prefix:
        env["CONDA_PREFIX"] = conda_prefix
    return env


def _crop_image_to_mask_bbox(image: np.ndarray, mask: np.ndarray, padding: int = 20) -> np.ndarray:
    """Crop image to bounding box of mask (padding applied). Returns cropped image only."""
    if mask.ndim == 3:
        mask = mask.squeeze()
    rows, cols = np.where(mask > 0)
    if rows.size == 0 or cols.size == 0:
        raise ValueError("Mask has no foreground pixels")
    rmin, rmax = int(rows.min()), int(rows.max()) + 1
    cmin, cmax = int(cols.min()), int(cols.max()) + 1
    h, w = image.shape[:2]
    rmin = max(0, rmin - padding)
    rmax = min(h, rmax + padding)
    cmin = max(0, cmin - padding)
    cmax = min(w, cmax + padding)
    return image[rmin:rmax, cmin:cmax].copy()


def _match_object_name(prompt: str, object_names: List[str]) -> Optional[str]:
    """Match user object_name to a segment name (object_1, object_2, or VLM name)."""
    prompt_lower = prompt.strip().lower().replace(" ", "_").replace("-", "_")
    if not prompt_lower:
        return object_names[0] if object_names else None
    for name in object_names:
        if name.lower() == prompt_lower:
            return name
    for name in object_names:
        if prompt_lower in name.lower():
            return name
    for name in object_names:
        if name.lower() in prompt_lower:
            return name
    if len(object_names) == 1:
        return object_names[0]
    return None


def _run_sam_segment() -> None:
    """Run SAM worker to produce all_masks.npy and object mapping in _output_dir."""
    all_masks_path = os.path.join(_output_dir, "all_masks.npy")
    sam_log_path = os.path.join(_output_dir, "sam_worker.log")
    env = prepare_env_with_conda_prefix(_sam_env_bin) if _sam_env_bin else os.environ.copy()
    with open(sam_log_path, "w") as log_file:
        subprocess.run(
            [_sam_env_bin, SAM_WORKER, "--image", _target_image, "--out", all_masks_path],
            cwd=ROOT,
            check=True,
            text=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
        )


def _load_segment_results() -> tuple:
    """Load or run SAM, return (image as numpy BGR, list of mask arrays, object_mapping)."""
    all_masks_path = os.path.join(_output_dir, "all_masks.npy")
    mapping_path = all_masks_path.replace(".npy", "_object_names.json")

    if not os.path.exists(all_masks_path) or not os.path.exists(mapping_path):
        _run_sam_segment()

    import cv2
    img = cv2.imread(_target_image)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {_target_image}")

    masks = np.load(all_masks_path, allow_pickle=True)
    if masks.dtype == object:
        masks = [m for m in masks]
    elif masks.ndim == 3:
        masks = [masks[i] for i in range(masks.shape[0])]
    else:
        masks = [masks]

    with open(mapping_path, "r") as f:
        object_mapping = json.load(f).get("object_mapping", [])
    while len(object_mapping) < len(masks):
        object_mapping.append(f"object_{len(object_mapping)}")

    return img, masks, object_mapping


def _comfyui_upload_image(cropped_path: str) -> str:
    """Upload image to ComfyUI; return filename as used in workflow."""
    url = f"{_comfyui_url.rstrip('/')}/upload/image"
    with open(cropped_path, "rb") as f:
        files = {"image": (os.path.basename(cropped_path), f, "image/png")}
        r = requests.post(url, files=files, timeout=60)
    r.raise_for_status()
    out = r.json()
    name = out.get("name") or out.get("filename") or os.path.basename(cropped_path)
    return name


def _comfyui_run_workflow(image_filename: str) -> Dict:
    """Run ComfyUI workflow with LoadImage node 76 set to image_filename; return history result."""
    with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
        workflow = json.load(f)

    # Set node 76 (LoadImage) input to uploaded image filename
    if LOAD_IMAGE_NODE_ID not in workflow:
        raise ValueError(f"Workflow missing LoadImage node {LOAD_IMAGE_NODE_ID}")
    workflow[LOAD_IMAGE_NODE_ID]["inputs"] = workflow[LOAD_IMAGE_NODE_ID].get("inputs", {})
    workflow[LOAD_IMAGE_NODE_ID]["inputs"]["image"] = image_filename

    # ComfyUI prompt format: prompt is the graph
    prompt_payload = {"prompt": workflow}
    r = requests.post(f"{_comfyui_url.rstrip('/')}/prompt", json=prompt_payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    prompt_id = data.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"ComfyUI did not return prompt_id: {data}")

    # Poll history until finished
    for _ in range(600):
        r = requests.get(f"{_comfyui_url.rstrip('/')}/history/{prompt_id}", timeout=10)
        r.raise_for_status()
        hist = r.json()
        if prompt_id in hist:
            return hist[prompt_id]
        time.sleep(1)
    raise TimeoutError("ComfyUI workflow did not finish in time")


def _comfyui_download_outputs(history: Dict, object_name: str) -> Optional[str]:
    """From ComfyUI history, find output files (e.g. GLB) and save to _assets_dir (Meshy-compatible). Return local path."""
    outputs = history.get("outputs", {})
    saved_path = None
    base = _comfyui_url.rstrip("/")
    # Save GLB to assets dir (same as Meshy) so path is output_dir/assets/object_name.glb
    save_dir = _assets_dir if _assets_dir else _output_dir

    def try_download(item: str, subfolder: str = "", type_name: str = "output") -> Optional[str]:
        try:
            r = requests.get(base + "/view", params={"filename": item, "subfolder": subfolder, "type": type_name}, timeout=30)
            r.raise_for_status()
            ext = os.path.splitext(item)[1]
            local = os.path.join(save_dir, f"{object_name}{ext}")
            os.makedirs(save_dir, exist_ok=True)
            with open(local, "wb") as f:
                f.write(r.content)
            return local
        except Exception:
            return None

    for node_id, node_out in outputs.items():
        if "gizmos" in node_out:
            continue
        for key, val in node_out.items():
            if key == "images" and isinstance(val, list):
                for entry in val:
                    if isinstance(entry, dict):
                        fn = entry.get("filename")
                        if isinstance(fn, str) and (fn.endswith(".glb") or fn.endswith(".ply") or fn.endswith(".png")):
                            path = try_download(fn, entry.get("subfolder", ""), entry.get("type", "output"))
                            if path and (saved_path is None or path.endswith(".glb")):
                                saved_path = path
            elif isinstance(val, list) and val and isinstance(val[0], dict):
                for entry in val:
                    fn = entry.get("filename")
                    if isinstance(fn, str) and (fn.endswith(".glb") or fn.endswith(".ply")):
                        path = try_download(fn, entry.get("subfolder", ""), entry.get("type", "output"))
                        if path:
                            saved_path = path
    return saved_path


@mcp.tool()
def initialize(args: Dict[str, object]) -> Dict[str, object]:
    """Initialize SAM segment-only flow (target image and output dir)."""
    global _target_image, _output_dir, _assets_dir, _log_file, _sam_env_bin
    try:
        _target_image = args["target_image_path"]
        base_out = args.get("output_dir", "")
        _output_dir = base_out + "/sam_init"
        _assets_dir = base_out + "/assets"  # GLB save dir, same as Meshy
        if os.path.exists(_output_dir):
            shutil.rmtree(_output_dir)
        os.makedirs(_output_dir, exist_ok=True)
        os.makedirs(_assets_dir, exist_ok=True)

        log_path = os.path.join(_output_dir, "sam_init.log")
        _log_file = open(log_path, "w", encoding="utf-8")
        log(f"[SAM_INIT] Initialized. Log file: {log_path}")

        _sam_env_bin = path_to_cmd.get("tools/sam3d/sam_worker.py")

        log("[SAM_INIT] sam init initialized (segment + export only)")
        return {"status": "success", "output": {"text": ["sam init initialized"], "tool_configs": tool_configs}}
    except Exception as e:
        error_msg = f"Error in initialize: {str(e)}"
        if _log_file:
            log(f"[SAM_INIT] {error_msg}")
        else:
            print(f"[SAM_INIT] {error_msg}", file=sys.stderr)
        return {"status": "error", "output": {"text": [error_msg]}}


@mcp.tool()
def get_better_object(object_name: str) -> Dict[str, object]:
    """Segment image, crop the requested object, export image and optionally send to ComfyUI."""
    global _target_image, _output_dir, _comfyui_url

    if not _target_image or not _output_dir:
        return {"status": "error", "output": {"text": ["Call initialize first"]}}

    normalized_name = object_name.strip().replace(" ", "_").replace("-", "_")

    try:
        import cv2
        img, masks, object_mapping = _load_segment_results()
        chosen = _match_object_name(normalized_name, object_mapping)
        if chosen is None:
            return {
                "status": "error",
                "output": {"text": [f"No object matching '{object_name}'. Found: {object_mapping}"]},
            }

        idx = object_mapping.index(chosen)
        mask = masks[idx]
        if mask.ndim > 2:
            mask = mask.squeeze()
        if mask.dtype != np.uint8 or mask.max() <= 1:
            mask = (mask > 0).astype(np.uint8) * 255

        cropped = _crop_image_to_mask_bbox(img, mask, padding=20)
        cropped_path = os.path.join(_output_dir, f"{normalized_name}_cropped.png")
        cv2.imwrite(cropped_path, cropped)
        log(f"[SAM_INIT] Cropped image saved: {cropped_path}")

        _comfyui_url = _comfyui_url or os.getenv("COMFYUI_API_URL")
        if _comfyui_url:
            filename = _comfyui_upload_image(cropped_path)
            log(f"[SAM_INIT] Uploaded to ComfyUI as {filename}")
            history = _comfyui_run_workflow(filename)
            result_path = _comfyui_download_outputs(history, normalized_name)
            if result_path:
                # Same response format as Meshy for generator memory ("downloaded to: " + path)
                return {
                    "status": "success",
                    "output": {"text": [f"Successfully generated static asset, downloaded to: {result_path}"]},
                }
            return {
                "status": "success",
                "output": {"text": [f"Cropped image sent to ComfyUI, saved to: {cropped_path}"]},
            }
        return {
            "status": "success",
            "output": {"text": [f"Cropped image saved to: {cropped_path}"]},
        }
    except Exception as e:
        log(f"[SAM_INIT] get_better_object failed: {e}")
        return {"status": "error", "output": {"text": [str(e)]}}


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        task = sys.argv[2] if len(sys.argv) > 2 else "test"
        initialize({
            "target_image_path": f"data/static_scene/{task}/target.png",
            "output_dir": os.path.join(ROOT, "output", "test", task),
        })
        result = get_better_object("object_1")
        print(json.dumps(result, indent=2), file=sys.stderr)
    else:
        mcp.run()


if __name__ == "__main__":
    main()
