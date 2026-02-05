"""SAM Bridge MCP Server.

Segment-only: SAM segments the image, cropped images are exported.
No 3D mesh/3DGS generation (use tools/sam3d/init.py with COMFYUI_API_URL for that).
"""

import json
import os
import subprocess
import sys
from typing import Dict, List, Optional

from mcp.server.fastmcp import FastMCP

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils._path import path_to_cmd

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

mcp = FastMCP("sam-bridge")

_target_image: Optional[str] = None
_output_dir: Optional[str] = None
_sam_env_bin: Optional[str] = None

try:
    _sam_env_bin = path_to_cmd.get("tools/sam3d/sam_worker.py")
except Exception:
    pass


def _crop_image_to_mask_bbox(image, mask, padding=20):
    import numpy as np
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


@mcp.tool()
def initialize(args: Dict[str, object]) -> Dict[str, object]:
    """Initialize the SAM bridge (target image and output dir)."""
    global _target_image, _output_dir, _sam_env_bin
    _target_image = args["target_image_path"]
    _output_dir = args.get("output_dir", "") + "/sam"
    os.makedirs(_output_dir, exist_ok=True)
    _sam_env_bin = path_to_cmd.get("tools/sam3d/sam_worker.py")
    return {
        "status": "success",
        "output": {"text": ["SAM bridge initialized (segment + export only)"], "tool_configs": tool_configs},
    }


@mcp.tool()
def get_better_object(object_name: str) -> Dict[str, object]:
    """Segment image, crop the requested object, and save cropped image. No 3D generation."""
    import numpy as np
    import cv2

    if not _target_image or not _output_dir:
        return {"status": "error", "output": {"text": ["Call initialize first"]}}

    normalized_name = object_name.strip().replace(" ", "_").replace("-", "_")
    all_masks_path = os.path.join(_output_dir, "all_masks.npy")
    mapping_path = all_masks_path.replace(".npy", "_object_names.json")

    try:
        if not os.path.exists(all_masks_path) or not os.path.exists(mapping_path):
            subprocess.run(
                [_sam_env_bin, SAM_WORKER, "--image", _target_image, "--out", all_masks_path],
                cwd=ROOT,
                check=True,
                text=True,
                capture_output=True,
            )

        img = cv2.imread(_target_image)
        if img is None:
            return {"status": "error", "output": {"text": [f"Could not load image: {_target_image}"]}}

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

        return {
            "status": "success",
            "output": {"text": [f"Cropped image saved to: {cropped_path}"]},
        }
    except Exception as e:
        return {"status": "error", "output": {"text": [str(e)]}}


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        initialize({
            "target_image_path": "data/static_scene/blackhouse25/target.jpeg",
            "output_dir": os.path.join(ROOT, "output", "test", "sam3"),
        })
        print(get_better_object("house"))
    else:
        mcp.run()


if __name__ == "__main__":
    main()
