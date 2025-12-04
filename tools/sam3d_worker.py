#!/usr/bin/env python
"""
Worker process that runs inside the `sam3d-objects` Conda environment to
reconstruct 3D assets using SAM-3D. It communicates via JSON lines over
stdin/stdout. Expected request format:
{
    "command": "reconstruct",
    "image_path": "/abs/path/to/img.png",
    "mask_path": "/abs/path/to/mask.npy",
    "glb_path": "/abs/path/to/output.glb",
    "seed": 42
}
The worker responds with:
{
    "status": "success",
    "glb_path": "/abs/path/to/output.glb",
    "translation": [...],
    "rotation": [...],
    "scale": [...]
}
"""

import argparse
import json
import os
import sys
import traceback

import numpy as np
from PIL import Image


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAM3D_PATH = os.path.join(REPO_ROOT, "utils", "sam3d", "notebook")
if SAM3D_PATH not in sys.path:
    sys.path.insert(0, SAM3D_PATH)

from inference import Inference  # noqa: E402


def log_exception(prefix: str, exc: Exception) -> None:
    traceback.print_exc()
    sys.stderr.write(f"{prefix}: {exc}\n")
    sys.stderr.flush()


def build_response(status: str, **kwargs) -> str:
    payload = {"status": status}
    payload.update(kwargs)
    return json.dumps(payload)


def sanitize_array(value):
    if value is None:
        return None
    if hasattr(value, "cpu"):
        value = value.cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    value = np.array(value)
    if value.size == 0:
        return None
    return value.flatten().tolist()


def load_mask(mask_path: str, target_hw):
    mask = np.load(mask_path)
    if mask.dtype != np.uint8:
        if mask.max() <= 1.0:
            mask = (mask * 255).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)

    if mask.ndim == 3:
        mask = mask[:, :, 0]

    if mask.shape[:2] != target_hw:
        mask_img = Image.fromarray(mask, mode="L")
        mask_img = mask_img.resize((target_hw[1], target_hw[0]), resample=Image.NEAREST)
        mask = np.array(mask_img)

    mask = (mask > 127).astype(np.uint8) * 255
    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True, help="Path to SAM-3D pipeline.yaml")
    args = parser.parse_args()

    inference = Inference(args.config_path, compile=False)

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            sys.stdout.write(build_response("error", message="Invalid JSON") + "\n")
            sys.stdout.flush()
            continue

        command = request.get("command")
        if command == "close":
            sys.stdout.write(build_response("success", message="closing") + "\n")
            sys.stdout.flush()
            break

        if command == "ping":
            sys.stdout.write(build_response("success", message="pong") + "\n")
            sys.stdout.flush()
            continue

        if command != "reconstruct":
            sys.stdout.write(build_response("error", message=f"Unknown command {command}") + "\n")
            sys.stdout.flush()
            continue

        image_path = request.get("image_path")
        mask_path = request.get("mask_path")
        glb_path = request.get("glb_path")
        seed = request.get("seed", 42)

        if not image_path or not os.path.exists(image_path):
            sys.stdout.write(build_response("error", message=f"Image not found: {image_path}") + "\n")
            sys.stdout.flush()
            continue

        if not mask_path or not os.path.exists(mask_path):
            sys.stdout.write(build_response("error", message=f"Mask not found: {mask_path}") + "\n")
            sys.stdout.flush()
            continue

        if not glb_path:
            sys.stdout.write(build_response("error", message="glb_path is required") + "\n")
            sys.stdout.flush()
            continue

        try:
            image = Image.open(image_path)
            image_array = np.array(image)
            mask = load_mask(mask_path, target_hw=image_array.shape[:2])

            output = inference(image_array, mask, seed=seed)
            translation = sanitize_array(output.get("translation"))
            rotation = sanitize_array(output.get("rotation"))
            scale = sanitize_array(output.get("scale"))

            glb_obj = output.get("glb")
            saved_glb_path = None
            if glb_obj is not None and hasattr(glb_obj, "export"):
                os.makedirs(os.path.dirname(glb_path), exist_ok=True)
                glb_obj.export(glb_path)
                saved_glb_path = glb_path

            response = {
                "status": "success",
                "glb_path": saved_glb_path,
                "translation": translation,
                "rotation": rotation,
                "scale": scale,
            }
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
        except Exception as exc:
            log_exception("[sam3d_worker] reconstruction failed", exc)
            sys.stdout.write(build_response("error", message=str(exc)) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()

