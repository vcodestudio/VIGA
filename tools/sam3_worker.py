#!/usr/bin/env python
"""
Worker process that runs inside the `sam3` Conda environment to extract masks
using SAM3. It communicates via JSON lines over stdin/stdout. Each incoming JSON
object must contain:
{
    "command": "extract_mask",
    "image_path": "/abs/path/to/image.png",
    "object_name": "chair",
    "mask_dir": "/abs/path/for/masks"
}
The worker responds with:
{
    "status": "success",
    "mask_path": "/abs/path/to/generated_mask.npy"
}
"""

import json
import os
import sys
import uuid
import traceback

import numpy as np
from PIL import Image


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAM3_PATH = os.path.join(REPO_ROOT, "utils", "sam3")
if SAM3_PATH not in sys.path:
    sys.path.insert(0, SAM3_PATH)

from sam3.model_builder import build_sam3_image_model  # noqa: E402
from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402


def log_exception(prefix: str, exc: Exception) -> None:
    traceback.print_exc()
    sys.stderr.write(f"{prefix}: {exc}\n")
    sys.stderr.flush()


def build_response(status: str, **kwargs) -> str:
    payload = {"status": status}
    payload.update(kwargs)
    return json.dumps(payload)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_mask(mask: np.ndarray, mask_dir: str) -> str:
    ensure_dir(mask_dir)
    mask_path = os.path.join(mask_dir, f"mask_{uuid.uuid4().hex}.npy")
    np.save(mask_path, mask.astype(np.uint8))
    return mask_path


def main():
    model = build_sam3_image_model()
    processor = Sam3Processor(model)

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

        if command != "extract_mask":
            sys.stdout.write(build_response("error", message=f"Unknown command {command}") + "\n")
            sys.stdout.flush()
            continue

        image_path = request.get("image_path")
        object_name = request.get("object_name")
        mask_dir = request.get("mask_dir") or os.path.dirname(image_path)

        if not image_path or not os.path.exists(image_path):
            sys.stdout.write(build_response("error", message=f"Image not found: {image_path}") + "\n")
            sys.stdout.flush()
            continue

        if not object_name:
            sys.stdout.write(build_response("error", message="object_name is required") + "\n")
            sys.stdout.flush()
            continue

        try:
            image = Image.open(image_path)
            inference_state = processor.set_image(image)
            output = processor.set_text_prompt(state=inference_state, prompt=object_name)
            masks = output.get("masks")
            scores = output.get("scores")

            if masks is None or len(masks) == 0:
                sys.stdout.write(build_response("error", message="No masks found") + "\n")
                sys.stdout.flush()
                continue

            best_idx = int(np.argmax(scores)) if scores is not None and len(scores) > 0 else 0
            mask = masks[best_idx]
            if hasattr(mask, "cpu"):
                mask = mask.cpu().numpy()
            elif hasattr(mask, "numpy"):
                mask = mask.numpy()

            mask = (mask > 0.5).astype(np.uint8) * 255
            mask_path = save_mask(mask, mask_dir)
            sys.stdout.write(build_response("success", mask_path=mask_path, score=float(scores[best_idx])) + "\n")
            sys.stdout.flush()
        except Exception as exc:
            log_exception("[sam3_worker] mask extraction failed", exc)
            sys.stdout.write(build_response("error", message=str(exc)) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()

