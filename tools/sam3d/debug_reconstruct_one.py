"""Run SAM then SAM3D for the first object only. Use to see full traceback when reconstruct fails.

Usage (from repo root, with sam3d conda env active):
  python tools/sam3d/debug_reconstruct_one.py
  python tools/sam3d/debug_reconstruct_one.py --image path/to/image.png
Default test image: tools/sam3d/debug_reconstruct_work/input.jpg (apple on white).
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK_DIR = ROOT / "tools" / "sam3d" / "debug_reconstruct_work"
DEFAULT_TEST_IMAGE = WORK_DIR / "input.jpg"
ASSET_APPLE = ROOT / "assets" / "c__PROJECTS_C_VIGA_new_tools_sam3d_debug_reconstruct_work_input.jpg"


def _default_image_path():
    if DEFAULT_TEST_IMAGE.exists():
        return DEFAULT_TEST_IMAGE
    if (WORK_DIR / "input.png").exists():
        return WORK_DIR / "input.png"
    if ASSET_APPLE.exists():
        return ASSET_APPLE
    p = ROOT / "data" / "static_scene" / "artist" / "target.png"
    if p.exists():
        return p
    return ROOT / "data" / "static_scene" / "test" / "target.png"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", default=None, help="Path to image (default: debug_reconstruct_work/input.jpg)")
    args = p.parse_args()

    image_path = Path(args.image) if args.image else _default_image_path()
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)

    work_dir = WORK_DIR
    work_dir.mkdir(parents=True, exist_ok=True)
    img_path = work_dir / "input.png"
    # Copy image so path is simple
    import shutil
    shutil.copy(str(image_path), str(img_path))
    masks_npy = work_dir / "all_masks.npy"
    mapping_path = work_dir / "all_masks_object_names.json"

    sam_worker = ROOT / "tools" / "sam3d" / "sam_worker.py"
    sam3d_worker = ROOT / "tools" / "sam3d" / "sam3d_worker.py"
    sam3d_config = ROOT / "utils" / "third_party" / "sam3d" / "checkpoints" / "hf" / "pipeline.yaml"

    print("Step 1: Running SAM worker (segment + VLM naming)...")
    r = subprocess.run(
        [sys.executable, str(sam_worker), "--image", str(img_path), "--out", str(masks_npy)],
        cwd=str(ROOT),
        capture_output=False,
    )
    if r.returncode != 0:
        print("SAM worker failed.")
        sys.exit(r.returncode)

    if not mapping_path.exists():
        print("SAM did not write mapping file. Possibly no objects after filtering or all identified as 'background'.")
        sys.exit(1)

    with open(mapping_path, "r") as f:
        mapping = json.load(f)
    object_mapping = mapping.get("object_mapping", [])
    if not object_mapping:
        print("object_mapping is empty.")
        sys.exit(1)

    name = object_mapping[0]
    mask_path = work_dir / f"{name}.npy"
    glb_path = work_dir / f"{name}.glb"
    info_path = work_dir / f"{name}.json"

    if not mask_path.exists():
        print(f"Mask file not found: {mask_path}")
        sys.exit(1)

    print(f"\nStep 2: Running SAM3D worker for first object: {name}")
    print("(stdout/stderr below will show the real error if it fails)\n")
    r = subprocess.run(
        [
            sys.executable, str(sam3d_worker),
            "--image", str(img_path),
            "--mask", str(mask_path),
            "--config", str(sam3d_config),
            "--glb", str(glb_path),
            "--info", str(info_path),
        ],
        cwd=str(ROOT),
        capture_output=False,
    )
    if r.returncode != 0:
        print("\nSAM3D worker failed (see traceback above).")
        sys.exit(r.returncode)

    print(f"\nSuccess: {glb_path}")


if __name__ == "__main__":
    main()
