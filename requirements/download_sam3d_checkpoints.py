"""Download SAM3D checkpoints from HuggingFace to utils/third_party/sam3d/checkpoints/hf.

Usage:
    python requirements/download_sam3d_checkpoints.py

Requires: pip install 'huggingface-hub[cli]<1.0' and HF token (hf auth login).
Repo: https://huggingface.co/facebook/sam-3d-objects (access may need to be requested).
"""
from __future__ import annotations

import os
import shutil
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAM3D_CHECKPOINTS = os.path.join(REPO_ROOT, "utils", "third_party", "sam3d", "checkpoints")
HF_REPO = "facebook/sam-3d-objects"
TAG = "hf"


def main() -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Install: pip install 'huggingface-hub[cli]<1.0'")
        sys.exit(1)

    download_dir = os.path.join(SAM3D_CHECKPOINTS, f"{TAG}-download")
    target_dir = os.path.join(SAM3D_CHECKPOINTS, TAG)

    if os.path.isdir(target_dir) and os.path.isfile(os.path.join(target_dir, "pipeline.yaml")):
        print(f"Checkpoints already present at {target_dir}")
        return

    print(f"Downloading {HF_REPO} to {download_dir} ...")
    snapshot_download(
        repo_id=HF_REPO,
        repo_type="model",
        local_dir=download_dir,
        max_workers=1,
    )

    # Move checkpoints subfolder from download to checkpoints/hf
    inner_checkpoints = os.path.join(download_dir, "checkpoints")
    if os.path.isdir(inner_checkpoints):
        os.makedirs(SAM3D_CHECKPOINTS, exist_ok=True)
        if os.path.isdir(target_dir):
            shutil.rmtree(target_dir)
        shutil.move(inner_checkpoints, target_dir)
        print(f"Checkpoints saved to {target_dir}")
    else:
        # Some repos have files at root; copy whole download to hf
        os.makedirs(target_dir, exist_ok=True)
        for name in os.listdir(download_dir):
            src = os.path.join(download_dir, name)
            dst = os.path.join(target_dir, name)
            if os.path.isdir(src):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        print(f"Checkpoints saved to {target_dir}")

    if os.path.isdir(download_dir):
        shutil.rmtree(download_dir)

    if not os.path.isfile(os.path.join(target_dir, "pipeline.yaml")):
        print("Warning: pipeline.yaml not found. Check repo layout.")


if __name__ == "__main__":
    main()
