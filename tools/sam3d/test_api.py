"""Test SAM3D API with a project image. Default: debug_reconstruct_work/input.jpg (apple)."""
import argparse
import base64
import json
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
    p.add_argument("--url", default="http://localhost:8002")
    p.add_argument("--image", default=None, help="Path to image (default: debug_reconstruct_work/input.jpg)")
    p.add_argument("--segment-only", action="store_true", help="Only test /segment (faster)")
    p.add_argument("--prompt", default=None, help="Test /reconstruct_by_prompt with this text (e.g. '사과')")
    p.add_argument("--max", type=int, default=3, help="Max objects to model in /reconstruct (default 3 for fast test; 0 = all)")
    args = p.parse_args()

    image_path = Path(args.image) if args.image else _default_image_path()
    if not image_path.exists():
        print(f"No image at {image_path}")
        sys.exit(1)

    try:
        import requests
    except ImportError:
        print("pip install requests")
        sys.exit(1)

    base_url = args.url.rstrip("/")
    print(f"API base: {base_url}")
    print(f"Image: {image_path}")

    # 1. Health
    print("\n1. GET /health ...")
    r = requests.get(f"{base_url}/health", timeout=60)
    r.raise_for_status()
    print("   ", r.json())

    # 2. Encode image
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    # 3. Segment (SAM only)
    print("\n2. POST /segment ...")
    r = requests.post(
        f"{base_url}/segment",
        json={"image": img_b64},
        timeout=300,
    )
    if r.status_code != 200:
        print(f"   ERROR {r.status_code}: {r.text[:500]}")
        r.raise_for_status()
    data = r.json()
    print(f"   masks: {len(data.get('masks', []))}, names: {data.get('names', [])}")

    if args.segment_only:
        print("\nOK (segment only).")
        return

    if args.prompt:
        # 3b. Reconstruct by prompt (segment by text -> crop -> SAM3D -> single GLB)
        print(f"\n3. POST /reconstruct_by_prompt (prompt={args.prompt!r}) ...")
        r = requests.post(
            f"{base_url}/reconstruct_by_prompt",
            json={"image": img_b64, "prompt": args.prompt},
            timeout=600,
        )
        if r.status_code != 200:
            print(f"   ERROR {r.status_code}: {r.text[:500]}")
            r.raise_for_status()
        data = r.json()
        glb_b64_len = len(data.get("glb", ""))
        print(f"   glb: {glb_b64_len} b64 chars, transform keys: {list(data.get('transform', {}).keys())}")
        print("\nOK (health + segment + reconstruct_by_prompt).")
        return

    # 4. Reconstruct (SAM + SAM3D, 객체별 크롭, 버텍스 컬러만, max 제한으로 테스트 시 빠르게)
    print(f"\n3. POST /reconstruct (max={args.max}) ...")
    r = requests.post(
        f"{base_url}/reconstruct",
        json={"image": img_b64, "max": args.max},
        timeout=3600,
    )
    r.raise_for_status()
    data = r.json()
    glbs = data.get("glb_files", [])
    print(f"   glb_files: {len(glbs)}, transforms: {len(data.get('transforms', []))}")
    for g in glbs:
        print(f"   - {g['name']} ({len(g['data'])} b64 chars)")

    print("\nOK (health + segment + reconstruct).")


if __name__ == "__main__":
    main()
