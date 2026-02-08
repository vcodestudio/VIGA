"""SAM → ComfyUI까지 실제 런너(init.py)와 동일한 방식으로 실행하는 테스트.

- SAM: init._run_sam_segment() (같은 path_to_cmd, prepare_env, subprocess).
- ComfyUI: NPZ 빌드 → 업로드 → 워크플로 실행 → 결과 다운로드 (init._run_comfyui_once_and_fill_assets).
타겟 이미지: data/static_scene/artist/target.png

Usage (from repo root):
  set COMFYUI_API_URL=http://localhost:8188
  python tools/sam3d/test_sam_only.py
  python tools/sam3d/test_sam_only.py --image data/static_scene/artist/target.png
  python tools/sam3d/test_sam_only.py --no-comfy   # SAM만
"""
from __future__ import annotations

import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except ImportError:
    pass

DEFAULT_IMAGE = "data/static_scene/artist/target.png"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run SAM + ComfyUI pipeline (same as init.py)")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Target image (relative to repo root or absolute)")
    parser.add_argument("--no-comfy", action="store_true", help="Run SAM only, skip ComfyUI")
    parser.add_argument("--use-existing", action="store_true", help="Use existing sam_init results (skip SAM), run ComfyUI only")
    args = parser.parse_args()

    target_image = os.path.join(ROOT, args.image) if not os.path.isabs(args.image) else args.image
    if not os.path.isfile(target_image):
        print(f"Error: image not found: {target_image}", file=sys.stderr)
        sys.exit(1)

    out_base = os.path.join(ROOT, "output", "static_scene", "test_sam_only", "artist")
    output_dir = os.path.join(out_base, "sam_init")
    assets_dir = os.path.join(out_base, "assets")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)

    import tools.sam3d.init as init_mod

    init_mod._target_image = target_image
    init_mod._output_dir = output_dir
    init_mod._assets_dir = assets_dir
    init_mod._sam_env_bin = init_mod.path_to_cmd.get("tools/sam3d/sam_worker.py")  # type: ignore
    init_mod._comfyui_url = os.getenv("COMFYUI_API_URL")
    init_mod._log_file = open(os.path.join(output_dir, "sam_init.log"), "w", encoding="utf-8")

    print(f"[TEST] Target image: {target_image}", flush=True)
    print(f"[TEST] Output dir:   {output_dir}", flush=True)
    print(f"[TEST] SAM Python:   {init_mod._sam_env_bin}", flush=True)
    if init_mod._comfyui_url and not args.no_comfy:
        print(f"[TEST] ComfyUI URL: {init_mod._comfyui_url}", flush=True)
    elif args.no_comfy:
        print("[TEST] ComfyUI: skipped (--no-comfy)", flush=True)
    else:
        print("[TEST] ComfyUI: skipped (COMFYUI_API_URL not set)", flush=True)

    total_start = time.perf_counter()

    # --- 1) SAM (실제와 동일) 또는 기존 결과 사용 ---
    if args.use_existing and os.path.isfile(os.path.join(output_dir, "all_masks.npy")):
        print("\n[TEST] Step 1: Using existing SAM results (skip run)", flush=True)
        t0 = time.perf_counter()
        t1 = t0
        sam_elapsed = 0.0
    else:
        print("\n[TEST] Step 1: SAM (init._run_sam_segment)...", flush=True)
        t0 = time.perf_counter()
        err = None
        try:
            init_mod._run_sam_segment()
        except Exception as e:
            err = e
        t1 = time.perf_counter()
        sam_elapsed = t1 - t0

        if err is not None:
            if getattr(init_mod._log_file, "close", None):
                init_mod._log_file.close()
            print(f"[TEST] SAM failed in {sam_elapsed:.1f}s: {err!r}", flush=True)
            print(f"[TEST] Log: {output_dir}/sam_worker.log", flush=True)
            sys.exit(1)
    print(f"[TEST] SAM finished in {sam_elapsed:.1f}s", flush=True)

    if args.no_comfy or not init_mod._comfyui_url:
        if getattr(init_mod._log_file, "close", None):
            init_mod._log_file.close()
        total_elapsed = time.perf_counter() - total_start
        print(f"\n[TEST] Total: {total_elapsed:.1f}s (SAM only)", flush=True)
        return

    # --- 2) ComfyUI (NPZ → 업로드 → 워크플로 → 다운로드, 실제와 동일) ---
    print("\n[TEST] Step 2: ComfyUI (init._run_comfyui_once_and_fill_assets)...", flush=True)
    t2 = time.perf_counter()
    try:
        init_mod._run_comfyui_once_and_fill_assets()
    except Exception as e:
        if getattr(init_mod._log_file, "close", None):
            init_mod._log_file.close()
        print(f"[TEST] ComfyUI failed: {e!r}", flush=True)
        sys.exit(1)
    t3 = time.perf_counter()
    comfy_elapsed = t3 - t2

    if getattr(init_mod._log_file, "close", None):
        init_mod._log_file.close()

    total_elapsed = time.perf_counter() - total_start
    n_assets = len(init_mod._comfy_assets or {})

    print(f"[TEST] ComfyUI finished in {comfy_elapsed:.1f}s", flush=True)
    print(f"[TEST] Downloaded {n_assets} assets -> {assets_dir}", flush=True)
    if init_mod._comfy_assets:
        for name, path in list(init_mod._comfy_assets.items())[:5]:
            print(f"       {name} -> {path}", flush=True)
        if n_assets > 5:
            print(f"       ... and {n_assets - 5} more", flush=True)
    print(f"\n[TEST] --- Timing ---", flush=True)
    print(f"       SAM:    {sam_elapsed:.1f}s", flush=True)
    print(f"       ComfyUI: {comfy_elapsed:.1f}s", flush=True)
    print(f"       Total:  {total_elapsed:.1f}s", flush=True)
    print(f"[TEST] comfy_assets.json: {os.path.join(assets_dir, 'comfy_assets.json')}", flush=True)


if __name__ == "__main__":
    main()
