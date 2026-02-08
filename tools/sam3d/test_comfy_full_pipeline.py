"""Test full pipeline: segment (SAM) -> NPZ -> ComfyUI API -> download. Reports per-step and total time.

Usage (from repo root):
  # 1) Create conda env 'sam' for SAM worker (required for segment step):
  #    conda create -n sam python=3.10 -y && conda activate sam
  #    pip install -r requirements/requirement_sam.txt
  # 2) Run test (agent env has requests/numpy/cv2 for this script; SAM subprocess uses 'sam' env):
  conda activate agent
  set COMFYUI_API_URL=http://localhost:8188
  python tools/sam3d/test_comfy_full_pipeline.py --image data/static_scene/artist/target.png

  # Use existing segment results (skip SAM, test NPZ+ComfyUI only):
  python tools/sam3d/test_comfy_full_pipeline.py --image data/static_scene/artist/target.png --use-existing-sam-init output/static_scene/TIMESTAMP/artist/sam_init
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

# Load .env for COMFYUI_API_URL
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except ImportError:
    pass

def main():
    parser = argparse.ArgumentParser(description="Test SAM -> NPZ -> ComfyUI full pipeline")
    parser.add_argument("--image", default="data/static_scene/artist/target.png", help="Input image path (relative to repo root)")
    parser.add_argument("--output-dir", default=None, help="Output dir (default: output/static_scene/test_comfy_pipeline)")
    parser.add_argument("--use-existing-sam-init", default=None, metavar="DIR", help="Use existing sam_init dir (all_masks.npy + _object_names.json); skip SAM run")
    args = parser.parse_args()

    image_path = os.path.join(ROOT, args.image) if not os.path.isabs(args.image) else args.image
    if not os.path.isfile(image_path):
        print(f"Error: image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    out_base = args.output_dir or os.path.join(ROOT, "output", "static_scene", "test_comfy_pipeline")
    sam_init_dir = os.path.join(out_base, "sam_init")
    assets_dir = os.path.join(out_base, "assets")
    os.makedirs(sam_init_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)

    comfy_url = os.getenv("COMFYUI_API_URL")
    if not comfy_url:
        print("Warning: COMFYUI_API_URL not set; will run SAM + NPZ only, skip ComfyUI.", file=sys.stderr)

    from utils._path import path_to_cmd
    sam_python = path_to_cmd.get("tools/sam3d/sam_worker.py")
    sam_worker_script = os.path.join(ROOT, "tools", "sam3d", "sam_worker.py")
    all_masks_path = os.path.join(sam_init_dir, "all_masks.npy")

    total_start = time.perf_counter()

    # --- 1) SAM segment (or use existing) ---
    use_existing = getattr(args, "use_existing_sam_init", None)
    if use_existing:
        if not os.path.isdir(use_existing):
            print(f"Error: --use-existing-sam-init path is not a directory: {use_existing}", file=sys.stderr)
            sys.exit(1)
        if not os.path.isfile(os.path.join(use_existing, "all_masks.npy")):
            print(f"Error: no all_masks.npy in {use_existing} (SAM did not complete there; need a sam_init dir with segment results)", file=sys.stderr)
            sys.exit(1)
        sam_init_dir = os.path.normpath(use_existing)
        out_base = os.path.dirname(sam_init_dir)
        assets_dir = os.path.join(out_base, "assets")
        os.makedirs(assets_dir, exist_ok=True)
        all_masks_path = os.path.join(sam_init_dir, "all_masks.npy")
        print("Step 1: Using existing SAM results at", sam_init_dir, flush=True)
        t0 = time.perf_counter()
        t1 = t0
    else:
        print("Step 1: Running SAM worker...", flush=True)
        print(f"  (using Python: {sam_python})", flush=True)
        t0 = time.perf_counter()
        log_path = os.path.join(sam_init_dir, "sam_worker.log")
        with open(log_path, "w") as log_file:
            ret = subprocess.run(
                [sam_python, sam_worker_script, "--image", image_path, "--out", all_masks_path],
                cwd=ROOT,
                text=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
        t1 = time.perf_counter()
        if ret.returncode != 0:
            print(f"  SAM worker failed in {t1 - t0:.1f}s (see {log_path})", flush=True)
            print("  Tip: create conda env 'sam' and install requirement_sam.txt (includes opencv).", flush=True)
            print(f"  Total elapsed: {t1 - total_start:.1f}s", flush=True)
            sys.exit(1)
        print(f"  SAM segment done in {t1 - t0:.1f}s", flush=True)

    # --- 2) Load segment results ---
    import numpy as np
    import cv2
    mapping_path = all_masks_path.replace(".npy", "_object_names.json")
    with open(mapping_path, "r") as f:
        obj_info = json.load(f)
    object_mapping = obj_info.get("object_mapping", obj_info.get("object_names", []))

    img = cv2.imread(image_path)
    if img is None:
        print("Error: failed to load image", file=sys.stderr)
        sys.exit(1)
    masks_np = np.load(all_masks_path, allow_pickle=True)
    if masks_np.dtype == object:
        masks = [masks_np[i] for i in range(len(masks_np))]
    elif masks_np.ndim == 3:
        masks = [masks_np[i] for i in range(masks_np.shape[0])]
    else:
        masks = [masks_np]
    while len(object_mapping) < len(masks):
        object_mapping.append(f"object_{len(object_mapping)}")
    t2 = time.perf_counter()
    print(f"  Load segment results: {len(masks)} masks, {len(object_mapping)} names ({t2 - t1:.2f}s)", flush=True)

    # --- 3) Build NPZ in ComfyUI-Multiband format ---
    print("Step 2: Building multiband NPZ (BCHW format)...", flush=True)
    npz_path = os.path.join(sam_init_dir, "scene_multiband.npz")
    h, w = img.shape[:2]
    n = len(masks)
    num_channels = 3 + n  # RGB + masks
    
    # Build samples array: (1, channels, H, W) - BCHW format
    samples = np.zeros((1, num_channels, h, w), dtype=np.float32)
    
    # RGB channels (BGR -> RGB, normalize to 0-1)
    img_rgb = img[:, :, ::-1] if img.shape[2] >= 3 else img  # BGR to RGB
    samples[0, 0, :, :] = img_rgb[:, :, 0].astype(np.float32) / 255.0  # R
    samples[0, 1, :, :] = img_rgb[:, :, 1].astype(np.float32) / 255.0  # G
    samples[0, 2, :, :] = img_rgb[:, :, 2].astype(np.float32) / 255.0  # B
    
    # Mask channels
    for i, m in enumerate(masks):
        m = m.squeeze() if getattr(m, "ndim", 0) > 2 else m
        mask_01 = (m > 0).astype(np.float32) if getattr(m, "dtype", None) != np.float32 else np.asarray(m, dtype=np.float32)
        if mask_01.max() > 1:
            mask_01 = mask_01 / 255.0
        samples[0, 3 + i, :, :] = mask_01
    
    # Build channel names
    channel_names = ['img_01_r', 'img_01_g', 'img_01_b']
    for i in range(n):
        channel_names.append(f'mask_{i+1:02d}')
    
    # Save in ComfyUI-Multiband format
    np.savez_compressed(
        npz_path,
        samples=samples,
        channel_names=np.array(channel_names, dtype=object)
    )
    t3 = time.perf_counter()
    print(f"  NPZ built in {t3 - t2:.2f}s -> {npz_path}", flush=True)
    print(f"  NPZ shape: {samples.shape}, channels: {channel_names[:5]}...", flush=True)

    if not comfy_url:
        total_elapsed = time.perf_counter() - total_start
        print(f"\nTotal (SAM + NPZ only): {total_elapsed:.1f}s", flush=True)
        return

    # --- 4) Upload NPZ to ComfyUI ---
    print("Step 3: Uploading NPZ to ComfyUI...", flush=True)
    import requests
    base = comfy_url.rstrip("/")
    t4 = time.perf_counter()
    with open(npz_path, "rb") as f:
        r = requests.post(f"{base}/upload/image", files={"image": ("scene_multiband.npz", f, "application/octet-stream")}, timeout=120)
    r.raise_for_status()
    upload_result = r.json()
    npz_filename = upload_result.get("name") or upload_result.get("filename") or "scene_multiband.npz"
    subfolder = upload_result.get("subfolder", "")
    if subfolder and str(subfolder).strip():
        npz_filename = f"{str(subfolder).strip('/')}/{npz_filename}"
    t5 = time.perf_counter()
    print(f"  Upload done in {t5 - t4:.1f}s -> {npz_filename}", flush=True)

    # --- 5) Run ComfyUI workflow ---
    print("Step 4: Running ComfyUI workflow...", flush=True)
    workflow_path = os.path.join(ROOT, "tools", "sam3d", "workflows", "comfy_workflow.json")
    with open(workflow_path, "r", encoding="utf-8") as f:
        workflow = json.load(f)
    if "75" not in workflow:
        print("Error: workflow missing node 75 (MultibandLoad)", file=sys.stderr)
        sys.exit(1)
    workflow["75"] = dict(workflow["75"])
    workflow["75"]["inputs"] = dict(workflow["75"].get("inputs", {}))
    workflow["75"]["inputs"]["image"] = npz_filename

    # Randomize seed in SAM3DSceneGenerate node (80) to avoid cache hit
    import random
    if "80" in workflow:
        seed = random.randint(0, 2**31 - 1)
        workflow["80"] = dict(workflow["80"])
        workflow["80"]["inputs"] = dict(workflow["80"].get("inputs", {}))
        workflow["80"]["inputs"]["seed"] = seed
        print(f"  Using random seed: {seed}", flush=True)

    r = requests.post(f"{base}/prompt", json={"prompt": workflow}, timeout=30)
    r.raise_for_status()
    prompt_id = r.json().get("prompt_id")
    if not prompt_id:
        print("Error: ComfyUI did not return prompt_id", file=sys.stderr)
        sys.exit(1)

    t6 = time.perf_counter()
    # Poll every 10 seconds, up to 90 iterations (15 minutes total)
    print(f"  Queued prompt_id: {prompt_id}", flush=True)
    print(f"  Polling ComfyUI every 10s (max 15 min)...", flush=True)
    history = None
    for i in range(90):
        r = requests.get(f"{base}/history/{prompt_id}", timeout=10)
        r.raise_for_status()
        hist = r.json()
        if prompt_id in hist:
            history = hist[prompt_id]
            print(f"  [Poll #{i+1}] Workflow DONE after {i * 10}s", flush=True)
            break
        elapsed = i * 10
        print(f"  [Poll #{i+1}] Waiting... ({elapsed}s elapsed)", flush=True)
        time.sleep(10)
    else:
        print("Warning: ComfyUI workflow did not finish in 15 minutes, trying direct download...", file=sys.stderr)
        history = {"outputs": {}}  # Empty history, will use direct fallback
    t7 = time.perf_counter()
    print(f"  Workflow finished in {t7 - t6:.1f}s", flush=True)

    # --- 6) Download outputs ---
    print("Step 5: Downloading outputs...", flush=True)
    import re
    outputs = history.get("outputs", {})
    by_subfolder = {}
    for _nid, node_out in outputs.items():
        if not isinstance(node_out, dict) or "gizmos" in node_out:
            continue
        for key, val in node_out.items():
            if not isinstance(val, list):
                continue
            for entry in val:
                if not isinstance(entry, dict):
                    continue
                fn = entry.get("filename")
                if not isinstance(fn, str) or not (fn.endswith(".glb") or fn.endswith(".ply") or fn.endswith(".obj")):
                    continue
                sub = entry.get("subfolder", "")
                by_subfolder.setdefault(sub, []).append((fn, entry.get("type", "output")))

    def pick_best(files):
        for ext in (".glb", ".ply", ".obj"):
            for (fn, typ) in files:
                if fn.endswith(ext):
                    return fn, typ
        return (files[0][0], files[0][1]) if files else None

    def sort_key(item):
        subfolder = item[0]
        m = re.search(r"object_(\d+)", subfolder, re.IGNORECASE)
        if m:
            return (int(m.group(1)), subfolder)
        return (999, subfolder)

    downloaded = []
    for idx, (subfolder, file_list) in enumerate(sorted(by_subfolder.items(), key=sort_key)):
        best = pick_best(file_list)
        if not best:
            continue
        filename, type_name = best
        r = requests.get(f"{base}/view", params={"filename": filename, "subfolder": subfolder, "type": type_name}, timeout=30)
        r.raise_for_status()
        obj_name = object_mapping[idx] if idx < len(object_mapping) else f"object_{idx}"
        safe = "".join(c for c in obj_name if c.isalnum() or c in "._- ") or "object"
        ext = os.path.splitext(filename)[1]
        local_path = os.path.join(assets_dir, f"{safe}{ext}")
        with open(local_path, "wb") as f:
            f.write(r.content)
        downloaded.append(local_path)

    # Fallback 1: Direct file system access to ComfyUI output (most reliable)
    import shutil
    comfyui_output_dir = os.getenv("COMFYUI_OUTPUT_DIR")
    if not downloaded and comfyui_output_dir and os.path.isdir(comfyui_output_dir):
        # Find the latest sam3d_inference_N folder
        inference_dirs = []
        for name in os.listdir(comfyui_output_dir):
            if name.startswith("sam3d_inference_"):
                try:
                    num = int(name.replace("sam3d_inference_", ""))
                    inference_dirs.append((num, os.path.join(comfyui_output_dir, name)))
                except ValueError:
                    pass
        if inference_dirs:
            inference_dirs.sort(key=lambda x: x[0], reverse=True)
            latest_dir = inference_dirs[0][1]
            print(f"  [Direct] Using latest ComfyUI output: {latest_dir}", flush=True)
            # Copy all object_N/mesh.glb files
            for i in range(len(object_mapping)):
                obj_folder = os.path.join(latest_dir, f"object_{i}")
                glb_path = os.path.join(obj_folder, "mesh.glb")
                if os.path.isfile(glb_path):
                    obj_name = object_mapping[i] if i < len(object_mapping) else f"object_{i}"
                    safe = "".join(c for c in obj_name if c.isalnum() or c in "._- ") or "object"
                    local_path = os.path.join(assets_dir, f"{safe}.glb")
                    shutil.copy2(glb_path, local_path)
                    downloaded.append(local_path)
            if downloaded:
                print(f"  [Direct] Copied {len(downloaded)} GLB files from {latest_dir}", flush=True)

    # Fallback 2: API-based download (if direct access failed)
    if not downloaded and object_mapping:
        for i in range(len(object_mapping)):
            obj_name = object_mapping[i] if i < len(object_mapping) else f"object_{i}"
            safe = "".join(c for c in obj_name if c.isalnum() or c in "._- ") or "object"
            local_path = os.path.join(assets_dir, f"{safe}.glb")
            for subfolder in [f"object_{i}", f"sam3d_inference_25/object_{i}", f"{prompt_id}/object_{i}" if prompt_id else ""]:
                if not subfolder:
                    continue
                try:
                    r = requests.get(f"{base}/view", params={"filename": "mesh.glb", "subfolder": subfolder, "type": "output"}, timeout=30)
                    r.raise_for_status()
                    with open(local_path, "wb") as f:
                        f.write(r.content)
                    downloaded.append(local_path)
                    break
                except Exception:
                    continue
        if downloaded:
            print(f"  [API Fallback] Downloaded {len(downloaded)} GLB via object_N/mesh.glb", flush=True)

    t8 = time.perf_counter()
    print(f"  Downloaded {len(downloaded)} files in {t8 - t7:.1f}s -> {assets_dir}", flush=True)
    if not downloaded:
        debug_path = os.path.join(assets_dir, "comfy_history_debug.json")
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        print(f"  [DEBUG] 0 outputs -> saved full history to {debug_path}", flush=True)

    # --- Summary ---
    total_elapsed = time.perf_counter() - total_start
    print("\n--- Timing summary ---", flush=True)
    print(f"  1. SAM segment:     {t1 - t0:.1f}s", flush=True)
    print(f"  2. Load + NPZ:      {t3 - t1:.1f}s", flush=True)
    print(f"  3. Upload NPZ:      {t5 - t4:.1f}s", flush=True)
    print(f"  4. ComfyUI run:     {t7 - t6:.1f}s", flush=True)
    print(f"  5. Download:        {t8 - t7:.1f}s", flush=True)
    print(f"  Total:              {total_elapsed:.1f}s", flush=True)
    print(f"\nOutputs: {downloaded[:5]}{'...' if len(downloaded) > 5 else ''}", flush=True)


if __name__ == "__main__":
    main()
