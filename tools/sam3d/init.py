"""Scene Initialization MCP Server.

Pipeline: Google Cloud Vision API → bbox crop → ComfyUI single workflow → GLB download.
No SAM segmentation — uses Vision API for object detection and bbox-based image cropping.
Each detected object is processed individually through comfy_workflow_single.json.
"""
# pyright: reportMissingModuleSource=false, reportMissingImports=false
from __future__ import annotations

import sys
print("[SAM_INIT] process started", flush=True, file=sys.stderr)
import base64
import json
import os
import random
import re
import shutil
import time
from typing import Dict, List, Optional, Tuple, Any

# Import requests eagerly to avoid import-lock deadlock in worker threads
import requests as _requests

def _get_requests():
    return _requests

print("[SAM_INIT] importing FastMCP...", flush=True, file=sys.stderr)
from mcp.server.fastmcp import FastMCP

# Defer cv2 to first use so MCP server can become "ready" quickly (cv2 import is slow on Windows).
# cv2 is used inside functions that run in ThreadPoolExecutor, avoiding asyncio import deadlock.
_cv2 = None

def _get_cv2():
    global _cv2
    if _cv2 is None:
        import cv2 as cv
        _cv2 = cv
    return _cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
print("[SAM_INIT] loading utils._path...", flush=True, file=sys.stderr)
from utils._path import path_to_cmd
print("[SAM_INIT] utils._path loaded", flush=True, file=sys.stderr)

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

# Load .env so API keys are available
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), ".env"))
except ImportError:
    pass

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORKFLOW_PATH = os.path.join(os.path.dirname(__file__), "workflows", "comfy_workflow_single.json")
IMAGE_NODE_ID = "76"   # LoadImage node: inputs.image = uploaded filename
SLAT_NODE_ID = "70"    # SAM3DGenerateSLAT node: inputs.seed = random value

# Architectural / non-object elements to filter out from Vision API results
ARCHITECTURE_FILTER = {
    "wall", "floor", "ceiling", "window", "door", "room", "building",
    "house", "roof", "stairs", "staircase", "hallway", "corridor",
    "fence", "sidewalk", "road", "sky", "person", "human", "man",
    "woman", "boy", "girl",
}

mcp = FastMCP("sam-init")

# ── Global State ────────────────────────────────────────────────────────
_target_image: Optional[str] = None
_output_dir: Optional[str] = None   # sam_init: crops + logs
_assets_dir: Optional[str] = None   # assets: GLB save dir (same as Meshy)
_log_file: Optional[object] = None
_comfyui_url: Optional[str] = os.getenv("COMFYUI_API_URL")
_comfy_assets: Optional[Dict[str, str]] = None   # object_name -> local GLB path
_detected_objects: Optional[List[Dict[str, Any]]] = None  # Vision API results (filtered)
_all_vision_objects: Optional[List[Dict[str, Any]]] = None  # Vision API ALL results (before filtering)

# Debug: log COMFYUI_API_URL at module load
print(f"[SAM_INIT] Module loaded. COMFYUI_API_URL={_comfyui_url}", file=sys.stderr)


def log(message: str) -> None:
    print(message, file=sys.stderr)
    if _log_file is not None:
        try:
            _log_file.write(message + "\n")
            _log_file.flush()
        except Exception:
            pass


def _match_object_name(prompt: str, object_names: List[str]) -> Optional[str]:
    """Match user object_name to a detected object name (fuzzy)."""
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


# ── Google Cloud Vision API ─────────────────────────────────────────────

def _detect_objects_with_vision_api(image_path: str) -> List[Dict[str, Any]]:
    """Use Google Cloud Vision API (OBJECT_LOCALIZATION) to detect objects and bboxes.

    Returns list of dicts: {"name": str, "display_name": str, "score": float, "bbox": [x1,y1,x2,y2]}
    Bounding boxes are in pixel coordinates. Architectural elements are filtered out.
    """
    api_key = (
        os.environ.get("GOOGLE_VISION_API_KEY")
        or os.environ.get("GEMINI_API_KEY", "")
    )
    if not api_key:
        log("[SAM_INIT] WARNING: No GOOGLE_VISION_API_KEY or GEMINI_API_KEY set")
        return []

    log("[SAM_INIT] Calling Google Cloud Vision API (Object Localization)...")

    # Read image and base64-encode
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    # Get image dimensions for coordinate conversion
    img = _get_cv2().imread(image_path)
    if img is None:
        log(f"[SAM_INIT] ERROR: Cannot read image: {image_path}")
        return []
    h, w = img.shape[:2]

    # REST call
    url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
    payload = {
        "requests": [{
            "image": {"content": img_b64},
            "features": [{"type": "OBJECT_LOCALIZATION", "maxResults": 20}],
        }]
    }

    try:
        r = _get_requests().post(url, json=payload, timeout=(10, 30))  # (connect, read)
        if not r.ok:
            log(f"[SAM_INIT] Vision API error {r.status_code}: {r.text[:500]}")
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log(f"[SAM_INIT] Vision API request failed: {e}")
        return []

    # Parse response
    responses = data.get("responses", [])
    if not responses:
        log("[SAM_INIT] Vision API returned no responses")
        return []

    error = responses[0].get("error")
    if error:
        log(f"[SAM_INIT] Vision API error in response: {error}")
        return []

    annotations = responses[0].get("localizedObjectAnnotations", [])
    if not annotations:
        log("[SAM_INIT] Vision API found no objects")
        return []

    log(f"[SAM_INIT] Vision API raw: {len(annotations)} objects detected")

    # Phase 1: Parse ALL annotations into unfiltered list (stored globally for on-demand)
    global _all_vision_objects
    all_objects: List[Dict[str, Any]] = []
    all_seen: set = set()

    for ann in annotations:
        name = ann.get("name", "").strip()
        score = float(ann.get("score", 0.0))
        vertices = ann.get("boundingPoly", {}).get("normalizedVertices", [])

        if not name or len(vertices) < 2:
            continue

        # Normalized (0-1) → pixel coordinates
        xs = [v.get("x", 0.0) for v in vertices]
        ys = [v.get("y", 0.0) for v in vertices]
        x1 = max(0, int(min(xs) * w))
        y1 = max(0, int(min(ys) * h))
        x2 = min(w, int(max(xs) * w))
        y2 = min(h, int(max(ys) * h))

        # Skip tiny boxes
        if (x2 - x1) < 20 or (y2 - y1) < 20:
            continue

        name_lower = name.lower()
        safe_name = name_lower.replace(" ", "_")
        if safe_name in all_seen:
            counter = 2
            while f"{safe_name}_{counter}" in all_seen:
                counter += 1
            safe_name = f"{safe_name}_{counter}"
        all_seen.add(safe_name)

        all_objects.append({
            "name": safe_name,
            "display_name": name,
            "score": score,
            "bbox": [x1, y1, x2, y2],
        })

    _all_vision_objects = all_objects
    log(f"[SAM_INIT] Total valid detections (unfiltered): {len(all_objects)}")

    # Phase 2: Apply filters for the main pipeline (architecture, low confidence)
    results: List[Dict[str, Any]] = []
    for obj in all_objects:
        name_lower = obj["display_name"].lower()
        if any(arch in name_lower for arch in ARCHITECTURE_FILTER):
            log(f"[SAM_INIT]   Filtered: {obj['display_name']} (score={obj['score']:.2f}) — architectural")
            continue
        if obj["score"] < 0.3:
            log(f"[SAM_INIT]   Filtered: {obj['display_name']} (score={obj['score']:.2f}) — low confidence")
            continue
        results.append(obj)
        log(f"[SAM_INIT]   OK: {obj['name']} (score={obj['score']:.2f}, bbox={obj['bbox']})")

    log(f"[SAM_INIT] After filtering: {len(results)} objects")
    return results


# ── Image Cropping ──────────────────────────────────────────────────────

def _crop_objects(image_path: str, objects: List[Dict[str, Any]], padding: int = 20) -> List[Dict[str, Any]]:
    """Crop original image based on each object's bbox. Save PNGs to _assets_dir/crops/.

    Returns a new list of dicts with an added 'crop_path' key.
    """
    img = _get_cv2().imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    h, w = img.shape[:2]

    # Save crops to assets dir for reuse across runs
    crops_dir = os.path.join(_assets_dir or _output_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    results: List[Dict[str, Any]] = []
    for obj in objects:
        x1, y1, x2, y2 = obj["bbox"]
        # Apply padding (expand box slightly for better 3D generation)
        x1p = max(0, x1 - padding)
        y1p = max(0, y1 - padding)
        x2p = min(w, x2 + padding)
        y2p = min(h, y2 + padding)

        cropped = img[y1p:y2p, x1p:x2p].copy()
        crop_path = os.path.join(crops_dir, f"{obj['name']}.png")
        _get_cv2().imwrite(crop_path, cropped)

        result = dict(obj)
        result["crop_path"] = crop_path
        results.append(result)
        log(f"[SAM_INIT] Cropped {obj['name']}: {crop_path} ({x2p - x1p}x{y2p - y1p}px)")

    return results


# ── ComfyUI Single-Object Processing ───────────────────────────────────

def _comfyui_upload_image(file_path: str, filename: Optional[str] = None) -> str:
    """Upload image to ComfyUI /upload/image; return the filename for workflow."""
    url = f"{_comfyui_url.rstrip('/')}/upload/image"
    name = filename or os.path.basename(file_path)
    with open(file_path, "rb") as f:
        files = {"image": (name, f, "image/png")}
        r = _get_requests().post(url, files=files, timeout=(15, 120))  # (connect, read)
    r.raise_for_status()
    out = r.json()
    uploaded_name = out.get("name") or out.get("filename") or name
    subfolder = out.get("subfolder", "")
    if subfolder and isinstance(subfolder, str) and subfolder.strip():
        uploaded_name = f"{subfolder.strip('/')}/{uploaded_name}"
    return uploaded_name


def _comfyui_run_single_workflow(uploaded_image_name: str) -> Tuple[str, Dict]:
    """Run comfy_workflow_single.json with the uploaded image.

    Sets node 76 (LoadImage) image, randomises node 70 seed.
    Returns (prompt_id, history_entry).
    """
    with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
        workflow = json.load(f)

    # Set image in LoadImage node (76)
    if IMAGE_NODE_ID not in workflow:
        raise ValueError(f"Workflow missing LoadImage node {IMAGE_NODE_ID}")
    workflow[IMAGE_NODE_ID] = dict(workflow[IMAGE_NODE_ID])
    workflow[IMAGE_NODE_ID]["inputs"] = dict(workflow[IMAGE_NODE_ID].get("inputs", {}))
    workflow[IMAGE_NODE_ID]["inputs"]["image"] = uploaded_image_name

    # Randomise seed in SAM3DGenerateSLAT node (70) to avoid cache hit
    if SLAT_NODE_ID in workflow:
        seed = random.randint(0, 2**31 - 1)
        workflow[SLAT_NODE_ID] = dict(workflow[SLAT_NODE_ID])
        workflow[SLAT_NODE_ID]["inputs"] = dict(workflow[SLAT_NODE_ID].get("inputs", {}))
        workflow[SLAT_NODE_ID]["inputs"]["seed"] = seed
        log(f"[SAM_INIT]   seed={seed}")

    # Submit
    prompt_payload = {"prompt": workflow}
    r = _get_requests().post(
        f"{_comfyui_url.rstrip('/')}/prompt", json=prompt_payload, timeout=(10, 30),
    )
    if not r.ok:
        try:
            log(f"[SAM_INIT] ComfyUI /prompt error {r.status_code}: {r.text[:500]}")
        except Exception:
            pass
    r.raise_for_status()
    data = r.json()
    prompt_id = data.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"ComfyUI did not return prompt_id: {data}")

    # Poll every 10s, up to 90 iterations (15 min)
    log(f"[SAM_INIT]   prompt_id={prompt_id}, polling...")
    for i in range(90):
        r = _get_requests().get(
            f"{_comfyui_url.rstrip('/')}/history/{prompt_id}", timeout=(5, 10),
        )
        r.raise_for_status()
        hist = r.json()
        if prompt_id in hist:
            log(f"[SAM_INIT]   DONE after {i * 10}s")
            return (prompt_id, hist[prompt_id])
        if i % 3 == 0:
            log(f"[SAM_INIT]   [Poll #{i + 1}] {i * 10}s elapsed...")
        time.sleep(10)
    raise TimeoutError("ComfyUI workflow did not finish in 15 minutes")


def _comfyui_download_glb(
    history: Dict, object_name: str, prompt_id: Optional[str] = None,
) -> Optional[str]:
    """Extract GLB from ComfyUI history for a single workflow run.

    Tries API /view first, then falls back to direct filesystem copy.
    Returns local GLB path or None.
    """
    save_dir = _assets_dir or _output_dir
    os.makedirs(save_dir, exist_ok=True)
    safe_name = "".join(c for c in object_name if c.isalnum() or c in "._- ") or "object"
    local_path = os.path.join(save_dir, f"{safe_name}.glb")
    base = _comfyui_url.rstrip("/")

    def _is_3d(fn: str) -> bool:
        return isinstance(fn, str) and fn.endswith((".glb", ".ply", ".obj"))

    def _is_textured(fn: str) -> bool:
        return isinstance(fn, str) and "textured" in fn.lower()

    # Try from history outputs — prefer textured GLB over raw mesh
    outputs = history.get("outputs", {})
    all_3d_entries: List[Dict] = []
    for _node_id, node_out in outputs.items():
        if "gizmos" in node_out:
            continue
        for _key, val in node_out.items():
            if not isinstance(val, list):
                continue
            for entry in val:
                if not isinstance(entry, dict):
                    continue
                fn = entry.get("filename")
                if not _is_3d(fn):
                    continue
                all_3d_entries.append(entry)

    # Sort: textured first, then by filename
    all_3d_entries.sort(key=lambda e: (0 if _is_textured(e.get("filename", "")) else 1, e.get("filename", "")))

    for entry in all_3d_entries:
        fn = entry.get("filename")
        subfolder = entry.get("subfolder", "")
        type_name = entry.get("type", "output")
        try:
            r = _get_requests().get(
                base + "/view",
                params={"filename": fn, "subfolder": subfolder, "type": type_name},
                timeout=(10, 60),
            )
            r.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(r.content)
            log(f"[SAM_INIT]   Downloaded {fn} via API → {local_path}")
            return local_path
        except Exception as e:
            log(f"[SAM_INIT]   API download failed for {fn}: {e}")

    # Fallback: direct filesystem access to COMFYUI_OUTPUT_DIR
    comfyui_output_dir = os.getenv("COMFYUI_OUTPUT_DIR")
    if comfyui_output_dir and os.path.isdir(comfyui_output_dir):
        inference_dirs: List[Tuple[int, str]] = []
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
            # Prefer mesh_textured.glb (with textures) over mesh.glb (raw)
            for candidate in [
                os.path.join(latest_dir, "mesh_textured.glb"),
                os.path.join(latest_dir, "object_0", "mesh_textured.glb"),
                os.path.join(latest_dir, "mesh.glb"),
                os.path.join(latest_dir, "object_0", "mesh.glb"),
            ]:
                if os.path.isfile(candidate):
                    shutil.copy2(candidate, local_path)
                    log(f"[SAM_INIT]   Filesystem fallback → {local_path}")
                    return local_path

    log(f"[SAM_INIT]   No GLB found for {object_name}")
    return None


def _comfyui_process_single_object(crop_path: str, object_name: str) -> Optional[str]:
    """Upload cropped image → run single workflow → poll → download GLB.

    Returns local GLB path or None.
    """
    try:
        log(f"[SAM_INIT] [ComfyUI] Processing: {object_name}")

        # 1. Upload
        uploaded_name = _comfyui_upload_image(crop_path)
        log(f"[SAM_INIT]   Uploaded as: {uploaded_name}")

        # 2. Run workflow + poll
        prompt_id, history = _comfyui_run_single_workflow(uploaded_name)

        # 3. Download GLB
        glb_path = _comfyui_download_glb(history, object_name, prompt_id=prompt_id)
        return glb_path
    except Exception as e:
        log(f"[SAM_INIT] [ComfyUI] Error for {object_name}: {e}")
        return None


# ── Full Pipeline ───────────────────────────────────────────────────────

def _run_full_pipeline() -> None:
    """Vision API → crop → sequential ComfyUI processing → download GLBs.

    Runs heavy I/O in a worker thread to prevent blocking the MCP asyncio loop.
    """
    global _comfy_assets, _detected_objects
    import concurrent.futures

    def _do_all_work() -> Dict[str, str]:
        """Entire pipeline in a worker thread."""
        t0 = time.time()

        # Step 1: Detect objects
        log("[SAM_INIT] ═══ Step 1: Google Vision API Object Detection ═══")
        objects = _detect_objects_with_vision_api(_target_image)
        _detected_objects_local = objects  # noqa: F841
        if not objects:
            log("[SAM_INIT] No objects detected — pipeline stopped")
            return {}

        # Step 2: Crop images
        log(f"[SAM_INIT] ═══ Step 2: Cropping {len(objects)} objects ═══")
        cropped = _crop_objects(_target_image, objects)

        # Save detected objects list (filtered) — to assets dir for reuse
        save_base = _assets_dir or _output_dir
        objects_json_path = os.path.join(save_base, "detected_objects.json")
        serialisable = [{k: v for k, v in o.items() if k != "crop_path"} for o in cropped]
        with open(objects_json_path, "w", encoding="utf-8") as f:
            json.dump(serialisable, f, indent=2)
        log(f"[SAM_INIT] Saved {objects_json_path}")

        # Save ALL vision API results (unfiltered) for on-demand lookup
        if _all_vision_objects:
            all_vision_path = os.path.join(save_base, "all_vision_objects.json")
            with open(all_vision_path, "w", encoding="utf-8") as f:
                json.dump(_all_vision_objects, f, indent=2)
            log(f"[SAM_INIT] Saved {all_vision_path} ({len(_all_vision_objects)} objects)")

        # Step 3: Process each object through ComfyUI (sequential queue)
        log(f"[SAM_INIT] ═══ Step 3: ComfyUI Queue ({len(cropped)} objects) ═══")
        assets: Dict[str, str] = {}
        for idx, obj in enumerate(cropped):
            log(f"[SAM_INIT] ── [{idx + 1}/{len(cropped)}] {obj['name']} ──")
            glb_path = _comfyui_process_single_object(obj["crop_path"], obj["name"])
            if glb_path:
                assets[obj["name"]] = glb_path
                log(f"[SAM_INIT] ── [{idx + 1}/{len(cropped)}] ✓ {obj['name']} ──")
            else:
                log(f"[SAM_INIT] ── [{idx + 1}/{len(cropped)}] ✗ {obj['name']} (no GLB) ──")

        elapsed = time.time() - t0
        log(f"[SAM_INIT] ═══ Pipeline complete: {len(assets)}/{len(cropped)} GLBs in {elapsed:.0f}s ═══")

        # Write comfy_assets.json for Blender import
        assets_list = [{"name": name, "glb": path, "glb_path": path} for name, path in assets.items()]
        assets_json_path = os.path.join(_assets_dir or _output_dir, "comfy_assets.json")
        with open(assets_json_path, "w", encoding="utf-8") as f:
            json.dump(assets_list, f, indent=2)
        log(f"[SAM_INIT] Wrote {assets_json_path}")

        return assets

    # Run in thread to avoid asyncio deadlock
    log("[SAM_INIT] Launching pipeline in worker thread...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_do_all_work)
        result = future.result(timeout=3600)  # 1 hour max for many objects
        _comfy_assets = result
        _detected_objects = None  # updated inside worker


# ── MCP Tools ───────────────────────────────────────────────────────────

@mcp.tool()
def initialize(args: Dict[str, object]) -> Dict[str, object]:
    """Initialise scene: detect objects via Vision API, crop, and run ComfyUI for each.

    Assets (crops + GLBs + comfy_assets.json) are saved to data/static_scene/<task>/assets/
    so they persist across runs. Only regenerated when missing.
    """
    global _target_image, _output_dir, _assets_dir, _log_file, _comfyui_url, _comfy_assets, _all_vision_objects
    try:
        config = args.get("args") if isinstance(args.get("args"), dict) else args

        # Close previous log file handle before deleting old output dir
        if _log_file is not None:
            try:
                _log_file.close()
            except Exception:
                pass
            _log_file = None

        _target_image = config.get("target_image_path") or ""
        base_out = os.path.normpath(config.get("output_dir") or "")
        _output_dir = os.path.join(base_out, "sam_init")

        # Assets dir: use data/static_scene/<task>/assets/ (persists across runs)
        # This comes from static_scene.py → task_config["assets_dir"]
        _assets_dir = os.path.normpath(config.get("assets_dir") or os.path.join(base_out, "assets"))
        log(f"[SAM_INIT] Assets directory: {_assets_dir}")

        # Check if assets already exist in task data dir → skip pipeline entirely
        assets_json_path = os.path.join(_assets_dir, "comfy_assets.json")
        if os.path.isfile(assets_json_path):
            try:
                with open(assets_json_path, "r", encoding="utf-8") as f:
                    assets_list = json.load(f)
                if assets_list and isinstance(assets_list, list):
                    _comfy_assets = {a["name"]: a["glb_path"] for a in assets_list if a.get("glb_path")}
                    _comfyui_url = _comfyui_url or os.getenv("COMFYUI_API_URL")
                    os.makedirs(_output_dir, exist_ok=True)
                    log_path = os.path.join(_output_dir, "sam_init.log")
                    _log_file = open(log_path, "a", encoding="utf-8")
                    log(f"[SAM_INIT] CACHED: Found existing {len(_comfy_assets)} assets in {_assets_dir} — skipping pipeline")
                    log(f"[SAM_INIT]   Assets: {list(_comfy_assets.keys())}")

                    # Load cached unfiltered Vision API results for on-demand lookups
                    all_vision_path = os.path.join(_assets_dir, "all_vision_objects.json")
                    # Fallback: check output dir if not in assets dir (legacy runs)
                    if not os.path.isfile(all_vision_path):
                        all_vision_path = os.path.join(_output_dir, "all_vision_objects.json")
                    if os.path.isfile(all_vision_path):
                        try:
                            with open(all_vision_path, "r", encoding="utf-8") as vf:
                                _all_vision_objects = json.load(vf)
                            log(f"[SAM_INIT] CACHED: Loaded {len(_all_vision_objects)} unfiltered Vision objects from {all_vision_path}")
                        except Exception:
                            _all_vision_objects = None
                    else:
                        log(f"[SAM_INIT] CACHED: No all_vision_objects.json found (on-demand may trigger Vision API re-run)")

                    avail = ", ".join(sorted(_comfy_assets.keys())) if _comfy_assets else "(none)"
                    return {
                        "status": "success",
                        "output": {"text": [f"sam init initialized (cached assets found). Available assets: {avail}. Do NOT call get_better_object for these; only call it for objects not in this list."], "tool_configs": tool_configs},
                    }
            except Exception as e:
                log(f"[SAM_INIT] Could not load existing assets, running fresh: {e}")

        if os.path.exists(_output_dir):
            try:
                shutil.rmtree(_output_dir)
            except OSError:
                pass
        os.makedirs(_output_dir, exist_ok=True)
        os.makedirs(_assets_dir, exist_ok=True)
        
        log_path = os.path.join(_output_dir, "sam_init.log")
        _log_file = open(log_path, "w", encoding="utf-8")
        log(f"[SAM_INIT] Initialised. Log: {log_path}")
        log(f"[SAM_INIT] Assets dir: {_assets_dir}")

        # Run full pipeline if target image + ComfyUI are available
        if _target_image and os.path.isfile(_target_image):
            _comfyui_url = _comfyui_url or os.getenv("COMFYUI_API_URL")
            log(f"[SAM_INIT] ComfyUI URL: {_comfyui_url}")
            if _comfyui_url:
                try:
                    _run_full_pipeline()
                except Exception as e:
                    log(f"[SAM_INIT] Pipeline failed: {e}")
            else:
                log("[SAM_INIT] No COMFYUI_API_URL — skipping pipeline")
        else:
            log("[SAM_INIT] No target image provided")

        avail = ", ".join(sorted(_comfy_assets.keys())) if _comfy_assets else "(none)"
        return {
            "status": "success",
            "output": {"text": [f"sam init initialized. Available assets: {avail}. Do NOT call get_better_object for these; only call it for objects not in this list."], "tool_configs": tool_configs},
        }
    except Exception as e:
        error_msg = f"Error in initialize: {str(e)}"
        if _log_file:
            log(f"[SAM_INIT] {error_msg}")
        else:
            print(f"[SAM_INIT] {error_msg}", file=sys.stderr)
        return {"status": "error", "output": {"text": [error_msg]}}


def _find_object_in_vision_results(object_name: str) -> Optional[Dict[str, Any]]:
    """Search the UNFILTERED Vision API results for a matching object.

    This allows finding objects that were filtered out (e.g. low confidence, 
    architecture filter) during the initial pipeline. Uses fuzzy name matching.
    Returns the matching object dict (with bbox) or None.
    """
    if not _all_vision_objects:
        log(f"[SAM_INIT] No Vision API results cached — cannot search for '{object_name}'")
        return None

    normalized = object_name.strip().lower().replace(" ", "_").replace("-", "_")
    all_names = [obj["name"] for obj in _all_vision_objects]

    # Exact match
    for obj in _all_vision_objects:
        if obj["name"] == normalized:
            log(f"[SAM_INIT] Found '{object_name}' in unfiltered Vision results (exact): {obj['name']} bbox={obj['bbox']}")
            return obj

    # Substring match: query in name
    for obj in _all_vision_objects:
        if normalized in obj["name"] or normalized in obj["display_name"].lower():
            log(f"[SAM_INIT] Found '{object_name}' in unfiltered Vision results (substring): {obj['name']} bbox={obj['bbox']}")
            return obj

    # Reverse substring: name in query
    for obj in _all_vision_objects:
        if obj["name"] in normalized or obj["display_name"].lower() in normalized:
            log(f"[SAM_INIT] Found '{object_name}' in unfiltered Vision results (reverse): {obj['name']} bbox={obj['bbox']}")
            return obj

    log(f"[SAM_INIT] '{object_name}' not found in unfiltered Vision results. All detected: {all_names}")
    return None


def _rerun_vision_api_for_object(object_name: str) -> Optional[Dict[str, Any]]:
    """Re-run Vision API with maxResults=50 to try finding an object missed in initial call.

    Returns matching object dict or None.
    """
    global _all_vision_objects

    api_key = (
        os.environ.get("GOOGLE_VISION_API_KEY")
        or os.environ.get("GEMINI_API_KEY", "")
    )
    if not api_key or not _target_image:
        return None

    log(f"[SAM_INIT] Re-running Vision API (maxResults=50) to find '{object_name}'...")

    with open(_target_image, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    img = _get_cv2().imread(_target_image)
    if img is None:
        return None
    h, w = img.shape[:2]

    url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
    payload = {
        "requests": [{
            "image": {"content": img_b64},
            "features": [{"type": "OBJECT_LOCALIZATION", "maxResults": 50}],
        }]
    }

    try:
        r = _get_requests().post(url, json=payload, timeout=(10, 30))  # (connect, read)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log(f"[SAM_INIT] Vision API re-run failed: {e}")
        return None

    responses = data.get("responses", [])
    if not responses:
        return None
    annotations = responses[0].get("localizedObjectAnnotations", [])

    # Build new full list and merge with existing
    new_objects: List[Dict[str, Any]] = []
    seen_names: set = set(obj["name"] for obj in (_all_vision_objects or []))

    for ann in annotations:
        name = ann.get("name", "").strip()
        score = float(ann.get("score", 0.0))
        vertices = ann.get("boundingPoly", {}).get("normalizedVertices", [])
        if not name or len(vertices) < 2:
            continue

        xs = [v.get("x", 0.0) for v in vertices]
        ys = [v.get("y", 0.0) for v in vertices]
        x1 = max(0, int(min(xs) * w))
        y1 = max(0, int(min(ys) * h))
        x2 = min(w, int(max(xs) * w))
        y2 = min(h, int(max(ys) * h))
        if (x2 - x1) < 20 or (y2 - y1) < 20:
            continue

        name_lower = name.lower()
        safe_name = name_lower.replace(" ", "_")
        if safe_name in seen_names:
            counter = 2
            while f"{safe_name}_{counter}" in seen_names:
                counter += 1
            safe_name = f"{safe_name}_{counter}"
        seen_names.add(safe_name)

        obj = {"name": safe_name, "display_name": name, "score": score, "bbox": [x1, y1, x2, y2]}
        new_objects.append(obj)

    # Merge with existing
    if _all_vision_objects is None:
        _all_vision_objects = new_objects
    else:
        existing_names = {o["name"] for o in _all_vision_objects}
        for obj in new_objects:
            if obj["name"] not in existing_names:
                _all_vision_objects.append(obj)

    log(f"[SAM_INIT] Re-run found {len(new_objects)} objects total, merged cache now {len(_all_vision_objects)}")

    # Now try matching again
    return _find_object_in_vision_results(object_name)


def _generate_on_demand(object_name: str) -> Optional[str]:
    """On-demand: find object in Vision API results → crop → ComfyUI → GLB.

    1. Search unfiltered Vision API cache
    2. If not found, re-run Vision API with higher maxResults
    3. Crop matched bbox → upload to ComfyUI → download GLB
    Returns local GLB path or None.
    """
    import concurrent.futures

    def _do_work() -> Optional[str]:
        log(f"[SAM_INIT] On-demand generation for '{object_name}'...")

        # 1. Search cached unfiltered Vision API results
        matched = _find_object_in_vision_results(object_name)

        # 2. If not found, re-run Vision API with more results
        if not matched:
            matched = _rerun_vision_api_for_object(object_name)

        if not matched:
            log(f"[SAM_INIT] Could not locate '{object_name}' in image via Vision API")
            return None

        # 3. Crop
        bbox = matched["bbox"]
        img = _get_cv2().imread(_target_image)
        if img is None:
            return None
        h, w = img.shape[:2]
        x1, y1, x2, y2 = bbox
        padding = 20
        x1p, y1p = max(0, x1 - padding), max(0, y1 - padding)
        x2p, y2p = min(w, x2 + padding), min(h, y2 + padding)
        cropped = img[y1p:y2p, x1p:x2p].copy()

        safe_name = object_name.strip().lower().replace(" ", "_").replace("-", "_")
        crops_dir = os.path.join(_assets_dir or _output_dir, "crops")
        os.makedirs(crops_dir, exist_ok=True)
        crop_path = os.path.join(crops_dir, f"{safe_name}.png")
        _get_cv2().imwrite(crop_path, cropped)
        log(f"[SAM_INIT] Cropped {safe_name}: {crop_path} ({x2p-x1p}x{y2p-y1p}px)")

        # 4. ComfyUI
        glb_path = _comfyui_process_single_object(crop_path, safe_name)
        return glb_path

    # Run in thread to avoid asyncio deadlock
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_do_work)
        return future.result(timeout=900)  # 15 min max


@mcp.tool()
def get_better_object(object_name: str) -> Dict[str, object]:
    """Return path to 3D GLB asset for the given object."""
    global _comfyui_url, _comfy_assets
    
    if not _target_image or not _output_dir:
        return {"status": "error", "output": {"text": ["Call initialize first"]}}

    normalized_name = object_name.strip().lower().replace(" ", "_").replace("-", "_")

    try:
        _comfyui_url = _comfyui_url or os.getenv("COMFYUI_API_URL")

        # Start: no assets yet → run full pipeline once (Vision API → crop → ComfyUI for all detected objects)
        if _comfy_assets is None and _comfyui_url:
            _run_full_pipeline()

        # 1. Try existing assets first
        if _comfy_assets:
            object_names = list(_comfy_assets.keys())
            chosen = _match_object_name(normalized_name, object_names)
            if chosen and chosen in _comfy_assets:
                result_path = _comfy_assets[chosen]
                return {
                    "status": "success",
                    "output": {"text": [
                        f"Successfully generated static asset, downloaded to: {result_path}"
                    ]},
                }

        # 2. Not in pre-generated assets → on-demand generation via Vision API + ComfyUI
        if _comfyui_url:
            # Avoid blocking in worker when on-demand would fail anyway (no Vision cache + no API key)
            vision_key = os.environ.get("GOOGLE_VISION_API_KEY") or os.environ.get("GEMINI_API_KEY")
            if _all_vision_objects is None and not vision_key:
                available = list(_comfy_assets.keys()) if _comfy_assets else []
                return {
                    "status": "error",
                    "output": {"text": [
                        f"Cannot generate '{object_name}' on-demand: no Vision API cache and no GOOGLE_VISION_API_KEY/GEMINI_API_KEY. Available: {available}"
                    ]},
                }
            log(f"[SAM_INIT] '{object_name}' not in pre-generated assets, trying on-demand...")
            glb_path = _generate_on_demand(normalized_name)
            if glb_path:
                # Cache for future calls
                if _comfy_assets is None:
                    _comfy_assets = {}
                _comfy_assets[normalized_name] = glb_path

                # Update comfy_assets.json
                assets_list = [{"name": n, "glb": p, "glb_path": p} for n, p in _comfy_assets.items()]
                assets_json_path = os.path.join(_assets_dir or _output_dir, "comfy_assets.json")
                with open(assets_json_path, "w", encoding="utf-8") as f:
                    json.dump(assets_list, f, indent=2)

                return {
                    "status": "success",
                    "output": {"text": [
                        f"Successfully generated static asset, downloaded to: {glb_path}"
                    ]},
                }
            else:
                available = list(_comfy_assets.keys()) if _comfy_assets else []
                return {
                    "status": "error",
                    "output": {"text": [
                        f"Could not generate '{object_name}' (not found in image). Available: {available}"
                    ]},
                }

        return {
            "status": "error",
            "output": {"text": ["No assets available (pipeline may have failed)"]},
        }
    except Exception as e:
        log(f"[SAM_INIT] get_better_object failed: {e}")
        return {"status": "error", "output": {"text": [str(e)]}}


# ── Main ────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        task = sys.argv[2] if len(sys.argv) > 2 else "test"
        initialize({
                "target_image_path": f"data/static_scene/{task}/target.png",
                "output_dir": os.path.join(ROOT, "output", "test", task),
        })
        result = get_better_object("chair")
        print(json.dumps(result, indent=2), file=sys.stderr)
    else:
        print("[SAM_INIT] entering mcp.run()...", flush=True, file=sys.stderr)
        mcp.run()


if __name__ == "__main__":
    main()
