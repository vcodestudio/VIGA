"""SAM Worker for Automatic Image Segmentation.

This script uses Segment Anything Model (SAM) to automatically generate
segmentation masks for all objects in an image, then uses a VLM to identify
and name each object.
"""
# First line: print immediately so user sees the process started (Python startup before this can take 30–60s on Windows)
print("[SAM_WORKER] process started", flush=True)

import sys
import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional

print("[SAM_WORKER] importing cv2, numpy, torch...", flush=True)
import cv2
import numpy as np
import torch
print("[SAM_WORKER] torch done", flush=True)
from PIL import Image

ROOT: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "utils", "third_party", "sam"))
sys.path.append(os.path.join(ROOT, "utils"))

# Load .env so API keys (GEMINI_API_KEY etc.) are available
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except ImportError:
    pass

print("[SAM_WORKER] importing segment_anything (may take 10–60s)...", flush=True)
from common import build_client, get_image_base64
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
print("[SAM_WORKER] segment_anything loaded", flush=True)


def panic_filtering_process(raw_masks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter and deduplicate SAM masks using greedy selection.

    Applies a multi-step filtering process:
    1. Pre-filtering: Remove masks below minimum area threshold
    2. Sorting: Sort by predicted IoU confidence score
    3. Greedy filling: Select masks that contribute sufficient new area
    4. Final selection: Keep top k masks by area

    Args:
        raw_masks: List of mask dictionaries, each containing:
            - 'segmentation': Binary mask (H, W)
            - 'predicted_iou': Confidence score (float)
            - 'area': Pixel area (int)

    Returns:
        Filtered list of mask dictionaries.
    """
    # Step 1: Pre-filtering
    # Remove masks that are too small (e.g., text on boxes) or too large (walls/floors)
    min_area_threshold = 100
    candidates = []
    if raw_masks:
        total_pixels = raw_masks[0]['segmentation'].shape[0] * raw_masks[0]['segmentation'].shape[1]
        max_area_ratio = 0.35  # Skip masks covering >35% of image (likely wall/floor/ceiling)
    else:
        total_pixels = 1
        max_area_ratio = 1.0
    for m in raw_masks:
        area_ratio = m['area'] / total_pixels
        if m['area'] > min_area_threshold and area_ratio < max_area_ratio:
            candidates.append(m)
        elif area_ratio >= max_area_ratio:
            print(f"    [Filter] Skipped large mask: {area_ratio*100:.1f}% of image (likely background)")

    if len(candidates) == 0:
        return []

    # Step 2: Sort by confidence score
    # Addresses inappropriate merging by prioritizing high-confidence masks
    candidates.sort(key=lambda x: x['predicted_iou'], reverse=True)

    # Step 3: Greedy filling
    # Select masks that contribute significant new area
    height, width = candidates[0]['segmentation'].shape
    occupancy_mask = np.zeros((height, width), dtype=bool)

    final_masks = []
    MIN_NEW_AREA_RATIO = 0.6

    for mask_data in candidates:
        current_seg = mask_data['segmentation']
        mask_area = mask_data['area']

        # Calculate overlap with already occupied regions
        intersection = np.logical_and(current_seg, occupancy_mask)
        intersection_area = np.count_nonzero(intersection)

        # Calculate how many new pixels this mask contributes
        new_area = mask_area - intersection_area

        # Calculate freshness ratio
        keep_ratio = new_area / mask_area if mask_area > 0 else 0

        # Decision: keep mask if it contributes enough new area
        if keep_ratio > MIN_NEW_AREA_RATIO:
            final_masks.append(mask_data)
            # Update occupancy map
            occupancy_mask = np.logical_or(occupancy_mask, current_seg)

    # Step 4: Keep only top k masks by area
    final_masks.sort(key=lambda x: x['area'], reverse=True)
    k = 15
    final_masks = final_masks[:k]

    return final_masks


def save_mask_as_png(
    mask_data: Dict[str, Any],
    original_image: np.ndarray,
    output_path: str
) -> str:
    """Save a single mask as a PNG image with transparency.

    Pixels where segmentation=1 retain original image colors.
    Pixels where segmentation=0 are set to transparent.

    Args:
        mask_data: Mask dictionary with 'segmentation' key (boolean array).
        original_image: Original RGB image array (H, W, 3).
        output_path: Path where the PNG will be saved.

    Returns:
        The output path where the image was saved.
    """
    m = mask_data['segmentation']  # Boolean array (H, W)
    h, w = m.shape

    # Resize original image if dimensions don't match
    if original_image.shape[:2] != (h, w):
        original_image = cv2.resize(original_image, (w, h))

    # Create RGBA image
    rgba_image = np.zeros((h, w, 4), dtype=np.uint8)

    # Where segmentation=1: use original colors, alpha=255 (opaque)
    rgba_image[m, :3] = original_image[m]
    rgba_image[m, 3] = 255

    # Where segmentation=0: keep as 0 (transparent)

    # Save as PNG
    pil_image = Image.fromarray(rgba_image, 'RGBA')
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    pil_image.save(output_path)
    return output_path


def sanitize_filename(name: str) -> str:
    """Sanitize a string for use as a filename.

    Replaces illegal characters with underscores, keeps only alphanumeric
    characters, underscores, and hyphens.

    Args:
        name: The original filename string.

    Returns:
        Sanitized filename string.
    """
    # Replace non-alphanumeric characters (except underscore and hyphen) with underscore
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    # Remove consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Remove leading and trailing underscores
    sanitized = sanitized.strip('_')
    # Use default name if empty
    if not sanitized:
        sanitized = "object"
    return sanitized


def get_object_name_from_vlm(
    image_path: str,
    ori_img_path: str,
    model: str = "gemini-2.5-flash",
    existing_names: Optional[List[str]] = None
) -> str:
    """Use a VLM to identify the object in an image and return a unique name.

    Args:
        image_path: Path to the segmented object PNG image.
        ori_img_path: Path to the original full image for context.
        model: VLM model name to use (default: gemini-2.5-flash).
        existing_names: List of already-used names to avoid duplicates.

    Returns:
        A unique object name string.
    """
    if existing_names is None:
        existing_names = []

    try:
        # Encode images
        image_b64 = get_image_base64(image_path)
        ori_img_b64 = get_image_base64(ori_img_path)
        # Initialize OpenAI-compatible client
        client = build_client(model)

        # Build prompt including existing names to avoid duplicates
        existing_names_str = ""
        if existing_names:
            existing_names_str = f"\n\nAlready identified objects (do not use these names): {', '.join(existing_names)}"

        # Create messages
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_b64
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": ori_img_b64
                        }
                    },
                    {
                        "type": "text",
                        "text": (
                            "You are classifying segmented regions from an interior/scene photo.\n\n"
                            "The first image shows a segmented region (transparent background = not part of this segment). "
                            "The second image shows the full original scene for context.\n\n"
                            "RULES:\n"
                            "- If this segment is a BACKGROUND element (wall, floor, ceiling, door frame, "
                            "window frame, baseboard, molding, carpet/rug that covers the whole floor, "
                            "large wall art that is part of decor, or any large architectural surface), "
                            "return EXACTLY: background\n"
                            "- If this segment is a distinct OBJECT (furniture, appliance, decoration item, "
                            "plant, lamp, pillow, vase, picture frame, etc.), return a short descriptive name "
                            "like 'red_chair', 'wooden_table', 'floor_lamp'.\n\n"
                            "Use only lowercase letters, numbers, and underscores. "
                            f"2-3 words max.{existing_names_str}\n\n"
                            "Respond with ONLY the name, nothing else."
                        )
                    }
                ]
            }
        ]

        # Call API
        response = client.chat.completions.create(model=model, messages=messages)

        # Parse response
        object_name = response.choices[0].message.content.strip()

        # Clean name: remove quotes, extra spaces
        object_name = object_name.strip('"\'')
        object_name = re.sub(r'\s+', '_', object_name)
        object_name = sanitize_filename(object_name)

        # Ensure uniqueness: add numeric suffix if duplicate
        base_name = object_name
        counter = 1
        while object_name in existing_names:
            object_name = f"{base_name}_{counter}"
            counter += 1

        return object_name

    except Exception as e:
        print(f"VLM naming failed: {e}, using fallback name")
        # Use default name if VLM call fails
        base_name = "object"
        counter = 1
        while f"{base_name}_{counter}" in existing_names:
            counter += 1
        return f"{base_name}_{counter}"


def run_guided_mode(args, sam, image: np.ndarray) -> None:
    """Guided mode: use Gemini-provided bboxes to segment specific objects with SamPredictor.
    
    Args:
        args: Parsed CLI arguments (must have .objects_json, .out, .image).
        sam: Loaded SAM model.
        image: RGB numpy array (H, W, 3).
    """
    from segment_anything import SamPredictor

    with open(args.objects_json, "r") as f:
        objects = json.load(f)
    
    if not objects:
        print("[SAM_WORKER] No objects in objects_json, nothing to segment", flush=True)
        return

    print(f"[SAM_WORKER] Guided mode: {len(objects)} objects from Gemini", flush=True)
    
    # Set image once — embeddings are computed once and reused for all predictions
    predictor = SamPredictor(sam)
    predictor.set_image(image)
    print("[SAM_WORKER] Image embeddings computed", flush=True)

    output_dir = os.path.dirname(args.out)
    os.makedirs(output_dir, exist_ok=True)

    object_names: List[str] = []
    mask_files: List[Dict[str, Any]] = []

    for idx, obj in enumerate(objects):
        name = obj["name"]
        bbox = obj["bbox"]  # [x1, y1, x2, y2]
        print(f"  [{idx+1}/{len(objects)}] {name} bbox={bbox}", flush=True)

        box_np = np.array(bbox)
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box_np,
            multimask_output=True,
        )
        # Pick best mask (highest score)
        best_idx = int(np.argmax(scores))
        mask_bool = masks[best_idx]  # (H, W) bool
        mask_uint8 = (mask_bool.astype(np.uint8)) * 255

        # Ensure unique name
        unique_name = name
        counter = 1
        while unique_name in object_names:
            unique_name = f"{name}_{counter}"
            counter += 1
        object_names.append(unique_name)

        # Save mask as PNG with transparency
        mask_data_dict = {"segmentation": mask_bool, "area": int(mask_bool.sum())}
        png_path = os.path.join(output_dir, f"{unique_name}.png")
        save_mask_as_png(mask_data_dict, image, png_path)

        # Save npy
        npy_path = os.path.join(output_dir, f"{unique_name}.npy")
        np.save(npy_path, mask_uint8)

        mask_files.append({
            "object_id": unique_name,
            "png_path": png_path,
            "npy_path": npy_path,
            "mask_data": mask_uint8,
        })
        print(f"    -> {unique_name} (score={scores[best_idx]:.3f}, area={mask_bool.sum()})", flush=True)

    # Save combined masks
    if mask_files:
        mask_arrays = np.stack([item["mask_data"] for item in mask_files], axis=0)
        np.save(args.out, mask_arrays)
        print(f"Saved combined masks: {args.out} shape={mask_arrays.shape}", flush=True)

    # Save object names mapping
    object_names_json_path = args.out.replace(".npy", "_object_names.json")
    object_mapping = [os.path.splitext(os.path.basename(item["npy_path"]))[0] for item in mask_files]
    with open(object_names_json_path, "w") as f:
        json.dump({"object_names": object_names, "object_mapping": object_mapping}, f, indent=2)
    print(f"Saved object names: {object_names_json_path}", flush=True)

    print(f"\nGuided segmentation: {len(mask_files)} objects", flush=True)
    print(f"Objects: {', '.join(object_names)}", flush=True)


def run_auto_mode(args, sam, image: np.ndarray) -> None:
    """Auto mode: original full-image segmentation with SamAutomaticMaskGenerator + VLM naming."""
    # Generate all masks
    mask_generator = SamAutomaticMaskGenerator(sam)
    raw_masks = mask_generator.generate(image)

    # Apply panic filtering process
    filtered_masks = panic_filtering_process(raw_masks)

    if len(filtered_masks) == 0:
        print("Warning: No masks after filtering")
        return

    # Determine output directory
    output_dir = os.path.dirname(args.out)
    os.makedirs(output_dir, exist_ok=True)

    # Save PNG for each mask and use VLM for naming
    object_names: List[str] = []
    mask_files: List[Dict[str, Any]] = []

    print(f"Processing {len(filtered_masks)} filtered masks...")
    for idx, mask_data in enumerate(filtered_masks):
        temp_png_path = os.path.join(output_dir, f"temp_mask_{idx}.png")
        ori_img_path = args.image
        save_mask_as_png(mask_data, image, temp_png_path)

        print(f"  Identifying object {idx+1}/{len(filtered_masks)}...")
        object_name = get_object_name_from_vlm(
            temp_png_path, ori_img_path, model=args.vlm_model, existing_names=object_names
        )
        if object_name == "background":
            os.remove(temp_png_path)
            continue
        object_names.append(object_name)

        final_png_path = os.path.join(output_dir, f"{object_name}.png")
        if os.path.exists(final_png_path):
            counter = 1
            while os.path.exists(final_png_path):
                final_png_path = os.path.join(output_dir, f"{object_name}_{counter}.png")
                counter += 1
        os.rename(temp_png_path, final_png_path)
        print(f"    Identified as: {object_name}")

        seg = mask_data['segmentation']
        mask_uint8 = (seg.astype(np.uint8)) * 255
        npy_path = os.path.join(output_dir, f"{object_name}.npy")
        if os.path.exists(npy_path):
            counter = 1
            while os.path.exists(npy_path):
                npy_path = os.path.join(output_dir, f"{object_name}_{counter}.npy")
                counter += 1
        np.save(npy_path, mask_uint8)

        mask_files.append({
            "object_id": object_name,
            "png_path": final_png_path,
            "npy_path": npy_path,
            "mask_data": mask_uint8
        })

    if args.out:
        mask_arrays = [item["mask_data"] for item in mask_files]
        if mask_arrays:
            mask_arrays = np.stack(mask_arrays, axis=0)
            np.save(args.out, mask_arrays)
            print(f"Also saved combined masks to: {args.out}")

        object_names_json_path = args.out.replace('.npy', '_object_names.json')
        object_mapping = [os.path.splitext(os.path.basename(item["npy_path"]))[0] for item in mask_files]
        object_names_info = {"object_names": object_names, "object_mapping": object_mapping}
        with open(object_names_json_path, 'w') as f:
            json.dump(object_names_info, f, indent=2)
        print(f"Also saved object names mapping to: {object_names_json_path}")

    print(f"\nGenerated {len(raw_masks)} raw masks, filtered to {len(filtered_masks)} masks")
    print(f"Identified objects: {', '.join(object_names)}")
    print(f"Saved {len(mask_files)} mask files to: {output_dir}")


def main() -> None:
    """Run the SAM worker to segment and identify objects in an image."""
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Path to input image")
    p.add_argument("--out", required=True, help="Path for output npy file")
    p.add_argument("--checkpoint", default=None, help="Path to SAM checkpoint")
    p.add_argument("--mode", default="guided", choices=["auto", "guided"], help="auto=full segment+VLM, guided=Gemini bbox+SamPredictor")
    p.add_argument("--objects-json", default=None, help="Path to JSON with objects+bboxes (required for guided mode)")
    p.add_argument("--vlm-model", default=os.environ.get("MODEL", "gemini-2.5-flash"), help="VLM model for object identification (auto mode only)")
    args = p.parse_args()

    # Model: vit_b (fast, ~375MB) or vit_h (slower, better quality). Override with env SAM_MODEL=vit_h.
    model_type = os.environ.get("SAM_MODEL", "vit_b").lower()
    if model_type not in ("vit_b", "vit_l", "vit_h"):
        model_type = "vit_b"
    default_checkpoints = {
        "vit_b": "sam_vit_b_01ec64.pth",
        "vit_l": "sam_vit_l_0b3195.pth",
        "vit_h": "sam_vit_h_4b8939.pth",
    }
    if args.checkpoint is None:
        args.checkpoint = os.path.join(
            ROOT, "utils", "third_party", "sam", default_checkpoints.get(model_type, "sam_vit_b_01ec64.pth")
        )

    print(f"[SAM_WORKER] Loading image: {args.image}", flush=True)
    image = cv2.imread(args.image)
    if image is None:
        raise ValueError(f"Failed to load image: {args.image}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SAM_WORKER] Loading SAM model ({model_type}) on {device}...", flush=True)
    sam = sam_model_registry[model_type](checkpoint=args.checkpoint)
    sam.to(device=device)

    if args.mode == "guided":
        if not args.objects_json or not os.path.isfile(args.objects_json):
            print(f"[SAM_WORKER] ERROR: guided mode requires --objects-json (got: {args.objects_json})", flush=True)
            print("[SAM_WORKER] Falling back to auto mode", flush=True)
            run_auto_mode(args, sam, image)
        else:
            run_guided_mode(args, sam, image)
    else:
        run_auto_mode(args, sam, image)


if __name__ == "__main__":
    main()
