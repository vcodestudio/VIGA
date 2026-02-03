import os
import sys
import json
import base64
import shutil
import tempfile
import argparse
import subprocess
import socket
import logging
from typing import List, Dict, Optional
from pathlib import Path

import numpy as np
import cv2
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn

# Add workspace root to sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("sam3d-api")

app = FastAPI(title="SAM3D Remote API Server")

# --- Models ---

class ReconstructRequest(BaseModel):
    image: str  # base64 encoded image

class SegmentRequest(BaseModel):
    image: str  # base64 encoded image

class ReconstructObjectRequest(BaseModel):
    image: str  # base64 encoded image
    mask: str   # base64 encoded mask (uint8 0/255)
    name: str

class GLBFile(BaseModel):
    name: str
    data: str  # base64 encoded GLB data

class ReconstructResponse(BaseModel):
    glb_files: List[GLBFile]
    transforms: List[Dict]

class SegmentResponse(BaseModel):
    masks: List[str]  # base64 encoded masks
    names: List[str]

class ReconstructObjectResponse(BaseModel):
    glb: str  # base64 encoded GLB data
    transform: Dict

# --- Helpers ---

def decode_base64_to_image(b64_str: str) -> np.ndarray:
    try:
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]
        img_data = base64.b64decode(b64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

def encode_file_to_base64(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

# --- Endpoints ---

@app.get("/health")
async def health():
    return {"status": "ok", "device": "cuda" if uvicorn.config.Config.device == "cuda" else "cpu"}

@app.post("/reconstruct", response_model=ReconstructResponse)
async def reconstruct(req: ReconstructRequest):
    """Full scene reconstruction: SAM -> SAM3D -> GLBs"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        img_path = tmp_path / "input.png"
        
        # Save input image
        img = decode_base64_to_image(req.image)
        cv2.imwrite(str(img_path), img)
        
        # 1. Run SAM Worker
        logger.info("Running SAM segmentation...")
        sam_worker = os.path.join(ROOT, "tools", "sam3d", "sam_worker.py")
        masks_npy = tmp_path / "all_masks.npy"
        
        # We need to use the correct python environment for SAM
        # Assuming the server is already running in an environment that has the dependencies
        # or we use the path_to_cmd logic if needed. For now, use sys.executable.
        try:
            subprocess.run(
                [sys.executable, sam_worker, "--image", str(img_path), "--out", str(masks_npy)],
                check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"SAM Worker failed: {e.stderr}")
            raise HTTPException(status_code=500, detail=f"SAM segmentation failed: {e.stderr}")

        # 2. Reconstruct each object
        logger.info("Running SAM3D reconstruction...")
        sam3d_worker = os.path.join(ROOT, "tools", "sam3d", "sam3d_worker.py")
        sam3d_config = os.path.join(ROOT, "utils", "third_party", "sam3d", "checkpoints", "hf", "pipeline.yaml")
        
        # Load object mapping
        mapping_path = tmp_path / "all_masks_object_names.json"
        if not mapping_path.exists():
            raise HTTPException(status_code=500, detail="SAM output mapping not found")
            
        with open(mapping_path, "r") as f:
            mapping_info = json.load(f)
            object_mapping = mapping_info.get("object_mapping", [])

        glb_files = []
        transforms = []
        
        for name in object_mapping:
            mask_path = tmp_path / f"{name}.npy"
            glb_path = tmp_path / f"{name}.glb"
            info_path = tmp_path / f"{name}.json"
            
            try:
                subprocess.run(
                    [
                        sys.executable, sam3d_worker,
                        "--image", str(img_path),
                        "--mask", str(mask_path),
                        "--config", sam3d_config,
                        "--glb", str(glb_path),
                        "--info", str(info_path)
                    ],
                    check=True, capture_output=True, text=True
                )
                
                # Collect results
                glb_files.append(GLBFile(name=f"{name}.glb", data=encode_file_to_base64(str(glb_path))))
                with open(info_path, "r") as f:
                    transforms.append(json.load(f))
                    
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to reconstruct object {name}: {e.stderr}")
                continue

        return ReconstructResponse(glb_files=glb_files, transforms=transforms)

@app.post("/segment", response_model=SegmentResponse)
async def segment(req: SegmentRequest):
    """SAM segmentation only"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        img_path = tmp_path / "input.png"
        img = decode_base64_to_image(req.image)
        cv2.imwrite(str(img_path), img)
        
        sam_worker = os.path.join(ROOT, "tools", "sam3d", "sam_worker.py")
        masks_npy = tmp_path / "all_masks.npy"
        
        try:
            subprocess.run(
                [sys.executable, sam_worker, "--image", str(img_path), "--out", str(masks_npy)],
                check=True, capture_output=True, text=True
            )
            
            mapping_path = tmp_path / "all_masks_object_names.json"
            with open(mapping_path, "r") as f:
                mapping_info = json.load(f)
                object_mapping = mapping_info.get("object_mapping", [])
            
            masks_b64 = []
            for name in object_mapping:
                mask_path = tmp_path / f"{name}.npy"
                mask_data = np.load(mask_path)
                _, buffer = cv2.imencode(".png", mask_data)
                masks_b64.append(base64.b64encode(buffer).decode())
                
            return SegmentResponse(masks=masks_b64, names=object_mapping)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/reconstruct_object", response_model=ReconstructObjectResponse)
async def reconstruct_object(req: ReconstructObjectRequest):
    """Single object reconstruction"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        img_path = tmp_path / "input.png"
        mask_path = tmp_path / "mask.npy"
        glb_path = tmp_path / "output.glb"
        info_path = tmp_path / "info.json"
        
        # Save input image
        img = decode_base64_to_image(req.image)
        cv2.imwrite(str(img_path), img)
        
        # Save mask
        mask_data = base64.b64decode(req.mask)
        mask_arr = cv2.imdecode(np.frombuffer(mask_data, np.uint8), cv2.IMREAD_GRAYCASE)
        np.save(str(mask_path), mask_arr)
        
        sam3d_worker = os.path.join(ROOT, "tools", "sam3d", "sam3d_worker.py")
        sam3d_config = os.path.join(ROOT, "utils", "third_party", "sam3d", "checkpoints", "hf", "pipeline.yaml")
        
        try:
            subprocess.run(
                [
                    sys.executable, sam3d_worker,
                    "--image", str(img_path),
                    "--mask", str(mask_path),
                    "--config", sam3d_config,
                    "--glb", str(glb_path),
                    "--info", str(info_path)
                ],
                check=True, capture_output=True, text=True
            )
            
            with open(info_path, "r") as f:
                transform = json.load(f)
                
            return ReconstructObjectResponse(
                glb=encode_file_to_base64(str(glb_path)),
                transform=transform
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

# --- CLI ---

def run_cloudflare_tunnel(port: int):
    logger.info("Starting Cloudflare Tunnel...")
    try:
        # Check if cloudflared is installed
        subprocess.run(["cloudflared", "--version"], check=True, capture_output=True)
        
        # Start tunnel
        process = subprocess.Popen(
            ["cloudflared", "tunnel", "--url", f"http://localhost:{port}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # Try to find the URL in the output
        import re
        url_pattern = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
        
        # Read output in a separate thread or just wait a bit
        logger.info("Waiting for Cloudflare URL...")
        for _ in range(20): # Try for 10 seconds
            line = process.stdout.readline()
            if not line: break
            match = url_pattern.search(line)
            if match:
                url = match.group(0)
                print(f"\n" + "="*50)
                print(f"CLOUDFLARE TUNNEL ACTIVE")
                print(f"URL: {url}")
                print("="*50 + "\n")
                return process
            import time
            time.sleep(0.5)
            
        logger.warning("Could not detect Cloudflare Tunnel URL. Check if cloudflared is running.")
        return process
    except Exception as e:
        logger.error(f"Failed to start Cloudflare Tunnel: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--cloudflare", action="store_true", help="Enable Cloudflare Tunnel")
    args = parser.parse_args()

    local_ip = get_local_ip()
    print(f"\n" + "="*50)
    print(f"SAM3D API Server starting...")
    print(f"Local IP: {local_ip}")
    print(f"Endpoint: http://{local_ip}:{args.port}")
    print("="*50 + "\n")

    cf_process = None
    if args.cloudflare:
        cf_process = run_cloudflare_tunnel(args.port)

    try:
        import torch
        uvicorn.config.Config.device = "cuda" if torch.cuda.is_available() else "cpu"
        uvicorn.run(app, host=args.host, port=args.port)
    finally:
        if cf_process:
            cf_process.terminate()
