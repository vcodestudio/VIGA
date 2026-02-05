import os
import sys
import json
import base64
import tempfile
import argparse
import subprocess
import socket
import logging
from typing import List, Dict
from pathlib import Path

import numpy as np
import cv2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Add workspace root to sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("sam3d-api")

app = FastAPI(title="SAM Segment API (segment / image export only)")

# --- Models (segment / image export only; no 3D) ---

class SegmentRequest(BaseModel):
    image: str  # base64 encoded image

class SegmentResponse(BaseModel):
    masks: List[str]  # base64 encoded masks
    names: List[str]

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
    
    cf_process = None
    if args.cloudflare:
        cf_process = run_cloudflare_tunnel(args.port)

    try:
        import torch
        uvicorn.config.Config.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Print with flush=True to ensure it appears in terminal immediately
        print("\n" + "="*50, flush=True)
        print("SAM3D API Server starting...", flush=True)
        print(f"Local IP: {local_ip}", flush=True)
        print(f"Endpoint: http://{local_ip}:{args.port}", flush=True)
        print("="*50 + "\n", flush=True)
        
        # Use log_level="info" and ensure no buffering
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        if cf_process:
            cf_process.terminate()
