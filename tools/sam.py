# sam.py
import atexit
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
from typing import Optional

from mcp.server.fastmcp import FastMCP

# tool_configs for agent (only the function w/ @mcp.tool)
tool_configs = [
    {
        "type": "function",
        "function": {
            "name": "extract_3d_object",
            "description": "Extract a 3D object from an image using SAM3 for mask extraction and SAM-3D for 3D reconstruction. Returns the GLB file path and position information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "object_name": {
                        "type": "string",
                        "description": "The name of the object to extract from the image (e.g., 'chair', 'table', 'lamp', etc.)"
                    }
                },
                "required": ["object_name"]
            }
        }
    }
]

mcp = FastMCP("sam-executor")
_sam3_worker = None
_sam3d_worker = None
_target_image_path = None
_output_dir = None
_mask_dir = None

SCRIPT_DIR = os.path.dirname(__file__)
SAM3_WORKER_PATH = os.path.join(SCRIPT_DIR, "sam3_worker.py")
SAM3D_WORKER_PATH = os.path.join(SCRIPT_DIR, "sam3d_worker.py")
DEFAULT_SAM3_ENV = "sam3"
DEFAULT_SAM3D_ENV = "sam3d-objects"


class JSONWorkerClient:
    """Simple JSON over stdio client for a background worker process."""

    def __init__(self, env_name: str, script_path: str, extra_args: Optional[list] = None, name: str = ""):
        self.env_name = env_name
        self.script_path = script_path
        self.extra_args = extra_args or []
        self.name = name or os.path.basename(script_path)
        self.process: Optional[subprocess.Popen] = None
        self.lock = threading.Lock()
        self._start()

    @staticmethod
    def _conda_exe() -> str:
        return os.environ.get("CONDA_EXE") or shutil.which("conda") or "conda"

    def _start(self) -> None:
        cmd = [
            self._conda_exe(),
            "run",
            "--no-capture-output",
            "-n",
            self.env_name,
            "python",
            self.script_path,
        ]
        cmd.extend(self.extra_args)
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
            bufsize=1,
            env=env,
        )

    def request(self, payload: dict) -> dict:
        if not self.process or self.process.poll() is not None:
            raise RuntimeError(f"{self.name} worker is not running")
        data = json.dumps(payload)
        with self.lock:
            self.process.stdin.write(data + "\n")
            self.process.stdin.flush()

            while True:
                line = self.process.stdout.readline()
                if line == "":
                    raise RuntimeError(f"{self.name} worker terminated unexpectedly")
                line = line.strip()
                if not line:
                    continue
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    logging.warning("[%s] Non-JSON worker output: %s", self.name, line)
                    continue

    def close(self) -> None:
        if self.process and self.process.poll() is None:
            try:
                self.process.stdin.write(json.dumps({"command": "close"}) + "\n")
                self.process.stdin.flush()
            except Exception:
                pass
            try:
                self.process.stdin.close()
            except Exception:
                pass
            try:
                self.process.terminate()
                self.process.wait(timeout=10)
            except Exception:
                pass
        self.process = None


def _close_worker(worker: Optional[JSONWorkerClient]) -> None:
    if worker:
        worker.close()


def _shutdown_workers() -> None:
    global _sam3_worker, _sam3d_worker
    _close_worker(_sam3_worker)
    _close_worker(_sam3d_worker)
    _sam3_worker = None
    _sam3d_worker = None


atexit.register(_shutdown_workers)


@mcp.tool()
def initialize(args: dict) -> dict:
    """
    Initialize SAM3 and SAM-3D models
    
    Args:
        args: Dictionary containing:
            - target_image_path: Path to the target image
            - output_dir: Directory to save output files
            - sam3d_config_path: (Optional) Path to SAM-3D config file
            - sam3_env_name: (Optional) Conda env name for SAM3 (default: sam3)
            - sam3d_env_name: (Optional) Conda env name for SAM-3D (default: sam3d-objects)
    """
    global _sam3_worker, _sam3d_worker, _target_image_path, _output_dir, _mask_dir
    
    try:
        target_image_path = args.get("target_image_path")
        output_dir = args.get("output_dir")
        sam3d_config_path = args.get("sam3d_config_path")
        sam3_env_name = args.get("sam3_env_name") or DEFAULT_SAM3_ENV
        sam3d_env_name = args.get("sam3d_env_name") or DEFAULT_SAM3D_ENV
        
        if not target_image_path:
            return {"status": "error", "output": {"text": ["target_image_path is required"]}}
        
        if not os.path.exists(target_image_path):
            return {"status": "error", "output": {"text": [f"Target image not found: {target_image_path}"]}}
        
        if not output_dir:
            return {"status": "error", "output": {"text": ["output_dir is required"]}}
        
        # Set default SAM-3D config path if not provided
        if not sam3d_config_path:
            sam3d_config_path = os.path.join(
                os.path.dirname(__file__), "..", "utils", "sam3d", "checkpoints", "hf", "pipeline.yaml"
            )
        
        if not os.path.exists(sam3d_config_path):
            return {"status": "error", "output": {"text": [f"SAM-3D config not found: {sam3d_config_path}"]}}
        
        # Store paths
        _target_image_path = target_image_path
        _output_dir = output_dir
        os.makedirs(_output_dir, exist_ok=True)
        _mask_dir = os.path.join(_output_dir, "masks")
        os.makedirs(_mask_dir, exist_ok=True)
        
        # Restart workers if already running
        _close_worker(_sam3_worker)
        _close_worker(_sam3d_worker)
        
        # Initialize SAM3 worker
        logging.info("[SAM] Starting SAM3 worker in env '%s'...", sam3_env_name)
        _sam3_worker = JSONWorkerClient(
            env_name=sam3_env_name,
            script_path=SAM3_WORKER_PATH,
            name="sam3",
        )
        _sam3_worker.request({"command": "ping"})
        
        # Initialize SAM-3D worker
        logging.info("[SAM] Starting SAM-3D worker in env '%s'...", sam3d_env_name)
        _sam3d_worker = JSONWorkerClient(
            env_name=sam3d_env_name,
            script_path=SAM3D_WORKER_PATH,
            extra_args=["--config-path", os.path.abspath(sam3d_config_path)],
            name="sam3d",
        )
        _sam3d_worker.request({"command": "ping"})
        
        logging.info("[SAM] Initialization completed successfully")
        return {"status": "success", "output": {"text": ["SAM initialize completed"], "tool_configs": tool_configs}}
    
    except Exception as e:
        logging.error(f"[SAM] Initialization failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"status": "error", "output": {"text": [str(e)]}}


@mcp.tool()
def extract_3d_object(object_name: str) -> dict:
    """
    Extract 3D object from the target image using SAM3 and SAM-3D
    
    Args:
        object_name: Name of the object to extract
        
    Returns:
        Dictionary with status and output containing glb_path and position information
    """
    global _sam3_worker, _sam3d_worker, _target_image_path, _output_dir, _mask_dir
    
    try:
        if _sam3_worker is None or _sam3d_worker is None:
            return {"status": "error", "output": {"text": ["SAM workers not initialized. Call initialize first."]}}
        
        if _target_image_path is None:
            return {"status": "error", "output": {"text": ["Target image path not set. Call initialize first."]}}
        
        if _output_dir is None:
            return {"status": "error", "output": {"text": ["Output directory not set. Call initialize first."]}}
        
        # Step 1: Extract mask using SAM3
        logging.info("[SAM] Extracting mask for object: %s", object_name)
        mask_response = _sam3_worker.request(
            {
                "command": "extract_mask",
                "image_path": _target_image_path,
                "object_name": object_name,
                "mask_dir": _mask_dir,
            }
        )

        if mask_response.get("status") != "success":
            message = mask_response.get("message") or f"Failed to extract mask for object: {object_name}"
            return {"status": "error", "output": {"text": [message]}}

        mask_path = mask_response.get("mask_path")
        if not mask_path or not os.path.exists(mask_path):
            return {"status": "error", "output": {"text": [f"Mask file missing: {mask_path}"]}}

        # Step 2: Reconstruct 3D using SAM-3D
        logging.info("[SAM] Reconstructing 3D object: %s", object_name)
        glb_path_candidate = os.path.join(_output_dir, f"{object_name}.glb")
        recon_response = _sam3d_worker.request(
            {
                "command": "reconstruct",
                "image_path": _target_image_path,
                "mask_path": mask_path,
                "glb_path": glb_path_candidate,
                "seed": 42,
            }
        )

        if recon_response.get("status") != "success":
            message = recon_response.get("message") or f"Failed to reconstruct 3D object: {object_name}"
            return {"status": "error", "output": {"text": [message]}}

        glb_path = recon_response.get("glb_path")
        translation = recon_response.get("translation")
        rotation = recon_response.get("rotation")
        scale = recon_response.get("scale")
        
        # Prepare output
        output_text = [f"Successfully extracted 3D object: {object_name}"]
        if glb_path:
            output_text.append(f"GLB file saved to: {glb_path}")
        
        output_data = {
            "glb_path": glb_path,
            "translation": translation,
            "rotation": rotation,
            "scale": scale,
        }
        
        return {
            "status": "success",
            "output": {
                "text": output_text,
                "data": output_data
            }
        }
    
    except Exception as e:
        logging.error(f"[SAM] Failed to extract 3D object: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"status": "error", "output": {"text": [str(e)]}}


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Test mode
        target_image_path = "data/static_scene/christmas/target.png"
        output_dir = "output/test/sam"
        sam3d_config_path = os.path.join(
            os.path.dirname(__file__), "..", "utils", "sam3d", "checkpoints", "hf", "pipeline.yaml"
        )
        
        init_payload = {
            "target_image_path": target_image_path,
            "output_dir": output_dir,
            "sam3d_config_path": sam3d_config_path,
            "sam3_env_name": DEFAULT_SAM3_ENV,
            "sam3d_env_name": DEFAULT_SAM3D_ENV,
        }
        result = initialize(init_payload)
        print("initialize result:", result)
        
        if result.get("status") == "success":
            result = extract_3d_object(object_name="chair")
            print("extract_3d_object result:", result)
    else:
        mcp.run()


if __name__ == "__main__":
    main()

