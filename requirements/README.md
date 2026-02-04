# Requirements

Conda environment setup for VIGA. Alternative to Docker-based deployment.

## Quick Start

```bash
# 1. Create main agent environment
conda create -n agent python=3.10 -y
conda activate agent
pip install -r requirements/requirement_agent.txt

# 2. Create tool environments (based on modes you need)
# For 3D modes (BlenderGym, BlenderBench, Static/Dynamic Scene)
conda create -n blender python=3.11 -y
conda activate blender
pip install -r requirements/requirement_blender.txt

# For SlideBench mode
conda create -n pptx python=3.10 -y
conda activate pptx
pip install -r requirements/requirement_pptx.txt
```

## Requirement Files

| File | Environment | Python | Modes |
|------|-------------|--------|-------|
| `requirement_agent.txt` | agent | 3.10 | All (main runtime) |
| `requirement_blender.txt` | blender | 3.11 | 3D modes |
| `requirement_pptx.txt` | pptx | 3.10 | SlideBench |
| `requirement_eval-blender.txt` | eval-blender | 3.11 | 3D evaluation |
| `requirement_eval-pptx.txt` | eval-pptx | 3.10 | Slides evaluation |
| `requirement_sam3d-objects.txt` | sam3d | 3.11 | SAM3D API (Static Scene 3D) |

### SAM3D API (Windows)

```powershell
# 1. Init submodule (if not done)
git submodule update --init utils/third_party/sam3d

# 2. Create env and install (PowerShell)
.\requirements\install_sam3d_win.ps1

# 3. Activate and run API server
conda activate sam3d
python tools/sam3d/api_server.py --port 8000

# 4. Test
curl http://localhost:8000/health   # -> {"status":"ok","device":"cuda"}

# 5. Download checkpoints (for /reconstruct; HuggingFace access may be required)
pip install "huggingface-hub[cli]<1.0"
python requirements/download_sam3d_checkpoints.py
```

Full inference (`/reconstruct`, `/segment`) is officially **Linux-only** (pytorch3d/gsplat). On Windows the API server runs and `/health` works; for full pipeline use WSL2 and `install_sam3d.sh`.

## External Dependencies

### Blender (for 3D modes)

```bash
cd utils/third_party
git clone https://github.com/princeton-vl/infinigen.git
cd infinigen
conda activate blender
bash scripts/install/interactive_blender.sh
```

### LibreOffice (for SlideBench)

```bash
sudo apt-get install -y libreoffice unoconv
```

## Configuration

Create two config files in `utils/`:

### `_api_keys.py`

```python
OPENAI_API_KEY = "your-key"
CLAUDE_API_KEY = "your-key"
GEMINI_API_KEY = "your-key"
MESHY_API_KEY = "your-key"
```

### `_path.py`

```python
path_to_cmd = {
    "tools/blender/exec.py": "/path/to/conda/envs/blender/bin/python",
    "tools/blender/investigator.py": "/path/to/conda/envs/blender/bin/python",
    "tools/slides/exec.py": "/path/to/conda/envs/pptx/bin/python",
}
```

Find your conda paths with `conda env list`.

## Verification

```bash
# Test agent environment
conda activate agent
python -c "import openai; import mcp; print('OK')"

# Test blender environment
conda activate blender
python -c "import bpy; print('OK')"

# Test pptx environment
conda activate pptx
python -c "import pptx; print('OK')"
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| ModuleNotFoundError | Ensure correct conda env is activated |
| Infinigen install fails | Check Python 3.11, install `build-essential cmake` |
| CUDA not found | Run `nvidia-smi`, reinstall PyTorch with correct CUDA |
| PPTX conversion fails | Reinstall LibreOffice |
| Wrong Python path | Update `_path.py` with `which python` output |

## Reference to Original Repositories

If you encounter installation issues, please refer to the original repositories:

- **Blender environment**: [BlenderGym](https://blendergym.github.io/) | [GitHub](https://github.com/para-lost/AutoPresent)
- **PPTX environment**: [AutoPresent](https://github.com/para-lost/AutoPresent)
- **SAM (Segment Anything Model)**: [Meta AI SAM](https://segment-anything.com/) | [GitHub](https://github.com/facebookresearch/segment-anything) | [SAM 2](https://ai.meta.com/sam2/)
- **vLLM**: [vLLM Official Site](https://vllm.ai/) | [GitHub](https://github.com/vllm-project/vllm)
