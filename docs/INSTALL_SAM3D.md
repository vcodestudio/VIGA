# SAM3D Environment Installation Guide

## Quick Install

```bash
# Method 1: Use the installation script (recommended)
bash scripts/install_sam3d_requirements.sh

# Method 2: Manual step-by-step installation
# Step 1: Install PyTorch first
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Step 2: Install packages that need torch at build time (with --no-build-isolation)
pip install --no-build-isolation git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git@59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d
pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae

# Step 3: Install remaining dependencies
pip install -r requirements/requirement_sam3d-objects.txt
```

## Why This Order and `--no-build-isolation`?

### The Problem

Some packages like `diff_gaussian_rasterization` and `nvdiffrast` require PyTorch to be installed **at build time** (not just runtime). They use torch in their `setup.py` to detect CUDA versions and compile extensions.

### Why Simple Ordering Doesn't Work

Even if you put torch at the top of requirements.txt, pip uses **PEP 517 build isolation**. This means:

1. Pip creates a **temporary isolated environment** (e.g., `/tmp/pip-build-env-xxx/`)
2. Builds packages in this isolated environment
3. This isolated environment doesn't have your already-installed torch!

Result: `ModuleNotFoundError: No module named 'torch'` even though torch is installed in your main environment.

### The Solution: `--no-build-isolation`

The `--no-build-isolation` flag tells pip to:
- Skip creating the isolated build environment
- Use the current environment's packages during build
- Allow the build process to access your installed torch

## Troubleshooting

### Error: "No module named 'torch'" during installation

**Symptom**: You see this error in `/tmp/pip-build-env-xxx/` even though you have torch installed.

**Cause**: PEP 517 build isolation - pip is building in an isolated environment without torch.

**Solution**: Use `--no-build-isolation` flag:
```bash
pip install --no-build-isolation git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git@59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d
```

### CUDA version mismatch

Make sure your CUDA version matches the torch version. For `torch==2.5.1+cu121`, you need CUDA 12.1.

Check your CUDA version:
```bash
nvidia-smi
```

### Build fails with CUDA errors

Ensure you have CUDA development tools installed:
```bash
nvcc --version
```

If `nvcc` is not found, install CUDA toolkit for your system.
