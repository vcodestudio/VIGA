# SAM3D API setup for Windows (Conda + PyTorch CUDA + API server)
# Full inference (reconstruct) is officially Linux-only; this script gets the API server running and /health testable.
# For full SAM3D inference on Windows, consider WSL2 and install_sam3d.sh.

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path $RepoRoot)) { $RepoRoot = $PSScriptRoot + "\.." }
$RepoRoot = (Resolve-Path $RepoRoot).Path

Write-Host "Repo root: $RepoRoot"
Write-Host ""

# 1. Create conda env
$envName = "sam3d"
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "Conda not found. Install Miniconda/Anaconda and ensure 'conda' is in PATH."
    exit 1
}
Write-Host "Creating conda env: $envName (python=3.11) ..."
conda create -n $envName python=3.11 -y
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# 2. Activate and install PyTorch (CUDA 12.1 for Windows)
Write-Host "Installing PyTorch with CUDA 12.1 ..."
& conda run -n $envName pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# 3. API server dependencies (minimal to run server and /health)
Write-Host "Installing API server deps (fastapi, uvicorn, opencv, ...) ..."
& conda run -n $envName pip install fastapi uvicorn opencv-python numpy pydantic python-multipart
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# 4. Try full SAM3D deps (may fail on Windows due to pytorch3d/gsplat/kaolin)
Write-Host "Installing SAM3D dependencies (may fail on Windows; use WSL for full inference) ..."
Push-Location $RepoRoot
& conda run -n $envName pip install -r requirements/requirement_sam3d-objects.txt --no-deps 2>$null
$fullOk = $LASTEXITCODE -eq 0
if (-not $fullOk) {
    Write-Host "Note: Full requirement_sam3d-objects.txt failed (common on Windows). API /health will work; /reconstruct needs Linux/WSL."
}
& conda run -n $envName pip install -e utils/third_party/sam3d 2>$null
$sam3dOk = $LASTEXITCODE -eq 0
Pop-Location

Write-Host ""
Write-Host "=== SAM3D API setup (Windows) ==="
Write-Host "Activate: conda activate $envName"
Write-Host "Run API:  python tools/sam3d/api_server.py --port 8000"
Write-Host "Test:    curl http://localhost:8000/health"
Write-Host ""
Write-Host "Download checkpoints (from repo root, after 'conda activate sam3d'):"
Write-Host "  pip install 'huggingface-hub[cli]<1.0'"
Write-Host "  huggingface-cli login   # if needed"
Write-Host "  python requirements/download_sam3d_checkpoints.py"
Write-Host ""
