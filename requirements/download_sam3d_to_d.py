"""Download SAM3D checkpoints to D:\sam3d_checkpoints and create symlink."""
from huggingface_hub import snapshot_download
import os
import shutil

download_dir = r'D:\sam3d_checkpoints\hf-download'
target_dir = r'D:\sam3d_checkpoints\hf'

print(f'Downloading to {download_dir}...')
snapshot_download(
    repo_id='facebook/sam-3d-objects',
    repo_type='model',
    local_dir=download_dir,
    max_workers=1,
)

inner = os.path.join(download_dir, 'checkpoints')
if os.path.isdir(inner):
    if os.path.isdir(target_dir):
        shutil.rmtree(target_dir)
    shutil.move(inner, target_dir)
    print(f'Moved to {target_dir}')
else:
    print('checkpoints subfolder not found, keeping download as-is')
    target_dir = download_dir

if os.path.isdir(download_dir) and download_dir != target_dir:
    shutil.rmtree(download_dir)

print(f'Checkpoints at: {target_dir}')
print('Done!')
