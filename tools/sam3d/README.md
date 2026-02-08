# SAM Segment + ComfyUI Tools

SAM으로 이미지에서 객체를 세그먼트하고, 크롭 이미지를 ComfyUI API로 보내 3D 에셋을 생성하는 도구입니다. (로컬 SAM3D 3D 복원은 사용하지 않음.)

## Files

| File | Description |
|------|-------------|
| `init.py` | MCP 서버: initialize, get_better_object |
| `bridge.py` | get_better_object 구현 (segment + crop + ComfyUI API) |
| `sam_worker.py` | SAM 세그먼트 워커 |
| `sam3_worker.py` | SAM3 워커 |

## Architecture

```
init.py (MCP Server)
    │
    ▼
bridge.py (get_better_object)
    │
    ├── sam_worker.py   (2D segmentation)
    └── ComfyUI API     (cropped image → 3D)
```

## Dependencies

- SAM 체크포인트: `utils/third_party/sam/` 또는 `download_sam_checkpoint.py`
- ComfyUI API: `COMFYUI_API_URL` (이미지 → 3D 워크플로우)

## Environment

- `sam3d-objects`: init.py, bridge.py
- `sam`: sam_worker.py
