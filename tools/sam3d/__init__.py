"""SAM segmentation and scene initialization (segment + ComfyUI for 3D).

This module provides:
- init: SAM scene initialization MCP server (initialize, get_better_object)
- bridge: SAM bridge for get_better_object (segment + crop + ComfyUI API)
- sam_worker: SAM segmentation worker
- sam3_worker: SAM3 segmentation worker
"""
