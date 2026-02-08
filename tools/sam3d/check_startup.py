"""서버 시작 지연 원인 점검: init.py와 동일한 import 순서로 단계별 소요 시간을 측정합니다.

사용법 (프로젝트 루트에서, sam3d-objects 환경):
  conda activate sam3d-objects
  python tools/sam3d/check_startup.py
"""
import sys
import time

def t(msg: str) -> float:
    elapsed = time.perf_counter() - t.start
    print(f"  [{elapsed:6.2f}s] {msg}", flush=True)
    return elapsed
t.start = time.perf_counter()

print("check_startup: measuring init.py-style imports", flush=True)

# 1) init.py 상단과 동일
import json
import os
import shutil
import subprocess
import time as time_mod
from typing import Dict, List, Optional
t("stdlib done")

# 2) FastMCP (mcp 패키지 - 여기서 오래 걸리는 경우 많음)
from mcp.server.fastmcp import FastMCP
t("mcp.server.fastmcp.FastMCP done")

# 3) utils._path (path_to_cmd)
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, repo_root)
from utils._path import path_to_cmd
t("utils._path.path_to_cmd done")

# 4) FastMCP 인스턴스 생성
mcp = FastMCP("sam-init")
t("FastMCP('sam-init') done")

total_import = time.perf_counter() - t.start
print(f"\nImport+setup 합계: {total_import:.2f}s\n", flush=True)

# ---------------------------------------------------------------------------
# 2) 실제 init.py 프로세스 띄우고 MCP session.initialize() 소요 시간 측정
# ---------------------------------------------------------------------------
print("=== 2) MCP 서버 연결 (init.py 프로세스 + session.initialize()) ===", flush=True)

async def measure_mcp_connect():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    init_script = os.path.join(repo_root, "tools", "sam3d", "init.py")
    cmd = path_to_cmd.get("tools/sam3d/init.py", sys.executable)
    args = [init_script]
    params = StdioServerParameters(command=cmd, args=args, env=os.environ.copy())

    start = time.perf_counter()
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
    return time.perf_counter() - start

try:
    import asyncio
    elapsed = asyncio.run(measure_mcp_connect())
    print(f"  session.initialize() 완료: {elapsed:.2f}s", flush=True)
    print(f"\n총 MCP 준비 시간: {elapsed:.2f}s (프로세스 시작 ~ initialize 응답)", flush=True)
except Exception as e:
    print(f"  오류: {e}", flush=True, file=sys.stderr)
    sys.exit(1)
