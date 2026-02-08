"""Measure Blender exec MCP server startup time. Run from repo root: python tools/blender/check_exec_startup.py"""
import asyncio
import os
import sys
import time

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, repo_root)

from utils._path import path_to_cmd

async def measure():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    script = os.path.join(repo_root, "tools", "blender", "exec.py")
    cmd = path_to_cmd.get("tools/blender/exec.py", sys.executable)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    params = StdioServerParameters(command=cmd, args=[script], env=env)

    start = time.perf_counter()
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
    return time.perf_counter() - start

def main():
    print("Blender exec MCP connect test (timeout 330s)...", flush=True)
    print("  command:", path_to_cmd.get("tools/blender/exec.py", "python"), flush=True)
    t0 = time.perf_counter()
    try:
        elapsed = asyncio.run(asyncio.wait_for(measure(), timeout=330))
        print("  OK: session.initialize() in", round(elapsed, 2), "s", flush=True)
        print("  total:", round(time.perf_counter() - t0, 2), "s", flush=True)
    except asyncio.TimeoutError:
        print("  TIMEOUT after 330s (server did not become ready)", flush=True)
        sys.exit(1)
    except Exception as e:
        print("  ERROR:", e, flush=True, file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
