"""Measure investigator MCP server startup. Run from repo root: python tools/blender/check_investigator_startup.py"""
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

    script = os.path.join(repo_root, "tools", "blender", "investigator.py")
    cmd = path_to_cmd.get("tools/blender/investigator.py", sys.executable)
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    params = StdioServerParameters(command=cmd, args=[script], env=env)

    start = time.perf_counter()
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
    return time.perf_counter() - start

def main():
    print("Investigator MCP connect test (timeout 120s)...", flush=True)
    print("  command:", path_to_cmd.get("tools/blender/investigator.py", "python"), flush=True)
    t0 = time.perf_counter()
    try:
        elapsed = asyncio.run(asyncio.wait_for(measure(), timeout=120))
        print("  OK: session.initialize() in", round(elapsed, 2), "s", flush=True)
    except asyncio.TimeoutError:
        print("  TIMEOUT after 120s", flush=True)
        sys.exit(1)
    except Exception as e:
        print("  ERROR:", e, flush=True, file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
