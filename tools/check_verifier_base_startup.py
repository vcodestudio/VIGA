"""Measure verifier_base MCP startup. Run from repo root: python tools/check_verifier_base_startup.py"""
import asyncio
import os
import sys
import time

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)

from utils._path import path_to_cmd

async def measure():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    script = os.path.join(repo_root, "tools", "verifier_base.py")
    cmd = path_to_cmd.get("tools/verifier_base.py", sys.executable)
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    params = StdioServerParameters(command=cmd, args=[script], env=env)

    start = time.perf_counter()
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
    return time.perf_counter() - start

def main():
    print("verifier_base MCP connect test (timeout 30s)...", flush=True)
    print("  command:", path_to_cmd.get("tools/verifier_base.py", "python"), flush=True)
    try:
        elapsed = asyncio.run(asyncio.wait_for(measure(), timeout=30))
        print("  OK: session.initialize() in", round(elapsed, 2), "s", flush=True)
    except asyncio.TimeoutError:
        print("  TIMEOUT", flush=True)
        sys.exit(1)
    except Exception as e:
        print("  ERROR:", e, flush=True, file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
