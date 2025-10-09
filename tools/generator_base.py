from mcp.server.fastmcp import FastMCP

mcp = FastMCP("generator-base")

# tool_configs for agent (only the function w/ @mcp.tools)
tool_configs = [
    {
        "type": "function",
        "function": {
            "name": "init_plan",
            "description": "Store the detailed scene plan to a file and return the path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "detailed_description": {"type": "string", "description": "Detailed scene plan"}
                },
                "required": ["detailed_description"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "end",
            "description": "No-op tool used to indicate the process should end.",
        }
    }
]

@mcp.tool()
def init_plan(detailed_description: str) -> dict:
    """
    Store the detailed scene plan to a file and return the path.
    """
    return {"status": "success", "output": detailed_description}

@mcp.tool()
def end() -> dict:
    """
    No-op tool used to indicate the process should end.
    """
    return {"status": "success", "output": "END THE PROCESS"}
