from mcp.server.fastmcp import FastMCP

mcp = FastMCP("verifier-base")

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
            "parameters": {
                "type": "object",
                "properties": {
                    "visual_difference": {"type": "string", "description": "Visual difference between the current scene and the target scene"},
                    "edit_suggestion": {"type": "string", "description": "Edit suggestion for the current scene"},
                    "image_paths": {"type": "string", "description": "Output the observation image path that you think will be helpful to the generator here"},
                },
                "required": ["visual_difference", "edit_suggestion"]
            }
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
def end(visual_difference: str, edit_suggestion: str) -> dict:
    """
    No-op tool used to indicate the process should end.
    """
    return {"status": "success", "output": "END THE PROCESS", "visual_difference": visual_difference, "edit_suggestion": edit_suggestion}