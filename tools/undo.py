from mcp.server.fastmcp import FastMCP

mcp = FastMCP("undo-last-step")

# tool_configs for agent (only the function w/ @mcp.tools)
tool_configs = [
    {
        "type": "function",
        "function": {
            "name": "undo_last_step",
            "description": "If you believe that your last action did not improve the current state, but instead moved it further away from the target state, you can call this tool to undo the last action.",
        }
    }
]

@mcp.tool()
def undo_last_step() -> dict:
    """
    Undo the last step.
    """
    return {"status": "success", "output": {"text": ["Last step undone successfully"]}}

@mcp.tool()
def initialize(args: dict) -> dict:
    """
    Initialize the undo last step.
    """
    return {"status": "success", "output": {"text": ["Undo last step initialized successfully"], "tool_configs": tool_configs}}

def main():
    mcp.run()

if __name__ == "__main__":
    main()