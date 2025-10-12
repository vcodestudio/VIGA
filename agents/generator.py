import asyncio
import os
import json
from openai import OpenAI
from typing import Dict, List, Optional, Any
import logging
import copy
from mcp.server.fastmcp import FastMCP
from agents.tool_client import ExternalToolClient
from agents.verifier import VerifierAgent
from prompts import prompt_manager
from utils.common import save_thought_process, get_scene_info, get_image_base64

class GeneratorAgent:
    """
    An MCP agent that takes code modification suggestions and implements them.
    This agent follows the MCP server pattern for better encapsulation and tool integration.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the Generator Agent.
        """
        self.config = kwargs
        self.memory = []
        
        # Connect to tool servers
        self.tool_client = ExternalToolClient()
        
        
        # Initialize OpenAI client
        client_kwargs = {"api_key": self.config.get("api_key"), 'base_url': self.config.get("api_base_url") or os.getenv("OPENAI_BASE_URL") or 'https://api.openai.com/v1'}
        self.client = OpenAI(**client_kwargs)
        
        # Initialize system prompt
        self.system_prompt = prompt_manager.get_all_prompts(self.config.get("mode"), "generator", self.config.get("task_name"), self.config.get("level")).get("system", "")

    async def run(self, verifier: VerifierAgent) -> Dict[str, Any]:
        """
        Generate code based on current memory and optional feedback.
        Now enforces tool calling and returns verifier flag.
        
        Args:
            verifier: Verifier agent instance
            
        Returns:
            Dict containing the generated code, metadata, and verifier flag
        """
        for i in range(self.max_rounds):
            memory = [self.system_prompt] + self.memory[-12:] if len(self.memory) > 12 else self.memory
            chat_args = {
                "model": self.model,
                "messages": memory,
            }
            tools = self._get_tools()
            if tools:
                chat_args['tools'] = tools
                if 'gpt' in self.model:
                    chat_args['parallel_tool_calls'] = False
                if self.model != 'Qwen2-VL-7B-Instruct':
                    chat_args['tool_choice'] = "auto"

            response = self.client.chat.completions.create(**chat_args)
            message = response.choices[0].message
            
            # Store assistant message in conversation history
            self.conversation_history.append(message.model_dump())
            
            if message.tool_calls:
                # Tool was called - this is what we want
                tool_called = True
                full_code = None
                execution_result = None
                call_verifier = False
                
                # Handle tool calls (only first one)
                for i, tool_call in enumerate(message.tool_calls):
                    if i > 0:
                        self.conversation_history.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.function.name,
                            "content": "You can only call a tool once per conversation round."
                        })
                        continue
                        
                    tool_response = await self._handle_tool_call(tool_call)
                    
                    # Store tool response in conversation history
                    self.conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": tool_response['text']
                    })
                    
                    # Check if this was an execute_and_evaluate tool call
                    if tool_call.function.name == "execute_and_evaluate":
                        call_verifier = True
                        execution_result = tool_response.get('execution_result')
                        full_code = tool_response.get('full_code')
                        # Internally handle feedback and add to memory
                        try:
                            if execution_result:
                                if execution_result.get("status") == "success":
                                    result_obj = execution_result.get("result", {})
                                    if result_obj.get("status") == "success":
                                        # Provide the render output (path/dir) back into memory
                                        await self.add_feedback(result_obj.get("output"))
                                    else:
                                        await self.add_feedback(f"Execution error: {result_obj.get('output')}")
                                else:
                                    await self.add_feedback(f"Execution error: {execution_result.get('error', 'Unknown error')}")
                            else:
                                await self.add_feedback("No execution result available. Please ensure you're calling the execute_and_evaluate tool.")
                        except Exception as e:
                            logging.error(f"Failed to add execution feedback to memory: {e}")
                    else:
                        execution_result = {"status": "success", "result": {"status": "success", "output": tool_response['text']}}
                    
                    break
                
                return {
                    "status": "success",
                    "code": full_code,
                    "response": tool_response['text'],
                    "round": self.current_round,
                    "execution_result": execution_result,
                    "call_verifier": call_verifier
                }
            else:
                # No tool called - this violates the requirement
                if attempt < max_attempts - 1:
                    # Add feedback to encourage tool calling
                    self.conversation_history.append({
                        "role": "user",
                        "content": "You must call a tool in each interaction. Please use one of the available tools to proceed with your task."
                    })
                    continue
                else:
                    # Final attempt failed - return error
                    return {
                        "status": "error",
                        "error": "No tool was called after multiple attempts. Tool calling is mandatory.",
                        "round": self.current_round
                    }
    
    async def cleanup(self) -> None:
        """Clean up external connections."""
        await self.tool_client.cleanup()


mcp = FastMCP("generator")
_generator = None

@mcp.tool()
async def initialize_generator(args: dict) -> dict:
    """
    Initialize a new Generator Agent with optional Blender or Slides executor setup.
    """
    try:
        global _generator
        _generator = GeneratorAgent(**args)
        return {
            "status": "success",
            "output": {"text": ["Generator Agent initialized successfully. Executor will be setup automatically when needed."]}
        }
    except Exception as e:
        return {"status": "error", "output": {"text": [str(e)]}}

@mcp.tool()
async def run(verifier: VerifierAgent) -> dict:
    """
    Generate code using the initialized Generator Agent.
    """
    try:
        global _generator
        if _generator is None:
            return {"status": "error", "output": {"text": ["Generator Agent not initialized. Call initialize_generator first."]}}
        result = await _generator.run(verifier=verifier)
        return result
    except Exception as e:
        return {"status": "error", "output": {"text": [str(e)]}}

@mcp.tool()
async def cleanup_generator() -> dict:
    """
    Clean up the Generator Agent and its connections.
    """
    try:
        global _generator
        if _generator is None:
            return {"status": "error", "output": {"text": ["Generator Agent not initialized. Call initialize_generator first."]}}
        await _generator.cleanup()
        return {"status": "success", "output": {"text": ["Generator Agent cleaned up successfully"]}}
    except Exception as e:
        return {"status": "error", "output": {"text": [str(e)]}}

def main():
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
