import json
from typing import Any, Dict, List

def get_openreward_system_prompt(tools: List[Dict[str, Any]]) -> str:
    """
    Dynamically generates a system prompt based on the tools available in the OpenReward task.
    """
    if not tools:
            "Complete the task as requested. If you need to guess or submit an answer, "
            "do so in plain text if no tools are available."

    tool_descriptions = []
    tool_examples = []

    for tool in tools:
        # OpenReward tools in 'openai' format are dicts
        name = tool.get("function", {}).get("name") or tool.get("name") or "unknown"
        description = tool.get("function", {}).get("description") or tool.get("description") or "No description provided."
        parameters = tool.get("function", {}).get("parameters") or tool.get("parameters") or {}
        
        # Build parameter string
        param_str = json.dumps(parameters, indent=2)
        tool_descriptions.append(f"- {name}: {description}\n  Parameters: {param_str}")
        
        # Build a generic example
        example_args = {}
        if isinstance(parameters, dict):
            props = parameters.get("properties", {})
            for p_name, p_info in props.items():
                p_type = p_info.get("type", "string")
                if p_type == "integer" or p_type == "number":
                    example_args[p_name] = 42
                elif p_type == "boolean":
                    example_args[p_name] = True
                else:
                    example_args[p_name] = "example_value"
        
        example_json = json.dumps({"name": name, "arguments": example_args})
        tool_examples.append(f"<tool_call>{example_json}</tool_call>")

    instr = (
        "You are the PLAYER in a reinforcement learning environment.\n"
        "Your goal is to solve the task provided in the user prompt using the tools listed below.\n"
        "You must interact with the environment by calling tools and analyzing the observations.\n\n"
        "DIRECT ACTION MANDATE:\n"
        "1. Your response MUST include a tool call in the XML format provided below.\n"
        "2. Do NOT say 'I am ready' or 'Please give me your guess'.\n"
        "ACT NOW by calling the tool. Do NOT wait.\n\n"
        "FEW-SHOT INTERACTION EXAMPLE:\n"
        "User: You are playing Guess The Number. Enter your guess. Use the guess_number tool.\n"
        "Assistant: <tool_call>{\"name\": \"guess_number\", \"arguments\": {\"number\": 10}}</tool_call>\n\n"
        "AVAILABLE TOOLS:\n\n"
        + "\n\n".join(tool_descriptions)
        + "\n\n"
        "To use a tool, you MUST use the following XML format:\n"
        "<tool_call>{\"name\": \"tool_name\", \"arguments\": {\"arg1\": \"value1\"}}</tool_call>\n\n"
        "One tool call at a time. Short responses.\n"
        "Examples of tool calls:\n"
        + "\n".join(tool_examples)
        + "\n\n"
        "CRITICAL: Do NOT hallucinate that you are finished. Only stop if the environment "
        "returns a 'finished: true' flag or you correctly solve the task.\n"
    )
    
    return instr
