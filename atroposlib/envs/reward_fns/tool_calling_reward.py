"""
Tool calling reward function that validates:
1. Format: Response contains <tool_call> tags
2. JSON: The content within the tags is valid parseable JSON
3. Tool existence: The tool called exists in the provided list of tools
4. Arguments: The arguments provided match the tool's expected parameters (optional)
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Union

from atroposlib.envs.reward_fns.registry import registry
from atroposlib.envs.reward_fns.reward_function import RewardFunction

logger = logging.getLogger(__name__)


@registry.register
class ToolCallingReward(RewardFunction):
    """
    Reward function for checking tool calling format and validity.
    
    Validates:
    1. Format: Response contains <tool_call> tags
    2. JSON: The content within the tags is valid parseable JSON
    3. Tool existence: The tool called exists in the provided list of tools
    4. Arguments: The arguments provided match the tool's expected parameters (optional)
    """
    
    def __init__(
        self, 
        tools: List[Dict], 
        preferred_tags: List[str] = None,
        check_arguments: bool = False,
        weight: float = 1.0, 
        name: Optional[str] = None, 
        **kwargs
    ):
        """
        Initialize the ToolCallingReward function.
        
        Args:
            tools: List of tool definitions with at least 'name' field
            preferred_tags: Tags to look for in the response (default: ['tool_call'])
            check_arguments: Whether to check if arguments match expected parameters
            weight: Importance factor when combining with other rewards
            name: Optional custom name for this reward function instance
            **kwargs: Additional configuration parameters
        """
        super().__init__(weight, name, **kwargs)
        self.tools = tools
        self.preferred_tags = preferred_tags or ["tool_call"]
        self.check_arguments = check_arguments
        
        # Extract valid tool names for quick lookup
        self.valid_tool_names = set()
        self.tool_parameters = {}
        
        for tool in tools:
            # Handle different tool definition formats
            if isinstance(tool, dict):
                if "name" in tool:
                    self.valid_tool_names.add(tool["name"])
                    
                    # Store parameters for argument validation if needed
                    if check_arguments and "function" in tool:
                        if "parameters" in tool["function"]:
                            self.tool_parameters[tool["name"]] = tool["function"]["parameters"]
                
                elif "function" in tool and "name" in tool["function"]:
                    self.valid_tool_names.add(tool["function"]["name"])
                    
                    # Store parameters for argument validation if needed
                    if check_arguments and "parameters" in tool["function"]:
                        self.tool_parameters[tool["function"]["name"]] = tool["function"]["parameters"]
        
        logger.info(f"Initialized ToolCallingReward with {len(self.valid_tool_names)} valid tools")
    
    def _extract_tool_call(self, text: str) -> Optional[str]:
        """Extract the content within the tool call tags."""
        for tag in self.preferred_tags:
            pattern = f"<{tag}>(.*?)</{tag}>"
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                return matches[0].strip()
        return None
    
    def _validate_arguments(self, tool_name: str, arguments: Dict) -> bool:
        """Validate that the arguments match the expected parameters for the tool."""
        if not self.check_arguments or tool_name not in self.tool_parameters:
            return True
            
        params = self.tool_parameters[tool_name]
        
        # Check required parameters
        if "required" in params:
            for required_param in params["required"]:
                if required_param not in arguments:
                    logger.debug(f"Missing required parameter: {required_param} for tool {tool_name}")
                    return False
        
        # Check parameter types if properties are defined
        if "properties" in params:
            for param_name, param_value in arguments.items():
                if param_name in params["properties"]:
                    prop = params["properties"][param_name]
                    
                    # Type checking (simplified)
                    if "type" in prop:
                        if prop["type"] == "string" and not isinstance(param_value, str):
                            return False
                        elif prop["type"] == "number" and not isinstance(param_value, (int, float)):
                            return False
                        elif prop["type"] == "integer" and not isinstance(param_value, int):
                            return False
                        elif prop["type"] == "boolean" and not isinstance(param_value, bool):
                            return False
                        elif prop["type"] == "array" and not isinstance(param_value, list):
                            return False
                        elif prop["type"] == "object" and not isinstance(param_value, dict):
                            return False
                    
                    # Enum validation
                    if "enum" in prop and param_value not in prop["enum"]:
                        return False
        
        return True
    
    def compute(self, completions: List[Any], **kwargs) -> List[float]:
        """
        Compute reward scores for the given completions.
        
        Scores:
        - 1.0: Perfect format with valid JSON and existing tool
        - 0.7: Valid format and JSON, but tool doesn't exist
        - 0.5: Valid format but JSON parsing error
        - 0.0: No tool call format found
        
        Args:
            completions: List of completions to evaluate
            
        Returns:
            List of reward scores, one for each completion
        """
        results = []
        
        for completion in completions:
            # Get the content from the completion
            content = self.get_content(completion)
            
            # Check if tool call format exists
            tool_call_content = self._extract_tool_call(content)
            if not tool_call_content:
                logger.debug(f"No tool call format found in: {content[:100]}...")
                results.append(0.0)
                continue
            
            # Try to parse the JSON
            try:
                # Safely handle single quotes or malformed JSON
                tool_call_content = tool_call_content.replace("'", '"')
                tool_call = json.loads(tool_call_content)
                
                # Check if the tool exists
                tool_name = tool_call.get("name", "")
                if not tool_name or tool_name not in self.valid_tool_names:
                    logger.debug(f"Tool '{tool_name}' not found in valid tools")
                    results.append(0.7)  # Valid JSON but invalid tool
                    continue
                
                # Check arguments if enabled
                if self.check_arguments:
                    arguments = tool_call.get("arguments", {})
                    if not self._validate_arguments(tool_name, arguments):
                        logger.debug(f"Invalid arguments for tool {tool_name}: {arguments}")
                        results.append(0.8)  # Valid tool but invalid arguments
                        continue
                
                # Everything is valid
                results.append(1.0)
                
            except json.JSONDecodeError as e:
                logger.debug(f"JSON parsing error: {e} in: {tool_call_content[:100]}...")
                results.append(0.5)  # Valid format but JSON error
                
        return results


# Legacy function for backward compatibility
def tool_calling_reward(
    completions: List[Any], tools: List[Dict], preferred_tags: List[str] = None, **kwargs
) -> List[float]:
    """
    Check if model responses correctly use tool calling format and call valid tools.
    
    Args:
        completions: List of completions to evaluate
        tools: List of tool definitions the model should be using
        preferred_tags: Tags to look for in the response (default: ['tool_call'])
        **kwargs: Additional parameters
        
    Returns:
        List of rewards (1.0 for correct format and valid tool, lower for issues)
    """
    reward_fn = ToolCallingReward(tools=tools, preferred_tags=preferred_tags)
    return reward_fn.compute(completions, **kwargs) 