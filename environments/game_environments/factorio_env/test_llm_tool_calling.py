#!/usr/bin/env python3
"""
Test script to evaluate LLM's ability to correctly call FLE tools.

This script tests whether Hermes-405B can correctly generate tool calls
for all discovered FLE tools, and identifies the practical limit for
the number of tools that can be reliably handled.
"""

import json
import os
import sys
import re
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass
import time

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from fle_tool_discovery import discover_fle_tools, format_tool_for_prompt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of testing a single tool."""
    tool_name: str
    success: bool
    generated_call: Optional[str] = None
    error: Optional[str] = None
    response_time: float = 0.0


class LLMToolTester:
    """Test LLM's ability to call FLE tools correctly."""
    
    def __init__(self, api_key: str, base_url: str = "https://inference-api.nousresearch.com/v1"):
        """Initialize the tester with API credentials."""
        self.api_key = api_key
        self.base_url = base_url
        self.tools = discover_fle_tools()
        
    async def test_single_tool(self, tool_name: str, tool_info: Dict[str, Any]) -> TestResult:
        """Test if LLM can correctly call a single tool."""
        start_time = time.time()
        
        try:
            # Create a scenario that requires this specific tool
            prompt = self._create_tool_prompt(tool_name, tool_info)
            
            # Format tool spec for the system prompt
            tool_spec = format_tool_for_prompt(tool_info)
            
            # Call the LLM
            response = await self._call_llm(prompt, [tool_spec])
            
            # Parse and validate the response
            tool_call = self._extract_tool_call(response)
            
            if tool_call:
                # Validate the tool call
                is_valid = self._validate_tool_call(tool_call, tool_name, tool_info)
                
                return TestResult(
                    tool_name=tool_name,
                    success=is_valid,
                    generated_call=json.dumps(tool_call),
                    response_time=time.time() - start_time
                )
            else:
                return TestResult(
                    tool_name=tool_name,
                    success=False,
                    error="No tool call found in response",
                    response_time=time.time() - start_time
                )
                
        except Exception as e:
            return TestResult(
                tool_name=tool_name,
                success=False,
                error=str(e),
                response_time=time.time() - start_time
            )
    
    async def test_tool_batch(self, tool_names_to_test: List[str], all_tool_names: Optional[List[str]] = None) -> Dict[str, TestResult]:
        """Test if LLM can pick the right tool from all available tools.
        
        Args:
            tool_names_to_test: Which tools to test
            all_tool_names: All tools to make available (defaults to all discovered tools)
        """
        results = {}
        
        # Use all tools if not specified
        if all_tool_names is None:
            all_tool_names = list(self.tools.keys())
        
        # Prepare ALL tool specs
        tool_specs = []
        for name in all_tool_names:
            if name in self.tools:
                tool_specs.append(format_tool_for_prompt(self.tools[name]))
        
        logger.info(f"Testing {len(tool_names_to_test)} tools with {len(tool_specs)} tools available")
        
        # Test each tool with ALL tools available
        for tool_name in tool_names_to_test:
            if tool_name not in self.tools:
                continue
                
            start_time = time.time()
            
            try:
                # Create scenario for this specific tool
                prompt = self._create_tool_prompt(tool_name, self.tools[tool_name])
                
                # Call LLM with ALL tools available
                response = await self._call_llm(prompt, tool_specs)
                
                # Parse and validate
                tool_call = self._extract_tool_call(response)
                
                if tool_call:
                    # Check if it called the right tool
                    called_tool = tool_call.get("name", "")
                    if called_tool == tool_name:
                        is_valid = self._validate_tool_call(tool_call, tool_name, self.tools[tool_name])
                        results[tool_name] = TestResult(
                            tool_name=tool_name,
                            success=is_valid,
                            generated_call=json.dumps(tool_call),
                            response_time=time.time() - start_time
                        )
                    else:
                        results[tool_name] = TestResult(
                            tool_name=tool_name,
                            success=False,
                            error=f"Called wrong tool: {called_tool}",
                            generated_call=json.dumps(tool_call),
                            response_time=time.time() - start_time
                        )
                else:
                    results[tool_name] = TestResult(
                        tool_name=tool_name,
                        success=False,
                        error="No tool call found",
                        response_time=time.time() - start_time
                    )
                    
            except Exception as e:
                results[tool_name] = TestResult(
                    tool_name=tool_name,
                    success=False,
                    error=str(e),
                    response_time=time.time() - start_time
                )
                
        return results
    
    def _create_tool_prompt(self, tool_name: str, tool_info: Dict[str, Any]) -> str:
        """Create a scenario prompt that requires using the specified tool."""
        
        # Tool-specific scenarios
        scenarios = {
            "get_entities": "I need to see what entities are within 50 tiles of my current position.",
            "place_entity": "Place an iron chest at position x=10, y=20.",
            "nearest": "Find the nearest iron ore deposit.",
            "craft_item": "Craft 5 transport belts.",
            "inspect_inventory": "Check what items I have in my inventory.",
            "move_to": "Move to position x=15, y=25.",
            "harvest_resource": "Mine the iron ore at my current position.",
            "insert_item": "Put 10 coal into the furnace next to me.",
            "extract_item": "Take the iron plates out of the furnace.",
            "connect_entities": "Connect the miner to the furnace with transport belts.",
            "pickup_entity": "Pick up the transport belt at position x=5, y=5.",
            "rotate_entity": "Rotate the inserter to face north.",
            "set_entity_recipe": "Set the assembling machine to produce iron gear wheels.",
            "can_place_entity": "Check if I can place a stone furnace at position x=0, y=0.",
            "get_resource_patch": "Find all the coal in the resource patch at my location.",
            "get_prototype_recipe": "What ingredients are needed to craft an iron gear wheel?",
            "place_entity_next_to": "Place an inserter next to the furnace.",
            "send_message": "Tell my teammate 'I found iron ore at the north base'.",
            "sleep": "Wait for 5 seconds.",
            "score": "Check my current score.",
            "launch_rocket": "Launch the rocket to win the game.",
            "get_research_progress": "Check the progress on automation research.",
            "set_research": "Start researching logistics.",
            "shift_entity": "Move the chest 2 tiles to the right.",
            "get_entity": "Get the entity at position x=10, y=10.",
            "get_connection_amount": "Check how many entities are connected in this power network.",
            "nearest_buildable": "Find the nearest location where I can build a mining drill.",
            "print": "Display the message 'Factory is operational'."
        }
        
        # Get scenario or create generic one
        scenario = scenarios.get(tool_name, f"Use the {tool_name} tool appropriately.")
        
        return f"Task: {scenario}\n\nGenerate ONLY the tool call JSON, no explanation."
    
    def _extract_tool_call(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract tool call JSON from LLM response."""
        
        # Try to find JSON in the response
        # Look for content between <tool_call> tags first
        tool_call_match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', response, re.DOTALL)
        if tool_call_match:
            try:
                return json.loads(tool_call_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find raw JSON
        json_match = re.search(r'\{[^{}]*"name"\s*:\s*"[^"]+[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        # Try to parse the entire response as JSON
        try:
            parsed = json.loads(response.strip())
            if isinstance(parsed, dict) and "name" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass
        
        return None
    
    def _validate_tool_call(self, tool_call: Dict[str, Any], expected_name: str, tool_info: Dict[str, Any]) -> bool:
        """Validate if the tool call is correctly formatted."""
        
        # Check tool name
        if tool_call.get("name") != expected_name:
            return False
        
        # Check if required parameters are present
        arguments = tool_call.get("arguments", {})
        for param in tool_info["parameters"]:
            if param["required"] and param["name"] not in arguments:
                logger.warning(f"Missing required parameter: {param['name']} for {expected_name}")
                return False
        
        return True
    
    async def _call_llm(self, prompt: str, tool_specs: List[Dict[str, Any]]) -> str:
        """Call the Hermes-405B API."""
        import aiohttp
        
        # Build system prompt with tools
        tools_text = "Available tools:\n" + "\n".join([f"- {spec}" for spec in tool_specs])
        
        system_prompt = f"""You are a Factorio automation assistant. You must respond with ONLY a tool call in JSON format.

{tools_text}

Response format must be exactly:
{{"name": "tool_name", "arguments": {{...}}, "expected_outcome": "..."}}"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": "Hermes-4-405B",
                    "messages": messages,
                    "max_tokens": 200,
                    "temperature": 0.1
                }
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["choices"][0]["message"]["content"]
                else:
                    raise Exception(f"API error: {response.status}")


async def main():
    """Run the tool calling tests."""
    
    # Get API key
    api_key = os.environ.get("HERMES_API_KEY", "sk-CRs4gcGL5Jai3ojQ2BKxxA")
    
    print("=" * 80)
    print("FLE Tool Calling Test for Hermes-405B")
    print("=" * 80)
    print()
    
    tester = LLMToolTester(api_key)
    
    # Test 1: Single tools (each tool in isolation)
    print("Test 1: Testing individual tools...")
    print("-" * 40)
    
    individual_results = {}
    test_tools = list(tester.tools.keys())[:5]  # Test first 5 tools
    
    for tool_name in test_tools:
        result = await tester.test_single_tool(tool_name, tester.tools[tool_name])
        individual_results[tool_name] = result
        status = "✓" if result.success else "✗"
        print(f"{status} {tool_name}: {result.success} ({result.response_time:.2f}s)")
        if not result.success:
            print(f"  Error: {result.error}")
    
    print()
    
    # Test 2: Test tool selection with different numbers of available tools
    print("Test 2: Testing tool selection with different numbers of available tools...")
    print("-" * 40)
    
    # Always test the same 5 tools, but with different numbers of tools available
    test_tools = ["get_entities", "place_entity", "nearest", "craft_item", "inspect_inventory"]
    batch_sizes = [5, 10, 15, 20, 28]  # Different numbers of available tools
    
    for batch_size in batch_sizes:
        available_tools = list(tester.tools.keys())[:batch_size]
        print(f"\nTesting with {batch_size} tools available (testing {len(test_tools)} tools):")
        
        # Test the same tools but with different numbers of tools available
        results = await tester.test_tool_batch(test_tools, available_tools)
        
        success_count = sum(1 for r in results.values() if r.success)
        print(f"  Success rate: {success_count}/{len(results)} ({100*success_count/len(results):.1f}%)")
        
        for tool_name, result in results.items():
            status = "✓" if result.success else "✗"
            print(f"  {status} {tool_name}")
            if not result.success and result.error:
                print(f"    Error: {result.error}")
    
    print()
    
    # Test 3: Stress test - test many tools with all available
    print("Test 3: Testing many tools with all 28 tools available...")
    print("-" * 40)
    
    all_tools = list(tester.tools.keys())
    # Test more tools to see if it can handle selection from all 28
    test_sample = all_tools[:10]  # Test 10 tools with all 28 available
    
    results = await tester.test_tool_batch(test_sample, all_tools)  # Pass all_tools explicitly
    success_count = sum(1 for r in results.values() if r.success)
    
    print(f"Success rate with all 28 tools available: {success_count}/{len(test_sample)} ({100*success_count/len(test_sample):.1f}%)")
    
    for tool_name, result in results.items():
        status = "✓" if result.success else "✗"
        if result.success:
            print(f"  {status} {tool_name}: Correct")
        else:
            print(f"  {status} {tool_name}: {result.error}")
    
    print()
    print("=" * 80)
    print("Test Complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())