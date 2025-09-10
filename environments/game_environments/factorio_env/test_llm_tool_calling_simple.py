#!/usr/bin/env python3
"""
Simplified test to see if LLM can pick the right tool from all 28 FLE tools.
"""

import json
import os
import sys
import re
import asyncio
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent))
from fle_tool_discovery import discover_fle_tools, format_tool_for_prompt

async def test_tool_selection():
    """Test if LLM can pick the right tool from all 28 available."""
    
    api_key = os.environ.get("HERMES_API_KEY", "sk-CRs4gcGL5Jai3ojQ2BKxxA")
    base_url = "https://inference-api.nousresearch.com/v1"
    
    # Discover all tools
    tools = discover_fle_tools()
    all_tool_specs = [format_tool_for_prompt(tools[name]) for name in sorted(tools.keys())]
    
    print(f"Testing with all {len(tools)} tools available")
    print("=" * 60)
    
    # Test scenarios - one for each tool type
    test_cases = [
        ("get_entities", "I need to see what entities are within 50 tiles of my position."),
        ("place_entity", "Place an iron chest at position x=10, y=20."),
        ("nearest", "Find the nearest iron ore deposit."),
        ("craft_item", "Craft 5 transport belts."),
        ("inspect_inventory", "Check what items I have in my inventory."),
        ("move_to", "Move to position x=15, y=25."),
        ("connect_entities", "Connect the miner to the furnace with transport belts."),
        ("harvest_resource", "Mine the iron ore at my current position."),
        ("insert_item", "Put 10 coal into the furnace."),
        ("extract_item", "Take the iron plates out of the furnace."),
    ]
    
    # Build system prompt with ALL tools
    tools_text = "Available tools:\n" + "\n".join([f"- {spec}" for spec in all_tool_specs])
    system_prompt = f"""You are a Factorio automation assistant. You must respond with ONLY a tool call in JSON format.

{tools_text}

Response format must be exactly:
{{"name": "tool_name", "arguments": {{...}}, "expected_outcome": "..."}}"""
    
    import aiohttp
    
    results = []
    async with aiohttp.ClientSession() as session:
        for expected_tool, scenario in test_cases:
            prompt = f"Task: {scenario}\n\nGenerate ONLY the tool call JSON, no explanation."
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            start = time.time()
            try:
                async with session.post(
                    f"{base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": "Hermes-4-405B",
                        "messages": messages,
                        "max_tokens": 200,
                        "temperature": 0.1
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        llm_response = data["choices"][0]["message"]["content"]
                        
                        # Extract tool name from response
                        tool_match = re.search(r'"name"\s*:\s*"([^"]+)"', llm_response)
                        if tool_match:
                            called_tool = tool_match.group(1)
                            success = called_tool == expected_tool
                            status = "✓" if success else "✗"
                            
                            if success:
                                print(f"{status} {expected_tool}: Correct (took {time.time()-start:.1f}s)")
                            else:
                                print(f"{status} {expected_tool}: Called '{called_tool}' instead (took {time.time()-start:.1f}s)")
                            
                            results.append(success)
                        else:
                            print(f"✗ {expected_tool}: No tool call found in response")
                            results.append(False)
                    else:
                        print(f"✗ {expected_tool}: API error {response.status}")
                        results.append(False)
                        
            except asyncio.TimeoutError:
                print(f"✗ {expected_tool}: Timeout")
                results.append(False)
            except Exception as e:
                print(f"✗ {expected_tool}: Error - {e}")
                results.append(False)
    
    print("=" * 60)
    success_count = sum(results)
    print(f"Overall: {success_count}/{len(results)} correct ({100*success_count/len(results):.1f}%)")
    print(f"The LLM had to choose from {len(tools)} available tools each time")

if __name__ == "__main__":
    asyncio.run(test_tool_selection())