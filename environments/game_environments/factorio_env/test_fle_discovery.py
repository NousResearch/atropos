#!/usr/bin/env python3
"""
Test script for FLE Tool Discovery Utility

This script tests the fle_tool_discovery module by discovering all available
FLE tools and displaying their signatures in various formats for debugging.
"""

import json
import sys
from pathlib import Path
from pprint import pprint
import logging

# Add current directory to path to import the utility
sys.path.insert(0, str(Path(__file__).parent))

from fle_tool_discovery import (
    discover_fle_tools,
    format_tool_for_prompt,
    get_tool_signatures_for_prompt
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def test_discovery():
    """Test the tool discovery functionality."""
    print("=" * 80)
    print("FLE Tool Discovery Test")
    print("=" * 80)
    print()
    
    # Discover tools
    print("1. Discovering FLE tools...")
    tools = discover_fle_tools()
    
    if not tools:
        print("ERROR: No tools discovered! Check the FLE path.")
        return
    
    print(f"✓ Discovered {len(tools)} tools\n")
    
    # List tool names
    print("2. Tool Names:")
    print("-" * 40)
    for i, name in enumerate(sorted(tools.keys()), 1):
        print(f"  {i:2d}. {name}")
    print()
    
    # Show detailed info for a few tools
    print("3. Detailed Tool Information (first 3 tools):")
    print("-" * 40)
    
    for tool_name in sorted(tools.keys())[:3]:
        tool_info = tools[tool_name]
        print(f"\n{tool_name}:")
        print(f"  Class: {tool_info['class_name']}")
        print(f"  Description: {tool_info['description']}")
        print(f"  Returns: {tool_info['returns']}")
        print(f"  Parameters:")
        
        if not tool_info['parameters']:
            print("    (no parameters)")
        else:
            for param in tool_info['parameters']:
                req = "required" if param['required'] else "optional"
                default = f" = {param['default']}" if param['default'] else ""
                print(f"    - {param['name']}: {param['type']}{default} ({req})")
    
    print()
    
    # Test specific important tools
    print("4. Testing Important Tools:")
    print("-" * 40)
    
    important_tools = ['get_entities', 'place_entity', 'nearest', 'craft_item', 'inspect_inventory']
    
    for tool_name in important_tools:
        if tool_name in tools:
            print(f"\n✓ {tool_name} found")
            tool = tools[tool_name]
            
            # Show parameters in a compact format
            params = []
            for p in tool['parameters']:
                param_str = f"{p['name']}: {p['type']}"
                if p['default']:
                    param_str += f" = {p['default']}"
                params.append(param_str)
            
            print(f"  Signature: {tool_name}({', '.join(params)})")
        else:
            print(f"\n✗ {tool_name} NOT FOUND")
    
    print()
    
    # Test prompt formatting
    print("5. LLM Prompt Format (first 3 tools):")
    print("-" * 40)
    
    for tool_name in sorted(tools.keys())[:3]:
        formatted = format_tool_for_prompt(tools[tool_name])
        print(f"\n{formatted}")
    
    print()
    
    # Export to JSON for inspection
    print("6. Exporting to JSON...")
    print("-" * 40)
    
    output_file = Path(__file__).parent / "fle_tools_discovered.json"
    
    # Clean up for JSON serialization
    json_safe = {}
    for name, info in tools.items():
        json_safe[name] = {
            "class_name": info["class_name"],
            "category": info["category"],
            "description": info["description"],
            "returns": info["returns"],
            "parameters": info["parameters"]
        }
    
    with open(output_file, 'w') as f:
        json.dump(json_safe, f, indent=2)
    
    print(f"✓ Exported to {output_file}")
    print()
    
    # Show statistics
    print("7. Statistics:")
    print("-" * 40)
    
    total_params = sum(len(tool['parameters']) for tool in tools.values())
    tools_with_no_params = sum(1 for tool in tools.values() if not tool['parameters'])
    required_params = sum(
        sum(1 for p in tool['parameters'] if p['required'])
        for tool in tools.values()
    )
    
    print(f"  Total tools: {len(tools)}")
    print(f"  Total parameters: {total_params}")
    print(f"  Average parameters per tool: {total_params / len(tools):.1f}")
    print(f"  Tools with no parameters: {tools_with_no_params}")
    print(f"  Required parameters: {required_params}")
    print(f"  Optional parameters: {total_params - required_params}")
    
    print()
    print("=" * 80)
    print("Test completed successfully!")
    print("=" * 80)


def test_specific_tool(tool_name: str):
    """Test a specific tool in detail."""
    print(f"\nDetailed test for tool: {tool_name}")
    print("-" * 40)
    
    tools = discover_fle_tools()
    
    if tool_name not in tools:
        print(f"Tool '{tool_name}' not found!")
        print("Available tools:", ", ".join(sorted(tools.keys())))
        return
    
    tool = tools[tool_name]
    
    print("Full information:")
    pprint(tool)
    
    print("\nFormatted for prompt:")
    print(format_tool_for_prompt(tool))
    
    print("\nExample usage:")
    params = []
    for p in tool['parameters']:
        if p['required']:
            # Show example value based on type
            if 'Position' in p['type']:
                params.append(f"{p['name']}=Position(x=0, y=0)")
            elif 'Prototype' in p['type']:
                params.append(f"{p['name']}=Prototype.ExampleEntity")
            elif 'Resource' in p['type']:
                params.append(f"{p['name']}=Resource.IronOre")
            elif p['type'] == 'int':
                params.append(f"{p['name']}=1")
            elif p['type'] == 'float':
                params.append(f"{p['name']}=1.0")
            else:
                params.append(f"{p['name']}=<{p['type']}>")
    
    print(f"  {tool_name}({', '.join(params)})")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Test specific tool if provided as argument
        test_specific_tool(sys.argv[1])
    else:
        # Run full test suite
        test_discovery()