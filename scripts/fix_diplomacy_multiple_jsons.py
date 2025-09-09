#!/usr/bin/env python3
"""
Fix Diplomacy dataset by splitting concatenated JSON objects into separate tool calls.

When a single assistant message contains multiple JSON objects (e.g., multiple messages
to different recipients), split them into separate tool_call XML blocks.
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional


def extract_json_objects(text: str) -> List[Dict[str, Any]]:
    """
    Extract multiple JSON objects from a text that may contain them concatenated.
    
    Returns a list of parsed JSON objects.
    """
    objects = []
    
    # Try to parse as single JSON first
    try:
        obj = json.loads(text)
        return [obj]
    except json.JSONDecodeError:
        pass
    
    # Look for pattern of multiple JSON objects
    # Split by '}\n{' pattern which indicates concatenated JSONs
    potential_jsons = []
    
    # Method 1: Split by newline between objects
    parts = text.split('}\n{')
    if len(parts) > 1:
        for i, part in enumerate(parts):
            # Add back the braces we split on
            if i > 0:
                part = '{' + part
            if i < len(parts) - 1:
                part = part + '}'
            potential_jsons.append(part)
    else:
        # Method 2: Use regex to find JSON objects
        # Look for complete JSON objects
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.finditer(json_pattern, text, re.DOTALL)
        potential_jsons = [match.group() for match in matches]
    
    # Try to parse each potential JSON
    for potential in potential_jsons:
        try:
            obj = json.loads(potential)
            objects.append(obj)
        except json.JSONDecodeError:
            # Try fixing common issues
            fixed = potential.replace('\n', '\\n')
            try:
                obj = json.loads(fixed)
                objects.append(obj)
            except:
                pass
    
    return objects


def format_as_tool_calls(objects: List[Dict[str, Any]]) -> str:
    """
    Format a list of JSON objects as tool calls.
    
    Each object becomes a separate <tool_call> block.
    """
    if not objects:
        return ""
    
    tool_calls = []
    
    for obj in objects:
        # Determine the tool name based on content
        if 'message_type' in obj:
            tool_name = 'send_message'
        elif 'reasoning' in obj and 'relationships' in obj:
            tool_name = 'analyze_phase'
        elif 'orders' in obj:
            tool_name = 'submit_orders'
        elif 'order_summary' in obj or 'negotiation_summary' in obj:
            tool_name = 'summarize_turn'
        else:
            tool_name = 'unknown_tool'
        
        # Create tool call
        tool_call = {
            "name": tool_name,
            "arguments": obj
        }
        
        # Format as XML
        tool_call_json = json.dumps(tool_call, ensure_ascii=False, indent=2)
        tool_calls.append(f"<tool_call>\n{tool_call_json}\n</tool_call>")
    
    return '\n'.join(tool_calls)


def process_diplomacy_file(input_path: Path, output_path: Path) -> Dict[str, int]:
    """
    Process the diplomacy file, converting concatenated JSONs to tool calls.
    """
    stats = {
        'total_samples': 0,
        'samples_with_single_json': 0,
        'samples_with_multiple_jsons': 0,
        'total_tool_calls_created': 0,
        'samples_with_errors': 0
    }
    
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line_num, line in enumerate(infile, 1):
            stats['total_samples'] += 1
            
            try:
                data = json.loads(line)
                
                # Process each message
                for msg_idx, msg in enumerate(data['messages']):
                    if msg['role'] == 'assistant':
                        content = msg['content']
                        
                        # Extract JSON objects
                        json_objects = extract_json_objects(content)
                        
                        if len(json_objects) == 0:
                            # No valid JSON found, keep original
                            pass
                        elif len(json_objects) == 1:
                            # Single JSON object - format as single tool call
                            stats['samples_with_single_json'] += 1
                            stats['total_tool_calls_created'] += 1
                            msg['content'] = format_as_tool_calls(json_objects)
                        else:
                            # Multiple JSON objects - format as multiple tool calls
                            stats['samples_with_multiple_jsons'] += 1
                            stats['total_tool_calls_created'] += len(json_objects)
                            msg['content'] = format_as_tool_calls(json_objects)
                            
                            if line_num <= 10:  # Show first few for verification
                                print(f"\nSample {line_num}: Found {len(json_objects)} JSON objects")
                                for obj in json_objects:
                                    obj_type = (
                                        obj.get('message_type', '') or
                                        ('analysis' if 'reasoning' in obj else '') or
                                        ('orders' if 'orders' in obj else '') or
                                        'unknown'
                                    )
                                    print(f"  - Type: {obj_type}")
                
                # Write the processed line
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                
            except Exception as e:
                stats['samples_with_errors'] += 1
                print(f"Error processing line {line_num}: {e}")
                # Write original line if processing fails
                outfile.write(line)
    
    return stats


def verify_output(output_path: Path, limit: int = 100):
    """Verify the output file has proper tool calls."""
    stats = {
        'samples_checked': 0,
        'samples_with_tool_calls': 0,
        'total_tool_calls': 0,
        'tool_types': {}
    }
    
    with open(output_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line_num > limit:
                break
            
            stats['samples_checked'] += 1
            
            try:
                data = json.loads(line)
                
                for msg in data['messages']:
                    if msg['role'] == 'assistant':
                        content = msg['content']
                        
                        # Count tool calls
                        tool_call_count = content.count('<tool_call>')
                        if tool_call_count > 0:
                            stats['samples_with_tool_calls'] += 1
                            stats['total_tool_calls'] += tool_call_count
                            
                            # Extract tool names
                            tool_calls = re.findall(r'<tool_call>.*?"name":\s*"([^"]+)".*?</tool_call>', 
                                                   content, re.DOTALL)
                            for tool_name in tool_calls:
                                stats['tool_types'][tool_name] = stats['tool_types'].get(tool_name, 0) + 1
                            
            except Exception as e:
                print(f"Verification error at line {line_num}: {e}")
    
    return stats


def main():
    """Main function."""
    input_file = Path('/home/maxpaperclips/atropos/data/diplomacy_italy_winning.jsonl')
    output_file = Path('/home/maxpaperclips/atropos/data/diplomacy_with_tool_calls.jsonl')
    
    print(f"Processing {input_file}...")
    print(f"Converting JSON objects to tool calls...")
    print(f"Output: {output_file}\n")
    
    # Process the file
    stats = process_diplomacy_file(input_file, output_file)
    
    print(f"\nProcessing complete!")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Samples with single JSON: {stats['samples_with_single_json']}")
    print(f"Samples with multiple JSONs: {stats['samples_with_multiple_jsons']}")
    print(f"Total tool calls created: {stats['total_tool_calls_created']}")
    print(f"Samples with errors: {stats['samples_with_errors']}")
    
    # Verify output
    print(f"\nVerifying output (first 100 samples)...")
    verify_stats = verify_output(output_file, limit=100)
    
    print(f"\nVerification results:")
    print(f"Samples checked: {verify_stats['samples_checked']}")
    print(f"Samples with tool calls: {verify_stats['samples_with_tool_calls']}")
    print(f"Total tool calls: {verify_stats['total_tool_calls']}")
    
    print(f"\nTool types found:")
    for tool_name, count in sorted(verify_stats['tool_types'].items(), 
                                  key=lambda x: -x[1]):
        print(f"  {tool_name}: {count}")


if __name__ == '__main__':
    main()