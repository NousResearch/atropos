#!/usr/bin/env python3
"""
Fix multi-line JSON messages in Diplomacy dataset - Version 2.
Better handling of newlines within content fields.
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple


def fix_multiline_json_v2(text: str) -> str:
    """
    Fix multi-line JSON by properly handling newlines in content fields.
    """
    # First check if it's already valid JSON
    try:
        obj = json.loads(text)
        return text  # Already valid
    except json.JSONDecodeError:
        pass
    
    # Check if it looks like JSON
    if not text.strip().startswith('{'):
        return text
    
    # Strategy: Find the content field and escape newlines within it
    # Look for pattern: "content": "..." where ... may contain newlines
    
    # First, let's try to identify the structure
    lines = text.split('\n')
    
    # Try to rebuild the JSON
    result = []
    in_content_field = False
    content_buffer = []
    brace_count = 0
    
    for line in lines:
        # Track brace depth
        brace_count += line.count('{') - line.count('}')
        
        # Check if we're starting a content field
        if '"content":' in line and not in_content_field:
            # Check if the content starts and ends on same line
            if line.strip().endswith('}') or line.strip().endswith('},'):
                # Single line content, just add it
                result.append(line)
            else:
                # Multi-line content starts here
                in_content_field = True
                content_buffer = [line]
        elif in_content_field:
            # We're inside a multi-line content field
            content_buffer.append(line)
            
            # Check if this line might end the content field
            # It should end with a quote followed by optional comma and/or closing brace
            if (line.strip().endswith('"') or 
                line.strip().endswith('",') or 
                line.strip().endswith('"}') or
                line.strip().endswith('"}')):
                # This might be the end of content field
                # Join the buffer and try to parse the whole thing so far
                temp_json = '\n'.join(result) + '\n' + '\n'.join(content_buffer)
                
                # Count quotes to see if we have a complete string
                content_str = '\n'.join(content_buffer)
                # Remove the "content": part to count quotes in the value
                content_value_part = content_str.split('"content":', 1)[1] if '"content":' in content_str else content_str
                
                # Count unescaped quotes
                unescaped_quotes = 0
                prev_char = ''
                for char in content_value_part:
                    if char == '"' and prev_char != '\\':
                        unescaped_quotes += 1
                    prev_char = char
                
                # If we have an even number of quotes, the string is complete
                if unescaped_quotes >= 2 and unescaped_quotes % 2 == 0:
                    # Join content buffer, escaping newlines
                    content_joined = ' '.join(content_buffer)
                    result.append(content_joined)
                    in_content_field = False
                    content_buffer = []
        else:
            # Normal line, just add it
            result.append(line)
    
    # If we still have content in buffer, add it
    if content_buffer:
        result.append(' '.join(content_buffer))
    
    # Join and try to parse
    fixed = '\n'.join(result)
    
    # Try more aggressive fixing - replace newlines in string values
    # This is a hacky but might work approach
    if fixed.strip().startswith('{'):
        # Replace newlines that appear to be inside strings
        # Look for patterns like: "...text\n...text..."
        
        # Split by quotes and process alternating segments
        parts = fixed.split('"')
        fixed_parts = []
        
        for i, part in enumerate(parts):
            if i % 2 == 1:  # This is inside quotes (odd indices)
                # Replace newlines with escaped newlines or spaces
                part = part.replace('\n', '\\n')
            fixed_parts.append(part)
        
        fixed = '"'.join(fixed_parts)
    
    # Try to parse the fixed version
    try:
        obj = json.loads(fixed)
        # Success! Re-serialize for consistency
        return json.dumps(obj, ensure_ascii=False)
    except json.JSONDecodeError:
        pass
    
    # If still failing, try ultra-aggressive approach
    # Remove ALL newlines except those at the very beginning/end
    if text.strip().startswith('{') and text.strip().endswith('}'):
        compact = text.replace('\n', ' ').replace('  ', ' ')
        try:
            obj = json.loads(compact)
            return json.dumps(obj, ensure_ascii=False)
        except:
            pass
    
    # Give up
    return text


def process_diplomacy_file_v2(input_path: Path, output_path: Path) -> Dict[str, int]:
    """Process the diplomacy JSONL file with better JSON fixing."""
    stats = {
        'total_samples': 0,
        'samples_processed': 0,
        'messages_fixed': 0,
        'messages_already_valid': 0,
        'messages_unfixable': 0
    }
    
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line_num, line in enumerate(infile, 1):
            stats['total_samples'] += 1
            
            try:
                data = json.loads(line)
                fixed_any = False
                
                # Process each message
                for msg_idx, msg in enumerate(data['messages']):
                    if msg['role'] == 'assistant':
                        original_content = msg['content']
                        
                        # Check if it's already valid JSON
                        try:
                            if original_content.strip().startswith('{'):
                                parsed = json.loads(original_content)
                                stats['messages_already_valid'] += 1
                                # Re-serialize for consistency
                                msg['content'] = json.dumps(parsed, ensure_ascii=False)
                        except json.JSONDecodeError:
                            # Try to fix it
                            fixed_content = fix_multiline_json_v2(original_content)
                            
                            # Check if fix worked
                            try:
                                if fixed_content.strip().startswith('{'):
                                    parsed = json.loads(fixed_content)
                                    msg['content'] = fixed_content
                                    stats['messages_fixed'] += 1
                                    fixed_any = True
                                    
                                    if line_num <= 3:
                                        print(f"Sample {line_num}: Successfully fixed message")
                                        print(f"  Type: {parsed.get('message_type', parsed.get('reasoning', 'unknown')[:20])}")
                                else:
                                    stats['messages_unfixable'] += 1
                            except:
                                stats['messages_unfixable'] += 1
                                if line_num <= 200 and '"message_type"' in original_content:
                                    print(f"Sample {line_num}: Still could not fix message")
                
                if fixed_any:
                    stats['samples_processed'] += 1
                
                # Write the processed line
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                # Write original line if processing fails
                outfile.write(line)
    
    return stats


def verify_output(output_path: Path):
    """Verify the fixed output."""
    stats = {'valid_json': 0, 'invalid_json': 0, 'non_json': 0}
    message_types = {}
    
    with open(output_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line_num > 500:  # Check first 500
                break
            try:
                data = json.loads(line)
                for msg in data['messages']:
                    if msg['role'] == 'assistant':
                        content = msg['content']
                        if content.strip().startswith('{'):
                            try:
                                obj = json.loads(content)
                                stats['valid_json'] += 1
                                
                                # Track message types
                                if 'message_type' in obj:
                                    msg_type = f"message_{obj['message_type']}"
                                elif 'reasoning' in obj:
                                    msg_type = 'analysis'
                                elif 'orders' in obj:
                                    msg_type = 'orders'
                                elif 'order_summary' in obj:
                                    msg_type = 'order_summary'
                                elif 'negotiation_summary' in obj:
                                    msg_type = 'negotiation_summary'
                                else:
                                    msg_type = 'other'
                                
                                message_types[msg_type] = message_types.get(msg_type, 0) + 1
                                
                            except:
                                stats['invalid_json'] += 1
                        else:
                            stats['non_json'] += 1
            except Exception as e:
                print(f"Verification error at line {line_num}: {e}")
    
    return stats, message_types


def main():
    """Main function."""
    input_file = Path('/home/maxpaperclips/atropos/data/diplomacy_italy_winning.jsonl')
    output_file = Path('/home/maxpaperclips/atropos/data/diplomacy_italy_winning_fixed.jsonl')
    
    print(f"Processing {input_file} with improved JSON fixing...")
    print(f"Output will be saved to {output_file}")
    
    stats = process_diplomacy_file_v2(input_file, output_file)
    
    print("\nProcessing complete!")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Samples processed: {stats['samples_processed']}")
    print(f"Messages already valid: {stats['messages_already_valid']}")
    print(f"Messages fixed: {stats['messages_fixed']}")
    print(f"Messages unfixable: {stats['messages_unfixable']}")
    
    # Verify
    print("\nVerifying output...")
    verify_stats, msg_types = verify_output(output_file)
    
    print(f"\nVerification results (first 500 samples):")
    print(f"Valid JSON messages: {verify_stats['valid_json']}")
    print(f"Invalid JSON messages: {verify_stats['invalid_json']}")
    print(f"Non-JSON messages: {verify_stats['non_json']}")
    
    print(f"\nMessage types found:")
    for msg_type, count in sorted(msg_types.items(), key=lambda x: -x[1]):
        print(f"  {msg_type}: {count}")


if __name__ == '__main__':
    main()