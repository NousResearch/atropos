#!/usr/bin/env python3
"""
Fix multi-line JSON messages in Diplomacy dataset.

Some assistant responses are JSON that got formatted across multiple lines,
making them fail to parse. This script identifies and fixes these cases.
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple


def is_likely_json_start(text: str) -> bool:
    """Check if text looks like the start of a JSON object."""
    text = text.strip()
    return text.startswith('{') and (
        '"message_type"' in text or
        '"reasoning"' in text or
        '"orders"' in text or
        '"order_summary"' in text or
        '"negotiation_summary"' in text
    )


def try_parse_json(text: str) -> Tuple[bool, Any]:
    """Try to parse text as JSON, return success flag and parsed object."""
    try:
        obj = json.loads(text)
        return True, obj
    except json.JSONDecodeError:
        return False, None


def fix_multiline_json(text: str) -> str:
    """
    Attempt to fix multi-line formatted JSON.
    
    Strategy:
    1. If it starts with { and contains expected keys, it's likely JSON
    2. Try to parse as-is first
    3. If that fails, try various fixes
    """
    # First, try parsing as-is
    success, obj = try_parse_json(text)
    if success:
        return text  # Already valid JSON
    
    # Check if it looks like JSON
    if not is_likely_json_start(text):
        return text  # Not JSON, return as-is
    
    # Strategy 1: Remove newlines within the JSON structure
    # But preserve newlines within string values
    fixed = text
    
    # Try to find string boundaries and preserve newlines only within them
    # This is tricky - we'll use a simple approach
    
    # First attempt: Just try to compact it naively
    # Remove newlines that are not within quotes
    lines = text.split('\n')
    
    # Rebuild the JSON, being careful with strings
    result_lines = []
    in_string = False
    escape_next = False
    
    for line in lines:
        processed_line = []
        for i, char in enumerate(line):
            if escape_next:
                escape_next = False
                processed_line.append(char)
                continue
                
            if char == '\\':
                escape_next = True
                processed_line.append(char)
                continue
                
            if char == '"' and not escape_next:
                in_string = not in_string
                processed_line.append(char)
            else:
                processed_line.append(char)
        
        result_lines.append(''.join(processed_line))
    
    # Try to parse the compacted version
    compacted = ' '.join(result_lines)
    success, obj = try_parse_json(compacted)
    if success:
        # Re-serialize to ensure consistent formatting
        return json.dumps(obj, ensure_ascii=False)
    
    # Strategy 2: More aggressive - look for the JSON structure
    # and try to extract it properly
    if text.startswith('{') and text.rstrip().endswith('}'):
        # Count braces to ensure we have a complete object
        open_braces = text.count('{')
        close_braces = text.count('}')
        
        if open_braces == close_braces:
            # Try removing all newlines and multiple spaces
            condensed = re.sub(r'\s+', ' ', text)
            success, obj = try_parse_json(condensed)
            if success:
                return json.dumps(obj, ensure_ascii=False)
    
    # If all else fails, return original
    print(f"WARNING: Could not fix JSON: {text[:100]}...")
    return text


def process_diplomacy_file(input_path: Path, output_path: Path) -> Dict[str, int]:
    """Process the diplomacy JSONL file and fix multi-line JSON issues."""
    stats = {
        'total_samples': 0,
        'samples_with_issues': 0,
        'messages_fixed': 0,
        'messages_already_valid': 0,
        'messages_unfixable': 0
    }
    
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line_num, line in enumerate(infile, 1):
            stats['total_samples'] += 1
            
            try:
                data = json.loads(line)
                sample_had_issues = False
                
                # Process each message
                for msg in data['messages']:
                    if msg['role'] == 'assistant':
                        original_content = msg['content']
                        
                        # Check if it's already valid JSON
                        success, _ = try_parse_json(original_content)
                        if success:
                            stats['messages_already_valid'] += 1
                        else:
                            # Try to fix it
                            fixed_content = fix_multiline_json(original_content)
                            
                            # Check if fix worked
                            success, _ = try_parse_json(fixed_content)
                            if success and fixed_content != original_content:
                                msg['content'] = fixed_content
                                stats['messages_fixed'] += 1
                                sample_had_issues = True
                                
                                # Log successful fix
                                if line_num <= 5:  # Log first few fixes for verification
                                    print(f"Sample {line_num}: Fixed multi-line JSON")
                                    print(f"  Original: {original_content[:100]}...")
                                    print(f"  Fixed: {fixed_content[:100]}...")
                            elif not success:
                                stats['messages_unfixable'] += 1
                                print(f"Sample {line_num}: Could not fix message")
                
                if sample_had_issues:
                    stats['samples_with_issues'] += 1
                
                # Write the processed line
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                # Write original line if processing fails
                outfile.write(line)
    
    return stats


def main():
    """Main function to fix the diplomacy dataset."""
    input_file = Path('/home/maxpaperclips/atropos/data/diplomacy_italy_winning.jsonl')
    output_file = Path('/home/maxpaperclips/atropos/data/diplomacy_italy_winning_fixed.jsonl')
    
    print(f"Processing {input_file}...")
    print(f"Output will be saved to {output_file}")
    
    stats = process_diplomacy_file(input_file, output_file)
    
    print("\nProcessing complete!")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Samples with issues: {stats['samples_with_issues']}")
    print(f"Messages already valid: {stats['messages_already_valid']}")
    print(f"Messages fixed: {stats['messages_fixed']}")
    print(f"Messages unfixable: {stats['messages_unfixable']}")
    
    # Verify the output
    print("\nVerifying output file...")
    verify_stats = {'valid': 0, 'invalid': 0}
    
    with open(output_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                for msg in data['messages']:
                    if msg['role'] == 'assistant':
                        content = msg['content']
                        if content.startswith('{'):
                            success, _ = try_parse_json(content)
                            if success:
                                verify_stats['valid'] += 1
                            else:
                                verify_stats['invalid'] += 1
            except Exception as e:
                print(f"Verification error at line {line_num}: {e}")
    
    print(f"\nVerification results:")
    print(f"Valid JSON messages: {verify_stats['valid']}")
    print(f"Invalid JSON messages: {verify_stats['invalid']}")


if __name__ == '__main__':
    main()