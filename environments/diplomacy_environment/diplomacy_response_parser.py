"""
Parser for Diplomacy agent responses.

Extracts tool calls, predictions, and AI_Diplomacy compatible JSON from agent responses.
"""

import json
import logging
import re
from typing import Dict, Any, Optional, Tuple

from atroposlib.utils.tool_call_parser import parse_tool_call

logger = logging.getLogger(__name__)


def parse_diplomacy_response(response: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Parse a diplomacy agent response and extract the AI_Diplomacy compatible part
    and predictions.
    
    Args:
        response: The full response from the agent including think/memory/tool_call blocks
        
    Returns:
        Tuple of (ai_diplomacy_json_str, predictions_dict)
        - ai_diplomacy_json_str: JSON string compatible with AI_Diplomacy (orders or messages)
        - predictions_dict: Expected outcomes for VR-CLI scoring (or None if not found)
    """
    try:
        # Try to parse tool_call
        tool_name, arguments, is_error = parse_tool_call(response)
        
        if is_error or tool_name != "diplomacy_action" or not arguments:
            # Fallback: try to extract JSON from the response
            logger.warning("Failed to parse tool_call, attempting fallback JSON extraction")
            return _fallback_json_extraction(response), None
            
        # Extract predictions
        predictions = arguments.get("expected_outcomes", {})
        
        # Build AI_Diplomacy compatible response
        if "orders" in arguments:
            # Order phase - AI_Diplomacy expects {"orders": [...]}
            ai_diplomacy_response = json.dumps({"orders": arguments["orders"]})
        elif "messages" in arguments:
            # Negotiation phase - AI_Diplomacy expects specific message format
            formatted_messages = []
            for msg in arguments["messages"]:
                formatted_msg = {
                    "message_type": msg.get("message_type", "private"),
                    "content": msg.get("content", "")
                }
                if msg.get("message_type") == "private" and "recipient" in msg:
                    formatted_msg["recipient"] = msg["recipient"]
                formatted_messages.append(formatted_msg)
            
            ai_diplomacy_response = json.dumps({"messages": formatted_messages})
        else:
            # Neither orders nor messages found
            logger.warning("No orders or messages found in tool_call arguments")
            return _fallback_json_extraction(response), predictions
            
        return ai_diplomacy_response, predictions
        
    except Exception as e:
        logger.error(f"Error parsing diplomacy response: {e}")
        return _fallback_json_extraction(response), None


def _fallback_json_extraction(response: str) -> str:
    """
    Fallback method to extract JSON from response when tool_call parsing fails.
    """
    # Try to find JSON blocks in the response
    json_patterns = [
        r'\{\{(.*?)\}\}',  # Double braces (AI_Diplomacy style)
        r'(?:```json\s*)(.*?)(?:```)',  # Markdown code blocks
        r'(\{[^{}]*\})',  # Simple JSON objects
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            for match in matches:
                try:
                    # Try to parse as JSON
                    if pattern == r'\{\{(.*?)\}\}':
                        # Add back the outer braces for double-brace format
                        json_str = '{' + match + '}'
                    else:
                        json_str = match
                        
                    parsed = json.loads(json_str)
                    
                    # Check if it has orders or messages
                    if isinstance(parsed, dict) and ("orders" in parsed or "messages" in parsed):
                        return json_str
                except json.JSONDecodeError:
                    continue
    
    # Last resort: return empty orders/messages
    if "order" in response.lower():
        return json.dumps({"orders": []})
    else:
        return json.dumps({"messages": []})


def extract_memory_content(response: str) -> Optional[str]:
    """
    Extract memory content from response.
    """
    memory_pattern = r'<memory>\s*(.*?)\s*</memory>'
    match = re.search(memory_pattern, response, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None