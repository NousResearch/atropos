"""
Tool definitions for Diplomacy agents.

This module defines the structured tool format for diplomacy actions,
including orders, negotiations, and outcome predictions.
"""

DIPLOMACY_TOOLS = [
    {
        "name": "diplomacy_action",
        "description": "Execute diplomacy actions (orders or negotiations) with outcome predictions",
        "parameters": {
            "type": "object",
            "properties": {
                "orders": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of orders for units (e.g., 'A PAR-BUR', 'F BRE-MAO'). Used in order phases.",
                },
                "messages": {
                    "type": "array", 
                    "items": {
                        "type": "object",
                        "properties": {
                            "message_type": {
                                "type": "string",
                                "enum": ["private", "broadcast"],
                                "description": "Type of message",
                            },
                            "recipient": {
                                "type": "string",
                                "description": "Recipient power (required for private messages)",
                            },
                            "content": {
                                "type": "string",
                                "description": "The message content",
                            },
                        },
                        "required": ["message_type", "content"],
                    },
                    "description": "List of messages to send. Used in negotiation phases.",
                },
                "expected_outcomes": {
                    "type": "object",
                    "properties": {
                        "negotiation_responses": {
                            "type": "object",
                            "additionalProperties": {"type": "string"},
                            "description": "Expected responses from other powers to your messages",
                        },
                        "board_changes": {
                            "type": "object",
                            "properties": {
                                "territories": {
                                    "type": "object",
                                    "additionalProperties": {"type": "string"},
                                    "description": "Expected territory control changes",
                                },
                                "unit_outcomes": {
                                    "type": "object",
                                    "additionalProperties": {"type": "string"},
                                    "description": "Expected outcomes for each unit's orders",
                                },
                            },
                            "description": "Expected changes to the game board",
                        },
                        "relationship_changes": {
                            "type": "object",
                            "additionalProperties": {"type": "string"},
                            "description": "Expected changes in trust/relationships with other powers",
                        },
                    },
                    "required": [],
                    "description": "Predictions about what will happen as a result of your actions",
                },
            },
            "required": ["expected_outcomes"],
        },
    }
]