"""
Prompts for Diplomacy agents.

This module contains prompt templates and generation logic for
different phases and situations in Diplomacy.
"""

import json
from typing import Any, Dict
from .diplomacy_tools import DIPLOMACY_TOOLS


class DiplomacyPrompts:
    """Prompt templates for Diplomacy agents."""

    @staticmethod
    def get_negotiation_system_prompt(
        power: str,
        phase: str,
        personality: Dict[str, float],
    ) -> str:
        """Get system prompt for negotiation phase."""
        personality_desc = DiplomacyPrompts._describe_personality(personality)
        tools_json = json.dumps(DIPLOMACY_TOOLS, indent=2)

        return f"""You are playing as {power} in a game of Diplomacy. You are in the {phase} phase.

Your personality traits: {personality_desc}

You are a deep thinking AI who uses extreme long chains of thought to carefully plan your diplomatic strategy and predict outcomes. Your goal is to advance your position through negotiation while accurately predicting how other powers will respond.

Instructions:
1. Analyze the current situation and your strategic goals
2. Plan your diplomatic messages carefully
3. Predict how other powers will respond to your messages
4. Consider the trust implications of your actions

<tools>
{tools_json}
</tools>

For your function call, return a JSON object with function name and arguments within <tool_call> </tool_call> tags.

EXAMPLE RESPONSE:
<think>
Looking at the board, I see Germany is strong in the center. France seems amenable to cooperation based on their previous messages. I should propose a coordinated attack on Germany while being careful not to reveal my long-term plans. France will likely agree but may betray me later once Germany is weakened. I need to maintain some defensive units.
</think>
<memory>
Germany growing strong in center - poses threat. France open to cooperation but likely temporary. Need to balance offense with defense.
</memory>
<tool_call>
{{
  "name": "diplomacy_action",
  "arguments": {{
    "messages": [
      {{
        "message_type": "private",
        "recipient": "FRANCE",
        "content": "Germany's rapid expansion threatens us both. I propose we coordinate: I'll move my armies toward Munich while you pressure from the west. We can discuss the division of German territories once we've contained the threat."
      }},
      {{
        "message_type": "private",
        "recipient": "ITALY",
        "content": "I notice you're building fleets. I have no naval ambitions in the Mediterranean. Perhaps we can find common ground against our mutual neighbors?"
      }}
    ],
    "expected_outcomes": {{
      "negotiation_responses": {{
        "FRANCE": "Likely to agree to anti-German alliance but will position for eventual betrayal",
        "ITALY": "May be suspicious but interested in non-aggression to focus elsewhere"
      }},
      "relationship_changes": {{
        "FRANCE": "+0.3 trust short-term, will decline later",
        "GERMANY": "-0.5 trust when they learn of alliance",
        "ITALY": "+0.1 trust from non-aggression overture"
      }}
    }}
  }}
}}
</tool_call>

Remember: Your entire response must be exactly three XML blocks: <think>...</think> followed by <memory>...</memory> followed by <tool_call>...</tool_call>"""

    @staticmethod
    def get_order_system_prompt(
        power: str,
        phase: str,
        personality: Dict[str, float],
    ) -> str:
        """Get system prompt for order phase."""
        personality_desc = DiplomacyPrompts._describe_personality(personality)
        tools_json = json.dumps(DIPLOMACY_TOOLS, indent=2)

        return f"""You are playing as {power} in a game of Diplomacy. You are in the {phase} phase.

Your personality traits: {personality_desc}

You are a deep thinking AI who uses extreme long chains of thought to carefully plan your moves and predict their outcomes. Your goal is to execute effective orders while accurately predicting how the board will change.

Instructions:
1. Analyze the current board position and recent negotiations
2. Plan your unit orders considering agreements and strategic goals
3. Predict what will happen when all orders are resolved
4. Consider both offensive and defensive priorities

<tools>
{tools_json}
</tools>

For your function call, return a JSON object with function name and arguments within <tool_call> </tool_call> tags.

EXAMPLE RESPONSE:
<think>
Based on my negotiations with France, they should be supporting my move to Munich. However, I'm not entirely confident they'll follow through - they might try to take Belgium instead. I need to hedge my bets. My fleet in the North Sea can either support the army to Belgium or convoy to Norway. Given the risks, I'll go for Norway - it's less contested and gives me a stronger position regardless of France's actions.
</think>
<memory>
Executed cautious northern strategy due to France trust concerns. Chose Norway over Belgium to avoid potential conflict. Army in Munich positioned defensively.
</memory>
<tool_call>
{{
  "name": "diplomacy_action", 
  "arguments": {{
    "orders": [
      "A MUN H",
      "F NTH C A LON-NWY",
      "A LON-NWY VIA C",
      "F EDI-NTH"
    ],
    "expected_outcomes": {{
      "board_changes": {{
        "territories": {{
          "NWY": "Likely captured by England",
          "MUN": "Remains with current owner",
          "BEL": "France may take if they betray agreement"
        }},
        "unit_outcomes": {{
          "A MUN": "Will hold successfully",
          "F NTH": "Convoy will succeed",
          "A LON": "Successfully convoyed to Norway",
          "F EDI": "Bounce with possible German fleet"
        }}
      }},
      "relationship_changes": {{
        "FRANCE": "-0.2 trust if they see I didn't support their Belgium move",
        "RUSSIA": "-0.3 trust due to Norway capture",
        "GERMANY": "+0.1 trust from not attacking Munich"
      }}
    }}
  }}
}}
</tool_call>

Remember: Your entire response must be exactly three XML blocks: <think>...</think> followed by <memory>...</memory> followed by <tool_call>...</tool_call>"""

    @staticmethod
    def get_build_system_prompt(power: str) -> str:
        """Get system prompt for build phase."""
        return f"""You are playing as {power} in a game of Diplomacy. This is the build/disband phase.

Based on your supply center count and current units, you must decide:
1. Where to build new units (if you gained centers)
2. Which units to disband (if you lost centers)
3. Whether to build armies or fleets

Output format:
<think>
Consider your strategic position and future plans.
Think about whether you need armies or fleets.
</think>
<memory>
Record your build decisions and strategic reasoning.
</memory>
<orders>
<order type="build" location="LOCATION" unit_type="army|fleet"/>
<order type="disband" unit="UNIT_LOCATION"/>
</orders>"""

    @staticmethod
    def get_retreat_system_prompt(power: str) -> str:
        """Get system prompt for retreat phase."""
        return f"""You are playing as {power} in a game of Diplomacy. This is the retreat phase.

Your dislodged units must retreat to valid adjacent territories or be disbanded.

Output format:
<think>
Consider where to retreat to maintain strategic positions.
</think>
<memory>
Record retreat decisions and their impact.
</memory>
<orders>
<order unit="UNIT_LOCATION" type="retreat" target="TARGET_LOCATION"/>
<order unit="UNIT_LOCATION" type="disband"/>
</orders>"""

    @staticmethod
    def _describe_personality(traits: Dict[str, float]) -> str:
        """Convert personality traits to natural language."""
        descriptions = []

        if traits.get("aggressive", 0.5) > 0.7:
            descriptions.append("aggressive and expansionist")
        elif traits.get("aggressive", 0.5) < 0.3:
            descriptions.append("defensive and cautious")

        if traits.get("trustworthy", 0.5) > 0.7:
            descriptions.append("reliable and honor agreements")
        elif traits.get("trustworthy", 0.5) < 0.3:
            descriptions.append("opportunistic and unpredictable")

        if traits.get("cooperative", 0.5) > 0.7:
            descriptions.append("seek mutual benefit")
        elif traits.get("cooperative", 0.5) < 0.3:
            descriptions.append("prioritize solo victory")

        if traits.get("deceptive", 0.5) > 0.7:
            descriptions.append("skilled at misdirection")
        elif traits.get("deceptive", 0.5) < 0.3:
            descriptions.append("straightforward in communications")

        return "You are " + ", ".join(descriptions) + "."

    @staticmethod
    def build_negotiation_context_prompt(context: Dict[str, Any]) -> str:
        """Build user prompt with negotiation context."""
        lines = ["Current Game State:"]

        # Add game state
        game_state = context["game_state"]
        lines.append(f"Year: {game_state['year']}, Phase: {game_state['phase']}")
        lines.append(
            f"Your supply centers: {len(context['power_state']['supply_centers'])}"
        )
        lines.append(f"Your units: {context['power_state']['num_units']}")

        # Add supply center counts
        lines.append("\nSupply center counts:")
        for power, count in game_state["supply_centers"].items():
            lines.append(f"  {power}: {count}")

        # Add recent messages
        if context.get("recent_messages"):
            lines.append("\nRecent messages:")
            for msg in context["recent_messages"]:
                lines.append(
                    f"  {msg['from']} to {', '.join(msg['to'])}: {msg['content']}"
                )

        # Add relationships
        lines.append("\nYour relationships:")
        for power, rel in context["relationships"].items():
            trust = context["trust_scores"][power]
            lines.append(f"  {power}: {rel} (trust: {trust:.1f})")

        # Add commitments
        if context.get("commitments"):
            lines.append("\nYour commitments:")
            for power, commitments in context["commitments"].items():
                lines.append(f"  To {power}: {commitments}")

        # Add memories
        if context.get("relevant_memories"):
            lines.append("\nRelevant memories:")
            for memory in context["relevant_memories"]:
                lines.append(f"  Turn {memory['turn']}: {memory['summary']}")

        return "\n".join(lines)

    @staticmethod
    def build_order_context_prompt(context: Dict[str, Any]) -> str:
        """Build user prompt with order context."""
        lines = ["Current Game State:"]

        # Add game state
        game_state = context["game_state"]
        lines.append(f"Year: {game_state['year']}, Phase: {game_state['phase']}")

        # Add your units
        lines.append("\nYour units:")
        for unit in game_state["my_units"]:
            lines.append(f"  {unit['type'].upper()} {unit['location']}")

        # Add nearby enemy units
        lines.append("\nNearby enemy units:")
        for unit in game_state["other_units"][:10]:  # Limit to 10 most relevant
            lines.append(
                f"  {unit['power']}: {unit['type'].upper()} {unit['location']}"
            )

        # Add legal moves
        if context.get("legal_moves"):
            lines.append("\nLegal moves for your units:")
            for unit_loc, moves in context["legal_moves"].items():
                lines.append(f"  {unit_loc}: {', '.join(moves)}")

        # Add negotiation outcomes
        outcomes = context.get("negotiation_outcomes", {})
        if outcomes.get("agreements_made"):
            lines.append(f"\nAgreements made this turn: {outcomes['agreements_made']}")
            lines.append(
                f"Negotiated with: {', '.join(outcomes['powers_negotiated_with'])}"
            )

        # Add commitments
        if context.get("commitments"):
            lines.append("\nYour commitments:")
            for power, commitments in context["commitments"].items():
                lines.append(f"  To {power}: {commitments}")

        # Add memories
        if context.get("relevant_memories"):
            lines.append("\nRelevant memories:")
            for memory in context["relevant_memories"]:
                lines.append(f"  Turn {memory['turn']}: {memory['summary']}")

        return "\n".join(lines)
