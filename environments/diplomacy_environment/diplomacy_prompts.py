"""
Prompts for Diplomacy agents.

This module contains prompt templates and generation logic for
different phases and situations in Diplomacy.
"""

from typing import Dict, Any


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
        
        return f"""You are playing as {power} in a game of Diplomacy. You are in the {phase} phase.

Your personality traits: {personality_desc}

You must generate diplomatic messages to other powers. Consider:
1. Your current position and goals
2. Potential alliances and threats
3. Your commitments from previous negotiations
4. The trustworthiness of other powers

Output format:
<think>
Analyze the current situation, your goals, and potential strategies.
Consider what messages to send and to whom.
</think>
<memory>
Record key decisions and insights about this negotiation round.
</memory>
<negotiation>
<message to="POWER_NAME" type="proposal">
Your message content here...
</message>
<message to="ANOTHER_POWER" type="negotiation">
Another message...
</message>
</negotiation>

Message types: "proposal" (specific agreement), "negotiation" (general discussion), "commitment" (binding promise)"""
    
    @staticmethod
    def get_order_system_prompt(
        power: str,
        phase: str,
        personality: Dict[str, float],
    ) -> str:
        """Get system prompt for order phase."""
        personality_desc = DiplomacyPrompts._describe_personality(personality)
        
        return f"""You are playing as {power} in a game of Diplomacy. You are in the {phase} phase.

Your personality traits: {personality_desc}

You must generate orders for your units. Consider:
1. Your negotiated agreements and commitments
2. The current board position
3. Your strategic goals
4. Potential betrayals or defenses

Output format:
<think>
Analyze the situation and plan your moves.
Consider agreements made and whether to honor them.
Think about offensive and defensive priorities.
</think>
<memory>
Record your strategic decisions and any betrayals or key moves.
</memory>
<orders>
<order unit="UNIT_LOCATION" type="move" target="TARGET_LOCATION"/>
<order unit="UNIT_LOCATION" type="support" supporting="UNIT_LOCATION" target="TARGET_LOCATION"/>
<order unit="UNIT_LOCATION" type="hold"/>
</orders>

Order types: "move", "support", "convoy", "hold", "build", "disband", "retreat"."""
    
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
        lines.append(f"Your supply centers: {len(context['power_state']['supply_centers'])}")
        lines.append(f"Your units: {context['power_state']['num_units']}")
        
        # Add supply center counts
        lines.append("\nSupply center counts:")
        for power, count in game_state["supply_centers"].items():
            lines.append(f"  {power}: {count}")
        
        # Add recent messages
        if context.get("recent_messages"):
            lines.append("\nRecent messages:")
            for msg in context["recent_messages"]:
                lines.append(f"  {msg['from']} to {', '.join(msg['to'])}: {msg['content']}")
        
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
            lines.append(f"  {unit['power']}: {unit['type'].upper()} {unit['location']}")
        
        # Add legal moves
        if context.get("legal_moves"):
            lines.append("\nLegal moves for your units:")
            for unit_loc, moves in context["legal_moves"].items():
                lines.append(f"  {unit_loc}: {', '.join(moves)}")
        
        # Add negotiation outcomes
        outcomes = context.get("negotiation_outcomes", {})
        if outcomes.get("agreements_made"):
            lines.append(f"\nAgreements made this turn: {outcomes['agreements_made']}")
            lines.append(f"Negotiated with: {', '.join(outcomes['powers_negotiated_with'])}")
        
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