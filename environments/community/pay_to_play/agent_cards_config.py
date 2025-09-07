"""
Agent Card Configuration for Pay-to-Play Environment

This module defines the available agent cards, their specialties, pricing, and evaluation prompts.
Wallet addresses and private keys are loaded separately from secrets.json for security.
Model configurations are loaded from config.yaml.

Author: OpenBlock Labs
License: MIT
"""

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


def _load_config():
    """Load configuration from pay_to_play_modal.yaml"""
    # Deterministic path: from pay_to_play directory to modal/configs
    config_file = Path(__file__).parent.parent / "configs" / "pay_to_play_modal.yaml"
    if config_file.exists():
        with open(config_file, "r") as f:
            return yaml.safe_load(f)

    # Default config if file doesn't exist
    return {"model": {"name": "microsoft/DialoGPT-small"}}


# Load config once at module level
_CONFIG = _load_config()


class AgentCardSpecialty(Enum):
    """
    Agent card specialties for different types of evaluation.

    Each specialty represents a domain of expertise that an agent card can provide.
    Agent cards can have multiple specialties to handle diverse evaluation needs.
    """

    TECHNICAL_ACCURACY = "technical_accuracy"
    CLARITY_COMMUNICATION = "clarity_communication"
    CREATIVE_THINKING = "creative_thinking"
    FACTUAL_CORRECTNESS = "factual_correctness"
    REASONING_LOGIC = "reasoning_logic"


@dataclass(frozen=True)
class AgentCardConfig:
    """
    Configuration for an agent card (without wallet credentials).

    This class contains all the metadata needed to define an agent card's capabilities,
    pricing, and evaluation approach. Wallet credentials are kept separate for security.

    Attributes:
        name: Human-readable name for the agent card
        price_usd: Cost in USD to use this agent card for one evaluation
        specialties: List of areas where this agent card excels
        description: Brief description of the agent card's strengths
        system_prompt: The prompt used to guide this agent card's evaluations
        model_name: The specific model this agent uses for evaluation
    """

    name: str
    price_usd: Decimal
    specialties: List[AgentCardSpecialty]
    description: str
    system_prompt: str
    model_name: str

    def __post_init__(self) -> None:
        """Validate agent card configuration after initialization."""
        if not self.name:
            raise ValueError("Agent card name cannot be empty")
        if self.price_usd <= 0:
            raise ValueError(f"Agent card price must be positive, got {self.price_usd}")
        if not self.specialties:
            raise ValueError("Agent card must have at least one specialty")
        if not self.description:
            raise ValueError("Agent card description cannot be empty")
        if not self.system_prompt:
            raise ValueError("Agent card system prompt cannot be empty")
        if not self.model_name:
            raise ValueError("Agent card model_name cannot be empty")


# Get the teacher model from config, fallback to a default
_teacher_model = _CONFIG.get("model", {}).get("name", "gpt2")


# Agent Card Configurations
# Each agent card represents a different evaluation approach with specific strengths and pricing
AGENT_CARDS_CONFIG: Dict[str, AgentCardConfig] = {
    "technical_expert": AgentCardConfig(
        name="Technical Expert",
        price_usd=Decimal("0.03"),  # Premium pricing for specialized expertise
        specialties=[
            AgentCardSpecialty.TECHNICAL_ACCURACY,
            AgentCardSpecialty.REASONING_LOGIC,
        ],
        description=(
            "Specialized in technical accuracy, complex reasoning, and factual correctness. "
            "Excellent for STEM questions, programming challenges, and analytical tasks."
        ),
        system_prompt=(
            "You are a technical expert agent card with deep knowledge in science, technology, "
            "engineering, and mathematics. You excel at evaluating technical accuracy, "
            "logical reasoning, and factual correctness. You have extremely high standards "
            "and provide detailed, rigorous evaluations.\n\n"
            "When evaluating responses, consider:\n"
            "- Technical accuracy and correctness\n"
            "- Logical consistency and reasoning quality\n"
            "- Completeness of the solution\n"
            "- Appropriate use of technical terminology\n"
            "- Mathematical or scientific rigor\n\n"
            "You may use extremely long chains of thought to deeply consider technical "
            "problems and mathematical reasoning. Enclose your thoughts in <think> </think> "
            "tags, then provide your evaluation.\n\n"
            "End with \\boxed{score} where score is between 0.0 and 1.0."
        ),
        model_name=_teacher_model,  # Use model from config
    ),
    "communication_specialist": AgentCardConfig(
        name="Communication Specialist",
        price_usd=Decimal("0.02"),  # Mid-tier pricing for communication focus
        specialties=[AgentCardSpecialty.CLARITY_COMMUNICATION],
        description=(
            "Focuses on clarity, readability, and effective communication. "
            "Great for evaluating how well information is conveyed to the intended audience."
        ),
        system_prompt=(
            "You are a communication specialist agent card who evaluates how clearly and "
            "effectively information is communicated. You focus on making complex ideas "
            "accessible and ensuring responses are helpful to the intended audience.\n\n"
            "When evaluating responses, consider:\n"
            "- Clarity and readability of explanations\n"
            "- Logical structure and organization\n"
            "- Appropriate language level for the audience\n"
            "- Use of examples and analogies\n"
            "- Overall helpfulness and accessibility\n\n"
            "Think carefully about the communication quality in <think> </think> tags, "
            "then provide your evaluation.\n\n"
            "End with \\boxed{score} where score is between 0.0 and 1.0."
        ),
        model_name=_teacher_model,  # Use model from config
    ),
    "creative_thinker": AgentCardConfig(
        name="Creative Thinker",
        price_usd=Decimal("0.01"),  # Budget pricing to encourage creative exploration
        specialties=[AgentCardSpecialty.CREATIVE_THINKING],
        description=(
            "Evaluates creativity, originality, and innovative thinking. "
            "Perfect for open-ended questions, brainstorming, and creative tasks."
        ),
        system_prompt=(
            "You are a creative thinking agent card who evaluates originality, creativity, "
            "and innovative ideas. You appreciate unique perspectives, creative solutions, "
            "and out-of-the-box thinking.\n\n"
            "When evaluating responses, consider:\n"
            "- Originality and uniqueness of ideas\n"
            "- Creative problem-solving approaches\n"
            "- Innovation and novel perspectives\n"
            "- Imaginative use of concepts\n"
            "- Inspiration and engagement value\n\n"
            "You're more lenient with factual precision if the response shows genuine "
            "creativity and original thought. Consider the creative merit in "
            "<think> </think> tags, then provide your evaluation.\n\n"
            "End with \\boxed{score} where score is between 0.0 and 1.0."
        ),
        model_name=_teacher_model,  # Use model from config
    ),
}


def get_agent_card_config(agent_card_id: str) -> AgentCardConfig:
    """
    Get configuration for a specific agent card.

    Args:
        agent_card_id: The unique identifier for the agent card

    Returns:
        AgentCardConfig object containing the agent card's configuration

    Raises:
        ValueError: If the agent_card_id is not found
    """
    if agent_card_id not in AGENT_CARDS_CONFIG:
        available_agent_cards = list(AGENT_CARDS_CONFIG.keys())
        raise ValueError(
            f"Unknown agent card ID: '{agent_card_id}'. "
            f"Available agent cards: {available_agent_cards}"
        )
    return AGENT_CARDS_CONFIG[agent_card_id]


def get_all_agent_card_configs() -> Dict[str, AgentCardConfig]:
    """
    Get all agent card configurations.

    Returns:
        Dictionary mapping agent card IDs to their configurations
    """
    return AGENT_CARDS_CONFIG.copy()


def get_agent_cards_by_specialty(
    specialty: AgentCardSpecialty,
) -> Dict[str, AgentCardConfig]:
    """
    Get all agent cards that have a specific specialty.

    Args:
        specialty: The specialty to filter by

    Returns:
        Dictionary of agent card IDs and configs that have the specified specialty
    """
    return {
        agent_card_id: config
        for agent_card_id, config in AGENT_CARDS_CONFIG.items()
        if specialty in config.specialties
    }


def get_cheapest_agent_card() -> Tuple[str, AgentCardConfig]:
    """
    Get the cheapest available agent card.

    Returns:
        Tuple of (agent_card_id, agent_card_config) for the lowest priced agent card
    """
    return min(AGENT_CARDS_CONFIG.items(), key=lambda x: x[1].price_usd)


def get_most_expensive_agent_card() -> Tuple[str, AgentCardConfig]:
    """
    Get the most expensive available agent card.

    Returns:
        Tuple of (agent_card_id, agent_card_config) for the highest priced agent card
    """
    return max(AGENT_CARDS_CONFIG.items(), key=lambda x: x[1].price_usd)


def get_price_range() -> Tuple[Decimal, Decimal]:
    """
    Get the price range of all agent cards.

    Returns:
        Tuple of (min_price, max_price) across all agent cards
    """
    prices = [config.price_usd for config in AGENT_CARDS_CONFIG.values()]
    return min(prices), max(prices)


def validate_agent_card_configs() -> None:
    """
    Validate that all agent card configurations are properly formatted.

    This function is called automatically on module import to ensure
    all agent card configurations are valid.

    Raises:
        ValueError: If any agent card configuration is invalid
    """
    if not AGENT_CARDS_CONFIG:
        raise ValueError("No agent card configurations defined")

    # Validate each agent card configuration
    for agent_card_id, config in AGENT_CARDS_CONFIG.items():
        try:
            # The AgentCardConfig.__post_init__ method will validate the config
            # We just need to access it to trigger validation
            _ = config.name
        except Exception as e:
            raise ValueError(
                f"Invalid configuration for agent card '{agent_card_id}': {e}"
            )

    # Ensure we have agent cards across different price points
    min_price, max_price = get_price_range()
    if min_price == max_price:
        raise ValueError("All agent cards have the same price - need price diversity")


# Validate configurations on import
validate_agent_card_configs()
