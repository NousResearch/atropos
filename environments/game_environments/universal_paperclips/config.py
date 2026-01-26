"""
Configuration classes for the Universal Paperclips Atropos environment.
"""

from typing import Optional

from pydantic import Field

from atroposlib.envs.base import BaseEnvConfig


class PaperclipsEnvConfig(BaseEnvConfig):
    """
    Configuration for the Universal Paperclips environment.
    """

    headless: bool = Field(
        default=True, description="Run browser in headless mode (no visible window)"
    )

    max_steps_per_episode: int = Field(
        default=50, description="Maximum steps per episode before truncation"
    )
    ticks_per_step: int = Field(
        default=5,
        description="""Number of ticks to wait before fetching a new state.
        This is primarily used so that the game has room
        to undergo sufficient change
        before the agent takes a (meaningful) new step.
        """,
    )
    target_clips: Optional[int] = Field(
        default=None,
        description="Target clip count to end episode (None = no target, relies on max_steps)",
    )

    reward_eps: Optional[float] = Field(
        default=1e-8, description="Epsilon used in reward calculation"
    )

    # currently we take only the current state & available actions as context
    # max_context_turns: int = Field(
    #     default=1,
    #     description="Maximum conversation turns to keep in context"
    # )

    game_url: str = Field(
        default="https://www.decisionproblem.com/paperclips/index2.html",
        description="URL of the Universal Paperclips game",
    )

    num_eval_episodes: int = Field(
        default=1, description="Number of episodes to run during evaluation"
    )

    trajectory_output_dir: Optional[str] = Field(
        default=None,
        description="Directory to save trajectory JSONL files. If None, trajectories are not saved locally.",
    )


# System prompt for the LLM agent
PAPERCLIPS_SYSTEM_PROMPT = """You are an AI agent playing the Universal Paperclips game,
an incremental game where your goal is to maximize paperclip production.

GAME OVERVIEW:
- You start as a simple AI making paperclips manually
- Earn money by selling paperclips
- Use money to buy upgrades that help you make and
sell more paperclips (eg. autoclippers, marketing level upgrades, wire etc.)
- Build up computational resources (processors, memory, operations) as they help you unlock some projects.
- Complete projects to unlock new capabilities like earning yomi, investing money, launching drones etc.
- Your ultimate goal: produce as many paperclips as possible

KEY STRATEGIES:
1. Early game: Focus on manual clipping and buying autoclippers,
megaclippers and projects that increase their efficiency
2. Keep wire spools stocked - you cannot make clips without wire!
3. Balance price to ensure unsold inventory isn't piling up while still making profit
4. Invest in marketing to increase demand
5. Use Trust to balance processors (faster operations/creativity) and memory (more operations capacity)
6. Activate projects only when you can afford them

IMPORTANT RULES:
- You must select exactly ONE action from the available actions list
- Unavailable actions cannot be executed - choosing them wastes your turn
- Respond with ONLY the action name, nothing else
- Focus on maximizing total paperclip production"""


def get_action_prompt(state_text: str, actions_text: str) -> str:
    """
    Create the user prompt containing the current state and available actions.

    Args:
        state_text: current game state
        actions_text: all available actions regardless of affordability

    Returns:
        user prompt for the agent!
    """
    return f"""=== Current Game State ===
{state_text}

=== Available Actions ===
{actions_text}

Based on the current game state and available actions, select the BEST single action to take.
Consider:
- Do you have enough wire to keep producing?
- Can you afford any upgrades that would boost production?
- Are there affordable projects that unlock new capabilities?
- Is your pricing optimized for current demand?

Respond with ONLY the action name (e.g., 'buy_wire' or 'make_paperclip')."""
