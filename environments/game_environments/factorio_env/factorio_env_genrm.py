#!/usr/bin/env python3
"""
Factorio Environment with LLM-as-a-Judge (GenRM) for Atropos

Generative Reward Model (GenRM) integration for Factorio Learning Environment.
Features:
- Best-of-n alternative generation per step 
- LLM-as-judge consensus scoring for action quality
- Outcome prediction evaluation for world modeling
- Step-by-step credit assignment with discounting
- Factorio-specific factory building evaluation
"""

import ast
import asyncio
import contextlib
import json
import logging
import os
import random
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
    ScoredDataItem,
)
from atroposlib.type_definitions import Item, Message
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

# Add FLE to path
fle_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fle")
sys.path.insert(0, fle_path)

import fle  # noqa: F401,E402  # Registers the environments
from fle.commons.models.game_state import GameState  # noqa: E402
from fle.env import FactorioInstance  # noqa: E402
from fle.env.gym_env.action import Action  # noqa: E402
from fle.env.gym_env.environment import FactorioGymEnv  # noqa: E402
from fle.env.gym_env.registry import (  # noqa: E402
    get_environment_info,
    list_available_environments,
)
from fle.env.tools import get_agent_tools  # noqa: E402
from fle.eval.tasks import TaskFactory  # noqa: E402

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # type: ignore
try:
    from factorio_rcon import (
        RCONClient,
    )  # lightweight RCON ping without resetting worlds
except Exception:  # pragma: no cover
    RCONClient = None  # type: ignore

logger = logging.getLogger(__name__)


class FactorioEnvConfig(BaseEnvConfig):
    """Configuration for the Factorio GenRM environment."""

    env_name: str = "factorio_genrm"
    wandb_name: str = "factorio-genrm-trainer"

    # Task settings
    task_names: List[str] = [
        "iron_ore_throughput",
        "iron_gear_wheel_throughput", 
        "iron_plate_throughput",
        "automation_science_pack_throughput",
    ]
    randomize_task_selection: bool = True
    max_steps_per_episode: int = 3  # Testing

    # Factorio server settings
    factorio_host: str = "localhost"
    factorio_tcp_port_base: int = 27000
    factorio_tcp_port: int = 27000
    factorio_fast_mode: bool = True
    factorio_total_servers: int = 32

    # Agent settings
    enable_self_planning: bool = True
    max_goals: int = 10

    # GenRM specific settings
    group_size: int = 4  # Reduced for testing
    max_num_workers: int = 10  # 10 workers for parallel data collection
    total_steps: int = 1000  # Collect 1000 trajectories
    max_token_length: int = 32768

    # Judge settings - LLM-as-a-judge for action evaluation
    action_judge_enabled: bool = True
    action_judge_weight: float = 0.4  # Weight for action quality judge
    action_judge_num_judges: int = 3  # Multiple judges for consensus

    # Outcome prediction judging - trains world modeling
    outcome_judge_enabled: bool = True
    outcome_judge_weight: float = 0.3  # Weight for outcome judge
    outcome_judge_num_judges: int = 3

    # Task progress judging - evaluates advancement toward factory goals
    progress_judge_enabled: bool = True  
    progress_judge_weight: float = 0.3  # Weight for progress judge
    progress_judge_num_judges: int = 3

    # Best-of-n selection parameters
    topk: int = 8  # Take top 8 alternatives
    bottomk: int = 8  # Take bottom 8 alternatives for exploration

    # Credit assignment
    credit_assignment_enabled: bool = True
    discount_factor: float = 0.99  # Discount future rewards

    # Scoring weights (base rewards before judge overlay)
    task_completion_weight: float = 10.0
    throughput_weight: float = 1.0
    efficiency_weight: float = 0.1

    # Data collection
    data_path_to_save_groups: Optional[str] = None  # Path to save ScoredDataGroups
    output_path: Optional[str] = None  # Path to save lightweight episode data (without tokens/masks)
    
    # Monitoring
    resource_log_interval_seconds: int = 15
    enable_resource_logging: bool = True
    preflight_timeout_seconds: int = 60
    skip_preflight_check: bool = True


class FactorioEnv(BaseEnv):
    """Factorio environment with LLM-as-a-judge (GenRM) for data generation."""

    name = "factorio_genrm"
    env_config_cls = FactorioEnvConfig
    
    # Class-level port allocation
    _port_status: Dict[int, bool] = {}

    def __init__(
        self,
        config: FactorioEnvConfig,
        server_configs: List[APIServerConfig],
        slurm: bool = False,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config: FactorioEnvConfig = config

        # Agent components
        self.current_goals: List[str] = []

        # Metrics tracking
        self.episode_outcomes_buffer = []
        self.episode_rewards_buffer = []
        self.episode_steps_buffer = []
        self.episode_task_types = []
        self.eval_metrics_custom = []

        self.tools_prompt = self._build_tools_prompt()

        # Initialize port status dictionary
        if not FactorioEnv._port_status:
            base = int(self.config.factorio_tcp_port_base)
            total = int(self.config.factorio_total_servers)
            for port in range(base, base + total):
                FactorioEnv._port_status[port] = False

        # Per-group aggregate metrics buffer
        self._group_metrics_buffer: List[Dict[str, float]] = []

    async def setup(self):
        """Initialize the Factorio environment for data generation."""
        logger.warning(
            f"Initialized Factorio GenRM environment with tasks: {self.config.task_names}"
        )
        logger.warning(f"Max steps per episode: {self.config.max_steps_per_episode}")
        logger.warning(f"Group size: {self.config.group_size}")
        logger.warning(f"Total trajectories to collect: {self.config.total_steps}")
        logger.warning(f"Judge settings: action={self.config.action_judge_enabled}, "
                      f"outcome={self.config.outcome_judge_enabled}, "
                      f"progress={self.config.progress_judge_enabled}")
        logger.warning("Factorio GenRM setup completed successfully")

    def _build_tools_prompt(self) -> str:
        """Build tools prompt for Factorio environment."""
        specs_lines = [
            (
                "- {'name': 'update_goals', "
                "'description': 'Update your goal list. Remove completed goals, add new ones.', "
                "'arguments': {'goals': 'list of strings'}}"
            )
        ]
        
        # Get Factorio-specific tools using the discovery utility
        try:
            # Import the discovery utility
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent))
            from fle_tool_discovery import discover_fle_tools, format_tool_for_prompt
            
            # Discover all agent tools
            discovered_tools = discover_fle_tools(tool_categories=["agent"])
            
            if discovered_tools:
                # Add each discovered tool to the specs
                for tool_name in sorted(discovered_tools.keys()):
                    tool_info = discovered_tools[tool_name]
                    formatted = format_tool_for_prompt(tool_info)
                    specs_lines.append(f"- {formatted}")
                logger.info(f"Loaded {len(discovered_tools)} FLE tools via discovery")
            else:
                raise Exception("No tools discovered")
                
        except Exception as e:
            logger.warning(f"Could not load Factorio tools via discovery: {e}")
            # Fallback to hardcoded tools if discovery fails
            specs_lines.extend([
                "- {'name': 'get_entities', 'description': 'Get entities within radius', 'arguments': {'radius': 'float', 'position': 'Position (optional)', 'entities': 'Prototype filter (optional)'}}",
                "- {'name': 'place_entity', 'description': 'Place a building/entity', 'arguments': {'entity': 'Prototype.EntityName', 'position': 'Position', 'direction': 'Direction (optional)'}}",
                "- {'name': 'nearest', 'description': 'Find nearest resource', 'arguments': {'resource': 'Resource.ResourceName'}}",
                "- {'name': 'craft_item', 'description': 'Craft items manually', 'arguments': {'item': 'Prototype.ItemName', 'count': 'int'}}",
                "- {'name': 'inspect_inventory', 'description': 'Check player inventory', 'arguments': {}}",
            ])

        return "Available tools:\n" + "\n".join(specs_lines)

    def _build_system_prompt(self, task_goal: str) -> str:
        """Build system prompt for factory building task."""
        return f"""You are an expert Factorio engineer working to optimize factory production.

Your current objective: {task_goal}

You have access to the following tools for factory building and management:
{self.tools_prompt}

RESPONSE FORMAT:
You must structure your response in exactly three XML blocks:

1. Enclose your thoughts and internal monologue inside <think> </think> tags. Use extremely long chains of thought to carefully consider the factory state, your objectives, and the likely outcomes of your actions.

2. Generate a memory summary inside <memory> </memory> tags that captures key information from this turn. Your memory should:
   - Build upon previous memories shown in 'Relevant Memories' if present
   - Note the outcome of your last action (did it match your prediction?)
   - Track important factory components: assemblers, belts, inserters, power systems
   - Remember resource locations and production rates
   - Keep your memory concise but informative for future turns

3. Execute exactly ONE action inside <tool_call> </tool_call> tags containing a JSON object with:
   - "name": The tool name to execute
   - "arguments": The arguments for the tool
   - "expected_outcome": Your prediction of what will happen

EXAMPLE RESPONSE 1 (Starting):
<think>
I need to produce 16 iron gear wheels per minute. Each gear wheel requires 2 iron plates, so I need 32 iron plates per minute. Let me check what's available in the factory. I should start by examining the current state and available resources.
</think>
<memory>
Goal: Produce 16 iron gear wheels/min. Need 32 iron plates/min input. Starting factory assessment.
</memory>
<tool_call>
{{"name": "get_entities", "arguments": {{"radius": 50}}, "expected_outcome": "I will see all entities within 50 tiles of my position, including any existing miners, furnaces, assemblers, and belts"}}
</tool_call>

EXAMPLE RESPONSE 2 (Building):
<think>
Based on my memory, I've located iron ore at position [15, -20]. I need to place miners there to extract it. Electric mining drills produce 30 ore/min each. Since I need 32 plates/min and furnaces smelt at 18.75 plates/min, I'll need 2 furnaces. For the ore, I'll need 32/18.75 * ore_per_plate = about 2 miners as well.
</think>
<memory>
Iron ore found at [15, -20]. Plan: 2 miners → 2 furnaces → 1 assembler for gears. Each miner: 30 ore/min. Each furnace: 18.75 plates/min.
</memory>
<tool_call>
{{"name": "place_entity", "arguments": {{"entity": "Prototype.ElectricMiningDrill", "position": {{"x": 15, "y": -20}}}}, "expected_outcome": "An electric mining drill will be placed on the iron ore patch at [15, -20] and will start extracting iron ore at 30 ore/min once powered"}}
</tool_call>

Remember: Your entire response must be exactly three XML blocks: <think>...</think> followed by <memory>...</memory> followed by <tool_call>...</tool_call>

FINAL REMINDER: After your <think> block and <memory> block, you MUST wrap your JSON function call in <tool_call> tags. The JSON goes INSIDE the <tool_call> tags, not after them."""



    async def _generate_alternatives(
        self, messages: List[Message], num_alternatives: int, trajectory_id: str, step: int
    ) -> List[Dict[str, Any]]:
        """Generate multiple alternative responses for current step using completion endpoint."""
        
        alternatives = []
        
        # Convert messages to prompt format for completions endpoint
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Add prefill to guide response format
        prefill = "<think>\n"
        prompt = prompt + prefill
        
        # Generate all alternatives in a single API call (like TextWorld does)
        try:
            logger.warning(f"[Episode {trajectory_id}] Step {step}: Generating {num_alternatives} alternatives in single call")
            
            response = await self.server.completion(
                prompt=prompt,
                n=num_alternatives,  # Get all alternatives at once
                max_tokens=32768,  # Model is trained to 32k context
                temperature=0.8,
                top_p=0.95,
                stop=["</tool_call>", "<|im_end|>", "<|endoftext|>"],  # Include </tool_call> as stop token
            )
            
            logger.warning(f"[Episode {trajectory_id}] Step {step}: API returned {len(response.choices)} choices")
            
            # Process each choice/alternative
            for i, choice in enumerate(response.choices):
                try:
                    # Get the generated text and prepend the prefill
                    generated_text = choice.text.strip()
                    content = prefill + generated_text
                    
                    
                    # If we have an opening tool_call tag but no closing tag (due to stop token), append it
                    if "<tool_call>" in content and "</tool_call>" not in content:
                        content += "\n</tool_call>"
                    
                    # Debug first few alternatives - show actual content
                    if i < 2:
                        logger.warning(f"[Episode {trajectory_id}] Step {step} Alt {i} RAW RESPONSE: {content}")
                    
                    # Parse the tool call from the response
                    parsed_action = self._parse_response_format(content)
                    if parsed_action:
                        alternatives.append({
                            "response": content,
                            "parsed_action": parsed_action,
                            "alternative_idx": i,
                        })
                    else:
                        # Include failed alternatives for GRPO training
                        logger.warning(f"[Episode {trajectory_id}] Step {step} Alt {i}: Failed to parse, including as invalid")
                        alternatives.append({
                            "response": content,
                            "parsed_action": {"name": "invalid", "arguments": {}, "expected_outcome": ""},
                            "alternative_idx": i,
                        })
                except Exception as e:
                    logger.warning(f"[Episode {trajectory_id}] Step {step}: Error processing choice {i}: {e}")
                    # Include as invalid alternative
                    alternatives.append({
                        "response": "",
                        "parsed_action": {"name": "invalid", "arguments": {}, "expected_outcome": ""},
                        "alternative_idx": i,
                    })
                    
        except Exception as e:
            logger.error(f"[Episode {trajectory_id}] Step {step}: Failed to generate alternatives: {e}")
            logger.error(f"[Episode {trajectory_id}] Step {step}: Exception type: {type(e)}")
            import traceback
            logger.error(f"[Episode {trajectory_id}] Step {step}: Traceback: {traceback.format_exc()}")
            raise e
        
        return alternatives

    async def _score_alternatives_with_judges(
        self, alternatives: List[Dict[str, Any]], obs_text: str, info: Dict,
        trajectory_id: str, step: int, task_goal: str
    ) -> Dict[str, Any]:
        """Score alternatives using LLM judges. Returns dict with scores to be converted to ScoredDataGroup later."""
        
        logger.warning(f"[Episode {trajectory_id}] Step {step}: ENTERED _score_alternatives_with_judges with {len(alternatives)} alternatives")
        
        scores = []
        items = []
        
        for i, alt in enumerate(alternatives):
            logger.warning(f"[Episode {trajectory_id}] Step {step}: Scoring alternative {i+1}/{len(alternatives)}")
            # Base score from environment/heuristics
            base_score = 0.5  # Neutral starting score
            
            # Judge scores
            judge_scores = {}
            
            if self.config.action_judge_enabled:
                logger.warning(f"[Episode {trajectory_id}] Step {step}: Calling action judge for alt {i+1}")
                try:
                    judge_scores["action"] = await self._judge_action_quality(
                        alt["parsed_action"], obs_text, task_goal, trajectory_id, step
                    )
                    logger.warning(f"[Episode {trajectory_id}] Step {step}: Action judge returned {judge_scores['action']}")
                except Exception as e:
                    logger.error(f"[Episode {trajectory_id}] Step {step}: Action judge error: {e}")
                    judge_scores["action"] = 0.5
            
            if self.config.progress_judge_enabled:
                logger.warning(f"[Episode {trajectory_id}] Step {step}: Calling progress judge for alt {i+1}")
                try:
                    judge_scores["progress"] = await self._judge_task_progress(
                        alt["parsed_action"], obs_text, task_goal, trajectory_id, step
                    )
                    logger.warning(f"[Episode {trajectory_id}] Step {step}: Progress judge returned {judge_scores['progress']}")
                except Exception as e:
                    logger.error(f"[Episode {trajectory_id}] Step {step}: Progress judge error: {e}")
                    judge_scores["progress"] = 0.5
            
            # Combine judge scores
            final_score = base_score
            total_weight = 1.0
            
            if "action" in judge_scores:
                final_score += self.config.action_judge_weight * judge_scores["action"]
                total_weight += self.config.action_judge_weight
                
            if "progress" in judge_scores:
                final_score += self.config.progress_judge_weight * judge_scores["progress"]
                total_weight += self.config.progress_judge_weight
            
            final_score = final_score / total_weight
            
            scores.append(final_score)
            items.append([])  # Will be populated with message history later
            
            logger.warning(f"[Episode {trajectory_id}] Step {step} Alt {alt['alternative_idx']}: score={final_score:.3f} judges={judge_scores}")
        
        # Return a dict for now - this will be properly converted to ScoredDataGroup
        # in _create_step_scored_group with tokenization
        return {"items": items, "scores": scores}

    async def _judge_action_quality(
        self, action: Dict[str, Any], obs_text: str, task_goal: str, 
        trajectory_id: str, step: int
    ) -> float:
        """Judge the quality of a factory building action."""
        
        logger.warning(f"[Episode {trajectory_id}] Step {step}: _judge_action_quality called")
        
        if not action:
            logger.warning(f"[Episode {trajectory_id}] Step {step}: No action provided, returning 0.0")
            return 0.0
            
        # Create judge prompt
        action_str = json.dumps(action, indent=2)
        judge_prompt = f"""You are evaluating a Factorio factory building action for efficiency and strategic value.

TASK GOAL: {task_goal}

CURRENT FACTORY STATE:
{obs_text}

PROPOSED ACTION:
{action_str}

Evaluate this action on a scale of 0-10 considering:

EFFICIENCY (0-3 points):
- Does this action efficiently use resources?
- Is the building placement optimal for throughput?
- Are there better alternatives for the same goal?

STRATEGY (0-4 points):
- Does this advance toward the task goal?
- Is this the right priority given current state?
- Does it consider future factory expansion needs?

TECHNICAL CORRECTNESS (0-3 points):
- Is the action technically feasible?
- Are prerequisites met (power, materials, technology)?
- Will this action actually work as intended?

Respond with just a number 0-10."""

        try:
            logger.warning(f"[Episode {trajectory_id}] Step {step}: Running {self.config.action_judge_num_judges} action judges")
            judge_scores = []
            for i in range(self.config.action_judge_num_judges):
                logger.warning(f"[Episode {trajectory_id}] Step {step}: Calling action judge {i+1}/{self.config.action_judge_num_judges}")
                
                # Use chat completion for judges (simpler and more reliable)
                messages = [{"role": "user", "content": judge_prompt}]
                
                response = await self.server.chat_completion(
                    model="Hermes-4-405B",  # Use the configured model
                    messages=messages,
                    max_tokens=10,
                    temperature=0.3,
                )
                
                judge_text = response.choices[0].message.content.strip()
                logger.warning(f"[Episode {trajectory_id}] Step {step}: Judge {i+1} response: {judge_text}")
                
                if not judge_text:
                    logger.warning(f"[Episode {trajectory_id}] Step {step}: Empty judge response")
                    judge_scores.append(0.5)
                    continue
                    
                try:
                    score = float(re.search(r'(\d+(?:\.\d+)?)', judge_text).group(1))
                    score = max(0.0, min(10.0, score)) / 10.0  # Normalize to 0-1
                    judge_scores.append(score)
                    logger.warning(f"[Episode {trajectory_id}] Step {step}: Judge {i+1} score: {score}")
                except:
                    logger.warning(f"[Episode {trajectory_id}] Step {step}: Failed to parse action judge score: {judge_text}")
                    judge_scores.append(0.5)  # Default neutral score
            
            return sum(judge_scores) / len(judge_scores) if judge_scores else 0.5
            
        except Exception as e:
            logger.error(f"[Episode {trajectory_id}] Step {step}: Action judge error: {e}")
            return 0.5

    async def _judge_task_progress(
        self, action: Dict[str, Any], obs_text: str, task_goal: str,
        trajectory_id: str, step: int
    ) -> float:
        """Judge how well an action advances toward the task goal."""
        
        if not action:
            return 0.0
            
        action_str = json.dumps(action, indent=2)
        judge_prompt = f"""You are evaluating how well a Factorio action advances toward a specific factory goal.

SPECIFIC GOAL: {task_goal}

CURRENT STATE:
{obs_text}

PROPOSED ACTION:
{action_str}

Rate from 0-10 how much this action advances toward the SPECIFIC GOAL:

0-2: Irrelevant or counterproductive to the goal
3-4: Loosely related but not directly helpful
5-6: Somewhat helpful, indirect progress
7-8: Good progress toward the goal
9-10: Directly and efficiently advances the goal

Consider:
- Does this build the right infrastructure for the goal?
- Is this the most impactful action available right now?
- Does this address current bottlenecks blocking progress?

Respond with just a number 0-10."""

        try:
            logger.warning(f"[Episode {trajectory_id}] Step {step}: Running {self.config.progress_judge_num_judges} progress judges")
            judge_scores = []
            for i in range(self.config.progress_judge_num_judges):
                logger.warning(f"[Episode {trajectory_id}] Step {step}: Calling progress judge {i+1}/{self.config.progress_judge_num_judges}")
                
                # Use chat completion for judges (simpler and more reliable)
                messages = [{"role": "user", "content": judge_prompt}]
                
                response = await self.server.chat_completion(
                    model="Hermes-4-405B",  # Use the configured model
                    messages=messages,
                    max_tokens=10,
                    temperature=0.3,
                )
                
                judge_text = response.choices[0].message.content.strip()
                logger.warning(f"[Episode {trajectory_id}] Step {step}: Progress judge {i+1} response: {judge_text}")
                
                if not judge_text:
                    logger.warning(f"[Episode {trajectory_id}] Step {step}: Empty judge response")
                    judge_scores.append(0.5)
                    continue
                    
                try:
                    score = float(re.search(r'(\d+(?:\.\d+)?)', judge_text).group(1))
                    score = max(0.0, min(10.0, score)) / 10.0
                    judge_scores.append(score)
                    logger.warning(f"[Episode {trajectory_id}] Step {step}: Progress judge {i+1} score: {score}")
                except:
                    logger.warning(f"[Episode {trajectory_id}] Step {step}: Failed to parse progress judge score: {judge_text}")
                    judge_scores.append(0.5)
            
            return sum(judge_scores) / len(judge_scores) if judge_scores else 0.5
            
        except Exception as e:
            logger.error(f"[Episode {trajectory_id}] Step {step}: Progress judge error: {e}")
            return 0.5

    async def _judge_outcome_prediction(
        self, action: Any, predicted_outcome: str, actual_outcome: str,
        trajectory_id: str, step: int
    ) -> float:
        """Judge accuracy of predicted vs actual outcome."""
        
        if not predicted_outcome or not actual_outcome:
            return 0.5
            
        judge_prompt = f"""You are evaluating how accurately a Factorio player predicted the outcome of their action.

ACTION TAKEN: {action}

PREDICTED OUTCOME:
"{predicted_outcome}"

ACTUAL OUTCOME:  
"{actual_outcome}"

Rate prediction accuracy from 0-10:
- 0-2: Completely wrong prediction
- 3-4: Major errors in understanding
- 5-6: Partially correct, got general idea
- 7-8: Mostly accurate prediction  
- 9-10: Excellent prediction, very accurate

Consider:
- Did they correctly predict factory state changes?
- Did they understand the action's effects?
- Were differences minor (details) or major (mechanics)?

Respond with just a number 0-10."""

        try:
            logger.warning(f"[Episode {trajectory_id}] Step {step}: Running {self.config.outcome_judge_num_judges} outcome judges")
            judge_scores = []
            for i in range(self.config.outcome_judge_num_judges):
                logger.warning(f"[Episode {trajectory_id}] Step {step}: Calling outcome judge {i+1}/{self.config.outcome_judge_num_judges}")
                
                # Use chat completion for judges (simpler and more reliable)
                messages = [{"role": "user", "content": judge_prompt}]
                
                response = await self.server.chat_completion(
                    model="Hermes-4-405B",  # Use the configured model
                    messages=messages,
                    max_tokens=10,
                    temperature=0.3,
                )
                
                judge_text = response.choices[0].message.content.strip()
                logger.warning(f"[Episode {trajectory_id}] Step {step}: Outcome judge {i+1} response: {judge_text}")
                
                if not judge_text:
                    logger.warning(f"[Episode {trajectory_id}] Step {step}: Empty judge response")
                    judge_scores.append(0.5)
                    continue
                    
                try:
                    score = float(re.search(r'(\d+(?:\.\d+)?)', judge_text).group(1))
                    score = max(0.0, min(10.0, score)) / 10.0
                    judge_scores.append(score)
                    logger.warning(f"[Episode {trajectory_id}] Step {step}: Outcome judge {i+1} score: {score}")
                except:
                    logger.warning(f"[Episode {trajectory_id}] Step {step}: Failed to parse outcome judge score: {judge_text}")
                    judge_scores.append(0.5)
            
            return sum(judge_scores) / len(judge_scores) if judge_scores else 0.5
            
        except Exception as e:
            logger.error(f"[Episode {trajectory_id}] Step {step}: Outcome judge error: {e}")
            return 0.5

    async def _select_best_alternative(self, step_group: ScoredDataGroup) -> int:
        """Select best alternative using top-k/bottom-k strategy."""
        
        if not step_group["scores"]:
            return 0
            
        # Sort alternatives by score
        scored_alts = [(i, score) for i, score in enumerate(step_group["scores"])]
        scored_alts.sort(key=lambda x: x[1], reverse=True)
        
        # Select from top-k for exploitation
        topk_indices = [idx for idx, _ in scored_alts[:self.config.topk]]
        
        # For now, always select the best (can add exploration later)
        selected_idx = topk_indices[0] if topk_indices else 0
        
        logger.warning(f"Selected alternative {selected_idx} with score {step_group['scores'][selected_idx]:.3f}")
        return selected_idx

    async def _execute_action(
        self, env: FactorioGymEnv, parsed_action: Dict[str, Any], 
        trajectory_id: str, step: int
    ) -> Tuple[Optional[Any], float]:
        """Execute parsed action in the Factorio environment."""
        
        logger.warning(f"[Episode {trajectory_id}] Step {step}: _execute_action called with parsed_action: {parsed_action}")
        
        if not parsed_action:
            logger.warning(f"[Episode {trajectory_id}] Step {step}: No parsed_action, returning failure")
            return None, -1.0
        
        # Check if it's an invalid action (failed to parse)
        if parsed_action.get("name") == "invalid":
            logger.warning(f"[Episode {trajectory_id}] Step {step}: Invalid action (failed to parse), returning failure")
            return None, -1.0
        
        try:
            tool_name = parsed_action.get("name", "")
            
            # Handle meta-tools (goal management)
            if tool_name == "update_goals":
                goals = parsed_action.get("arguments", {}).get("goals", [])
                self.current_goals = goals[:self.config.max_goals]
                logger.info(f"[Episode {trajectory_id}] Step {step}: Updated goals: {self.current_goals}")
                return "goals_updated", 0.1  # Small positive reward for planning
            
            # Handle Factorio game actions
            else:
                # Remove connection check - it's causing issues and connection seems stable
                
                # Convert to FLE Action object
                logger.warning(f"[Episode {trajectory_id}] Step {step}: Getting GameState...")
                current_game_state = GameState.from_instance(env.instance)
                logger.warning(f"[Episode {trajectory_id}] Step {step}: GameState obtained")
                
                # Create code representation of the action
                action_code = self._convert_to_factorio_code(parsed_action)
                logger.warning(f"[Episode {trajectory_id}] Step {step}: Generated code: {action_code}")
                
                action = Action(
                    agent_idx=0,
                    code=action_code,
                    game_state=current_game_state
                )
                logger.warning(f"[Episode {trajectory_id}] Step {step}: Action created, returning")
                
                return action, 0.0  # Environment will provide reward
                
        except Exception as e:
            logger.error(f"[Episode {trajectory_id}] Step {step}: Action execution failed: {e}")
            return None, -1.0

    def _convert_to_factorio_code(self, parsed_action: Dict[str, Any]) -> str:
        """Convert parsed tool call to Python code for FLE execution."""
        
        tool_name = parsed_action.get("name", "")
        args = parsed_action.get("arguments", {})
        
        # If no tool name, return a no-op
        if not tool_name:
            logger.warning("No tool name found in parsed action, returning no-op")
            return "print(inspect_inventory())"
        
        # Build Python function call - EXACT COPY from factorio_env_minimal.py
        arg_strs = []
        for k, v in args.items():
            if isinstance(v, dict) and "x" in v and "y" in v:
                # Only convert dicts with x,y to Position
                arg_strs.append(f"{k}=Position(x={v['x']}, y={v['y']})")
            elif isinstance(v, str):
                # Handle Resource/Prototype/Direction enums
                if any(v.startswith(prefix) for prefix in ["Resource.", "Prototype.", "Direction."]):
                    arg_strs.append(f"{k}={v}")
                else:
                    arg_strs.append(f"{k}={repr(v)}")
            else:
                # Pass everything else through as-is (lists, numbers, etc.)
                arg_strs.append(f"{k}={v}")
        
        # Return Python function call wrapped in print()
        return f"print({tool_name}({', '.join(arg_strs)}))"

    def _parse_tool_call(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse tool call from LLM response."""
        
        try:
            # Look for JSON in tool_call tags first
            json_match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', response, re.DOTALL)
            
            if not json_match:
                # Look for JSON that contains "name" field - handle nested braces properly
                json_match = re.search(r'\{(?:[^{}]|{[^{}]*})*"name"(?:[^{}]|{[^{}]*})*\}', response, re.DOTALL)
            
            if not json_match:
                # Try to find any JSON-like object in the response
                json_match = re.search(r'\{.*?"name".*?\}', response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1) if json_match.groups() else json_match.group(0)
                return json.loads(json_str)
                
        except Exception as e:
            logger.warning(f"Failed to parse tool call: {e}")
        
        return None

    def _parse_response_format(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse the structured response format to extract action and expected outcome."""
        import json
        import re
        
        try:
            # Extract tool_call JSON
            tool_call_match = re.search(r'<tool_call>\s*(.*?)\s*</tool_call>', response, re.DOTALL)
            if not tool_call_match:
                # Debug: Log if no tool_call tags found
                if "<tool_call>" not in response:
                    logger.warning(f"Parse fail: No <tool_call> tags in response. First 200 chars: {response[:200]}")
                else:
                    logger.warning(f"Parse fail: Has <tool_call> but regex failed. First 200 chars: {response[:200]}")
                return None
                
            tool_call_json = tool_call_match.group(1).strip()
            logger.warning(f"Extracted tool_call JSON: {tool_call_json[:200]}")
            
            # Parse the JSON
            tool_call = json.loads(tool_call_json)
            
            # Validate structure
            if "name" not in tool_call:
                logger.warning(f"Parse fail: No 'name' field in tool_call: {tool_call}")
                return None
                
            # Extract expected outcome if present (might be in arguments)
            args = tool_call.get("arguments", {})
            expected_outcome = args.get("expected_outcome", "")
            
            # Return parsed action with expected outcome
            return {
                "name": tool_call["name"],
                "arguments": args,
                "expected_outcome": expected_outcome
            }
            
        except json.JSONDecodeError as e:
            logger.warning(f"Parse fail: JSON decode error: {e}")
            return None
        except Exception as e:
            logger.warning(f"Parse fail: Unexpected error: {e}")
            return None
    
    def _strip_thinking_blocks(self, response: str) -> str:
        """Remove thinking blocks from response."""
        # Remove <think>...</think> blocks
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        return response.strip()

    async def _save_episode_data(
        self, 
        scored_groups: List[ScoredDataGroup], 
        trajectory_id: str,
        task_name: str,
        total_reward: float,
        episode_outcome: float
    ):
        """Save episode data to JSONL file."""
        import json
        import os
        
        # Use configured path
        data_path = self.config.data_path_to_save_groups
        if not data_path:
            return
            
        # Create directory if needed
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        
        # Track success metrics
        self.episode_count = getattr(self, 'episode_count', 0) + 1
        self.success_count = getattr(self, 'success_count', 0)
        if episode_outcome > 0:  # Task completed successfully
            self.success_count += 1
        
        success_rate = (self.success_count / self.episode_count * 100) if self.episode_count > 0 else 0
        
        # Log episode outcome
        outcome_str = "SUCCESS" if episode_outcome > 0 else "FAILURE"
        logger.warning(f"[Episode {self.episode_count}] {outcome_str} - Task: {task_name}, Reward: {total_reward:.2f}, Steps: {len(scored_groups)}")
        logger.warning(f"[Stats] Episodes: {self.episode_count}, Successes: {self.success_count}, Success Rate: {success_rate:.1f}%")
        
        # Prepare episode data - list of step groups with metadata
        episode_data = {
            "trajectory_id": trajectory_id,
            "task_name": task_name,
            "total_reward": total_reward,
            "episode_outcome": episode_outcome,
            "num_steps": len(scored_groups),
            "episode_number": self.episode_count,
            "success": episode_outcome > 0,
            "steps": []
        }
        
        # Add each step's data (without tokens/masks)
        for step_idx, group in enumerate(scored_groups):
            step_data = {
                "step": step_idx,
                "scores": group.get("scores", []),
                "num_alternatives": len(group.get("scores", [])),
                # Don't include tokens or masks
            }
            
            # Add messages if configured
            if self.config.include_messages and "items" in group:
                step_data["messages"] = group["items"]
                
            episode_data["steps"].append(step_data)
        
        # Append to JSONL file
        try:
            with open(data_path, "a") as f:
                f.write(json.dumps(episode_data) + "\n")
            logger.info(f"Saved episode {trajectory_id} with {len(scored_groups)} steps to {data_path}")
        except Exception as e:
            logger.error(f"Failed to save episode data: {e}")
    
    async def _save_lightweight_episode(
        self, scored_groups: List[ScoredDataGroup], trajectory_id: str,
        task_name: str, total_reward: float, episode_outcome: float
    ):
        """Save episode data without tokens/masks (similar to TextWorld's _save_winning_episode)."""
        import json
        import os
        
        # Use configured output path
        data_path = self.config.output_path
        if not data_path:
            return
            
        # Create directory if needed
        os.makedirs(os.path.dirname(data_path) if os.path.dirname(data_path) else ".", exist_ok=True)
        
        # Convert ScoredDataGroups to serializable format, excluding tokens and masks
        episode_array = []
        for sdg in scored_groups:
            # ScoredDataGroup is a dict, so copy it
            clean_sdg = dict(sdg).copy()
            
            # Remove tokens and masks
            clean_sdg.pop('tokens', None)
            clean_sdg.pop('masks', None)
            
            # Keep scores, messages, and other metadata
            episode_array.append(clean_sdg)
        
        # Create episode metadata
        episode_data = {
            "trajectory_id": trajectory_id,
            "task_name": task_name,
            "total_reward": total_reward,
            "episode_outcome": episode_outcome,
            "num_steps": len(scored_groups),
            "success": episode_outcome > 0,
            "steps": episode_array
        }
        
        # Append to JSONL file
        try:
            with open(data_path, "a") as f:
                f.write(json.dumps(episode_data) + "\n")
            logger.info(f"Saved lightweight episode {trajectory_id} to {data_path}")
        except Exception as e:
            logger.error(f"Failed to save lightweight episode data: {e}")
    
    def _format_observation(self, obs: Dict, info: Dict) -> str:
        """Format Factorio observation for the LLM."""
        
        parts = []
        
        # Current goals
        if self.current_goals:
            parts.append("Current goals:")
            for i, goal in enumerate(self.current_goals[:5]):
                parts.append(f"  {i+1}. {goal}")
        
        # Game state from observation
        if isinstance(obs, dict):
            if "inventory" in obs:
                inv = obs["inventory"]
                if inv:
                    inv_str = ", ".join([f"{item['type']}:{item['quantity']}" for item in inv[:10]])
                    parts.append(f"Inventory: {inv_str}")
            
            if "entities" in obs:
                parts.append(f"Nearby entities: {len(obs['entities'])}")
                
            if "resources" in obs:
                resources = obs["resources"]
                if resources:
                    res_str = ", ".join([f"{r['type']}:{r['amount']}" for r in resources[:5]])
                    parts.append(f"Nearby resources: {res_str}")
        
        # Task progress from info
        if info:
            if "task_completed" in info:
                parts.append(f"Task completed: {info['task_completed']}")
            if "throughput" in info:
                parts.append(f"Current throughput: {info['throughput']}")
            if "error" in info:
                parts.append(f"Error: {info['error']}")
        
        return "\n".join(parts) if parts else "No specific information available"

    async def _apply_credit_assignment(
        self, step_groups: List[ScoredDataGroup], step_rewards: List[float],
        selected_indices: List[int], trajectory_id: str
    ) -> List[ScoredDataGroup]:
        """Apply Monte Carlo credit assignment to step groups."""
        
        if not self.config.credit_assignment_enabled:
            return step_groups
            
        # Calculate discounted returns
        returns = []
        discounted_return = 0.0
        
        for reward in reversed(step_rewards):
            discounted_return = reward + self.config.discount_factor * discounted_return
            returns.append(discounted_return)
        
        returns.reverse()
        
        # Apply returns to selected alternatives in each step
        for step_idx, (step_group, selected_idx, ret) in enumerate(zip(step_groups, selected_indices, returns)):
            if step_idx < len(step_groups) and selected_idx < len(step_group["scores"]):
                # Boost selected alternative's score with discounted return
                original_score = step_group["scores"][selected_idx]
                step_group["scores"][selected_idx] = original_score + 0.1 * ret  # Scale return influence
                
                logger.warning(f"[Episode {trajectory_id}] Step {step_idx}: Credit assignment +{0.1 * ret:.3f} (return={ret:.2f})")
        
        return step_groups

    async def _reserve_ports(self, n: int) -> List[int]:
        """Reserve n distinct RCON ports from the configured cluster pool."""
        
        logger.warning(f"[DEBUG] _reserve_ports called with n={n}, total_servers={self.config.factorio_total_servers}")
        
        # Simple case: single instance
        if n == 1 and self.config.factorio_total_servers == 1:
            port = self.config.factorio_tcp_port
            logger.warning(f"[DEBUG] Single instance mode, returning port {port}")
            return [port]
        
        # Multi-instance case
        deadline = time.time() + 300  # 5 minute timeout
        while True:
            free_ports = [port for port, in_use in FactorioEnv._port_status.items() if not in_use]
            logger.warning(f"[DEBUG] Free ports available: {len(free_ports)}, needed: {n}")
            
            if len(free_ports) >= n:
                chosen = free_ports[:n]
                for port in chosen:
                    FactorioEnv._port_status[port] = True
                logger.warning(f"[DEBUG] Reserved ports: {chosen}")
                return chosen
                
            if time.time() > deadline:
                logger.error(f"Failed to reserve {n} ports after 5 minutes")
                return [self.config.factorio_tcp_port]  # Fallback
                
            logger.warning(f"Waiting for {n} free ports (have {len(free_ports)})")
            await asyncio.sleep(2)

    async def _release_ports(self, ports: List[int]):
        """Release reserved ports back to the pool."""
        for port in ports:
            if port in FactorioEnv._port_status:
                FactorioEnv._port_status[port] = False

    async def get_next_item(self) -> Item:
        """Get the next factory task configuration."""
        import time
        
        # Generate a unique seed for this task instance
        unique_seed = int(time.time() * 1000000) + random.randint(0, 1000000)
        
        # Randomly select a task
        if len(self.config.task_names) == 1:
            task_name = self.config.task_names[0]
        else:
            task_name = random.choice(self.config.task_names)
        
        # Return task configuration as a dictionary (Item = Any, so just return dict)
        return {
            "task_name": task_name,
            "seed": unique_seed,
            "environment": "factorio_genrm",
        }

    async def collect_trajectories(
        self, item: Item
    ) -> Tuple[List[ScoredDataGroup], List[Item]]:
        """Collect trajectories using step-by-step generation with parallel alternatives."""
        task_name = item["task_name"]
        seed = item["seed"]
        
        logger.warning(f"Creating Factorio task for: {task_name} with seed: {seed}")
        
        # Collect episode using step-by-step generation
        scored_data_groups = await self._collect_episode_step_by_step(
            task_name, seed
        )
        
        logger.warning("Factorio episode complete, cleaning up")
        return scored_data_groups, []
    
    async def postprocess_histories(
        self,
        trajectories: Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]]],
    ) -> Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]]]:
        """
        Postprocess the histories, this is called after the collect_trajectories method
        
        If you don't need to do anything to the trajectories, you may safely ignore this.
        """
        return trajectories
    
    async def cleanup(self):
        """
        Optional: Cleanup the Factorio environment
        """
        # Clean up any Docker containers or resources if needed
        pass

    async def _collect_episode_step_by_step(
        self, task_name: str, seed: int
    ) -> List[ScoredDataGroup]:
        """Collect ONE episode with group_size alternatives at EACH step.
        
        Returns a List[ScoredDataGroup] where each group contains alternatives for one step.
        This enables step-level training where the model learns good decisions at each step.
        """
        logger.warning(f"[DEBUG] Starting step-by-step episode collection for {task_name}")
        
        try:
            # Create Factorio task and reserve ports - map task name to JSON file path
            logger.warning(f"[DEBUG] Creating task from {task_name}")
            task_json_path = f"lab_play/{task_name}.json"
            task = TaskFactory.create_task(task_json_path)
            logger.warning(f"[DEBUG] Task created successfully: {type(task)}")
            
            logger.warning(f"[DEBUG] Reserving 1 port for episode...")
            ports = await self._reserve_ports(1)  # Reserve one port for this episode
            tcp_port = ports[0]
            logger.warning(f"[DEBUG] Reserved port: {tcp_port}")
            
            try:
                # Create Factorio instance with timeout
                logger.warning(f"[DEBUG] Creating FactorioInstance on {self.config.factorio_host}:{tcp_port}")
                
                # Wrap in asyncio timeout to prevent hanging
                import asyncio
                
                try:
                    # FactorioInstance is synchronous, so use executor
                    loop = asyncio.get_event_loop()
                    instance = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            lambda: FactorioInstance(
                                address=self.config.factorio_host,
                                tcp_port=tcp_port,
                                fast=self.config.factorio_fast_mode,
                                num_agents=1,
                            )
                        ),
                        timeout=30.0  # 30 second timeout
                    )
                    logger.warning(f"[DEBUG] FactorioInstance created successfully")
                except asyncio.TimeoutError:
                    logger.error(f"[DEBUG] FactorioInstance creation timed out after 30s on port {tcp_port}")
                    await self._release_ports([tcp_port])
                    return []
                except Exception as e:
                    logger.error(f"[DEBUG] FactorioInstance creation failed: {e}")
                    await self._release_ports([tcp_port])
                    return []
                
                logger.warning(f"[DEBUG] Creating FactorioGymEnv...")
                env = FactorioGymEnv(instance=instance, task=task)
                logger.warning(f"[DEBUG] FactorioGymEnv created")
                
                logger.warning(f"[DEBUG] Resetting environment...")
                obs, info = await asyncio.wait_for(
                    loop.run_in_executor(None, env.reset),
                    timeout=30.0
                )
                logger.warning(f"[DEBUG] Environment reset complete - obs keys: {obs.keys() if obs else 'None'}")
                
                # Initialize conversation
                task_goal = getattr(task, "goal_description", "Complete the factory task")
                system_prompt = self._build_system_prompt(task_goal)
                messages = [{"role": "system", "content": system_prompt}]
                
                obs_text = self._format_observation(obs, info)
                messages.append({"role": "user", "content": obs_text})
                
                done = False
                total_reward = 0.0
                step = 0
                scored_data_groups = []  # One ScoredDataGroup per step
                step_rewards = []  # Track rewards for each step
                selected_indices = []  # Track selected alternative for each step
                
                logger.warning(f"[DEBUG] Starting episode loop, max_steps={self.config.max_steps_per_episode}")
                
                while not done and step < self.config.max_steps_per_episode:
                    step += 1
                    logger.warning(f"[DEBUG] Starting step {step}/{self.config.max_steps_per_episode}")
                    
                    # Token management - implement sliding window
                    MAX_GENERATION_TOKENS = 2000  # Conservative for Factorio
                    available_budget = self.config.max_token_length - MAX_GENERATION_TOKENS
                    current_tokens = self._count_tokens(messages)
                    
                    if current_tokens > available_budget:
                        logger.warning(f"[DEBUG] Step {step}: Token count {current_tokens} exceeds budget {available_budget}, applying sliding window")
                        
                        # Keep system prompt and recent context
                        while len(messages) > 3 and current_tokens > available_budget:
                            dropped_msg = messages.pop(1)
                            logger.warning(f"Dropped message: {dropped_msg['role']} - {dropped_msg['content'][:50]}...")
                            current_tokens = self._count_tokens(messages)
                        
                        logger.warning(f"[DEBUG] After sliding window: {len(messages)} messages, {current_tokens} tokens")
                    
                    # Generate group_size alternatives for this step
                    logger.warning(f"[DEBUG] Step {step}: Generating {self.config.group_size} alternatives...")
                    alternatives = await self._generate_alternatives_for_step(
                        messages, step
                    )
                    logger.warning(f"[DEBUG] Step {step}: Generated {len(alternatives) if alternatives else 0} alternatives")
                    if not alternatives:
                        logger.warning(f"No alternatives generated at step {step}, ending episode")
                        break
                    
                    # Pad alternatives if we have fewer than expected
                    if len(alternatives) < self.config.group_size:
                        logger.warning(f"[DEBUG] Step {step}: Only {len(alternatives)}/{self.config.group_size} alternatives, padding with duplicates")
                        # Duplicate existing alternatives to reach group_size
                        original_count = len(alternatives)
                        while len(alternatives) < self.config.group_size:
                            duplicate_idx = len(alternatives) % original_count
                            duplicate = alternatives[duplicate_idx].copy()
                            duplicate["alternative_idx"] = len(alternatives)
                            alternatives.append(duplicate)
                        logger.warning(f"[DEBUG] Step {step}: Padded to {len(alternatives)} alternatives")
                    
                    # Score and package alternatives into a ScoredDataGroup for this step
                    step_group = await self._create_step_scored_group(
                        alternatives, messages, env, task_goal, step
                    )
                    if step_group:
                        scored_data_groups.append(step_group)
                        
                        # Select best alternative to continue the episode
                        selected_idx = await self._select_best_alternative(step_group)
                        selected_alt = alternatives[selected_idx]
                        
                        # Execute selected action in environment
                        action, reward = await self._execute_action(
                            env, selected_alt["parsed_action"], f"{task_name}_{seed}", step
                        )
                        
                        if action is None:
                            logger.warning(f"Failed to execute action at step {step}")
                            break
                        
                        # Get new observation from environment
                        try:
                            obs, reward_env, terminated, truncated, info = env.step(action)
                            done = terminated or truncated
                            reward += reward_env
                            total_reward += reward
                        except Exception as e:
                            logger.error(f"Environment step failed: {e}")
                            break
                        
                        # Update step group with outcome judge if enabled
                        if self.config.outcome_judge_enabled and selected_alt["parsed_action"].get("expected_outcome"):
                            actual_outcome = self._format_observation(obs, info)
                            outcome_judge_score = await self._judge_outcome_prediction(
                                action,
                                selected_alt["parsed_action"]["expected_outcome"],
                                actual_outcome,
                                f"{task_name}_{seed}",
                                step
                            )
                            
                            # Update selected alternative's score with outcome judge
                            original_score = step_group["scores"][selected_idx]
                            step_group["scores"][selected_idx] = (
                                (1.0 - self.config.outcome_judge_weight) * original_score +
                                self.config.outcome_judge_weight * outcome_judge_score
                            )
                        
                        # Track for credit assignment
                        step_rewards.append(reward)
                        selected_indices.append(selected_idx)
                        
                        # Add selected response to conversation
                        stripped_response = self._strip_thinking_blocks(selected_alt["response"])
                        messages.append({"role": "assistant", "content": stripped_response})
                        
                        # Add new observation for next step
                        if not done:
                            obs_text = self._format_observation(obs, info)
                            messages.append({"role": "user", "content": obs_text})
                        
                        logger.warning(f"Step {step}: Selected alt {selected_idx}, reward={reward:.2f}, done={done}")
                
                env.close()
                
                # Apply credit assignment to scored groups
                if self.config.credit_assignment_enabled and scored_data_groups and step_rewards:
                    await self._apply_credit_assignment_to_groups(
                        scored_data_groups, step_rewards, selected_indices, f"{task_name}_{seed}"
                    )
                
                # Calculate episode outcome based on task completion
                episode_outcome = 1.0 if info.get("task_completed", False) else 0.0
                
                # Log completion with task success/failure
                trajectory_id = f"{task_name}_{seed}"
                if episode_outcome > 0:
                    logger.warning(f"[Episode {trajectory_id}] ✅ TASK COMPLETED! Steps: {step}, Reward: {total_reward:.2f}")
                else:
                    logger.warning(f"[Episode {trajectory_id}] ❌ Task failed. Steps: {step}, Reward: {total_reward:.2f}")
                
                # Save regular data if configured
                if self.config.data_path_to_save_groups and scored_data_groups:
                    await self._save_episode_data(
                        scored_data_groups,
                        trajectory_id,
                        task_name,
                        total_reward,
                        episode_outcome
                    )
                
                # Save lightweight data without tokens/masks
                if self.config.output_path and scored_data_groups:
                    await self._save_lightweight_episode(
                        scored_data_groups,
                        trajectory_id,
                        task_name,
                        total_reward,
                        episode_outcome
                    )
                
                logger.warning(f"Episode completed: {step} steps, total_reward={total_reward:.2f}, groups={len(scored_data_groups)}")
                return scored_data_groups
                
            finally:
                # Always release ports
                await self._release_ports(ports)
                
        except Exception as e:
            logger.error(f"Fatal error in episode collection: {e}")
            logger.error(traceback.format_exc())
            return []

    async def _generate_alternatives_for_step(
        self, messages: List[Message], step: int
    ) -> List[Dict[str, Any]]:
        """Generate multiple alternative responses for current step (adapted from existing method)."""
        return await self._generate_alternatives(messages, self.config.group_size, f"step_{step}", step)

    async def _create_step_scored_group(
        self, alternatives: List[Dict[str, Any]], messages: List[Message], 
        env: Any, task_goal: str, step: int
    ) -> Optional[ScoredDataGroup]:
        """Create a ScoredDataGroup for this step's alternatives."""
        if not alternatives:
            return None
        
        # Get current observation for judge context
        obs_text = messages[-1]["content"] if messages else ""
        
        # Score alternatives using judges
        scores = []
        tokenized_items = []
        messages_list = []  # Store message history for each alternative
        
        for alt in alternatives:
            # Judge scoring (reuse existing methods)
            base_score = 0.5
            judge_scores = {}
            
            if self.config.action_judge_enabled:
                judge_scores["action"] = await self._judge_action_quality(
                    alt["parsed_action"], obs_text, task_goal, f"step_{step}", step
                )
            
            if self.config.progress_judge_enabled:
                judge_scores["progress"] = await self._judge_task_progress(
                    alt["parsed_action"], obs_text, task_goal, f"step_{step}", step
                )
            
            # Combine judge scores
            final_score = base_score
            total_weight = 1.0
            
            if "action" in judge_scores:
                final_score += self.config.action_judge_weight * judge_scores["action"]
                total_weight += self.config.action_judge_weight
            
            if "progress" in judge_scores:
                final_score += self.config.progress_judge_weight * judge_scores["progress"]
                total_weight += self.config.progress_judge_weight
            
            final_score = final_score / total_weight
            scores.append(final_score)
            
            # Create conversation with this alternative
            step_messages = messages + [{"role": "assistant", "content": alt["response"]}]
            
            # Store messages for this alternative
            messages_list.append(step_messages)
            
            # Tokenize for trainer
            try:
                tokenization_result = tokenize_for_trainer(
                    tokenizer=self.tokenizer,
                    chat=step_messages,
                    train_on_all_assistant_turns=True,
                )
                tokenized_items.append({
                    "tokens": tokenization_result["tokens"],
                    "masks": tokenization_result["masks"],
                })
            except Exception as e:
                logger.warning(f"Tokenization failed for alternative {len(tokenized_items)}: {e}")
                # Provide fallback empty tokenization
                tokenized_items.append({
                    "tokens": [],
                    "masks": [],
                })
        
        # Create ScoredDataGroup (matching TextWorld's structure)
        return ScoredDataGroup(
            tokens=[item["tokens"] for item in tokenized_items],
            masks=[item["masks"] for item in tokenized_items], 
            scores=scores,
            messages=messages_list if self.config.include_messages else None,
            advantages=None,
            ref_logprobs=None,
            group_overrides={},
            overrides=None,
            images=None,
        )

    def _count_tokens(self, messages: List[Message]) -> int:
        """Count tokens in a message list."""
        try:
            return len(self.tokenizer.apply_chat_template(messages, tokenize=True))
        except Exception as e:
            logger.warning(f"Token counting failed: {e}")
            # Rough estimation fallback
            return sum(len(msg["content"].split()) * 1.3 for msg in messages)

    async def _apply_credit_assignment_to_groups(
        self, scored_data_groups: List[ScoredDataGroup], 
        step_rewards: List[float], selected_indices: List[int],
        trajectory_id: str
    ) -> None:
        """Apply Monte Carlo credit assignment to scored data groups."""
        if not self.config.credit_assignment_enabled or not step_rewards or not selected_indices:
            return
        
        # Calculate discounted returns backwards
        returns = []
        discounted_return = 0.0
        
        for reward in reversed(step_rewards):
            discounted_return = reward + self.config.discount_factor * discounted_return
            returns.append(discounted_return)
        
        returns.reverse()
        
        # Apply returns to selected alternatives in each step group
        for step_idx, (step_group, selected_idx, ret) in enumerate(zip(scored_data_groups, selected_indices, returns)):
            if step_idx < len(scored_data_groups) and selected_idx < len(step_group["scores"]):
                # Boost selected alternative's score with discounted return
                original_score = step_group["scores"][selected_idx]
                step_group["scores"][selected_idx] = original_score + 0.1 * ret  # Scale return influence
                
                logger.warning(f"[{trajectory_id}] Step {step_idx}: Credit assignment +{0.1 * ret:.3f} (return={ret:.2f})")

    async def evaluate(self, num_items: int) -> Dict[str, Any]:
        """Evaluate the model - not implemented for data generation environment."""
        logger.warning("Evaluation not implemented in Factorio GenRM data generation environment")
        return {"message": "Evaluation not implemented - this environment is for data collection only"}

    @classmethod
    def config_init(cls) -> Tuple[FactorioEnvConfig, List[APIServerConfig]]:
        """Initialize default configuration for Factorio GenRM data generation."""
        
        env_config = FactorioEnvConfig(
            tokenizer_name=os.getenv("MODEL_NAME", "NousResearch/Hermes-4-Qwen3-14B-1-e3"),
            group_size=4,  # Reduced for debugging - normally 16
            max_num_workers=10,  # 10 parallel workers for data collection
            use_wandb=True,
            wandb_name="factorio-genrm-data",
            max_token_length=32768,
            total_steps=1000,  # Collect 1000 trajectories
            task_names=["iron_ore_throughput", "iron_gear_wheel_throughput"],  # Start with basic tasks
            max_steps_per_episode=4,  # Reduced for testing - normally 250
            factorio_total_servers=1,  # Single container for sequential processing
            
            # GenRM settings
            action_judge_enabled=True,
            outcome_judge_enabled=True, 
            progress_judge_enabled=True,
            credit_assignment_enabled=True,
            
            # Data collection path
            data_path_to_save_groups="/home/maxpaperclips/atropos/data/factorio_genrm_hermes405b.jsonl",
        )
        
        # API server configs for Hermes-405B judge calls
        server_configs = [
            APIServerConfig(
                model_name="Hermes-4-405B",
                base_url="https://inference-api.nousresearch.com/v1", 
                api_key=os.getenv("HERMES_API_KEY", "dummy-key"),
                num_requests_for_eval=128,
            ),
        ]
        
        return env_config, server_configs

    async def wandb_log(self, wandb_metrics: Optional[Dict[str, float]] = None):
        """Log metrics to Weights & Biases."""
        
        if wandb_metrics is None:
            wandb_metrics = {}
        
        # Episode-level metrics
        if self.episode_rewards_buffer:
            total_episodes = len(self.episode_rewards_buffer)
            avg_reward = sum(self.episode_rewards_buffer) / total_episodes
            avg_steps = sum(self.episode_steps_buffer) / total_episodes if self.episode_steps_buffer else 0
            completion_rate = sum(self.episode_outcomes_buffer) / total_episodes if self.episode_outcomes_buffer else 0
            
            wandb_metrics.update({
                f"{self.name}/train/total_episodes": total_episodes,
                f"{self.name}/train/avg_reward": avg_reward,
                f"{self.name}/train/avg_steps": avg_steps,
                f"{self.name}/train/completion_rate": completion_rate,
            })
            
            # Task breakdown
            task_counts = {}
            for task in self.episode_task_types:
                task_counts[task] = task_counts.get(task, 0) + 1
            for task, count in task_counts.items():
                wandb_metrics[f"{self.name}/train/task_{task}_count"] = count
        
        # Per-group aggregates
        if self._group_metrics_buffer:
            try:
                num_groups = len(self._group_metrics_buffer)
                avg_reward = sum(g["avg_reward"] for g in self._group_metrics_buffer) / num_groups
                avg_steps = sum(g["avg_steps"] for g in self._group_metrics_buffer) / num_groups
                completion_rate = sum(g["completion_rate"] for g in self._group_metrics_buffer) / num_groups
                
                wandb_metrics.update({
                    f"{self.name}/train/group/num_groups": num_groups,
                    f"{self.name}/train/group/avg_reward": avg_reward,
                    f"{self.name}/train/group/avg_steps": avg_steps, 
                    f"{self.name}/train/group/completion_rate": completion_rate,
                })
            finally:
                self._group_metrics_buffer = []
        
        # Clear buffers
        self.episode_rewards_buffer = []
        self.episode_steps_buffer = []
        self.episode_outcomes_buffer = []
        self.episode_task_types = []
        
        # Add eval metrics
        for key, value in self.eval_metrics_custom:
            wandb_metrics[key] = value
        self.eval_metrics_custom = []
        
        await super().wandb_log(wandb_metrics)


if __name__ == "__main__":
    # Allow running directly for testing
    FactorioEnv.cli()