
"""
Multi-Turn Tool-Calling Environment with Turn-Level Advantages
==============================================================

Extends the multi-turn tool-calling environment to implement turn-level credit assignment
following the MT-GRPO approach from "Turn-Level Credit Assignment for Multi-Turn LLM Agents".

Key differences from the base multiturn environment:
    • Computes both turn-level rewards (R_T) and outcome-level rewards (R_O)  
    • Implements MT-GRPO advantage computation:
      - Turn 1: A_T_1 + λ * A_O
      - Turn 2: A_T_2 + λ * A_O  
      - Turn 3: A_O (outcome only)
    • Populates per-token advantages in ScoredDataGroup instead of just scores
    • Enables fine-grained credit assignment across turns

Dataset columns expected
------------------------
* `conversations` – list[dict] with keys `from` and `value`
"""

import json
import random
import re
import asyncio
import ast
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from collections import Counter

import wandb
from datasets import load_dataset
from tqdm.asyncio import tqdm_asyncio
from pydantic import Field

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    EvalHandlingEnum,
    Item,
    ScoredDataGroup,
)
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

# Easy-to-change constants for experimentation
WRONG_CALL_PENALTY = -0.2
MAX_GEN_PER_TURN = 1024
MAX_TOOL_CALL_TURNS_CAP = 3
VALIDATE_THINK_BLOCKS = True
GENERATE_ALL_GPT_TURNS = True
ADD_DYNAMIC_FEW_SHOT = True
USE_PARALLEL_REQUESTS = False # For when api provider doesn't support n*prompt for 1st turn
TURN_LEVEL_ADVANTAGE_LAMBDA = 0.5  # Paper uses 1.0, experiment with 0.1, 0.5, 1.0
SCENARIO_CATEGORY = "multistep"

# For filtering completed tasks
#COMPLETED_DATASET_ID = "interstellarninja/hermes_reasoning_tool_use"
COMPLETED_DATASET_ID = "interstellarninja/toolace_sequential_tool_use_reasoning"
try:
    _done_ds = load_dataset(COMPLETED_DATASET_ID, split="train")
    COMPLETED_TASKS: set[str] = set(_done_ds["task"])
    print(f"[filter] Loaded {len(COMPLETED_TASKS):,} completed tasks from {COMPLETED_DATASET_ID}")
except Exception as _exc:
    COMPLETED_TASKS = set()
    print(f"[filter] Could not load completed-task dataset: {_exc}. No skipping will occur.")


class MTGRPOEnvConfig(BaseEnvConfig):
    """Configuration for Multi-Turn Tool Calling with Turn-Level Advantages Environment."""
    max_tool_call_turns_cap: Optional[int] = Field(
        default=2,
        description="Upper cap on assistant TOOL‑CALLING turns per episode (None → no cap)"
    )
    validate_think_blocks: bool = Field(
        default=True,
        description="Whether to validate that all GPT messages have <think> blocks"
    )
    generate_all_gpt_turns: bool = Field(
        default=True,
        description="Generate GPT turns after tool responses, including summaries"
    )
    max_gen_per_turn: int = Field(
        default=1024,
        description="Hard cap on how many new tokens the model may generate in a single turn"
    )
    wrong_call_penalty: float = Field(
        default=-0.2,
        description="Negative reward applied when the first mismatched tool-call causes early termination"
    )
    skip_completed: bool = Field(
        default=True,
        description="Skip any conversation whose first user prompt appears in COMPLETED_TASKS"
    )
    scenario_category: str = Field(
        default="multistep",
        description='BFCL‑style scenario type: "single", "multistep", or "multiturn"'
    )
    add_dynamic_few_shot: bool = Field(
        default=True,
        description="Insert most‑recent harvested example into system prompt"
    )
    use_parallel_requests: bool = Field(
        default=True,
        description="Whether to use parallel requests even for identical first-turn prompts"
    )
    turn_level_advantage_lambda: float = Field(
        default=0.5,
        description="Turn-level advantage coefficient (λ in MT-GRPO paper)"
    )

system_prompt = (
    "You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the "
    "problem and deliberate with yourself via systematic reasoning processes to help come to a correct "
    "solution prior to answering. You should enclose your thoughts and internal monologue inside <think> "
    "</think> tags, and then provide your solution or response to the problem."
)

few_shot_example = (
    "Example Reasoning format:\n"
    "<think>\n"
    "Let's be sure of the definite integral ∫₀¹ x² dx. It's easy by hand "
    "I know the antiderivative of x² is x³/3, so evaluating from 0 to 1 should give 1/3.\n"
    "Let me verify this calculation with SymPy to be completely certain.\n"
    "</think>\n"
    "<tool_call>\n"
    '{"name":"python_interpreter","arguments":{"code":"import sympy as sp\\nx=sp.symbols(\'x\')\\nprint(sp.integrate(x**2,(x,0,1)))"}}\n'
    "</tool_call>\n"
)

SEQ_TOOL_HELPER = (
    "You are in sequential tool calling mode. "
    "Call exactly **one** tool, wait for its <tool_response>, "
    "then decide whether to call another. "
    "Never bundle multiple <tool_call> blocks in the same message. "
    "Do not generate <tool_response> blocks by yourself since system will provide. "
    "When you get the <tool_response> and if you believe the task is complete, "
    "return your reasoning in <think> blocks followed by plain text summary. "
    "You are on a limited token budget so perform compressed thinking steps."
)

def _normalize_tool_call_json(txt: str) -> str:
    """
    Normalize the entire response structure:
    - Preserve <think> block
    - Convert Python dict style tool calls to proper JSON format
    - Ensure consistent newlines around tags
    """
    # First extract the think block
    think_match = re.match(r"^\s*(<think>[\s\S]*?</think>)\s*", txt)
    if not think_match:
        return txt
    think_block = think_match.group(1)

    # Ensure think block ends with newline
    if not think_block.endswith('\n'):
        think_block += '\n'

    # Then normalize tool calls
    def replace_tool_call(match):
        content = match.group(1).strip()
        try:
            obj = ast.literal_eval(content)
            return f"\n<tool_call>\n{json.dumps(obj, separators=(',', ':'))}\n</tool_call>\n"
        except Exception:
            pass
            
        try:
            json_str = re.sub(r"'([^']*)':", r'"\1":', content)  # Handle dict keys
            json_str = re.sub(r':\s*\'([^\']*)\'', r': "\1"', json_str)  # Handle string values
            json.loads(json_str)  # Validate
            return f"\n<tool_call>\n{json_str}\n</tool_call>\n"
        except Exception:
            print(f"Failed to normalize JSON: {content}")
            return match.group(0)
    
    # Replace tool calls after the think block
    rest_of_text = txt[len(think_match.group(0)):]
    normalized_calls = re.sub(r"<tool_call>\s*(.*?)\s*</tool_call>", replace_tool_call, rest_of_text, flags=re.DOTALL)

    # Clean up any multiple newlines
    result = think_block + normalized_calls
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result

def _validate_think_only(txt: str) -> bool:
    """
    A narration / summary turn must:
    • start with exactly one <think> … </think> block
    • contain **no** <tool_call> tags anywhere
    • contain no additional <think> blocks
    Anything after the </think> (user‑visible answer) is allowed.
    """
    txt = _normalize_tool_call_json(txt)
    if not isinstance(txt, str):
        return False

    # Must begin with exactly one think block and no more think blocks after
    think_blocks = re.findall(r"<think>[\s\S]*?</think>", txt, flags=re.IGNORECASE)
    if len(think_blocks) != 1:
        return False

    # Must be at the start
    if not re.match(r"^\s*<think>", txt, flags=re.IGNORECASE):
        return False

    # Must not contain any <tool_call>
    if re.search(r"<tool_call\s*>", txt, flags=re.IGNORECASE):
        return False

    return True


def _validate_think_plus_calls(txt: str):
    """
    Validate a GPT reply that should contain exactly one <think> … </think> followed by
    one or more <tool_call> … </tool_call> blocks.

    Returns normalized JSON objects with proper double quotes.
    """
    txt = _normalize_tool_call_json(txt)

    # Check for exactly one think block
    think_blocks = re.findall(r"<think>[\s\S]*?</think>", txt, flags=re.IGNORECASE)
    if len(think_blocks) != 1:
        return None

    # Must start with the think block
    if not re.match(r"^\s*<think>", txt, flags=re.IGNORECASE):
        return None

    # Must be followed by at least one tool call
    tool_calls = re.findall(
        r"<tool_call>\s*([\s\S]*?)\s*</tool_call>", txt, flags=re.IGNORECASE
    )
    if not tool_calls:
        return None

    # Parse and normalize tool calls to proper JSON
    tool_jsons = []
    for raw in tool_calls:
        try:
            # First try direct JSON parse
            obj = json.loads(raw)
        except Exception:
            try:
                # If that fails, try literal_eval and convert to JSON
                obj = ast.literal_eval(raw)
            except Exception:
                # If both fail, try crude string replacement
                json_like = re.sub(r"'([^']*)':", r'"\1":', raw)
                json_like = re.sub(r":\s*'([^']*)'", r':"\1"', json_like)
                try:
                    obj = json.loads(json_like)
                except Exception:
                    return None
        tool_jsons.append(obj)
    return tool_jsons


def _json_objects_match(model_json, expected_json):
    """
    True when every key/value in expected_json appears exactly in model_json.
    Nested dicts handled recursively.
    """
    if not isinstance(model_json, dict) or not isinstance(expected_json, dict):
        return False
    for k, v in expected_json.items():
        if k not in model_json:
            return False
        if isinstance(v, dict):
            if not _json_objects_match(model_json[k], v):
                return False
        else:
            if model_json[k] != v:
                return False
    return True

def _check_sequential_tools(conv: List[Dict[str, str]]) -> bool:
    """
    Return True when every assistant tool‑calling turn is followed only by the
    corresponding <tool_response> messages from the system before the next
    assistant <tool_call>. Allow concluding narration after all tool calls are done.
    """
    tool_indices = [
        i
        for i, m in enumerate(conv)
        if m["from"] in ("gpt", "assistant") and "<tool_call>" in m["value"].lower()
    ]

    # No tool calls at all
    if not tool_indices:
        return False

    # Check sequences between tool calls
    for i in range(len(tool_indices) - 1):
        start, end = tool_indices[i], tool_indices[i + 1]
        # Messages strictly between two tool‑calling turns
        in_between = conv[start + 1 : end]
        # Only <tool_response> allowed between tool calls
        if any(m["from"] != "tool" for m in in_between):
            return False

    # For the last tool call, only check up to the next tool response
    # (allow narration after that)
    last_tool_idx = tool_indices[-1]
    next_responses = [
        i
        for i, m in enumerate(conv[last_tool_idx + 1 :], start=last_tool_idx + 1)
        if m["from"] == "tool"
    ]
    if not next_responses:  # No tool response after last tool call
        return False

    # Check sequence after last tool call up to its response
    last_response_idx = next_responses[0]
    in_between = conv[last_tool_idx + 1 : last_response_idx + 1]
    if any(m["from"] != "tool" for m in in_between):
        return False

    return True

def _detect_conversation_pattern(conv: List[Dict[str, str]]) -> Optional[str]:
    """
    Detect the conversation pattern and return the scenario category.

    Returns:
    - "single": exactly one tool-calling turn
    - "multistep": ≥2 sequential tool-calling turns, no human interruption
    - "multiturn": ≥1 tool-calling turn with later human interaction
    - None: if conversation doesn't match any valid pattern
    """
    # Find all tool-calling turns
    tool_indices = [
        i for i, m in enumerate(conv)
        if m["from"] in ("gpt", "assistant") and "<tool_call>" in m["value"].lower()
    ]

    if not tool_indices:
        return None

    # Find first assistant message
    first_assistant_idx = next(
        (i for i, m in enumerate(conv[2:], start=2)
        if m["from"] in ("gpt", "assistant")),
        None
    )

    # Check for human messages after first tool call
    human_after_first_tool = any(
        i > tool_indices[0] and m["from"] == "human"
        for i, m in enumerate(conv)
    )

    # Determine pattern
    if len(tool_indices) == 1:
        return "single"
    elif len(tool_indices) >= 2:
        if (first_assistant_idx == tool_indices[0]  # First assistant is tool call
            and not human_after_first_tool  # No human interruption
            and _check_sequential_tools(conv)):  # Follows sequential pattern
            return "multistep"
        elif human_after_first_tool:
            return "multiturn"

    return None

def _extract_inter_turn_context(
    conv: List[Dict[str, str]],
    current_idx: int,
    next_idx: int,
    scenario: str
) -> List[Dict[str, str]]:
    """
    Extract the context messages between two tool-calling turns.

    Args:
        conv: Full conversation
        current_idx: Index of current tool-calling turn
        next_idx: Index of next tool-calling turn (or None for last turn)
        scenario: The conversation scenario category

    Returns:
        List of context messages to include between turns
    """
    if scenario == "multistep":
        # Only include immediate tool response
        tool_response = conv[current_idx + 1]
        return [{
            "role": "tool",
            "content": tool_response["value"],
        }]
    else:
        # Include all messages until next tool call
        return [
            {"role": m["from"].replace("gpt", "assistant"),
            "content": m["value"]}
            for m in conv[current_idx:next_idx]
        ]

def _build_system_prompt(
    base_system: str,
    tools_block: str,
    dynamic_example: Optional[str] = None,
    scenario: str = "multistep"
) -> str:
    """
    Build the complete system prompt with appropriate components.

    Args:
        base_system: Base system message
        tools_block: The tools definition block
        dynamic_example: Optional dynamic few-shot example
        scenario: The conversation scenario category

    Returns:
        Complete system prompt
    """
    example = dynamic_example if dynamic_example else few_shot_example
    combined = f"{system_prompt}\n\n{base_system}\n\n{example}"

    # Add sequential helper for multistep scenario
    #if scenario == "multistep":
    #    combined += f"\n\n{SEQ_TOOL_HELPER}"

    return combined

class MultiTurnToolUseTurnLevelAdvantageEnv(BaseEnv):

    name = "multiturn_tool_use_turnlevel_advantage"

    def __init__(self, config: MTGRPOEnvConfig, server_configs: List[APIServerConfig],
                slurm: bool = True, testing: bool = False):
        super().__init__(config, server_configs, slurm, testing)
        #self.ds = load_dataset("interstellarninja/toolace_hermes_sequential_tool_use", split="train")
        self.ds = load_dataset("interstellarninja/hermes_salesforce_apigen_tool_use", split="train")

        # Existing buffers
        self.percent_correct_buffer: List[float] = []
        self.raw_score_buffer: List[float] = []
        self.eval_metrics: List[Tuple[str, float]] = []
        self.rollouts_for_wandb: List = []

        # New: track dynamic examples
        self.dynamic_example: Optional[str] = None

        # List of (messages_tuple, expected_calls_by_turn, inter_turns) triples
        self.train_items: List[
            Tuple[Tuple[frozenset, ...], List[List[str]], List[List[Dict[str, str]]]]
        ] = []
        self.test_items: List[
            Tuple[Tuple[frozenset, ...], List[List[str]], List[List[Dict[str, str]]]]
        ] = []
        self.iter = 0

    @classmethod
    def config_init(cls) -> Tuple[MTGRPOEnvConfig, List[APIServerConfig]]:
        env_cfg = MTGRPOEnvConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
            group_size=16,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=2000,
            batch_size=1024,
            steps_per_eval=20,
            max_token_length=1024 * 64,
            inference_weight=1.0,
            wandb_name="multiturn_tool_use_turnlevel_advantage",
            eval_handling=EvalHandlingEnum.LIMIT_TRAIN,
            eval_limit_ratio=0.1,
            # Override config defaults with experimental constants
            wrong_call_penalty=WRONG_CALL_PENALTY,
            max_gen_per_turn=MAX_GEN_PER_TURN,
            max_tool_call_turns_cap=MAX_TOOL_CALL_TURNS_CAP,  # Updated from max_tool_call_turns
            validate_think_blocks=VALIDATE_THINK_BLOCKS,
            generate_all_gpt_turns=GENERATE_ALL_GPT_TURNS,  # Added
            add_dynamic_few_shot=ADD_DYNAMIC_FEW_SHOT,  # Added
            scenario_category=SCENARIO_CATEGORY,  # Added
            use_parallel_requests=USE_PARALLEL_REQUESTS,
            turn_level_advantage_lambda=TURN_LEVEL_ADVANTAGE_LAMBDA,
        )
        server_cfgs = [
            APIServerConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
                base_url="http://localhost:9004/v1",
                api_key="x",
                num_max_requests_at_once=32,
                num_requests_for_eval=256,
            )
        ]
        return env_cfg, server_cfgs

    async def setup(self):
        ds = self.ds.shuffle()

        # Track statistics by scenario type
        counts = Counter()
        scenario_counts = {
            "single": Counter(),
            "multistep": Counter(),
            "multiturn": Counter()
        }

        for row in ds:
            conv = row["conversations"]
            num_turns = 0
            for msg in conv:
                if msg["from"] in ("gpt", "assistant") and re.search(
                    r"<tool_call>", msg["value"], re.IGNORECASE
                ):
                    num_turns += 1
            counts[num_turns] += 1

            # Categorize by scenario
            pattern = _detect_conversation_pattern(conv)
            if pattern:
                scenario_counts[pattern][num_turns] += 1

        print("Tool-call distribution (tool_calls_per_convo → examples):")
        for k in sorted(counts):
            print(f"  {k:2d} → {counts[k]} total")
            for scenario in ["single", "multistep", "multiturn"]:
                if k in scenario_counts[scenario]:
                    print(f"       {scenario}: {scenario_counts[scenario][k]}")

        split = ds.train_test_split(0.02)
        split["train"] = split["train"].shuffle()
        split["test"] = split["test"].shuffle()
        self._prep_items(split["train"], is_train=True)
        self._prep_items(split["test"], is_train=False)

    def _prep_items(self, dataset, *, is_train: bool):
        target = self.train_items if is_train else self.test_items
        before_len = len(target)

        for row in dataset:
            conv = row["conversations"]

            # Basic validation
            if len(conv) < 3 or conv[0]["from"] != "system" or conv[1]["from"] != "human":
                continue

            # Skip completed tasks
            if self.config.skip_completed and COMPLETED_TASKS:
                if conv[1]["value"].strip() in COMPLETED_TASKS:
                    continue

            # Detect conversation pattern
            pattern = _detect_conversation_pattern(conv)
            if pattern != self.config.scenario_category:
                continue

            # Build system prompt
            running_msgs = []
            system_content = _build_system_prompt(
                conv[0]["value"],
                self.dynamic_example if self.config.add_dynamic_few_shot else None,
                scenario=pattern
            )
            running_msgs.append(frozenset({"role": "system", "content": system_content}.items()))
            # Add user message with sequential helper for multistep
            user_content = conv[1]["value"]
            if pattern == "multistep":
                user_content = f"{user_content}\n\n{SEQ_TOOL_HELPER}"
            running_msgs.append(frozenset({"role": "user", "content": user_content}.items()))

            # Process tool-calling turns
            tool_indices = [
                i for i, m in enumerate(conv)
                if m["from"] in ("gpt", "assistant") and "<tool_call>" in m["value"].lower()
            ]

            expected_calls_by_turn = []
            inter_turns = []

            # Extract tool calls using direct regex like before
            for i, tool_idx in enumerate(tool_indices):
                matches = re.findall(
                    r"<tool_call>\s*(.*?)\s*</tool_call>",
                    conv[tool_idx]["value"],
                    re.DOTALL | re.IGNORECASE
                )

                # Don't skip on validation failure, try to parse what we can
                turn_calls = []
                for raw in matches:
                    try:
                        obj = json.loads(raw)
                        turn_calls.append(json.dumps(obj, separators=(",", ":")))
                    except Exception:
                        # If JSON parsing fails, use the raw string
                        turn_calls.append(raw)

                if turn_calls:  # Only add if we found any calls
                    expected_calls_by_turn.append(turn_calls)

                    # Build inter-turn context
                    if i < len(tool_indices) - 1:
                        inter_turn = _extract_inter_turn_context(
                            conv, tool_idx, tool_indices[i + 1], pattern
                        )
                        inter_turns.append(inter_turn)

            # Handle final summary turn if needed
            if self.config.generate_all_gpt_turns:
                last_response_idx = tool_indices[-1] + 1
                if (last_response_idx + 1 < len(conv)
                    and conv[last_response_idx + 1]["from"] in ("gpt", "assistant")
                    and "<tool_call>" not in conv[last_response_idx + 1]["value"]):
                    expected_calls_by_turn.append([])
                    inter_turns.append([{
                        "role": "tool",
                        "content": conv[last_response_idx]["value"]
                    }])

            # Apply turn cap
            if (self.config.max_tool_call_turns_cap is not None
                and len(expected_calls_by_turn) > self.config.max_tool_call_turns_cap):
                cut = self.config.max_tool_call_turns_cap
                expected_calls_by_turn = expected_calls_by_turn[:cut]
                inter_turns = inter_turns[:cut - 1]

            # Only add if we have valid tool calls for a multistep scenario
            if pattern == "multistep" and len(expected_calls_by_turn) >= 2:
                target.append((tuple(running_msgs), expected_calls_by_turn, inter_turns))

        print(f"[prep_items] {'train' if is_train else 'test'}: added {len(target)-before_len} items.")

    def _compute_turn_and_outcome_rewards(
        self,
        responses_by_turn: List[str],
        pred_calls_by_turn: List[List],
        expected_calls_by_turn: List[List[str]]
    ) -> Tuple[List[float], float]:
        """
        Compute turn-level rewards (R_T) and outcome-level reward (R_O).

        Turn-level rewards: Based on proper <think> blocks + <tool_call> blocks + tool call matches
        Outcome reward:
        - 1.0 if reached and validated final turn successfully
        - 0.5 if reached final turn but didn't validate perfectly
        - 0.0 if didn't reach final turn
        """
        turn_rewards = []
        completed_turns = 0

        # Process each turn that was attempted
        for turn_idx, (response, pred_turn) in enumerate(zip(responses_by_turn, pred_calls_by_turn)):
            expected_turn = expected_calls_by_turn[turn_idx]
            turn_reward = 0.0

            # Validate based on whether this turn expects tool calls
            if expected_turn:  # Tool-calling turn
                validation_result = _validate_think_plus_calls(response)
                has_valid_structure = validation_result is not None

                # Check tool call matches if structure is valid
                tool_calls_match = False
                if has_valid_structure and pred_turn:
                    # Handle early termination mismatch
                    actual_pred_turn = pred_turn[:-1] if pred_turn[-1] == "__MISMATCH__" else pred_turn

                    # Parse expected calls
                    exp_jsons = []
                    for raw in expected_turn:
                        try:
                            exp_jsons.append(json.loads(raw))
                        except json.JSONDecodeError:
                            exp_jsons.append(ast.literal_eval(raw))

                    # Check all tool calls match
                    if len(actual_pred_turn) == len(exp_jsons):
                        correct = sum(1 for p, e in zip(actual_pred_turn, exp_jsons)
                                    if _json_objects_match(p, e))
                        tool_calls_match = (correct == len(exp_jsons))

            else:  # Summary/narration turn
                has_valid_structure = _validate_think_only(response)
                tool_calls_match = True  # Not applicable for summary turns

            # Compute turn reward
            if has_valid_structure:
                turn_reward += 0.5  # Reward for proper structure
            if tool_calls_match:
                turn_reward += 0.5  # Reward for correct tool calls
                completed_turns += 1

            # Apply mismatch penalty if needed
            if pred_turn and pred_turn[-1] == "__MISMATCH__":
                turn_reward += self.config.wrong_call_penalty

            turn_rewards.append(turn_reward)

        # Fill in -1.0 rewards for turns that weren't attempted
        while len(turn_rewards) < len(expected_calls_by_turn):
            turn_rewards.append(-1.0)

        # Outcome reward: binary signal for MT-GRPO advantage computation
        outcome_reward = 1.0 if completed_turns == len(expected_calls_by_turn) else 0.0

        return turn_rewards, outcome_reward

    def _compute_mt_grpo_advantages(self, turn_rewards_batch: List[List[float]], outcome_rewards_batch: List[float]) -> List[List[float]]:
        """
        Compute MT-GRPO advantages following the paper:
        - Turn 1: A_T_1 + λ * A_O
        - Turn 2: A_T_2 + λ * A_O  
        - Turn 3: A_O (outcome only)
        
        Args:
            turn_rewards_batch: List of turn rewards for each rollout [num_rollouts x num_turns]
            outcome_rewards_batch: List of outcome rewards for each rollout [num_rollouts]
            
        Returns:
            List of advantages for each rollout [num_rollouts x num_turns]
        """
        if not turn_rewards_batch:
            return []

        # Find the maximum number of turns across all rollouts
        max_turns = max(len(rewards) for rewards in turn_rewards_batch)

        # Compute standardized turn advantages (A_T_i) for each turn
        turn_advantages_by_turn = []
        for turn_idx in range(max_turns):
            # Collect rewards for this turn across all rollouts
            turn_rewards = [
                rewards[turn_idx] if turn_idx < len(rewards) else 0.0
                for rewards in turn_rewards_batch
            ]
            # Standardize
            mean_turn = np.mean(turn_rewards)
            std_turn = np.std(turn_rewards)
            if std_turn == 0:
                std_turn = 1.0
            turn_advantages_by_turn.append((np.array(turn_rewards) - mean_turn) / std_turn)

        # Compute standardized outcome advantages (A_O)
        mean_outcome = np.mean(outcome_rewards_batch)
        std_outcome = np.std(outcome_rewards_batch)
        if std_outcome == 0:
            std_outcome = 1.0
        outcome_advantages = (np.array(outcome_rewards_batch) - mean_outcome) / std_outcome

        # Combine according to MT-GRPO formula
        mt_grpo_advantages = []
        for rollout_idx in range(len(turn_rewards_batch)):
            rollout_advantages = []
            actual_num_turns = len(turn_rewards_batch[rollout_idx])

            for turn_idx in range(actual_num_turns):
                if turn_idx < actual_num_turns - 1:  # Not the final turn
                    # A_T_i + λ * A_O for all non-final turns
                    adv = (turn_advantages_by_turn[turn_idx][rollout_idx] +
                        self.config.turn_level_advantage_lambda * outcome_advantages[rollout_idx])
                else:  # Final turn
                    # A_O only for final turn
                    adv = outcome_advantages[rollout_idx]
                rollout_advantages.append(adv)

            mt_grpo_advantages.append(rollout_advantages)
        
        return mt_grpo_advantages

    def _assign_advantages_to_tokens(self, contexts: List[List[Dict[str, str]]], advantages_by_turn: List[List[float]]) -> List[List[float]]:
        """
        Assign turn-level advantages to tokens in a turn-based manner.
        
        Our approach:
        - Each assistant message = one turn (tool call round)
        - Turn 1, 2, ..., N-1: A_T_i + λ * A_O (turn + outcome advantages)
        - Turn N (final): A_O only (outcome advantage)
        - Assign the turn's advantage to ALL tokens in that assistant message
        
        Args:
            contexts: List of conversation contexts for each rollout
            advantages_by_turn: List of advantages for each turn for each rollout
            
        Returns:
            List of per-token advantages for each rollout
        """
        per_token_advantages = []
        
        for rollout_idx, (context, turn_advantages) in enumerate(zip(contexts, advantages_by_turn)):
            # Tokenize the full conversation to get tokens and masks
            out = tokenize_for_trainer(
                tokenizer=self.tokenizer,
                chat=context,
                include_messages=self.config.include_messages,
            )
            tokens = out["tokens"]
            masks = out["masks"]
            
            # Initialize advantages for all tokens
            token_advantages = [0.0] * len(tokens)
            
            # Find assistant message boundaries and assign turn-specific advantages
            assistant_turn_idx = 0
            current_advantage = turn_advantages[0] if turn_advantages else 0.0
            
            # Track if we're currently in an assistant message
            in_assistant_msg = False
            assistant_start_idx = -1
            
            for i, (token_id, mask) in enumerate(zip(tokens, masks)):
                if mask != -100:  # Non-padding token
                    # Decode a small window to detect message boundaries
                    start_idx = max(0, i - 10)
                    end_idx = min(len(tokens), i + 10)
                    token_window = self.tokenizer.decode(tokens[start_idx:end_idx])
                    
                    # Detect start of new assistant message
                    if "assistant" in token_window.lower() and not in_assistant_msg:
                        in_assistant_msg = True
                        assistant_start_idx = i
                        # Use current turn advantage
                        if assistant_turn_idx < len(turn_advantages):
                            current_advantage = turn_advantages[assistant_turn_idx]
                    
                    # Detect end of assistant message (start of next role)
                    elif ("user" in token_window.lower() or "tool" in token_window.lower()) and in_assistant_msg:
                        # Apply advantage to all tokens in the assistant message we just finished
                        for j in range(assistant_start_idx, i):
                            if masks[j] != -100:
                                token_advantages[j] = current_advantage
                        
                        in_assistant_msg = False
                        assistant_turn_idx += 1
                    
                    # If we're in an assistant message, apply current advantage
                    elif in_assistant_msg:
                        token_advantages[i] = current_advantage
            
            # Handle case where conversation ends with assistant message
            if in_assistant_msg and assistant_start_idx != -1:
                for j in range(assistant_start_idx, len(tokens)):
                    if masks[j] != -100:
                        token_advantages[j] = current_advantage
            
            per_token_advantages.append(token_advantages)
        
        return per_token_advantages

    @staticmethod
    def _score_episode(pred_calls: list, exp_calls: list, lam: float = 0.5) -> float:
        """
        Legacy method for compatibility - now used primarily for evaluation.
        For training, we use turn-level rewards instead.
        """
        exp_jsons: List[dict] = []
        for raw in exp_calls:
            try:
                exp_jsons.append(json.loads(raw))
            except json.JSONDecodeError:
                exp_jsons.append(ast.literal_eval(raw))
        mismatch_penalty = 0.0
        if pred_calls and pred_calls[-1] == "__MISMATCH__":
            pred_calls = pred_calls[:-1]
            mismatch_penalty = self.config.wrong_call_penalty
        correct = sum(
            1 for p, e in zip(pred_calls, exp_jsons) if _json_objects_match(p, e)
        )
        dense = correct / max(1, len(exp_jsons))
        bonus = 1.0 if correct == len(exp_jsons) else 0.0
        return dense + lam * bonus + mismatch_penalty

    async def rollout_and_score_eval(self, item) -> float:
        messages_tuple, expected_calls_by_turn, inter_turns = item
        base_ctx = [dict(m) for m in messages_tuple]
        ctx = list(base_ctx)
        preds = []

        # Iterate through turns
        for turn_idx, expected_turn_calls in enumerate(expected_calls_by_turn):
            if turn_idx > 0 and turn_idx - 1 < len(inter_turns):
                ctx.extend(inter_turns[turn_idx - 1])

            prompt = self.tokenizer.apply_chat_template(ctx, add_generation_prompt=True, tokenize=False)
            max_toks = max(1, self.config.max_token_length - len(prompt))
            comp = await self.server.completion(
                prompt=prompt,
                n=1,
                max_tokens=max_toks,  # Use computed max_toks
                temperature=0.0,
                split="eval"
            )
            reply = comp.choices[0].text
            reply = _normalize_tool_call_json(reply)  # Important: normalize first

            # Validate based on whether this turn expects tool calls
            if expected_turn_calls:  # Tool-calling turn
                tool_jsons = _validate_think_plus_calls(reply)
                if tool_jsons is None:
                    break
                preds.extend(tool_jsons)
            else:  # Summary turn
                if not _validate_think_only(reply):
                    break

            # Only append valid replies to context
            ctx.append({"role": "assistant", "content": reply})

            # Check if we've processed enough turns
            if turn_idx >= len(expected_calls_by_turn) - 1:
                break
        
        # Flatten expected calls for scoring
        expected_calls_flat = [call for turn_calls in expected_calls_by_turn for call in turn_calls]
        score = self._score_episode(
            preds,
            expected_calls_flat,
            wrong_call_penalty=self.config.wrong_call_penalty
        )
        return score

    async def evaluate(self, *_, **__):
        subset = self.test_items[: min(128, len(self.test_items))]
        scores = await tqdm_asyncio.gather(*[self.rollout_and_score_eval(it) for it in subset])
        avg_reward = sum(scores) / len(scores)
        pct_exact = sum(1 for s in scores if s >= 1.0) / len(scores)
        self.eval_metrics.append(("eval/avg_reward", avg_reward))
        self.eval_metrics.append(("eval/percent_correct", pct_exact))

    async def get_next_item(self):
        """
        Return the next training item in a strictly sequential (non‐wrapping) order.
        Once we've gone through all items, reshuffle and start over.
        """
        if not self.train_items:
            raise ValueError("train_items is empty – dataset preparation failed.")

        if self.iter >= len(self.train_items):
            random.shuffle(self.train_items)
            self.iter = 0

        itm = self.train_items[self.iter]
        self.iter += 1
        return itm

    async def _build_turn_contexts(
        self,
        turn_idx: int,
        contexts: List[List[Dict[str, str]]],
        inter_turns: List[List[Dict[str, str]]],
        active: List[bool],
    ) -> Tuple[List[str], List[int]]:
        """Build prompts for the current turn from active rollout contexts."""
        # Add inter-turn context if not the first turn
        if turn_idx > 0 and turn_idx - 1 < len(inter_turns):
            filler = inter_turns[turn_idx - 1]
            for r in range(len(contexts)):
                if active[r]:
                    contexts[r].extend(filler)

        # Build prompts for active rollouts
        prompts, ridx_map = [], []
        for r in range(len(contexts)):
            if not active[r]:
                continue
            ptxt = self.tokenizer.apply_chat_template(
                contexts[r],
                add_generation_prompt=True,
                tokenize=False,
            )
            prompts.append(ptxt)
            ridx_map.append(r)

        return prompts, ridx_map

    async def _execute_turn_inference(
        self,
        turn_idx: int,
        prompts: List[str],
        ridx_map: List[int],
        expected_calls_by_turn: List[List[str]]
    ) -> List[str]:
        """Execute inference for a turn using optimal batching strategy."""
        print(f"\n\033[95m=== Expected Tool Calls for Turn {turn_idx+1} ===\033[0m")
        print(f"\033[95m{expected_calls_by_turn[turn_idx]}\033[0m\n")

        if turn_idx == 0 and not self.config.use_parallel_requests:
            choices = await self._batch_identical_prompts(prompts[0], len(ridx_map), turn_idx)
        else:
            choices = await self._batch_heterogeneous_prompts(prompts, turn_idx)

        return choices

    async def _batch_identical_prompts(self, prompt: str, count: int, turn_idx: int) -> List[str]:
        """Handle identical prompts efficiently using n parameter."""
        print(f"    \033[93m→ TURN {turn_idx+1} prompt full:\033[0m \033[92m{prompt}\033[0m")

        gen_limit = self.config.max_gen_per_turn
        resp = await self.server.completion(
            prompt=prompt,
            n=count,
            max_tokens=gen_limit,
            temperature=0.8,
        )
        choices = [c.text for c in resp.choices]
        
        # Debug: print each rollout
        for i, raw in enumerate(choices):
            print(f"    \033[93m· turn {turn_idx+1} rollout raw [{i}]:\033[0m \033[94m{raw}\033[0m")
            if not raw.strip():
                print(f"      → (empty or error string returned for rollout {i})")
        print("    → All turn 1 rollouts printed; moving on.\n" + "-"*48)
        
        return choices

    async def _batch_heterogeneous_prompts(self, prompts: List[str], turn_idx: int) -> List[str]:
        """Handle heterogeneous prompts using parallel requests."""
        if turn_idx == 1:
            print("=== DEBUG: Now parallelizing Turn 2 prompts ===")
        print(f"    → Parallelizing {len(prompts)} prompts at turn {turn_idx+1}")
        
        # Print each prompt
        for idx_p, p_str in enumerate(prompts):
            print(f"    \033[93m→ TURN-{turn_idx+1} prompt[{idx_p}] full:\033[0m \033[92m{p_str}\033[0m")

        async def _call_single(prompt_str: str) -> str:
            try:
                gen_limit = self.config.max_gen_per_turn
                comp = await self.server.completion(
                    prompt=prompt_str,
                    n=1,
                    max_tokens=gen_limit,
                    temperature=0.8,
                )
                return comp.choices[0].text
            except Exception as e:
                print(f"    → Turn {turn_idx+1} _call_single exception: {e}")
                return ""

        tasks = [_call_single(p) for p in prompts]
        results = await asyncio.gather(*tasks)

        # Debug: print results for all turns
        choices = []
        for i, rtext in enumerate(results):
            raw = rtext or ""
            print(f"    \033[93m· rollout {i} (Turn {turn_idx+1}) full reply:\033[0m \033[94m{raw}\033[0m\n" + "-"*48)
            if not raw:
                print(f"    → Rollout {i} returned empty or error string")
            choices.append(raw)
        
        return choices

    async def _process_turn_responses(
        self,
        turn_idx: int,
        choices: List[str],
        ridx_map: List[int],
        contexts: List[List[Dict[str, str]]],
        preds_by_turn: List[List[List]],
        responses_by_turn: List[List[str]],
        active: List[bool],
        expected_calls_by_turn: List[List[str]]
    ) -> None:
        """Process and validate responses for a single turn."""
        for txt, r in zip(choices, ridx_map):
            print(f"\n\033[93m=== Processing Response {r} ===\033[0m")
            raw_txt = txt or ""
            norm_txt = _normalize_tool_call_json(raw_txt)

            # Store response for reward computation
            responses_by_turn[r].append(norm_txt)

            expected_turn_calls = expected_calls_by_turn[turn_idx]
            is_valid = False

            if expected_turn_calls:  # Turn SHOULD have tool calls
                calls = _validate_think_plus_calls(norm_txt)
                print(f"\033[95mExtracted tool calls:\033[0m {calls}")
                print(f"\033[95mExpected tool calls:\033[0m {expected_turn_calls}")

                if calls is None:
                    print("\033[91m[DEBUG] Invalid tool call turn: missing <think> or <tool_call>\033[0m")
                    preds_by_turn[r][turn_idx].append("__MISMATCH__")
                    active[r] = False
                    continue

                # Check number of calls and content matches
                if len(calls) != len(expected_turn_calls):
                    print(f"\033[91m[DEBUG] Call‑count mismatch — model={len(calls)} vs exp={len(expected_turn_calls)}\033[0m")
                    preds_by_turn[r][turn_idx].append("__MISMATCH__")
                    active[r] = False
                    continue

                mismatch = False
                for mdl, exp_raw in zip(calls, expected_turn_calls):
                    try:
                        exp_obj = json.loads(exp_raw)
                    except Exception:
                        exp_obj = ast.literal_eval(exp_raw)
                    if not _json_objects_match(mdl, exp_obj):
                        mismatch = True
                        break

                if mismatch:
                    print("\033[91m[DEBUG] Tool‑call field mismatch detected\033[0m")
                    preds_by_turn[r][turn_idx].append("__MISMATCH__")
                    active[r] = False
                else:
                    print("\033[92m[DEBUG] Valid tool call turn\033[0m")
                    is_valid = True
                    # Store normalized JSON versions
                    preds_by_turn[r][turn_idx].extend([json.dumps(call, separators=(',', ':')) for call in calls])
                    if self.config.add_dynamic_few_shot and calls:
                        self.dynamic_example = norm_txt.strip()

            else:  # Narration / summary turn
                print(f"\033[95mValidating summary turn for rollout {r}\033[0m")
                if not _validate_think_only(norm_txt):
                    print(f"\033[91m[DEBUG] Invalid summary: missing <think> or contains <tool_call>\033[0m")
                    active[r] = False
                else:
                    print(f"\033[92m[DEBUG] Valid summary turn - keeping rollout active\033[0m")
                    is_valid = True
                    preds_by_turn[r][turn_idx] = []  # Empty list for summary turn

            # Only append to context if validation passed
            if is_valid:
                print(f"\033[92m[DEBUG] Adding response to context for rollout {r}\033[0m")
                contexts[r].append({"role": "assistant", "content": norm_txt})

    async def collect_trajectories(
        self,
        item: Tuple[
            Tuple[frozenset, ...],
            List[List[str]],
            List[List[Dict[str, str]]],
        ],
    ) -> Tuple[Optional[ScoredDataGroup], List[Item]]:
        """
        Roll-out multi-turn tool-calling with turn-level advantage computation.
        """
        messages_tuple, expected_calls_by_turn, inter_turns = item
        base_ctx = [dict(m) for m in messages_tuple]

        num_rollouts = self.config.group_size
        contexts: List[List[Dict[str, str]]] = [list(base_ctx) for _ in range(num_rollouts)]
        preds_by_turn: List[List[List]] = [[[] for _ in range(len(expected_calls_by_turn))] for _ in range(num_rollouts)]
        responses_by_turn: List[List[str]] = [[] for _ in range(num_rollouts)]
        active = [True] * num_rollouts

        max_turns = len(expected_calls_by_turn)

        for turn_idx in range(max_turns):
            prompts, ridx_map = await self._build_turn_contexts(turn_idx, contexts, inter_turns, active)

            # Only break if no prompts AND no active rollouts completed all turns
            if not prompts:
                break

            # Execute inference for this turn
            choices = await self._execute_turn_inference(
                turn_idx,
                prompts,
                ridx_map,
                expected_calls_by_turn
            )

            # Process and validate responses
            await self._process_turn_responses(
                turn_idx, choices, ridx_map, contexts, preds_by_turn,
                responses_by_turn, active, expected_calls_by_turn
            )

            if not any(active):
                break

        # Create scored group even if some rollouts failed
        scored = ScoredDataGroup(tokens=[], masks=[], scores=[], advantages=[])

        for r in range(num_rollouts):
            # Compute turn and outcome rewards
            turn_rewards, outcome_reward = self._compute_turn_and_outcome_rewards(
                responses_by_turn[r], preds_by_turn[r], expected_calls_by_turn
            )

            # Compute MT-GRPO advantages
            mt_grpo_advantages = self._compute_mt_grpo_advantages(
                [turn_rewards], [outcome_reward]
            )[0]  # Get advantages for this rollout

            # Tokenize and store results
            out = tokenize_for_trainer(
                tokenizer=self.tokenizer,
                chat=contexts[r],
                include_messages=self.config.include_messages,
            )
            scored["tokens"].append(out["tokens"])
            scored["masks"].append(out["masks"])
            scored["scores"].append(sum(turn_rewards))
            scored["advantages"].append(
                self._assign_advantages_to_tokens([contexts[r]], [mt_grpo_advantages])[0]
            )

        # Apply length penalty if needed
        if scored["scores"]:
            if all(s > 0.99 for s in scored["scores"]):
                cutoff = self.config.max_token_length * 0.5
                for i, ln in enumerate([len(t) for t in scored["tokens"]]):
                    if ln > cutoff:
                        frac = min((ln - cutoff) / (self.config.max_token_length - cutoff), 1.0)
                        scored["scores"][i] = max(0.0, scored["scores"][i] - frac)

        # Update metrics
        for s in scored["scores"]:
            self.raw_score_buffer.append(s)
            self.percent_correct_buffer.append(1.0 if s >= len(expected_calls_by_turn) else 0.0)

        # Debug prints for group scoring
        print("\n\033[95m=== Group Score Distribution ===\033[0m")
        print(f"\033[96mScores: {scored['scores']}\033[0m")
        print(f"\033[96mActive rollouts at end: {sum(1 for a in active if a)}\033[0m")
        print(f"\033[96mSuccessful completions: {sum(1 for s in scored['scores'] if s > 0.99)}\033[0m")
        print(f"\033[96mTotal rollouts: {len(scored['scores'])}\033[0m")

        # Group validation with float sums
        if len(scored["tokens"]) < self.config.group_size or not any(s >= 1.0 for s in scored["scores"]):
            return None, []

        await self.add_rollouts_for_wandb(scored, item)
        return scored, []

    async def create_rollout_table(self, wdict):
        if self.rollouts_for_wandb:
            table = wandb.Table(columns=["generation", "score", "expected_tool_call"])
            for grp in self.rollouts_for_wandb:
                for g, sc, exp in grp:
                    exp_str = json.dumps(exp, separators=(",", ":"))
                    table.add_data(g, sc, exp_str)
            wdict["train/rollouts"] = table
        self.rollouts_for_wandb = []
        return wdict

    async def add_rollouts_for_wandb(
        self, scored: ScoredDataGroup, item: Item, *, num_keep: int = 1
    ):
        num_keep = min(num_keep, len(scored["tokens"]))
        # Flatten expected calls for wandb logging
        expected_calls_flat = [call for turn_calls in item[1] for call in turn_calls]
        self.rollouts_for_wandb.append(
            [
                (
                    self.tokenizer.decode(scored["tokens"][i]),
                    scored["scores"][i],
                    expected_calls_flat,
                )
                for i in range(num_keep)
            ]
        )
        if len(self.rollouts_for_wandb) > self.config.num_rollouts_to_keep:
            self.rollouts_for_wandb.pop(0)

    async def wandb_log(self, metrics: Optional[Dict] = None):
        metrics = metrics or {}
        metrics = await self.create_rollout_table(metrics)
        if self.raw_score_buffer:
            avg_reward = sum(self.raw_score_buffer) / len(self.raw_score_buffer)
            pct_correct = (
                sum(self.percent_correct_buffer) / len(self.percent_correct_buffer)
            )
            metrics["train/avg_reward"] = avg_reward
            metrics["train/percent_correct"] = pct_correct
            self.raw_score_buffer.clear()
            self.percent_correct_buffer.clear()
        for k, v in self.eval_metrics:
            metrics[k] = v
        await super().wandb_log(metrics)


if __name__ == "__main__":
    MultiTurnToolUseTurnLevelAdvantageEnv.cli()
