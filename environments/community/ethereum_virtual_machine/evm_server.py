#!/usr/bin/env python3
"""
EVM Environment for Atropos: Ethereum Virtual Machine Transaction Agent Training

This environment trains language models to generate and execute profitable Ethereum transactions
using Anvil (Foundry's local blockchain simulation).
"""

import json
import logging
import os
import random
import re
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple

from anvil import AnvilBackend, AnvilConfig
from evm_config import EVMEnvConfig
from openai import OpenAI
from utils import cleanup_blockchain, cleanup_manager, setup_evm_error_message

from atroposlib.envs.base import BaseEnv, ScoredDataGroup
from atroposlib.envs.server_handling.server_manager import APIServerConfig
from atroposlib.type_definitions import Item
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

# Add logger
logger = logging.getLogger(__name__)

# System prompt for EVM transaction agent
system_prompt = (
    "You are a deep thinking AI, you may use extremely long chains of thought "
    "to deeply consider the problem and deliberate with yourself via systematic "
    "reasoning processes to help come to a correct solution prior to answering. "
    "You should enclose your thoughts and internal monologue inside <think> </think> "
    "tags, and then provide your solution or response to the problem.\n\n"
)

system_prompt += """You are allowed to use a maximum of 2048 tokens. Please strive to use less.

You are here to assist a user execute transfers of both ETH and ERC-20 tokens as requested.
Your job is to generate correct Ethereum transaction data for the requested action.

IMPORTANT: After your thinking, your response must include a valid JSON transaction object:
{"to": "0x...", "value": "amount_in_wei", "data": "0x..."}

- 'to': The recipient address (contract or EOA)
- 'value': Amount of ETH to send in wei (string)
- 'data': Transaction data

If you do not provide a valid JSON transaction object, your submission will be ignored and you \
will receive a score of -1.0.

Example 1:
{
    "to": "0xe688b84b23f322a994A53dbF8E15FA82CDB71127",
    "value": "0.01",
    "data": "0x"
}

Example 2:
{
    "to": "0xEA29e9da69317d80075fBfc836E843C6d65971F5",
    "value": "0x",
    "data": "0xa9059cbb000000000000000000000000ea29e9da69317d80075fbfc836e843c6d65971f50000000000000000000000000000000000000000000000000000000005f5e100"  # noqa: E501
}
"""


class EVMEnv(BaseEnv):
    """EVM Transaction Environment for training agents to interact with Ethereum"""

    name = "evm_agent"
    env_config_cls = EVMEnvConfig

    def __init__(
        self,
        config: EVMEnvConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        """Initialize the EVM environment"""
        super().__init__(config, server_configs, slurm, testing)

        # Set up minimal logging - only for essential operations
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.logger.setLevel(logging.WARNING)  # Only warnings and errors
        self.logger.propagate = False
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(message)s")  # Clean format
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Suppress base environment logs
        if config.suppress_base_env_logs:
            base_logger = logging.getLogger("atroposlib.envs.base")
            base_logger.setLevel(logging.WARNING)

        # Load Anvil configuration
        self.anvil_config = AnvilConfig(config.anvil_config_path)

        # Initialize blockchain handler
        self.blockchain = AnvilBackend(self.anvil_config)

        # Performance tracking for adaptive question selection
        self.question_performance = {qtype: [] for qtype in config.question_types}
        self.current_question_type = None

        # Store current prompt data for scoring
        self.current_prompt_data = None

        # Register cleanup with the global cleanup manager
        cleanup_manager.register_cleanup(cleanup_blockchain, self.blockchain)

    async def setup(self):
        """Setup the EVM environment and start Anvil"""
        try:
            print("Starting Anvil blockchain simulation...")
            self.blockchain.start()
            self.blockchain.setup_wallet()
            print("EVM environment setup completed successfully.")
        except Exception as e:
            error_message = setup_evm_error_message(self.anvil_config, e)
            print(error_message)

            # Cleanup and exit
            cleanup_blockchain(self.blockchain)
            sys.exit(1)

    async def get_next_item(self) -> Optional[Item]:
        """Generate the next transaction challenge for the agent"""
        try:
            # Select question type based on performance (exploration vs exploitation)
            question_type = self._select_question_type()
            self.current_question_type = question_type

            # Generate question prompt and get structured data
            prompt_text, prompt_data = await self._generate_question_prompt(
                question_type
            )

            # Store the prompt data for scoring
            self.current_prompt_data = prompt_data

            # Display Generated Input
            self.logger.debug("\n=== Generated Input ===")
            self.logger.debug(prompt_text)
            self.logger.debug("=" * 50)

            prompt = tuple(
                [frozenset({"role": "user", "content": prompt_text}.items())]
            )

            return (prompt, None, None)

        except Exception as e:
            print(f"Error in get_next_item: {e}")
            traceback.print_exc()
            return None

    def _select_question_type(self) -> str:
        """Select question type using weakness-targeting strategy with 80/20 ratio"""
        # If no performance data yet, select randomly
        if not any(self.question_performance.values()):
            return random.choice(self.config.question_types)

        # Calculate average scores for each question type
        avg_scores = {}
        for qtype, scores in self.question_performance.items():
            if scores:
                avg_scores[qtype] = sum(scores) / len(scores)
            else:
                avg_scores[qtype] = 0.0  # Prioritize untested question types

        # Split into weak and strong areas based on configurable performance threshold
        weak_threshold = self.config.weak_performance_threshold

        weak_qtypes = [
            qtype for qtype, score in avg_scores.items() if score < weak_threshold
        ]
        strong_qtypes = [
            qtype for qtype, score in avg_scores.items() if score >= weak_threshold
        ]

        # Configurable focus on weak areas vs strong areas for mastery maintenance
        if random.random() < self.config.weak_area_focus_ratio and weak_qtypes:
            selected_type = random.choice(weak_qtypes)
        elif strong_qtypes:
            selected_type = random.choice(strong_qtypes)
        else:
            selected_type = random.choice(list(avg_scores.keys()))

        return selected_type

    async def _generate_question_prompt(
        self, question_type: str
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Generate a dynamic question prompt using LLM based on the question type"""

        # Initialize OpenAI client
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        # Create prompt for LLM to generate a request
        llm_prompt = f"""You are generating a natural language transaction request for an Ethereum blockchain agent.

TRANSACTION TYPE: "{question_type}"

CONTEXT:
- Wallet Address: {self.anvil_config.funding.custom_wallet}
- Current Balances: {json.dumps(self.blockchain.get_wallet_balances(), indent=2)}

TASK:
Generate a realistic, conversational transaction request that:
1. Matches the specified transaction type exactly
2. Does not use more than current wallet balances and typically would be small transfers or possibly larger,
   like how a real person may use their assets
3. Includes all necessary details (token, amount, destination address)
4. Sounds like how a real user would naturally request a transaction
5. Varies in tone and style (casual, formal, urgent, etc.)

REQUIREMENTS:
- Use realistic destination addresses (not placeholder text like
  "0x1234567890123456789012345678901234567890")
- Does not specify amounts larger than 50% of the current balance
- Make the request executable

Generate ONE natural language request that matches the transaction type "{question_type}".
Respond with a JSON object with the following fields:
- question_type: The type of transaction to generate
- request: The natural language request text
- destination_address: The destination address
- transfer_token: The token to transfer
- transfer_amount: The amount to transfer

Examples:
1. ETH transfer
{{
    "question_type": "ETH transfer",
    "request": "yo, can i send 0.01 ETH to my buddy jasper?  His address is jasper.eth"
    "destination_address": "jasper.eth"
    "transfer_token": "ETH"
    "transfer_amount": "0.01"
}}
2. ERC-20 transfer using 18 decimal token
{{
    "question_type": "ERC-20 transfer using 18 decimal token",
    "request": "Send 100 CRV to 0xe688b84b23f322a994A53dbF8E15FA82CDB71127"
    "destination_address": "0xe688b84b23f322a994A53dbF8E15FA82CDB71127"
    "transfer_token": "CRV"
    "transfer_amount": "100"
}}
3. ERC-20 transfer using a non-18 decimal token (e.g. USDT)
{{
    "question_type": "ERC-20 transfer using a non-18 decimal token",
    "request": "give 100 tether to 0xea29e9da69317d80075fbfc836e843c6d65971f5"
    "destination_address": "0xea29e9da69317d80075fbfc836e843c6d65971f5"
    "transfer_token": "USDT"
    "transfer_amount": "100"
}}
"""

        try:
            # Generate multiple responses in a single call for efficiency
            response = client.chat.completions.create(
                model=self.config.question_generation_model,
                messages=[{"role": "user", "content": llm_prompt}],
                temperature=self.config.question_generation_temperature,
                max_tokens=self.config.question_generation_max_tokens,
                n=self.config.question_generation_n,
            )

            # Try each response until we find a valid one
            for i, choice in enumerate(response.choices):
                generated_content = choice.message.content.strip()

                # Extract JSON from response using generic function
                prompt_data = self._extract_json_from_response(
                    generated_content, ["question_type", "request"], "prompt"
                )

                # Validate required fields
                if prompt_data and self._validate_prompt_data(
                    prompt_data, question_type
                ):
                    return prompt_data["request"], prompt_data

            # All choices failed, use fallback
            fallback_data = {
                "question_type": question_type,
                "request": "Transfer 0.01 ETH to 0x0000000000000000000000000000000000000000",
                "destination_address": "0x0000000000000000000000000000000000000000",
                "transfer_token": "ETH",
                "transfer_amount": "0.01",
            }
            return fallback_data["request"], fallback_data

        except Exception:
            fallback_data = {
                "question_type": question_type,
                "request": "Transfer 0.01 ETH to 0x0000000000000000000000000000000000000000",
                "destination_address": "0x0000000000000000000000000000000000000000",
                "transfer_token": "ETH",
                "transfer_amount": "0.01",
            }
            return fallback_data["request"], fallback_data

    def _extract_json_from_response(
        self, response: str, required_keys: List[str], json_type: str = "JSON"
    ) -> Optional[Dict[str, Any]]:
        """Generic JSON extraction from LLM response, handling thinking tags"""
        if not isinstance(response, str):
            return None

        # First, try to extract content after thinking tags (following SWE pattern)
        content_after_think = response
        think_end_match = re.search(r"</think>", response, re.IGNORECASE)
        if think_end_match:
            content_after_think = response[think_end_match.end() :].strip()

        # Create patterns based on required keys
        if len(required_keys) >= 2:
            key1, key2 = required_keys[0], required_keys[1]
            json_patterns = [
                rf'\{{[^{{}}]*"{key1}"[^{{}}]*"{key2}"[^{{}}]*\}}',  # Simple pattern with first two keys
                rf'\{{.*?"{key1}".*?"{key2}".*?\}}',  # Flexible pattern with first two keys
                r"\{.*?\}",  # Any JSON object
            ]
        else:
            json_patterns = [r"\{.*?\}"]  # Fallback to any JSON

        for pattern in json_patterns:
            matches = re.findall(pattern, content_after_think, re.DOTALL)
            for match in matches:
                try:
                    # Clean up the JSON string
                    json_str = match.strip()

                    # Parse the JSON
                    obj = json.loads(json_str)

                    # Verify it has the expected structure
                    if isinstance(obj, dict) and required_keys[0] in obj:
                        return obj

                except json.JSONDecodeError:
                    continue

        return None

    def _extract_transaction_json(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract transaction JSON from LLM response"""
        return self._extract_json_from_response(
            response, ["to", "value"], "transaction"
        )

    def _validate_prompt_data(
        self, prompt_data: Dict[str, Any], expected_question_type: str
    ) -> bool:
        """Validate that prompt data has all required fields and correct question type"""
        required_fields = [
            "question_type",
            "request",
            "destination_address",
            "transfer_token",
            "transfer_amount",
        ]

        # Check all required fields are present
        if not all(field in prompt_data for field in required_fields):
            return False

        # Check question type matches what we requested
        if prompt_data["question_type"] != expected_question_type:
            return False

        # Check that fields are not empty
        for field in required_fields:
            if not prompt_data[field] or str(prompt_data[field]).strip() == "":
                return False

        return True

    async def collect_trajectories(
        self, item: Item
    ) -> Tuple[Optional[ScoredDataGroup], List[Item]]:
        """Collect trajectories by having the agent generate transactions"""
        to_score = []
        to_backlog = []

        system_msg = {
            "role": "system",
            "content": system_prompt,
        }

        user_msg = {"role": "user", "content": dict(item[0][0])["content"]}

        messages = [system_msg, user_msg]

        try:
            # Use proper Atropos framework pattern like humor generation
            chat_completions = await self.server.chat_completion(
                messages=messages,
                n=self.config.group_size,
                max_tokens=2048,
            )

            # Store completions for output saving
            self.last_completions = []

            for i, choice in enumerate(chat_completions.choices):
                # Store the completion
                self.last_completions.append(choice.message.content)

                # Display Generated Output
                self.logger.debug(f"\n=== Generated Output {i+1} ===")
                self.logger.debug(choice.message.content)
                self.logger.debug("=" * 50)

                history = [
                    {"role": "system", "content": system_msg["content"]},
                    {"role": "user", "content": user_msg["content"]},
                    {"role": "assistant", "content": choice.message.content},
                ]
                to_score.append((history, item[1], None))

        except Exception as e:
            print(f"Error in collect_trajectories: {e}")
            traceback.print_exc()
            to_backlog.append(item)

        if not to_score:
            return None, to_backlog

        scored_data = await self.score(to_score)
        return scored_data, to_backlog

    async def score(self, rollout_group_data) -> Optional[ScoredDataGroup]:
        """Score the generated transactions by executing them on Anvil"""
        if not rollout_group_data:
            return None

        scores = ScoredDataGroup()
        scores["tokens"] = []
        scores["masks"] = []
        scores["scores"] = []
        scores["advantages"] = None
        scores["ref_logprobs"] = None
        scores["messages"] = None
        scores["group_overrides"] = {"group_size": self.config.group_size}
        scores["overrides"] = None
        scores["ground_truths"] = []

        for i, item in enumerate(rollout_group_data):
            out = tokenize_for_trainer(self.tokenizer, item[0])
            tokens = out["tokens"]
            masks = out["masks"]

            try:
                # Extract the agent's response (transaction JSON)
                agent_response = item[0][-1]["content"].strip()
                ground_truth = item[1] if isinstance(item[1], str) else ""

                # Score the transaction
                score = await self._score_transaction(agent_response)

                # Display Score
                self.logger.debug(f"\n=== Score {i+1} ===")
                self.logger.debug(f"{score}")
                self.logger.debug("=" * 50)

                # Track performance for this question type
                if self.current_question_type:
                    self.question_performance[self.current_question_type].append(score)
                    # Keep only last 10 scores per question type
                    if len(self.question_performance[self.current_question_type]) > 10:
                        self.question_performance[self.current_question_type].pop(0)

            except Exception as e:
                score = -1.0
                ground_truth = item[1] if isinstance(item[1], str) else ""

                # Display Score for error case
                print(f"\n=== Score {i+1} ===")
                print(f"{score} (Error: {e})")
                print("=" * 50)

            # Skip if too few tokens
            if len([i for i in masks if i != -100]) < 10:
                continue

            scores["tokens"].append(tokens)
            scores["masks"].append(masks)
            scores["scores"].append(score)
            scores["ground_truths"].append(ground_truth)

            if len(scores["tokens"]) >= self.config.group_size:
                break

        if not scores["tokens"]:
            return None

        return scores

    async def _score_transaction(self, agent_response: str) -> float:
        """Score a transaction based on multiple criteria"""
        try:
            # First, extract JSON from the response (handling thinking tags)
            tx_obj = self._extract_transaction_json(agent_response)
            if tx_obj is None:
                return -1.0  # Could not extract valid JSON

            # Validate required fields
            if not all(field in tx_obj for field in ["to", "value", "data"]):
                return -1.0  # Missing required fields

            # Get expected transfer details from stored prompt data
            if not hasattr(self, "current_prompt_data") or not self.current_prompt_data:
                return -1.0

            expected_token = self.current_prompt_data.get("transfer_token", "ETH")
            expected_amount = self.current_prompt_data.get("transfer_amount", "0")
            expected_destination = self.current_prompt_data.get(
                "destination_address", ""
            )

            # Get sender and destination addresses
            sender_address = self.anvil_config.funding.custom_wallet
            destination_address = tx_obj.get("to", "")

            # Get relevant tokens to check
            relevant_tokens = ["ETH"]
            if expected_token != "ETH":
                relevant_tokens.append(expected_token)

            # Take a snapshot before execution
            snapshot_id = self.blockchain.snapshot()

            # Get pre-execution balances for both addresses
            pre_balances = {
                "sender": self.blockchain.get_wallet_balances(
                    sender_address, relevant_tokens
                ),
                "destination": self.blockchain.get_wallet_balances(
                    destination_address, relevant_tokens
                ),
            }

            try:
                # Execute the transaction
                result = self.blockchain.execute_transaction(tx_obj)

                # Get post-execution balances
                post_balances = {
                    "sender": self.blockchain.get_wallet_balances(
                        sender_address, relevant_tokens
                    ),
                    "destination": self.blockchain.get_wallet_balances(
                        destination_address, relevant_tokens
                    ),
                }

                # Calculate score based on execution result and balance changes
                score = self._calculate_transaction_score(
                    tx_obj,
                    result,
                    agent_response,
                    pre_balances,
                    post_balances,
                    expected_token,
                    expected_amount,
                    expected_destination,
                )

                # Revert to snapshot to maintain clean state
                self.blockchain.revert(snapshot_id)

                return score

            except Exception:
                # Revert on any error
                self.blockchain.revert(snapshot_id)
                return -1.0  # Execution error

        except Exception:
            return -1.0  # General error

    def _calculate_transaction_score(
        self,
        tx_obj: Dict[str, Any],
        result: Dict[str, Any],
        agent_response: str,
        pre_balances: Dict[str, Dict[str, Any]],
        post_balances: Dict[str, Dict[str, Any]],
        transfer_token: str,
        transfer_amount: str,
        destination_address: str,
    ) -> float:
        """Calculate score based on transaction execution results and balance changes"""
        base_score = 0.0

        # 1. Successful execution (0.3 points)
        if result.get("status") == "0x1":
            base_score += 0.3  # Transaction succeeded

        # 2. Correct transaction - exact balance verification (0.5 points)
        balance_score = self._verify_expected_transfers(
            pre_balances,
            post_balances,
            transfer_token,
            transfer_amount,
            destination_address,
        )
        base_score += balance_score

        # 3. Thinking quality (max 0.1 points, with negative for missing thinking)
        thinking_score = self._analyze_thinking_quality(agent_response)
        base_score += thinking_score  # Range: -0.2 to +0.1

        # 4. To field verification (0.05 points)
        tx_to = tx_obj.get("to", "").lower()
        expected_to = destination_address.lower()
        if tx_to == expected_to:
            base_score += 0.05

        # 5. Data field verification (0.05 points)
        data_field = tx_obj.get("data", "0x")
        if transfer_token == "ETH":
            # ETH transfer should have empty data field
            if data_field == "0x":
                base_score += 0.05
        else:
            # ERC-20 transfer should have transfer function call data
            if data_field.startswith("0xa9059cbb"):  # transfer function selector
                base_score += 0.05

        return base_score

    def _verify_expected_transfers(
        self,
        pre_balances: Dict[str, Dict[str, Any]],
        post_balances: Dict[str, Dict[str, Any]],
        expected_token: str,
        expected_amount: str,
        expected_destination: str,
    ) -> float:
        """Verify that the expected transfer amounts occurred"""
        try:
            expected_amount_float = float(expected_amount)

            # For ETH transfers - only check destination address
            if expected_token == "ETH":
                # Extract balance values from the nested dictionary structure
                dest_pre_balance = (
                    pre_balances["destination"].get("ETH", {}).get("balance", 0)
                )
                dest_post_balance = (
                    post_balances["destination"].get("ETH", {}).get("balance", 0)
                )

                dest_eth_change = float(dest_post_balance) - float(dest_pre_balance)

                # Check if destination gained exactly the expected amount
                if abs(dest_eth_change - expected_amount_float) < 1e-10:
                    return 0.5

            # For ERC-20 transfers - check both sender and destination
            else:
                # Extract balance values from the nested dictionary structure
                sender_pre_balance = (
                    pre_balances["sender"].get(expected_token, {}).get("balance", 0)
                )
                sender_post_balance = (
                    post_balances["sender"].get(expected_token, {}).get("balance", 0)
                )
                dest_pre_balance = (
                    pre_balances["destination"]
                    .get(expected_token, {})
                    .get("balance", 0)
                )
                dest_post_balance = (
                    post_balances["destination"]
                    .get(expected_token, {})
                    .get("balance", 0)
                )

                sender_token_change = float(sender_post_balance) - float(
                    sender_pre_balance
                )
                dest_token_change = float(dest_post_balance) - float(dest_pre_balance)

                # For ERC-20, expect exact amounts (no gas costs in token)
                if (
                    abs(sender_token_change + expected_amount_float) < 1e-6
                    and abs(dest_token_change - expected_amount_float) < 1e-6
                ):
                    return 0.5

            return 0.0  # Transfer amounts don't match expectations

        except (ValueError, TypeError):
            return 0.0

    def _analyze_thinking_quality(self, response: str) -> float:
        """Evaluate thinking tag quality with max 0.1 points, negative for missing thinking"""
        thinking_score = 0.0

        # Check for thinking tags
        if "<think>" not in response or "</think>" not in response:
            return -0.2  # Penalty for no thinking tags

        # Extract thinking content
        try:
            thinking_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
            if not thinking_match:
                return -0.2  # No thinking content found

            thinking_content = thinking_match.group(1).strip()
            if not thinking_content:
                return -0.1  # Empty thinking tags

            # Basic quality assessment for positive score (max 0.1)
            word_count = len(thinking_content.split())

            # Award points based on thinking depth
            if word_count >= 50:  # Substantial thinking
                thinking_score += 0.1
            elif word_count >= 20:  # Moderate thinking
                thinking_score += 0.05
            elif word_count >= 5:  # Minimal thinking
                thinking_score += 0.02

            return thinking_score

        except Exception:
            return -0.1  # Error in processing thinking

    async def evaluate(self, *args, **kwargs):
        """Evaluation method - could implement portfolio performance tracking"""
        return

    def close(self):
        """Clean up resources"""
        cleanup_blockchain(self.blockchain)

    @classmethod
    def config_init(cls) -> Tuple[EVMEnvConfig, List[APIServerConfig]]:
        """Initialize configuration for EVM environment"""
        # pydantic-settings automatically loads from YAML file
        env_config = EVMEnvConfig(
            tokenizer_name="NousResearch/Hermes-3-Llama-3.1-8B",
            group_size=4,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=500,
            batch_size=16,
            steps_per_eval=50,
            max_token_length=2048,
            wandb_name="evm-agent",
            anvil_config_path="configs/token_transfers.yaml",
        )

        # API server configuration
        server_configs = [
            APIServerConfig(
                model_name="gpt-4o-mini",
                base_url=None,  # Use OpenAI directly
                api_key=os.environ.get("OPENAI_API_KEY"),
                num_requests_for_eval=64,
            ),
        ]

        return env_config, server_configs


if __name__ == "__main__":
    EVMEnv.cli()
