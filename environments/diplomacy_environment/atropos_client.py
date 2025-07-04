"""
AtroposClient - LLM Client Proxy for AI_Diplomacy Integration

This client implements the AI_Diplomacy BaseModelClient interface and forwards
all LLM requests to an Atropos policy server.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "AI_Diplomacy"))

import asyncio
import json
import logging
import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
from ai_diplomacy.clients import BaseModelClient

if TYPE_CHECKING:
    from environments.diplomacy_environment.diplomacy_env_grpo import DiplomacyEnvGRPO

from atroposlib.envs.base import ScoredDataGroup
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

logger = logging.getLogger(__name__)


class AtroposClient(BaseModelClient):
    """
    Proxy client that forwards LLM requests to Atropos policy server.
    Implements the AI_Diplomacy BaseModelClient interface.

    In GRPO mode, this client intercepts requests and performs best-of-N
    selection while accumulating trajectory data.
    """

    def __init__(
        self,
        model_name: str,
        server_url: str = "http://localhost:8000",
        env: Optional["DiplomacyEnvGRPO"] = None,
        is_training: bool = False,
        power: Optional[str] = None,
    ):
        super().__init__(model_name)
        self.server_url = server_url
        self.episode_id: Optional[str] = None
        self.power: Optional[str] = power
        self.client = httpx.AsyncClient(timeout=60.0)

        # GRPO mode configuration
        self.env = env  # Reference to parent environment (for GRPO mode)
        self.is_training = is_training
        self.trajectory_data: List[ScoredDataGroup] = []
        self.canonical_history: List[Dict] = []  # Selected responses only
        self.decision_count = 0

        logger.info(
            f"Initialized AtroposClient for model {model_name} at {server_url} (GRPO mode: {is_training})"
        )

    def set_context(self, episode_id: str, power: str):
        """Set the current game context for this client."""
        self.episode_id = episode_id
        self.power = power
        logger.debug(f"Set context: episode={episode_id}, power={power}")

    async def generate_response(
        self, prompt: str, temperature: float = 0.0, inject_random_seed: bool = True
    ) -> str:
        """
        Forward the prompt to Atropos server and return the response.

        This is the main method that AI_Diplomacy calls for all LLM interactions.
        In GRPO mode, this implements best-of-N selection and data collection.
        """
        # If not in training mode or no env, use normal forwarding
        if not self.is_training or not self.env:
            return await self._normal_generate_response(
                prompt, temperature, inject_random_seed
            )

        # GRPO mode: best-of-N selection
        logger.info(
            f"[AtroposClient GRPO] generate_response called! power={self.power}, model={self.model_name}"
        )

        self.decision_count += 1
        task_type = self._infer_task_type(prompt)

        logger.info(
            f"[AtroposClient GRPO] Decision {self.decision_count} for {self.power}: "
            f"task_type={task_type}"
        )

        # Step 1: Build messages from canonical history + current prompt
        messages = self._build_messages(prompt)

        # Step 2: Sample N responses from policy server
        group_size = self.env.config.group_size
        responses, all_logprobs = await self._sample_n_responses(
            prompt, group_size, temperature
        )

        if not responses or len(responses) < group_size:
            logger.error(
                f"Failed to sample {group_size} responses, got {len(responses)}"
            )
            # Fallback to getting a single response
            single_responses, _ = await self._sample_n_responses(prompt, 1, temperature)
            return (
                single_responses[0]
                if single_responses
                else self._generate_fallback_response(prompt)
            )

        # Step 3: Score each response
        raw_scores = []
        for i, response in enumerate(responses):
            logprobs = all_logprobs[i] if i < len(all_logprobs) else []
            score = self._score_response(response, prompt, task_type, logprobs)
            raw_scores.append(score)

        # Normalize scores based on whether we're using LaTRo rewards
        use_latro = getattr(self.env.config, "use_latro_rewards", True)
        if use_latro and any(all_logprobs):
            # For LaTRo rewards, compute advantages as in the paper
            mean_score = np.mean(raw_scores)
            advantages = [s - mean_score for s in raw_scores]

            # Normalize to [0, 1] range
            min_adv = min(advantages)
            max_adv = max(advantages)
            if max_adv > min_adv:
                scores = [(a - min_adv) / (max_adv - min_adv) for a in advantages]
            else:
                scores = [0.5] * len(advantages)
        else:
            # For heuristic scores, just ensure they're in [0, 1]
            scores = [min(1.0, max(0.0, s)) for s in raw_scores]

        # Step 4: Tokenize alternatives for ScoredDataGroup
        alt_tokens = []
        alt_masks = []
        alt_messages = []

        for response in responses:
            # Build full message history for this alternative
            alt_msgs = messages + [{"role": "assistant", "content": response}]
            alt_messages.append(alt_msgs)

            # Tokenize
            tokenized = tokenize_for_trainer(self.env.tokenizer, alt_msgs)
            alt_tokens.append(tokenized["tokens"])
            alt_masks.append(tokenized["masks"])

        # Step 5: Create ScoredDataGroup
        scored_group = ScoredDataGroup(
            tokens=alt_tokens,
            masks=alt_masks,
            scores=scores,
            messages=alt_messages if self.env.config.include_messages else None,
            group_overrides={
                "power": self.power,
                "episode_id": self.episode_id,
                "decision": self.decision_count,
                "task_type": task_type,
                "phase": self._extract_phase(prompt),
            },
        )

        self.trajectory_data.append(scored_group)

        # Step 6: Select best response
        best_idx = np.argmax(scores)
        selected_response = responses[best_idx]

        logger.info(
            f"[AtroposClient GRPO] Selected response {best_idx} with score {scores[best_idx]:.3f}"
        )

        # Log LaTRo details if enabled
        if use_latro and any(all_logprobs):
            logger.debug(
                f"[AtroposClient GRPO] LaTRo raw scores: {[f'{s:.3f}' for s in raw_scores]}"
            )
            logger.debug(
                f"[AtroposClient GRPO] LaTRo normalized scores: {[f'{s:.3f}' for s in scores]}"
            )

        # Step 7: Update canonical history with selected response
        self.canonical_history.append(
            {
                "prompt": prompt,
                "response": selected_response,
                "task_type": task_type,
                "selected_idx": best_idx,
                "all_scores": scores,
            }
        )

        return selected_response

    async def _normal_generate_response(
        self, prompt: str, temperature: float = 0.0, inject_random_seed: bool = True
    ) -> str:
        """Normal (non-GRPO) response generation."""
        request_data = {
            "prompt": prompt,
            "model": self.model_name,
            "temperature": temperature,
            "episode_id": self.episode_id,
            "power": self.power,
            "metadata": {
                "task_type": self._infer_task_type(prompt),
                "inject_random_seed": inject_random_seed,
            },
        }

        try:
            logger.debug(
                f"Sending request for {self.power}: task_type={request_data['metadata']['task_type']}"
            )
            response = await self.client.post(
                f"{self.server_url}/v1/completions", json=request_data
            )
            response.raise_for_status()

            result = response.json()
            logger.debug(
                f"Received response for {self.power}: {len(result.get('text', ''))} chars"
            )
            return result["text"]

        except httpx.ConnectError:
            logger.error(f"Failed to connect to Atropos server at {self.server_url}")
            # Return a fallback response for development
            return self._generate_fallback_response(prompt)
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def _infer_task_type(self, prompt: str) -> str:
        """Infer the type of task from the prompt content."""
        prompt_lower = prompt.lower()

        if "orders for this turn" in prompt_lower or "submit orders" in prompt_lower:
            return "orders"
        elif (
            "conversation" in prompt_lower
            or "message" in prompt_lower
            or "respond to" in prompt_lower
        ):
            return "negotiation"
        elif (
            "plan" in prompt_lower
            or "strategy" in prompt_lower
            or "goals" in prompt_lower
        ):
            return "planning"
        elif "diary" in prompt_lower or "private thoughts" in prompt_lower:
            return "diary"
        else:
            return "general"

    def _generate_fallback_response(self, prompt: str) -> str:
        """
        Generate a simple fallback response for development/testing.
        This allows the system to run even without a connected Atropos server.
        """
        task_type = self._infer_task_type(prompt)

        if task_type == "orders":
            # Return empty orders (AI_Diplomacy will use defaults)
            return json.dumps(
                {
                    "orders": {},
                    "explanations": {
                        "general": "Fallback response - no server connected"
                    },
                }
            )
        elif task_type == "negotiation":
            return json.dumps(
                {
                    "messages": [],
                    "explanations": {
                        "general": "Fallback response - no server connected"
                    },
                }
            )
        elif task_type == "planning":
            return json.dumps(
                {
                    "plans": {"immediate": "Hold all positions"},
                    "explanations": {
                        "general": "Fallback response - no server connected"
                    },
                }
            )
        else:
            return "Fallback response - Atropos server not connected"

    def _build_messages(self, current_prompt: str) -> List[Dict]:
        """Build message history from canonical history + current prompt."""
        messages = []

        # Add system prompt if this is the first decision
        if not self.canonical_history:
            # Get system prompt from AI_Diplomacy's agent setup
            # For now, use a simple prompt
            messages.append(
                {
                    "role": "system",
                    "content": f"You are playing as {self.power} in a game of Diplomacy.",
                }
            )

        # Add canonical history (selected responses only)
        for entry in self.canonical_history:
            messages.append({"role": "user", "content": entry["prompt"]})
            messages.append({"role": "assistant", "content": entry["response"]})

        # Add current prompt
        messages.append({"role": "user", "content": current_prompt})

        return messages

    async def _sample_n_responses(
        self, prompt: str, n: int, temperature: float
    ) -> Tuple[List[str], List[List[float]]]:
        """Sample N responses from the actual policy server and return text + logprobs."""
        try:
            # Build the full prompt with message history
            messages = self._build_messages(prompt)

            # Make N separate API calls to get N responses
            responses = []
            all_logprobs = []
            actual_model = self.env.server.servers[0].config.model_name
            logger.info(f"[AtroposClient GRPO] Using model: {actual_model}")

            for i in range(n):
                try:
                    # Try chat_completion first (better for logprobs)
                    completion = await self.env.server.chat_completion(
                        messages=messages,
                        n=1,  # Request one response at a time
                        max_tokens=self.env.config.max_token_length,
                        temperature=(
                            temperature
                            if temperature > 0
                            else self.env.config.temperature
                        ),
                        top_p=self.env.config.top_p,
                        model=actual_model,
                        logprobs=True,  # Enable logprobs for LaTRo rewards
                        top_logprobs=5,  # Get top 5 logprobs per token
                    )

                    # Extract the text and logprobs from chat completion
                    if completion.choices:
                        choice = completion.choices[0]
                        response_text = choice.message.content
                        responses.append(response_text)

                        # Extract logprobs from chat format
                        if (
                            hasattr(choice, "logprobs")
                            and choice.logprobs
                            and choice.logprobs.content
                        ):
                            token_logprobs = []
                            for token_data in choice.logprobs.content:
                                if (
                                    hasattr(token_data, "logprob")
                                    and token_data.logprob is not None
                                ):
                                    token_logprobs.append(token_data.logprob)
                            all_logprobs.append(token_logprobs)
                            logger.debug(
                                f"[AtroposClient GRPO] Got {len(token_logprobs)} logprobs from chat API"
                            )
                        else:
                            logger.warning(
                                f"No logprobs in chat completion for response {i}"
                            )
                            all_logprobs.append([])

                except Exception as chat_error:
                    logger.debug(
                        f"Chat completion failed, trying completion API: {chat_error}"
                    )

                    # Fall back to completion API with proper tokenization
                    # Use the tokenizer to format messages into a single prompt
                    formatted_prompt = self.env.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,  # We want the string, not tokens
                        add_generation_prompt=True,  # Add the assistant prompt
                    )

                    completion = await self.env.server.completion(
                        prompt=formatted_prompt,
                        n=1,
                        max_tokens=self.env.config.max_token_length,
                        temperature=(
                            temperature
                            if temperature > 0
                            else self.env.config.temperature
                        ),
                        top_p=self.env.config.top_p,
                        model=actual_model,
                        logprobs=5,  # Request top 5 logprobs
                        echo=False,  # Don't include prompt in response
                    )

                    # Extract from completion API format
                    if completion.choices:
                        choice = completion.choices[0]
                        responses.append(choice.text)

                        # Extract logprobs if available
                        if hasattr(choice, "logprobs") and choice.logprobs is not None:
                            if (
                                hasattr(choice.logprobs, "token_logprobs")
                                and choice.logprobs.token_logprobs
                            ):
                                token_logprobs = choice.logprobs.token_logprobs
                                # Filter out None values (first token often has None)
                                token_logprobs = [
                                    lp for lp in token_logprobs if lp is not None
                                ]
                                all_logprobs.append(token_logprobs)
                            else:
                                logger.warning(
                                    f"Logprobs object missing token_logprobs for response {i}"
                                )
                                all_logprobs.append([])
                        else:
                            logger.warning(f"No logprobs returned for response {i}")
                            all_logprobs.append([])

            return responses, all_logprobs

        except Exception as e:
            logger.error(f"Error sampling responses: {e}")
            # Fallback: return n copies of a fallback response with empty logprobs
            fallback = self._generate_fallback_response(prompt)
            return [fallback] * n, [[]] * n

    def _compute_latro_reward(self, logprobs_sequence: List[float]) -> float:
        """
        Compute LaTRo reward as the sum of log probabilities.

        This implements r(z) = log π(z|x) from the paper.
        Higher values indicate the model is more confident about this response.
        """
        if not logprobs_sequence:
            # No logprobs available, return a neutral value
            return -10.0  # Reasonable default for missing logprobs

        # Sum log probabilities across all tokens
        total_logprob = sum(logprobs_sequence)
        return total_logprob

    def _score_response(
        self, response: str, prompt: str, task_type: str, logprobs: List[float]
    ) -> float:
        """
        Score a response using LaTRo rewards when available, with fallback to heuristics.

        The score will be normalized to [0, 1] range later by the caller.
        """
        # Check if we should use LaTRo rewards
        use_latro = getattr(self.env.config, "use_latro_rewards", True)

        if use_latro and logprobs:
            # Use LaTRo reward based on log probabilities
            return self._compute_latro_reward(logprobs)
        else:
            # Fallback to heuristic scoring
            base_score = random.random()

            # Add small bonuses for valid-looking responses
            if task_type == "orders":
                # Check if response contains order-like patterns
                if "orders" in response.lower() and "{" in response:
                    base_score += 0.1

            elif task_type == "negotiation":
                # Check if response contains message content
                if "messages" in response.lower() and len(response) > 100:
                    base_score += 0.1

            # Return raw score (will be normalized later)
            return base_score

    def _extract_phase(self, prompt: str) -> str:
        """Extract game phase from prompt if possible."""
        # Look for phase indicators like "Spring 1901" or "S1901M"
        import re

        # Pattern for standard phase notation
        phase_match = re.search(r"[SF]\d{4}[MRB]", prompt)
        if phase_match:
            return phase_match.group()

        # Pattern for verbose phase
        verbose_match = re.search(r"(Spring|Fall) \d{4}", prompt)
        if verbose_match:
            return verbose_match.group()

        return "unknown"

    def get_trajectory_data(self) -> List[ScoredDataGroup]:
        """Get accumulated trajectory data for training."""
        return self.trajectory_data

    def clear_trajectory(self):
        """Clear trajectory data for a new episode."""
        self.trajectory_data = []
        self.canonical_history = []
        self.decision_count = 0
        self.episode_id = f"episode-{random.randint(1000, 9999)}"

    async def close(self):
        """Clean up resources."""
        await self.client.aclose()


class AtroposModelRegistry:
    """
    Registry for managing multiple AtroposClient instances.
    Allows different powers to use different models/configurations.
    """

    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url
        self.clients: Dict[str, AtroposClient] = {}

    def get_client(self, model_name: str) -> AtroposClient:
        """Get or create a client for the specified model."""
        if model_name not in self.clients:
            self.clients[model_name] = AtroposClient(model_name, self.server_url)
        return self.clients[model_name]

    async def close_all(self):
        """Close all client connections."""
        for client in self.clients.values():
            await client.close()


# Integration with AI_Diplomacy's model loading system
def register_atropos_models(env=None):
    """
    Monkey-patch AI_Diplomacy's model loading to recognize Atropos models.
    This should be called before running any games.

    Args:
        env: Optional DiplomacyEnvGRPO instance for GRPO mode
    """
    from ai_diplomacy import clients

    original_load = clients.load_model_client
    logger.info(f"[register_atropos_models] Called with env: {env is not None}")

    def load_model_client_with_atropos(model_id: str) -> BaseModelClient:
        logger.info(
            f"[load_model_client_with_atropos] Called with model_id: {model_id}"
        )

        if (
            model_id == "atropos-training-policy"
            and env is not None
            and hasattr(env, "config")
        ):
            # For GRPO environment, use our client in training mode
            power_name = getattr(env, "training_power", env.config.training_power)
            logger.info(
                f"[load_model_client_with_atropos] Creating AtroposClient in GRPO mode for {power_name}"
            )
            # Create client that wraps the actual policy model
            return AtroposClient(
                model_id,
                env.server.servers[
                    0
                ].config.base_url,  # Use actual server URL from config
                env=env,
                is_training=True,
                power=power_name,
            )
        elif model_id.startswith("atropos-"):
            # Regular Atropos client (for evaluation or non-training agents)
            server_url = os.environ.get("ATROPOS_SERVER_URL", "http://localhost:8000")
            return AtroposClient(model_id, server_url)
        else:
            return original_load(model_id)

    clients.load_model_client = load_model_client_with_atropos
    logger.info("Registered Atropos model loader")


if __name__ == "__main__":
    # Simple test of the client
    async def test_client():
        client = AtroposClient("atropos-test", "http://localhost:8000")
        client.set_context("test-episode", "FRANCE")

        # Test different prompt types
        test_prompts = [
            "What are your orders for this turn?",
            "Send a message to England about an alliance.",
            "What is your strategic plan for the next few turns?",
        ]

        for prompt in test_prompts:
            print(f"\nPrompt: {prompt[:50]}...")
            try:
                response = await client.generate_response(prompt)
                print(f"Response: {response[:100]}...")
            except Exception as e:
                print(f"Error: {e}")

        await client.close()

    asyncio.run(test_client())
