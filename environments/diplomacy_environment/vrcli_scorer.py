"""
VR-CLI (Verifiable Rewards via Completion Likelihood Improvement) scorer for Diplomacy.

This module implements VR-CLI scoring to evaluate prediction quality by measuring
how well predictions improve the model's ability to predict actual outcomes.
"""

import asyncio
import logging
import math
from typing import Dict, Any, List, Optional, Tuple

from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class VRCLIScorer:
    """
    VR-CLI scorer for evaluating prediction quality in Diplomacy.
    
    Measures how well predictions improve perplexity of actual outcomes,
    following the VR-CLI paper methodology.
    """
    
    def __init__(self, server, tokenizer: PreTrainedTokenizerBase):
        """
        Initialize VR-CLI scorer.
        
        Args:
            server: The inference server (with completion/chat_completion methods)
            tokenizer: The tokenizer for the model
        """
        self.server = server
        self.tokenizer = tokenizer
        
    async def calculate_perplexity(self, messages: List[Dict[str, str]]) -> float:
        """
        Calculate perplexity using logprobs from the inference server.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            
        Returns:
            Perplexity value (exp of negative mean log probability)
        """
        # Apply chat template to get the full formatted text
        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        
        try:
            # Use completion mode to get logprobs for the entire sequence
            response = await self.server.completion(
                prompt=full_text,
                max_tokens=0,  # We're not generating, just getting logprobs
                echo=True,  # Return logprobs for the input
                logprobs=1,  # Return log probability of the selected tokens
                temperature=0.0,
            )
            
            # Find where the last message starts in the tokenized sequence
            if len(messages) > 1:
                prefix_messages = messages[:-1]
                prefix_text = self.tokenizer.apply_chat_template(
                    prefix_messages, tokenize=False, add_generation_prompt=True
                )
                prefix_tokens = self.tokenizer.encode(prefix_text, add_special_tokens=False)
                
                # Extract logprobs for the last message (the completion we're evaluating)
                all_logprobs = response.choices[0].logprobs.token_logprobs
                completion_logprobs = all_logprobs[len(prefix_tokens):]
            else:
                # If there's only one message, evaluate the whole thing
                completion_logprobs = response.choices[0].logprobs.token_logprobs
            
            # Calculate perplexity: exp(-mean(log_probs))
            if completion_logprobs:
                # Filter out None values (first token often has None)
                valid_logprobs = [lp for lp in completion_logprobs if lp is not None]
                if valid_logprobs:
                    mean_logprob = sum(valid_logprobs) / len(valid_logprobs)
                    return math.exp(-mean_logprob)
                    
        except Exception as e:
            logger.warning(f"Failed to calculate perplexity: {e}")
            
        return float("inf")
        
    async def score_prediction(
        self,
        game_state: str,
        predictions: Dict[str, Any],
        actual_outcomes: Dict[str, Any],
        prediction_type: str = "negotiation"
    ) -> float:
        """
        Score prediction quality using VR-CLI methodology.
        
        Args:
            game_state: Current game state context
            predictions: Predicted outcomes
            actual_outcomes: What actually happened
            prediction_type: Type of prediction ("negotiation", "board", "trust")
            
        Returns:
            VR-CLI score between 0 and 1
        """
        # Format predictions and outcomes based on type
        if prediction_type == "negotiation":
            predicted_text = self._format_negotiation_predictions(predictions)
            actual_text = self._format_negotiation_outcomes(actual_outcomes)
        elif prediction_type == "board":
            predicted_text = self._format_board_predictions(predictions)
            actual_text = self._format_board_outcomes(actual_outcomes)
        elif prediction_type == "trust":
            predicted_text = self._format_trust_predictions(predictions)
            actual_text = self._format_trust_outcomes(actual_outcomes)
        else:
            logger.warning(f"Unknown prediction type: {prediction_type}")
            return 0.0
            
        if not predicted_text or not actual_text:
            return 0.0
            
        # Create message lists for base and prediction-conditioned perplexity
        base_messages = [
            {
                "role": "user",
                "content": f"Game state:\n{game_state}\n\nWhat happens next?",
            },
            {"role": "assistant", "content": actual_text},
        ]
        
        prediction_messages = [
            {
                "role": "user",
                "content": f"Game state:\n{game_state}\n\nPredicted: {predicted_text}\n\nWhat actually happens?",
            },
            {"role": "assistant", "content": actual_text},
        ]
        
        # Calculate perplexities
        base_ppl = await self.calculate_perplexity(base_messages)
        pred_ppl = await self.calculate_perplexity(prediction_messages)
        
        # Calculate percentage improvement using VR-CLI formula
        # Improvement = [1 - PPL(y|x,a)/PPL(y|x)] × 100
        if base_ppl == 0 or base_ppl == float('inf'):
            return 0.0
            
        improvement = (1 - pred_ppl / base_ppl) * 100
        
        # Map to discrete reward levels (following VR-CLI paper)
        if improvement < 0.05:
            return 0.0  # Negligible improvement
        elif improvement < 1.0:
            return 0.5  # Small improvement
        elif improvement < 2.0:
            return 0.9  # Moderate improvement
        else:
            return 1.0  # Significant improvement
            
    async def score_turn_predictions(
        self,
        turn_data: Dict[str, Any],
        actual_outcomes: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Score all predictions from a turn.
        
        Args:
            turn_data: Turn data including game state and predictions
            actual_outcomes: Actual outcomes from the game
            
        Returns:
            Dict mapping prediction types to scores
        """
        scores = {}
        game_state = turn_data.get("prompt", "")
        predictions = turn_data.get("predictions", {})
        
        # Score each type of prediction
        if "negotiation_responses" in predictions and "negotiation_responses" in actual_outcomes:
            scores["negotiation"] = await self.score_prediction(
                game_state,
                predictions["negotiation_responses"],
                actual_outcomes["negotiation_responses"],
                "negotiation"
            )
            
        if "board_changes" in predictions and "board_changes" in actual_outcomes:
            scores["board"] = await self.score_prediction(
                game_state,
                predictions["board_changes"],
                actual_outcomes["board_changes"],
                "board"
            )
            
        if "relationship_changes" in predictions and "relationship_changes" in actual_outcomes:
            scores["trust"] = await self.score_prediction(
                game_state,
                predictions["relationship_changes"],
                actual_outcomes["relationship_changes"],
                "trust"
            )
            
        return scores
        
    def _format_negotiation_predictions(self, predictions: Dict[str, str]) -> str:
        """Format negotiation predictions into text."""
        if not predictions:
            return ""
            
        lines = []
        for power, response in predictions.items():
            lines.append(f"{power}: {response}")
        return "Predicted responses:\n" + "\n".join(lines)
        
    def _format_negotiation_outcomes(self, outcomes: Dict[str, Any]) -> str:
        """Format actual negotiation outcomes into text."""
        if not outcomes:
            return ""
            
        lines = []
        for power, messages in outcomes.items():
            if isinstance(messages, list):
                for msg in messages:
                    lines.append(f"{power}: {msg}")
            else:
                lines.append(f"{power}: {messages}")
        return "Actual responses:\n" + "\n".join(lines)
        
    def _format_board_predictions(self, predictions: Dict[str, Any]) -> str:
        """Format board state predictions into text."""
        if not predictions:
            return ""
            
        lines = []
        
        if "territories" in predictions:
            lines.append("Territory changes:")
            for territory, change in predictions["territories"].items():
                lines.append(f"  {territory}: {change}")
                
        if "unit_outcomes" in predictions:
            lines.append("Unit outcomes:")
            for unit, outcome in predictions["unit_outcomes"].items():
                lines.append(f"  {unit}: {outcome}")
                
        return "\n".join(lines)
        
    def _format_board_outcomes(self, outcomes: Dict[str, Any]) -> str:
        """Format actual board outcomes into text."""
        # Similar to predictions but with actual results
        return self._format_board_predictions(outcomes)
        
    def _format_trust_predictions(self, predictions: Dict[str, str]) -> str:
        """Format trust/relationship predictions into text."""
        if not predictions:
            return ""
            
        lines = ["Predicted relationship changes:"]
        for power, change in predictions.items():
            lines.append(f"  {power}: {change}")
        return "\n".join(lines)
        
    def _format_trust_outcomes(self, outcomes: Dict[str, Any]) -> str:
        """Format actual trust changes into text."""
        if not outcomes:
            return ""
            
        lines = ["Actual relationship changes:"]
        for power, change in outcomes.items():
            if isinstance(change, (int, float)):
                lines.append(f"  {power}: {'+' if change > 0 else ''}{change:.2f} trust")
            else:
                lines.append(f"  {power}: {change}")
        return "\n".join(lines)