"""
Scoring system for Diplomacy environment.

Implements composite scoring based on:
- VR-CLI scores for action prediction quality
- Game outcome scores (territory control, survival)
- Negotiation quality scores (trust, deception, coordination)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .diplomacy_types import (
    DiplomacyAction,
    DiplomacyEpisodeState,
    GameState,
    NegotiationMessage,
    Order,
    PowerRelationship,
)

logger = logging.getLogger(__name__)


class DiplomacyScorer:
    """
    Scoring system for Diplomacy actions and outcomes.

    Combines multiple scoring components:
    1. VR-CLI: Rewards accurate predictions of action outcomes
    2. Game outcomes: Territory control, elimination, survival
    3. Negotiation quality: Trust building, successful coordination
    """

    def __init__(
        self,
        vrcli_weight: float = 0.3,
        outcome_weight: float = 0.5,
        negotiation_weight: float = 0.2,
        discount_factor: float = 0.99,
    ):
        self.vrcli_weight = vrcli_weight
        self.outcome_weight = outcome_weight
        self.negotiation_weight = negotiation_weight
        self.discount_factor = discount_factor

        # VR-CLI thresholds
        self.vrcli_thresholds = {
            "negligible": 0.05,
            "small": 1.0,
            "moderate": 2.0,
        }

        # VR-CLI reward levels
        self.vrcli_rewards = {
            "negligible": 0.0,
            "small": 0.5,
            "moderate": 0.9,
            "significant": 1.0,
        }

    def calculate_vrcli_score(
        self,
        prediction_perplexity: float,
        baseline_perplexity: float,
    ) -> float:
        """
        Calculate VR-CLI score based on perplexity improvement.

        Formula: [1 - PPL(y|x,a)/PPL(y|x)] × 100
        """
        if baseline_perplexity <= 0:
            return 0.0

        improvement = (1 - prediction_perplexity / baseline_perplexity) * 100

        # Map to discrete reward levels
        if improvement < self.vrcli_thresholds["negligible"]:
            return self.vrcli_rewards["negligible"]
        elif improvement < self.vrcli_thresholds["small"]:
            return self.vrcli_rewards["small"]
        elif improvement < self.vrcli_thresholds["moderate"]:
            return self.vrcli_rewards["moderate"]
        else:
            return self.vrcli_rewards["significant"]

    def calculate_outcome_score(
        self,
        episode_state: DiplomacyEpisodeState,
        power: str,
        initial_centers: int,
        final_centers: int,
        game_ended: bool,
        winner: Optional[str] = None,
    ) -> float:
        """Calculate game outcome score for a power."""
        score = 0.0

        # Victory bonus
        if winner == power:
            score += 10.0

        # Survival bonus
        if final_centers > 0:
            score += 2.0

        # Territory control
        center_change = final_centers - initial_centers
        score += center_change * 0.5

        # Relative strength
        total_centers = sum(episode_state.game_state.supply_centers.values())
        if total_centers > 0:
            control_ratio = final_centers / total_centers
            score += control_ratio * 3.0

        # Normalize to 0-1 range
        return np.clip(score / 15.0, 0.0, 1.0)

    def calculate_negotiation_score(
        self,
        messages_sent: List[NegotiationMessage],
        messages_received: List[NegotiationMessage],
        agreements_made: int,
        agreements_kept: int,
        betrayals_executed: int,
        betrayals_suffered: int,
    ) -> float:
        """Calculate negotiation quality score."""
        score = 0.0

        # Communication activity
        communication_score = min(
            1.0, (len(messages_sent) + len(messages_received)) / 20
        )
        score += communication_score * 0.2

        # Agreement success rate
        if agreements_made > 0:
            trust_score = agreements_kept / agreements_made
            score += trust_score * 0.4
        else:
            score += 0.2  # Neutral if no agreements

        # Strategic deception (successful betrayals vs suffered)
        deception_score = 0.5  # Neutral baseline
        if betrayals_executed > 0 or betrayals_suffered > 0:
            net_betrayals = betrayals_executed - betrayals_suffered
            deception_score = 0.5 + np.tanh(net_betrayals * 0.3) * 0.5
        score += deception_score * 0.4

        return np.clip(score, 0.0, 1.0)

    def score_negotiation_action(
        self,
        episode_state: DiplomacyEpisodeState,
        power: str,
        action: DiplomacyAction,
        outcome: Dict[str, Any],
    ) -> float:
        """Score a negotiation action."""
        # Base score components
        vrcli_score = outcome.get("vrcli_score", 0.0)

        # Message quality scoring
        message_scores = []
        for message in action.messages or []:
            msg_score = self._score_message_quality(
                message,
                episode_state.power_states[power],
                episode_state,
            )
            message_scores.append(msg_score)

        message_quality = np.mean(message_scores) if message_scores else 0.5

        # Combine scores
        total_score = (
            self.vrcli_weight * vrcli_score + self.negotiation_weight * message_quality
        )

        return total_score

    def score_order_action(
        self,
        episode_state: DiplomacyEpisodeState,
        power: str,
        action: DiplomacyAction,
        outcome: Dict[str, Any],
    ) -> float:
        """Score an order action."""
        # Base score components
        vrcli_score = outcome.get("vrcli_score", 0.0)

        # Order validity and success
        order_scores = []
        for order_result in outcome.get("order_results", []):
            if order_result["success"]:
                order_scores.append(1.0)
            elif order_result["valid"]:
                order_scores.append(0.5)  # Valid but failed
            else:
                order_scores.append(0.0)  # Invalid order

        order_success = np.mean(order_scores) if order_scores else 0.0

        # Strategic value (placeholder - would need game analysis)
        strategic_value = 0.5

        # Combine scores
        total_score = (
            self.vrcli_weight * vrcli_score
            + self.outcome_weight * order_success * 0.5
            + self.outcome_weight * strategic_value * 0.5
        )

        return total_score

    def _score_message_quality(
        self,
        message: Dict[str, Any],
        power_state: Any,
        episode_state: DiplomacyEpisodeState,
    ) -> float:
        """Score the quality of a negotiation message."""
        score = 0.5  # Baseline

        # Check if message aligns with relationships
        to_powers = message.get("to_powers", [])
        for to_power in to_powers:
            relationship = power_state.relationships.get(
                to_power, PowerRelationship.NEUTRAL
            )

            # Adjust based on message type and relationship
            msg_type = message.get("type", "negotiation")
            if msg_type == "commitment":
                if relationship in [PowerRelationship.ALLY, PowerRelationship.FRIENDLY]:
                    score += 0.1
                else:
                    score -= 0.1  # Risky to commit to non-allies
            elif msg_type == "proposal":
                if relationship != PowerRelationship.ENEMY:
                    score += 0.05

        # Length and content checks
        content = message.get("content", "")
        if 20 < len(content.split()) < 100:
            score += 0.1  # Good length

        return np.clip(score, 0.0, 1.0)

    def calculate_episode_scores(
        self,
        episode_state: DiplomacyEpisodeState,
        turn_scores: Dict[int, Dict[str, float]],
    ) -> Dict[str, float]:
        """Calculate final scores for an episode."""
        final_scores = {}

        for power in episode_state.power_assignment:
            # Get initial and final center counts
            initial_centers = self._get_initial_centers(power)
            final_centers = episode_state.game_state.supply_centers.get(power, 0)

            # Check if power won
            winner = self._determine_winner(episode_state)

            # Calculate outcome score
            outcome_score = self.calculate_outcome_score(
                episode_state,
                power,
                initial_centers,
                final_centers,
                game_ended=True,
                winner=winner,
            )

            # Aggregate turn scores with discounting
            discounted_turn_score = 0.0
            turn_count = 0
            for turn, scores in sorted(turn_scores.items()):
                if power in scores:
                    discounted_turn_score += scores[power] * (
                        self.discount_factor**turn_count
                    )
                    turn_count += 1

            if turn_count > 0:
                avg_turn_score = discounted_turn_score / turn_count
            else:
                avg_turn_score = 0.0

            # Combine scores
            final_scores[power] = (
                self.outcome_weight * outcome_score
                + (1 - self.outcome_weight) * avg_turn_score
            )

        return final_scores

    def _get_initial_centers(self, power: str) -> int:
        """Get initial supply center count for a power."""
        # Standard Diplomacy starting positions
        initial_centers = {
            "ENGLAND": 3,
            "FRANCE": 3,
            "GERMANY": 3,
            "ITALY": 3,
            "AUSTRIA": 3,
            "RUSSIA": 4,
            "TURKEY": 3,
        }
        return initial_centers.get(power, 3)

    def _determine_winner(self, episode_state: DiplomacyEpisodeState) -> Optional[str]:
        """Determine if there's a winner."""
        for power, centers in episode_state.game_state.supply_centers.items():
            if centers >= 18:  # Standard victory condition
                return power

        # No winner - check for leader
        if episode_state.final_outcome:
            return episode_state.final_outcome.get("winner")

        return None

    def calculate_credit_assignment(
        self,
        episode_state: DiplomacyEpisodeState,
        power: str,
        action_history: List[Tuple[int, DiplomacyAction, float]],
        final_outcome: float,
    ) -> List[float]:
        """
        Calculate credit assignment for actions in an episode.

        Uses discounted returns from final outcome.
        """
        credits = []
        num_actions = len(action_history)

        for i, (turn, action, immediate_score) in enumerate(action_history):
            # Distance to end of episode
            steps_to_end = num_actions - i - 1

            # Discounted return from final outcome
            discounted_return = final_outcome * (self.discount_factor**steps_to_end)

            # Combine immediate and future rewards
            total_credit = immediate_score + discounted_return * 0.5

            credits.append(total_credit)

        return credits
