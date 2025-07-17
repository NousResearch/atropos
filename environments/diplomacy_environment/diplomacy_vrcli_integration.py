"""
Integration module for VR-CLI scoring in the Diplomacy environment.

This module handles the collection of predictions, tracking of actual outcomes,
and calculation of VR-CLI scores for credit assignment.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

from .vrcli_scorer import VRCLIScorer

logger = logging.getLogger(__name__)


class DiplomacyVRCLIIntegration:
    """
    Manages VR-CLI scoring integration for Diplomacy environment.
    
    Tracks predictions, collects actual outcomes, and calculates scores
    for use in GRPO credit assignment.
    """
    
    def __init__(self, vrcli_scorer: VRCLIScorer, vrcli_weight: float = 0.3):
        """
        Initialize VR-CLI integration.
        
        Args:
            vrcli_scorer: The VR-CLI scorer instance
            vrcli_weight: Weight for VR-CLI scores in final reward (0-1)
        """
        self.scorer = vrcli_scorer
        self.vrcli_weight = vrcli_weight
        
        # Track predictions by episode and phase
        self.episode_predictions: Dict[str, Dict[str, List[Dict]]] = defaultdict(
            lambda: defaultdict(list)
        )
        
        # Track actual outcomes by episode and phase
        self.episode_outcomes: Dict[str, Dict[str, Dict]] = defaultdict(dict)
        
        # Cache scores to avoid recomputation
        self.score_cache: Dict[Tuple[str, str, int], Dict[str, float]] = {}
        
    def store_predictions(
        self,
        episode_id: str,
        power: str,
        phase: str,
        decision_num: int,
        predictions: Dict[str, Any],
        prompt: str
    ):
        """
        Store predictions made by an agent.
        
        Args:
            episode_id: Episode identifier
            power: Power making the prediction
            phase: Game phase (e.g., "S1901M")
            decision_num: Decision number within episode
            predictions: The predictions made
            prompt: The prompt/game state when prediction was made
        """
        key = f"{power}_{phase}_{decision_num}"
        self.episode_predictions[episode_id][key].append({
            "power": power,
            "phase": phase,
            "decision": decision_num,
            "predictions": predictions,
            "prompt": prompt,
        })
        
    def store_actual_outcomes(
        self,
        episode_id: str,
        phase: str,
        outcomes: Dict[str, Any]
    ):
        """
        Store actual outcomes after a phase completes.
        
        Args:
            episode_id: Episode identifier
            phase: Game phase that completed
            outcomes: Actual outcomes including messages, board changes, trust changes
        """
        self.episode_outcomes[episode_id][phase] = outcomes
        
    async def calculate_scores_for_episode(
        self,
        episode_id: str,
        powers: List[str]
    ) -> Dict[str, List[float]]:
        """
        Calculate VR-CLI scores for all predictions in an episode.
        
        Args:
            episode_id: Episode identifier
            powers: List of powers to calculate scores for
            
        Returns:
            Dict mapping power to list of scores for their decisions
        """
        power_scores = defaultdict(list)
        
        if episode_id not in self.episode_predictions:
            logger.warning(f"No predictions found for episode {episode_id}")
            return power_scores
            
        # Process each power's predictions
        for power in powers:
            # Get all predictions for this power
            power_predictions = []
            for key, pred_list in self.episode_predictions[episode_id].items():
                if key.startswith(f"{power}_"):
                    power_predictions.extend(pred_list)
                    
            # Sort by decision number
            power_predictions.sort(key=lambda p: p["decision"])
            
            # Calculate scores for each prediction
            for pred_data in power_predictions:
                phase = pred_data["phase"]
                cache_key = (episode_id, phase, pred_data["decision"])
                
                # Check cache first
                if cache_key in self.score_cache:
                    scores = self.score_cache[cache_key]
                else:
                    # Get actual outcomes for this phase
                    if phase in self.episode_outcomes[episode_id]:
                        outcomes = self.episode_outcomes[episode_id][phase]
                        scores = await self.scorer.score_turn_predictions(
                            pred_data, outcomes
                        )
                        self.score_cache[cache_key] = scores
                    else:
                        logger.debug(f"No outcomes yet for phase {phase}")
                        scores = {}
                        
                # Average the different score types
                if scores:
                    avg_score = sum(scores.values()) / len(scores)
                    power_scores[power].append(avg_score)
                else:
                    # No outcomes yet or prediction didn't match phase
                    power_scores[power].append(0.0)
                    
        return power_scores
        
    def apply_vrcli_to_rewards(
        self,
        base_rewards: List[float],
        vrcli_scores: List[float]
    ) -> List[float]:
        """
        Combine base rewards with VR-CLI scores.
        
        Args:
            base_rewards: Original rewards (e.g., from LaTRo)
            vrcli_scores: VR-CLI scores for predictions
            
        Returns:
            Combined rewards
        """
        if len(base_rewards) != len(vrcli_scores):
            logger.warning(
                f"Reward length mismatch: {len(base_rewards)} vs {len(vrcli_scores)}"
            )
            return base_rewards
            
        combined = []
        for base, vrcli in zip(base_rewards, vrcli_scores):
            # Weighted combination
            combined_score = (
                base * (1 - self.vrcli_weight) + 
                vrcli * self.vrcli_weight
            )
            combined.append(combined_score)
            
        return combined
        
    def clear_episode_data(self, episode_id: str):
        """Clear stored data for an episode."""
        if episode_id in self.episode_predictions:
            del self.episode_predictions[episode_id]
        if episode_id in self.episode_outcomes:
            del self.episode_outcomes[episode_id]
            
        # Clear relevant cache entries
        keys_to_remove = [k for k in self.score_cache if k[0] == episode_id]
        for key in keys_to_remove:
            del self.score_cache[key]
            
    def extract_negotiation_outcomes(
        self,
        messages_this_phase: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extract negotiation outcomes from messages.
        
        Args:
            messages_this_phase: All messages sent in a phase
            
        Returns:
            Dict mapping sender to their messages
        """
        outcomes = defaultdict(list)
        
        for msg in messages_this_phase:
            sender = msg.get("sender", "")
            content = msg.get("content", "")
            if sender and content:
                outcomes[sender].append(content)
                
        return dict(outcomes)
        
    def extract_board_outcomes(
        self,
        previous_state: Dict[str, Any],
        current_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract board change outcomes by comparing states.
        
        Args:
            previous_state: Board state before orders
            current_state: Board state after orders
            
        Returns:
            Dict with territory and unit changes
        """
        outcomes = {
            "territories": {},
            "unit_outcomes": {}
        }
        
        # Compare territory ownership
        prev_ownership = previous_state.get("ownership", {})
        curr_ownership = current_state.get("ownership", {})
        
        for territory in set(prev_ownership) | set(curr_ownership):
            prev_owner = prev_ownership.get(territory)
            curr_owner = curr_ownership.get(territory)
            if prev_owner != curr_owner:
                outcomes["territories"][territory] = f"{prev_owner} -> {curr_owner}"
                
        # Track unit outcomes (simplified - could be expanded)
        prev_units = previous_state.get("units", {})
        curr_units = current_state.get("units", {})
        
        # Units that moved
        for power in prev_units:
            for unit in prev_units.get(power, []):
                unit_loc = unit.get("location")
                if unit_loc:
                    # Check if unit still exists and where
                    still_exists = False
                    for curr_unit in curr_units.get(power, []):
                        if curr_unit.get("type") == unit.get("type"):
                            new_loc = curr_unit.get("location")
                            if new_loc != unit_loc:
                                unit_key = f"{unit['type']} {unit_loc}"
                                outcomes["unit_outcomes"][unit_key] = f"moved to {new_loc}"
                            still_exists = True
                            break
                    if not still_exists:
                        unit_key = f"{unit['type']} {unit_loc}"
                        outcomes["unit_outcomes"][unit_key] = "destroyed"
                        
        return outcomes
        
    def extract_trust_outcomes(
        self,
        previous_trust: Dict[str, Dict[str, float]],
        current_trust: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Extract trust/relationship changes.
        
        Args:
            previous_trust: Trust scores before phase
            current_trust: Trust scores after phase
            
        Returns:
            Dict mapping power pairs to trust changes
        """
        outcomes = {}
        
        for power_a in current_trust:
            for power_b in current_trust.get(power_a, {}):
                prev_trust = previous_trust.get(power_a, {}).get(power_b, 0.5)
                curr_trust = current_trust.get(power_a, {}).get(power_b, 0.5)
                
                if abs(curr_trust - prev_trust) > 0.01:  # Significant change
                    key = f"{power_a}->{power_b}"
                    outcomes[key] = curr_trust - prev_trust
                    
        return outcomes