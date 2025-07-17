#!/usr/bin/env python3
"""
Test script for VR-CLI integration in Diplomacy environment.

Tests prediction storage, outcome extraction, and scoring.
"""

import asyncio
import logging
from transformers import AutoTokenizer

from vrcli_scorer import VRCLIScorer
from diplomacy_vrcli_integration import DiplomacyVRCLIIntegration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Mock server for testing
class MockServer:
    async def completion(self, **kwargs):
        # Return mock logprobs
        class MockChoice:
            class MockLogprobs:
                token_logprobs = [-0.5, -0.3, -0.4, -0.2, -0.6]  # Mock log probabilities
            logprobs = MockLogprobs()
            
        class MockResponse:
            choices = [MockChoice()]
            
        return MockResponse()


async def test_vrcli_integration():
    """Test VR-CLI integration components."""
    
    # Initialize components
    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Hermes-4-Qwen3-14B-1-e3")
    server = MockServer()
    scorer = VRCLIScorer(server, tokenizer)
    integration = DiplomacyVRCLIIntegration(scorer, vrcli_weight=0.3)
    
    # Test 1: Store predictions
    logger.info("Test 1: Storing predictions")
    integration.store_predictions(
        episode_id="test_game_001",
        power="FRANCE",
        phase="S1901M",
        decision_num=1,
        predictions={
            "negotiation_responses": {
                "ENGLAND": "Likely to agree to DMZ in English Channel",
                "GERMANY": "May be suspicious but interested in Burgundy deal"
            },
            "board_changes": {
                "territories": {
                    "BEL": "France likely to capture",
                    "HOL": "Germany likely to capture"
                }
            },
            "relationship_changes": {
                "ENGLAND": "+0.2 trust",
                "GERMANY": "-0.1 trust"
            }
        },
        prompt="Game state: Spring 1901 Movement phase..."
    )
    
    # Test 2: Store actual outcomes
    logger.info("Test 2: Storing actual outcomes")
    integration.store_actual_outcomes(
        episode_id="test_game_001",
        phase="S1901M",
        outcomes={
            "negotiation_responses": {
                "ENGLAND": ["Agreed to DMZ in Channel. Let's focus on Germany."],
                "GERMANY": ["I'm moving to Holland as discussed. Good luck in Belgium!"]
            },
            "board_changes": {
                "territories": {
                    "BEL": "Unoccupied -> FRANCE",
                    "HOL": "Unoccupied -> GERMANY"
                }
            },
            "relationship_changes": {
                "ENGLAND->FRANCE": 0.3,
                "GERMANY->FRANCE": 0.0
            }
        }
    )
    
    # Test 3: Calculate VR-CLI scores
    logger.info("Test 3: Calculating VR-CLI scores")
    scores = await integration.calculate_scores_for_episode(
        episode_id="test_game_001",
        powers=["FRANCE"]
    )
    
    logger.info(f"VR-CLI scores for FRANCE: {scores['FRANCE']}")
    
    # Test 4: Test score combination
    logger.info("Test 4: Testing score combination")
    base_rewards = [0.6, 0.4, 0.8, 0.5]
    vrcli_scores = [0.9, 0.5, 0.9, 0.0]
    
    combined = integration.apply_vrcli_to_rewards(base_rewards, vrcli_scores)
    logger.info(f"Base rewards: {base_rewards}")
    logger.info(f"VR-CLI scores: {vrcli_scores}")
    logger.info(f"Combined (weight=0.3): {combined}")
    
    # Test 5: Test outcome extraction methods
    logger.info("Test 5: Testing outcome extraction")
    
    # Test negotiation outcome extraction
    messages = [
        {"sender": "ENGLAND", "content": "I agree to the alliance"},
        {"sender": "FRANCE", "content": "Great! Let's coordinate"},
        {"sender": "ENGLAND", "content": "Move to Belgium, I'll support"}
    ]
    negotiation_outcomes = integration.extract_negotiation_outcomes(messages)
    logger.info(f"Negotiation outcomes: {negotiation_outcomes}")
    
    # Test board outcome extraction
    prev_state = {
        "ownership": {"PAR": "FRANCE", "LON": "ENGLAND", "BER": "GERMANY"},
        "units": {
            "FRANCE": [{"type": "A", "location": "PAR"}],
            "ENGLAND": [{"type": "F", "location": "LON"}]
        }
    }
    curr_state = {
        "ownership": {"PAR": "FRANCE", "LON": "ENGLAND", "BER": "GERMANY", "BEL": "FRANCE"},
        "units": {
            "FRANCE": [{"type": "A", "location": "BEL"}],
            "ENGLAND": [{"type": "F", "location": "NTH"}]
        }
    }
    board_outcomes = integration.extract_board_outcomes(prev_state, curr_state)
    logger.info(f"Board outcomes: {board_outcomes}")
    
    # Test trust outcome extraction
    prev_trust = {
        "FRANCE": {"ENGLAND": 0.5, "GERMANY": 0.5},
        "ENGLAND": {"FRANCE": 0.5, "GERMANY": 0.4}
    }
    curr_trust = {
        "FRANCE": {"ENGLAND": 0.7, "GERMANY": 0.4},
        "ENGLAND": {"FRANCE": 0.8, "GERMANY": 0.3}
    }
    trust_outcomes = integration.extract_trust_outcomes(prev_trust, curr_trust)
    logger.info(f"Trust outcomes: {trust_outcomes}")
    
    logger.info("All tests completed successfully!")


if __name__ == "__main__":
    asyncio.run(test_vrcli_integration())