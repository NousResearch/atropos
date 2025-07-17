#!/usr/bin/env python3
"""
Filter TextWorld dataset by score and convert to different formats.

This script processes the raw JSONL output from TextWorld rejection sampling
and filters/converts it for downstream use.
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def extract_game_info_from_messages(messages: List[Dict[str, str]]) -> Dict[str, str]:
    """Extract game information from conversation messages."""
    info = {"game_type": "unknown", "objective": "unknown", "difficulty": "unknown"}

    # Look for objective in the first user message
    for msg in messages:
        if msg["role"] == "user" and "Objective:" in msg["content"]:
            # Extract objective
            lines = msg["content"].split("\n")
            for line in lines:
                if line.strip().startswith("Objective:"):
                    info["objective"] = line.split("Objective:", 1)[1].strip()
                    break

            # Try to infer game type from objective
            objective_lower = info["objective"].lower()
            if "cook" in objective_lower or "recipe" in objective_lower:
                info["game_type"] = "cooking"
            elif "coin" in objective_lower or "collect" in objective_lower:
                info["game_type"] = "coin_collector"
            elif "treasure" in objective_lower:
                info["game_type"] = "treasure_hunter"
            elif "find" in objective_lower and "take" in objective_lower:
                info["game_type"] = "simple_quest"
            elif "puzzle" in objective_lower or "solve" in objective_lower:
                info["game_type"] = "puzzle"
            elif "navigate" in objective_lower or "maze" in objective_lower:
                info["game_type"] = "navigation"
            else:
                info["game_type"] = "mixed"

            break

    return info


def convert_to_sft_format(scored_group: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert a scored data group to ShareGPT format conversations."""
    sft_conversations = []

    # Get messages for each alternative
    messages_list = scored_group.get("messages", [])
    scores = scored_group.get("scores", [])

    # Extract game info from metadata if available
    metadata = scored_group.get("metadata", {})
    episode_id = metadata.get("episode_id", "unknown")
    turn_number = metadata.get("turn_number", 0)
    vrcli_scores = metadata.get("vrcli_scores", [])
    env_rewards = metadata.get("env_rewards", [])

    for i, (messages, score) in enumerate(zip(messages_list, scores)):
        if not messages or score < 0:  # Skip invalid or negative scores
            continue

        # Extract game info
        game_info = extract_game_info_from_messages(messages)

        # Convert to ShareGPT format
        conversation = []
        for msg in messages:
            if msg["role"] == "system":
                conversation.append({"from": "system", "value": msg["content"]})
            elif msg["role"] == "user":
                conversation.append({"from": "human", "value": msg["content"]})
            elif msg["role"] == "assistant":
                conversation.append({"from": "gpt", "value": msg["content"]})

        # Create the SFT entry
        sft_entry = {
            "conversations": conversation,
            "score": score,
            "game_type": game_info["game_type"],
            "objective": game_info["objective"],
            "episode_id": episode_id,
            "turn_number": turn_number,
            "alternative_index": i,
            "source": "textworld",
        }

        # Add VR-CLI and environment reward info if available
        if i < len(vrcli_scores):
            sft_entry["vrcli_score"] = vrcli_scores[i]
        if i < len(env_rewards):
            sft_entry["env_reward"] = env_rewards[i]

        sft_conversations.append(sft_entry)

    return sft_conversations


def main():
    parser = argparse.ArgumentParser(description="Filter TextWorld dataset by score")
    parser.add_argument(
        "input_file", type=str, help="Input JSONL file from TextWorld data generation"
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum score threshold (default: 0.0)",
    )
    parser.add_argument(
        "--format",
        choices=["sft", "raw"],
        default="sft",
        help="Output format (default: sft)",
    )
    parser.add_argument(
        "--output", type=str, help="Output file path (default: input_file with suffix)"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Determine output file path
    input_path = Path(args.input_file)
    if args.output:
        output_path = Path(args.output)
    else:
        suffix = f"_filtered_{args.format}.jsonl"
        output_path = input_path.with_suffix("").with_suffix(suffix)

    # Process the file
    total_groups = 0
    total_alternatives = 0
    filtered_conversations = []
    game_type_counts = defaultdict(int)
    score_distribution = defaultdict(int)

    logger.info(f"Processing {input_path}")
    logger.info(f"Filtering with min_score >= {args.min_score}")

    with open(input_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                data = json.loads(line)
                total_groups += 1

                # Count alternatives
                if "messages" in data:
                    total_alternatives += len(data.get("messages", []))

                if args.format == "sft":
                    # Convert to SFT format and filter by score
                    sft_convs = convert_to_sft_format(data)
                    for conv in sft_convs:
                        if conv["score"] >= args.min_score:
                            filtered_conversations.append(conv)
                            game_type_counts[conv["game_type"]] += 1
                            # Bucket scores for distribution
                            score_bucket = int(conv["score"] * 10) / 10
                            score_distribution[score_bucket] += 1
                else:
                    # Raw format - filter entire groups
                    scores = data.get("scores", [])
                    if any(s >= args.min_score for s in scores):
                        filtered_conversations.append(data)

            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line {line_num}: {e}")
            except Exception as e:
                logger.warning(f"Error processing line {line_num}: {e}")

    # Write output
    logger.info(f"Writing {len(filtered_conversations)} items to {output_path}")

    with open(output_path, "w") as f:
        for item in filtered_conversations:
            f.write(json.dumps(item) + "\n")

    # Print statistics
    logger.info("\n=== Dataset Statistics ===")
    logger.info(f"Total scored groups: {total_groups}")
    logger.info(f"Total alternatives: {total_alternatives}")
    logger.info(f"Filtered conversations: {len(filtered_conversations)}")
    if total_alternatives > 0:
        retention_rate = len(filtered_conversations) / total_alternatives * 100
        logger.info(f"Retention rate: {retention_rate:.1f}%")

    if args.format == "sft":
        logger.info("\n=== Game Type Distribution ===")
        for game_type, count in sorted(game_type_counts.items(), key=lambda x: -x[1]):
            logger.info(f"{game_type}: {count} conversations")

        logger.info("\n=== Score Distribution ===")
        for score, count in sorted(score_distribution.items()):
            logger.info(f"Score {score:.1f}: {count} conversations")

        # Calculate average score
        if filtered_conversations:
            avg_score = sum(c["score"] for c in filtered_conversations) / len(
                filtered_conversations
            )
            logger.info(f"\nAverage score: {avg_score:.3f}")


if __name__ == "__main__":
    main()
