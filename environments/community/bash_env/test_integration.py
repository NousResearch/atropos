#!/usr/bin/env python3
"""
Integration test for NL2Bash Environment that works with OpenAI-compatible APIs.

This test verifies:
1. NL2SH-ALFA dataset loading
2. Bash command generation from LLM
3. Bash extraction from \\boxed{}
4. String matching verification
5. Scoring logic
"""

import asyncio
import json
import random
from typing import Optional

import openai

# Import local modules
from bash_utils import commands_match, extract_boxed_bash
from nl2bash_loader import load_nl2bash_split

# System prompt from the environment
SYSTEM_PROMPT = (
    "You are a deep thinking AI, you may use extremely long chains of thought "
    "to deeply consider the problem and deliberate with yourself via systematic "
    "reasoning processes to help come to a correct solution prior to answering. "
    "You should enclose your thoughts and internal monologue inside <think> </think> "
    "tags, and then provide your solution or response to the problem.\n\n"
    "You are a Bash command expert. Given a natural language instruction, "
    "generate the appropriate Bash command.\n\n"
    "You are allocated a maximum of 1024 tokens, please strive to use less.\n\n"
    "Provide your Bash command inside \\boxed{} like this: "
    '\\boxed{find . -name "*.txt"}\n\n'
    "Important:\n"
    "- Generate a single, complete Bash command\n"
    "- Do not include explanatory text outside of <think> tags\n"
    "- Ensure your command is valid Bash syntax\n\n"
    "So please end your answer with \\boxed{your bash command here}"
)


def format_instruction(nl: str) -> str:
    """Format the natural language instruction for the prompt."""
    return f"Instruction: {nl}"


def score_bash(
    generated_bash: str,
    gold_bash: str,
    alt_bash: Optional[str] = None,
) -> dict:
    """Score bash by string matching."""
    result = {
        "generated_bash": generated_bash,
        "gold_bash": gold_bash,
        "alt_bash": alt_bash,
        "score": -1.0,
        "match": False,
        "error": None,
    }

    if not generated_bash:
        result["error"] = "No Bash command extracted from response"
        return result

    if commands_match(generated_bash, gold_bash, alt_bash):
        result["score"] = 1.0
        result["match"] = True
    else:
        result["error"] = "Command does not match gold or alternative"

    return result


async def test_single_item(client, model_name: str, item: dict, item_idx: int) -> dict:
    """Test a single NL2Bash item."""
    user_content = format_instruction(item["nl"])

    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_tokens=1024,
            temperature=0.6,
        )

        response_content = response.choices[0].message.content

        # Extract Bash
        generated_bash = extract_boxed_bash(response_content)

        # Score
        score_result = score_bash(generated_bash, item["bash"], item.get("bash2"))

        return {
            "item_idx": item_idx,
            "instruction": item["nl"],
            "difficulty": item.get("difficulty"),
            "response": (
                response_content[:500] + "..."
                if len(response_content) > 500
                else response_content
            ),
            **score_result,
        }

    except Exception as e:
        return {
            "item_idx": item_idx,
            "instruction": item["nl"],
            "error": str(e),
            "score": -1.0,
        }


async def run_integration_test(
    base_url: str,
    model_name: str,
    api_key: str = "x",
    num_samples: int = 10,
    use_test_set: bool = True,
):
    """Run the integration test."""
    print(f"\n{'='*60}")
    print("NL2Bash Environment Integration Test")
    print(f"{'='*60}")
    print(f"Server: {base_url}")
    print(f"Model: {model_name}")
    print(f"Samples: {num_samples}")
    print(f"Split: {'test' if use_test_set else 'train'}")
    print()

    # Load dataset
    split = "test" if use_test_set else "train"
    print(f"Loading NL2SH-ALFA {split} data...")
    data = load_nl2bash_split(split)
    print(f"Loaded {len(data)} examples")

    # Initialize OpenAI client
    client = openai.AsyncClient(
        base_url=base_url,
        api_key=api_key,
        timeout=120.0,
    )

    # Sample random items
    if num_samples < len(data):
        test_items = random.sample(data, num_samples)
    else:
        test_items = data

    # Run tests
    print(f"\nTesting {len(test_items)} samples...\n")
    results = []

    for i, item in enumerate(test_items):
        print(f"[{i+1}/{len(test_items)}] Testing: {item['nl'][:60]}...")
        result = await test_single_item(client, model_name, item, i)
        results.append(result)

        # Print result
        if result["score"] == 1.0:
            print(f"  ✓ CORRECT - {result.get('generated_bash', 'N/A')[:60]}")
        else:
            print(f"  ✗ INCORRECT - {result.get('error', 'Unknown error')}")
            if result.get("generated_bash"):
                print(f"    Generated: {result['generated_bash'][:60]}")
            print(f"    Gold: {result.get('gold_bash', 'N/A')[:60]}")
            if result.get("alt_bash"):
                print(f"    Alt:  {result.get('alt_bash', '')[:60]}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    correct = sum(1 for r in results if r["score"] == 1.0)
    total = len(results)

    print(f"Overall Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")

    # Difficulty breakdown (for test set)
    if use_test_set:
        difficulty_names = {0: "Easy", 1: "Medium", 2: "Hard"}
        for diff, name in difficulty_names.items():
            diff_results = [r for r in results if r.get("difficulty") == diff]
            if diff_results:
                diff_correct = sum(1 for r in diff_results if r["score"] == 1.0)
                print(
                    f"  {name}: {diff_correct}/{len(diff_results)} "
                    f"({100*diff_correct/len(diff_results):.1f}%)"
                )

    # Save results
    output_file = "integration_test_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NL2Bash Environment Integration Test")
    parser.add_argument(
        "--base_url",
        type=str,
        default="http://localhost:8000/v1",
        help="Base URL for OpenAI-compatible API",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Model name",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="x",
        help="API key",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to test",
    )
    parser.add_argument(
        "--use_train",
        action="store_true",
        help="Use training set instead of test set",
    )

    args = parser.parse_args()

    asyncio.run(
        run_integration_test(
            base_url=args.base_url,
            model_name=args.model,
            api_key=args.api_key,
            num_samples=args.num_samples,
            use_test_set=not args.use_train,
        )
    )
