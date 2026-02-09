#!/usr/bin/env python3
"""
Integration test for SQL Query Environment that works with OpenAI-compatible APIs.

This test verifies:
1. WikiSQL dataset loading
2. SQL generation from LLM
3. SQL extraction from \boxed{}
4. SQL execution and result comparison
5. Scoring logic
"""

import asyncio
import json
from typing import Any, List

import openai

# Import local modules
from sql_executor import (
    create_table_from_wikisql,
    execute_sql,
    extract_boxed_sql,
    quote_identifiers_in_sql,
    results_match,
)
from wikisql_loader import load_wikisql_split

# System prompt from the environment
SYSTEM_PROMPT = (
    "You are a deep thinking AI, you may use extremely long chains of thought "
    "to deeply consider the problem and deliberate with yourself via systematic "
    "reasoning processes to help come to a correct solution prior to answering. "
    "You should enclose your thoughts and internal monologue inside <think> </think> "
    "tags, and then provide your solution or response to the problem.\n\n"
    "You are a SQL expert. Given a table schema and a natural language question, "
    "generate a SQL query that answers the question.\n\n"
    "You are allocated a maximum of 1024 tokens, please strive to use less.\n\n"
    "Provide your SQL query inside \\boxed{} like this: "
    "\\boxed{SELECT column FROM table WHERE condition}\n\n"
    "Important:\n"
    "- Use only the columns provided in the table schema\n"
    '- The table is always named "data"\n'
    "- Ensure your SQL syntax is valid SQLite\n\n"
    "So please end your answer with \\boxed{your SQL query here}"
)


def format_table_schema(
    header: List[str], rows: List[List[Any]], max_rows: int = 3
) -> str:
    """Format table schema for the prompt."""
    schema = f"Table: data\nColumns: {', '.join(header)}\n"
    if rows:
        schema += "Sample data:\n"
        for row in rows[:max_rows]:
            row_str = " | ".join(str(v) for v in row)
            schema += f"  {row_str}\n"
    return schema


def score_sql(
    generated_sql: str,
    gold_sql: str,
    header: List[str],
    rows: List[List[Any]],
) -> dict:
    """Score SQL by execution comparison."""
    result = {
        "generated_sql": generated_sql,
        "gold_sql": gold_sql,
        "score": -1.0,
        "execution_success": False,
        "gen_result": None,
        "gold_result": None,
        "error": None,
    }

    if not generated_sql:
        result["error"] = "No SQL extracted from response"
        return result

    # Create in-memory table
    try:
        conn = create_table_from_wikisql(header, rows)
    except Exception as e:
        result["error"] = f"Table creation failed: {e}"
        return result

    # Quote identifiers in gold SQL that need quoting
    quoted_gold_sql = quote_identifiers_in_sql(gold_sql, header)
    result["quoted_gold_sql"] = quoted_gold_sql

    # Execute gold SQL
    gold_result = execute_sql(conn, quoted_gold_sql)
    if gold_result is None:
        conn.close()
        result["error"] = f"Gold SQL execution failed: {quoted_gold_sql}"
        return result
    result["gold_result"] = str(gold_result)

    # Execute generated SQL
    gen_result = execute_sql(conn, generated_sql)
    conn.close()

    if gen_result is None:
        result["error"] = "Generated SQL execution failed"
        return result

    result["gen_result"] = str(gen_result)
    result["execution_success"] = True

    # Compare results
    if results_match(gen_result, gold_result):
        result["score"] = 1.0
    else:
        result["error"] = "Results do not match"

    return result


async def test_single_item(client, model_name: str, item: dict, item_idx: int) -> dict:
    """Test a single WikiSQL item."""
    table_schema = format_table_schema(item["header"], item["rows"])
    user_content = f"{table_schema}\nQuestion: {item['question']}"

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

        # Extract SQL
        generated_sql = extract_boxed_sql(response_content)

        # Score
        score_result = score_sql(
            generated_sql, item["gold_sql"], item["header"], item["rows"]
        )

        return {
            "item_idx": item_idx,
            "question": item["question"],
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
            "question": item["question"],
            "error": str(e),
            "score": -1.0,
        }


async def run_integration_test(
    base_url: str,
    model_name: str,
    api_key: str = "x",
    num_samples: int = 10,
):
    """Run the integration test."""
    print(f"\n{'='*60}")
    print("SQL Query Environment Integration Test")
    print(f"{'='*60}")
    print(f"Server: {base_url}")
    print(f"Model: {model_name}")
    print(f"Samples: {num_samples}")
    print()

    # Load dataset
    print("Loading WikiSQL training data...")
    train_data = load_wikisql_split("train")
    print(f"Loaded {len(train_data)} training examples")

    # Initialize OpenAI client
    client = openai.AsyncClient(
        base_url=base_url,
        api_key=api_key,
        timeout=120.0,
    )

    # Run tests
    print(f"\nTesting {num_samples} samples...\n")
    results = []

    for i in range(min(num_samples, len(train_data))):
        item = train_data[i]
        print(f"[{i+1}/{num_samples}] Testing: {item['question'][:60]}...")
        result = await test_single_item(client, model_name, item, i)
        results.append(result)

        # Print result
        if result["score"] == 1.0:
            print(f"  ✓ CORRECT - Generated: {result.get('generated_sql', 'N/A')[:80]}")
        else:
            print(f"  ✗ INCORRECT - {result.get('error', 'Unknown error')}")
            if result.get("generated_sql"):
                print(f"    Generated: {result['generated_sql'][:80]}")
            print(f"    Gold: {result.get('gold_sql', 'N/A')[:80]}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    correct = sum(1 for r in results if r["score"] == 1.0)
    executed = sum(1 for r in results if r.get("execution_success", False))

    print(f"Correct:          {correct}/{num_samples} ({100*correct/num_samples:.1f}%)")
    print(
        f"Execution Success: {executed}/{num_samples} ({100*executed/num_samples:.1f}%)"
    )

    # Save results
    output_file = "integration_test_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="SQL Query Environment Integration Test"
    )
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

    args = parser.parse_args()

    asyncio.run(
        run_integration_test(
            base_url=args.base_url,
            model_name=args.model,
            api_key=args.api_key,
            num_samples=args.num_samples,
        )
    )
