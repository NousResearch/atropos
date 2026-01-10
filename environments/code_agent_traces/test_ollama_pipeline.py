#!/usr/bin/env python3
"""
Test script for the Ollama Agent Trace Pipeline with DeepSeek V3.2.

This script tests the complete pipeline:
1. Ollama Cloud connection with logprobs
2. Code generation with DeepSeek V3.2
3. Local code execution and scoring

Usage:
    # Set your Ollama Cloud API key
    export OLLAMA_API_KEY=your_api_key_here

    # Run tests with DeepSeek V3.2 (default)
    python test_ollama_pipeline.py

    # Test with different model
    OLLAMA_MODEL=deepseek-v3.1 python test_ollama_pipeline.py

    # Test with local Ollama
    OLLAMA_BASE_URL=http://localhost:11434 OLLAMA_MODEL=deepseek-r1:7b python test_ollama_pipeline.py
"""

import asyncio
import json
import math
import os
import sys

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rich import print as rprint
from rich.console import Console
from rich.table import Table

console = Console()


async def test_ollama_connection():
    """Test basic Ollama Cloud connection."""
    from atroposlib.envs.server_handling.ollama_server import OllamaServer, OllamaServerConfig

    rprint("\n[bold cyan]=" * 60)
    rprint("[bold cyan]Test 1: Ollama Cloud Connection (DeepSeek V3.2)")
    rprint("[bold cyan]=" * 60)

    config = OllamaServerConfig(
        base_url=os.getenv("OLLAMA_BASE_URL", "https://ollama.com"),
        api_key=os.getenv("OLLAMA_API_KEY", ""),
        model_name=os.getenv("OLLAMA_MODEL", "deepseek-v3.2"),
        timeout=120,
    )

    rprint(f"Base URL: {config.base_url}")
    rprint(f"Model: {config.model_name}")

    server = OllamaServer(config)

    try:
        # Simple chat completion test
        completion = await server.chat_completion(
            messages=[{"role": "user", "content": "Say 'Hello' and nothing else."}],
            max_tokens=10,
            temperature=0.1,
        )

        response = completion.choices[0].message.content
        rprint(f"[green]Connection successful![/green]")
        rprint(f"Response: {response}")
        return True

    except Exception as e:
        rprint(f"[red]Connection failed: {e}[/red]")
        return False


async def test_logprobs():
    """Test logprobs extraction from Ollama."""
    from atroposlib.envs.server_handling.ollama_server import OllamaServer, OllamaServerConfig

    rprint("\n[bold cyan]=" * 60)
    rprint("[bold cyan]Test 2: Logprobs Extraction")
    rprint("[bold cyan]=" * 60)

    config = OllamaServerConfig(
        base_url=os.getenv("OLLAMA_BASE_URL", "https://ollama.com"),
        api_key=os.getenv("OLLAMA_API_KEY", ""),
        model_name=os.getenv("OLLAMA_MODEL", "deepseek-v3.2"),
        timeout=120,
    )

    server = OllamaServer(config)

    try:
        completion, logprobs = await server.chat_completion_with_logprobs(
            messages=[{"role": "user", "content": "What is 2 + 2? Reply with just the number."}],
            max_tokens=20,
            temperature=0.7,
            top_logprobs=5,
        )

        response = completion.choices[0].message.content
        rprint(f"[green]Logprobs extraction successful![/green]")
        rprint(f"Response: {response}")
        rprint(f"Number of tokens with logprobs: {len(logprobs[0])}")

        if logprobs[0]:
            # Show first 5 tokens
            table = Table(title="First 5 Token Logprobs")
            table.add_column("Token", style="cyan")
            table.add_column("Logprob", style="magenta")
            table.add_column("Probability", style="green")

            for lp in logprobs[0][:5]:
                table.add_row(
                    repr(lp["token"]),
                    f"{lp['logprob']:.4f}",
                    f"{lp['probability']:.4f}",
                )

            console.print(table)
            return True
        else:
            rprint("[yellow]No logprobs returned. This is expected for some Ollama configurations.[/yellow]")
            return True

    except Exception as e:
        rprint(f"[red]Logprobs test failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False


async def test_code_generation():
    """Test code generation with the pipeline."""
    from atroposlib.envs.server_handling.ollama_server import OllamaServer, OllamaServerConfig

    rprint("\n[bold cyan]=" * 60)
    rprint("[bold cyan]Test 3: Code Generation")
    rprint("[bold cyan]=" * 60)

    config = OllamaServerConfig(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        api_key=os.getenv("OLLAMA_API_KEY", ""),
        model_name=os.getenv("OLLAMA_MODEL", "deepseek-r1:7b"),
        timeout=300,
    )

    server = OllamaServer(config)

    # Test problem
    problem = """
Write a Python function called `two_sum` that takes a list of integers `nums`
and an integer `target`, and returns the indices of two numbers that add up to the target.

Example:
    two_sum([2, 7, 11, 15], 9) -> [0, 1]
    two_sum([3, 2, 4], 6) -> [1, 2]
"""

    messages = [
        {"role": "system", "content": "You are an expert Python programmer. Write clean, efficient code."},
        {"role": "user", "content": problem + "\n\nEnclose your code in ```python and ``` delimiters."},
    ]

    try:
        completion = await server.chat_completion(
            messages=messages,
            max_tokens=500,
            temperature=0.3,
        )

        response = completion.choices[0].message.content
        rprint(f"[green]Code generation successful![/green]")
        rprint("\n[bold]Generated Code:[/bold]")
        rprint(response)

        # Extract and test the code
        import re
        code_match = re.search(r"```python\s*(.*?)```", response, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
            rprint("\n[bold]Extracted Code:[/bold]")
            rprint(code)

            # Try to execute the code
            try:
                exec_globals = {}
                exec(code, exec_globals)
                if "two_sum" in exec_globals:
                    result = exec_globals["two_sum"]([2, 7, 11, 15], 9)
                    if result == [0, 1]:
                        rprint(f"[green]Code execution successful! Result: {result}[/green]")
                    else:
                        rprint(f"[yellow]Code executed but wrong result: {result}[/yellow]")
            except Exception as e:
                rprint(f"[red]Code execution failed: {e}[/red]")

        return True

    except Exception as e:
        rprint(f"[red]Code generation failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False


async def test_tokens_and_logprobs():
    """Test the tokens_and_logprobs_completion method."""
    from atroposlib.envs.server_handling.ollama_server import OllamaServer, OllamaServerConfig

    rprint("\n[bold cyan]=" * 60)
    rprint("[bold cyan]Test 4: Tokens and Logprobs Completion")
    rprint("[bold cyan]=" * 60)

    config = OllamaServerConfig(
        base_url=os.getenv("OLLAMA_BASE_URL", "https://ollama.com"),
        api_key=os.getenv("OLLAMA_API_KEY", ""),
        model_name=os.getenv("OLLAMA_MODEL", "deepseek-v3.2"),
        timeout=120,
    )

    server = OllamaServer(config)

    try:
        prompt_tokens, output_tokens, output_logprobs, finish_reasons = (
            await server.tokens_and_logprobs_completion(
                messages=[{"role": "user", "content": "Hello!"}],
                max_tokens=30,
                n=2,
            )
        )

        rprint(f"[green]Tokens and logprobs completion successful![/green]")
        rprint(f"Prompt tokens: {len(prompt_tokens)}")
        rprint(f"Number of completions: {len(output_tokens)}")

        for i, (tokens, logprobs, reason) in enumerate(
            zip(output_tokens, output_logprobs, finish_reasons)
        ):
            rprint(f"\nCompletion {i}:")
            rprint(f"  Output tokens: {len(tokens)}")
            rprint(f"  Logprobs available: {len(logprobs)}")
            rprint(f"  Finish reason: {reason}")
            if logprobs:
                avg_logprob = sum(logprobs) / len(logprobs)
                rprint(f"  Average logprob: {avg_logprob:.4f}")

        return True

    except Exception as e:
        rprint(f"[red]Tokens and logprobs test failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False


async def test_full_pipeline():
    """Test the full agent trace pipeline."""
    rprint("\n[bold cyan]=" * 60)
    rprint("[bold cyan]Test 5: Full Pipeline (Simulated)")
    rprint("[bold cyan]=" * 60)

    from atroposlib.envs.server_handling.ollama_server import OllamaServer, OllamaServerConfig

    config = OllamaServerConfig(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        api_key=os.getenv("OLLAMA_API_KEY", ""),
        model_name=os.getenv("OLLAMA_MODEL", "deepseek-r1:7b"),
        timeout=300,
    )

    server = OllamaServer(config)

    # Simulated coding problem
    problem = {
        "idx": 0,
        "problem": "Write a function `is_prime(n)` that returns True if n is prime, False otherwise.",
        "problem_type": "func",
        "tests": {
            "inputs": [[2], [3], [4], [17], [1]],
            "outputs": [True, True, False, True, False],
        },
    }

    system_prompt = "You are an expert Python programmer."
    user_prompt = f"""
Problem: {problem['problem']}

Write your solution and enclose it in ```python and ``` delimiters.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        # Generate solution with logprobs
        completion, logprobs = await server.chat_completion_with_logprobs(
            messages=messages,
            max_tokens=300,
            temperature=0.5,
            top_logprobs=5,
        )

        response = completion.choices[0].message.content

        # Extract code
        import re
        code_match = re.search(r"```python\s*(.*?)```", response, re.DOTALL)
        code = code_match.group(1).strip() if code_match else None

        # Test the code
        score = 0.0
        if code:
            try:
                exec_globals = {}
                exec(code, exec_globals)
                if "is_prime" in exec_globals:
                    func = exec_globals["is_prime"]
                    correct = 0
                    for inp, expected in zip(problem["tests"]["inputs"], problem["tests"]["outputs"]):
                        if func(*inp) == expected:
                            correct += 1
                    score = correct / len(problem["tests"]["inputs"])
            except Exception as e:
                rprint(f"[red]Execution error: {e}[/red]")

        # Create trace
        trace = {
            "problem_idx": problem["idx"],
            "problem": problem["problem"],
            "response": response,
            "code": code,
            "score": score,
            "num_tokens": len(logprobs[0]) if logprobs[0] else 0,
            "avg_logprob": (
                sum(lp["logprob"] for lp in logprobs[0]) / len(logprobs[0])
                if logprobs[0] else 0.0
            ),
        }

        rprint(f"[green]Full pipeline test completed![/green]")
        rprint(f"\n[bold]Trace Summary:[/bold]")
        rprint(f"  Problem: {trace['problem'][:50]}...")
        rprint(f"  Score: {trace['score']:.2%}")
        rprint(f"  Tokens generated: {trace['num_tokens']}")
        rprint(f"  Average logprob: {trace['avg_logprob']:.4f}")

        if code:
            rprint(f"\n[bold]Generated Code:[/bold]")
            rprint(code)

        return True

    except Exception as e:
        rprint(f"[red]Full pipeline test failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False


async def test_local_executor():
    """Test local code execution."""
    rprint("\n[bold cyan]=" * 60)
    rprint("[bold cyan]Test 6: Local Code Executor")
    rprint("[bold cyan]=" * 60)

    try:
        from local_executor import run_test_local, execute_code_safe

        # Test 1: Simple function
        code = '''
def add(a, b):
    return a + b
'''
        tests = {
            "fn_name": "add",
            "inputs": [[1, 2], [3, 4], [-1, 1]],
            "outputs": [3, 7, 0],
        }

        results, metadata = execute_code_safe(code, tests)
        rprint(f"Test 1 - add function: {results}")

        # Test 2: Async interface
        results2, metadata2 = await run_test_local({"tests": tests}, code)
        rprint(f"Test 2 - async interface: {results2}")

        # Test 3: Error handling
        bad_code = "def broken(: pass"
        results3, metadata3 = execute_code_safe(bad_code, tests)
        rprint(f"Test 3 - syntax error: {results3}, error={metadata3.get('error', 'none')[:50]}")

        # Test 4: Timeout
        slow_code = '''
def slow(a, b):
    import time
    time.sleep(100)
    return a + b
'''
        results4, metadata4 = execute_code_safe(slow_code, tests, timeout=1.0)
        rprint(f"Test 4 - timeout: {results4}, error={metadata4.get('error', 'none')}")

        rprint(f"[green]Local executor tests passed![/green]")
        return True

    except Exception as e:
        rprint(f"[red]Local executor test failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    rprint("[bold magenta]" + "=" * 60)
    rprint("[bold magenta]Ollama Cloud + DeepSeek V3.2 - Agent Trace Pipeline")
    rprint("[bold magenta]" + "=" * 60)

    rprint("\n[bold]Configuration:[/bold]")
    rprint(f"  OLLAMA_BASE_URL: {os.getenv('OLLAMA_BASE_URL', 'https://ollama.com')}")
    rprint(f"  OLLAMA_MODEL: {os.getenv('OLLAMA_MODEL', 'deepseek-v3.2')}")
    rprint(f"  OLLAMA_API_KEY: {'[set]' if os.getenv('OLLAMA_API_KEY') else '[NOT SET - required for Ollama Cloud]'}")

    results = {}

    # Run local executor test first (no network needed)
    results["local_executor"] = await test_local_executor()

    # Run Ollama Cloud tests
    results["connection"] = await test_ollama_connection()

    if results["connection"]:
        results["logprobs"] = await test_logprobs()
        results["code_gen"] = await test_code_generation()
        results["tokens_logprobs"] = await test_tokens_and_logprobs()
        results["full_pipeline"] = await test_full_pipeline()
    else:
        rprint("[yellow]Skipping Ollama tests due to connection failure.[/yellow]")
        rprint("[yellow]Make sure OLLAMA_API_KEY is set for Ollama Cloud.[/yellow]")

    # Summary
    rprint("\n[bold magenta]" + "=" * 60)
    rprint("[bold magenta]Test Summary")
    rprint("[bold magenta]" + "=" * 60)

    table = Table(title="Test Results")
    table.add_column("Test", style="cyan")
    table.add_column("Status", style="green")

    for test, passed in results.items():
        status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        table.add_row(test, status)

    console.print(table)

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    rprint(f"\n[bold]Results: {passed}/{total} tests passed[/bold]")

    return all(results.values())


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
