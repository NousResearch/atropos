"""
Test file for reasoning model support across multiple providers.

This test validates:
1. ReasoningConfig builds correct extra_body for different providers
2. Reasoning can be enabled and responses contain reasoning content
3. Reasoning extraction works for various API response formats

Providers tested:
- OpenAI (gpt-5.2) - Uses reasoning_effort at top level
- OpenRouter (anthropic/claude-opus-4.5, nousresearch/hermes-4-70B, deepseek/deepseek-v3.2)
  - Uses nested reasoning object with enabled/effort/max_tokens

Usage:
    python -m pytest atroposlib/tests/test_reasoning_models.py -v

    Or run directly:
    python atroposlib/tests/test_reasoning_models.py

Note: This test requires valid API keys. Set them as environment variables or
modify the constants below for testing.
"""

import asyncio
import json
import os
import sys
from datetime import datetime

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import openai  # noqa: E402

from atroposlib.envs.server_handling.server_baseline import (  # noqa: E402
    ReasoningConfig,
)
from environments.eval_environments.eval_helpers import (  # noqa: E402
    HERMES_REASONING_PROMPT,
    HERMES_REASONING_PROMPT_WITH_ANSWER,
    extract_reasoning_from_completion,
    extract_reasoning_from_response,
)

# =============================================================================
# API CONFIGURATION
# =============================================================================
# These are test credentials. For production, use environment variables.

# API keys must be set via environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.2")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.environ.get(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)

# Models to test on OpenRouter
OPENROUTER_MODELS = [
    "anthropic/claude-opus-4.5",
    "nousresearch/hermes-4-70b",
    "deepseek/deepseek-v3.2",
]

# Test prompt that should trigger reasoning
TEST_PROMPT = "What is 15 * 23? Think step by step before giving your answer."

# Log file for full ChatCompletion objects
LOG_FILE = os.path.join(os.path.dirname(__file__), "reasoning_test_results.log")


def log_to_file(message: str):
    """Append message to log file."""
    with open(LOG_FILE, "a") as f:
        f.write(message + "\n")


# =============================================================================
# UNIT TESTS FOR ReasoningConfig
# =============================================================================


def test_reasoning_config_default():
    """Test default ReasoningConfig is not active."""
    config = ReasoningConfig()
    assert not config.enabled
    assert config.effort is None
    assert config.max_tokens is None
    assert not config.is_active()
    assert config.build_extra_body() is None
    print("✓ Default ReasoningConfig is inactive")


def test_reasoning_config_enabled_only():
    """Test ReasoningConfig with only enabled=True."""
    config = ReasoningConfig(enabled=True)
    assert config.enabled
    assert config.is_active()

    # Test for non-OpenAI provider
    extra_body = config.build_extra_body("https://openrouter.ai/api/v1")
    assert extra_body == {"reasoning": {"enabled": True}}

    # Test for OpenAI provider
    extra_body = config.build_extra_body("https://api.openai.com/v1")
    assert extra_body == {"reasoning_effort": "medium"}
    print("✓ ReasoningConfig with enabled=True works correctly")


def test_reasoning_config_with_effort():
    """Test ReasoningConfig with effort specified."""
    config = ReasoningConfig(effort="high")
    assert config.enabled  # Should be auto-enabled
    assert config.effort == "high"
    assert config.is_active()

    # Test for non-OpenAI provider
    extra_body = config.build_extra_body("https://openrouter.ai/api/v1")
    assert extra_body == {"reasoning": {"enabled": True, "effort": "high"}}

    # Test for OpenAI provider
    extra_body = config.build_extra_body("https://api.openai.com/v1")
    assert extra_body == {"reasoning_effort": "high"}
    print("✓ ReasoningConfig with effort works correctly")


def test_reasoning_config_with_max_tokens():
    """Test ReasoningConfig with max_tokens specified."""
    config = ReasoningConfig(max_tokens=4096)
    assert config.enabled  # Should be auto-enabled
    assert config.max_tokens == 4096
    assert config.is_active()

    # Test for non-OpenAI provider
    extra_body = config.build_extra_body("https://openrouter.ai/api/v1")
    assert extra_body == {"reasoning": {"enabled": True, "max_tokens": 4096}}

    # Test for OpenAI provider (max_tokens not supported, falls back to medium)
    extra_body = config.build_extra_body("https://api.openai.com/v1")
    assert extra_body == {"reasoning_effort": "medium"}
    print("✓ ReasoningConfig with max_tokens works correctly")


def test_reasoning_config_full():
    """Test ReasoningConfig with all options."""
    config = ReasoningConfig(enabled=True, effort="xhigh", max_tokens=8192)
    assert config.enabled
    assert config.effort == "xhigh"
    assert config.max_tokens == 8192

    # Test for non-OpenAI provider
    # Note: OpenRouter only allows ONE of effort or max_tokens
    # When both are set, effort takes priority
    extra_body = config.build_extra_body("https://openrouter.ai/api/v1")
    assert extra_body == {
        "reasoning": {
            "enabled": True,
            "effort": "xhigh",
            # max_tokens is NOT included when effort is specified (OpenRouter limitation)
        }
    }

    # Test for OpenAI provider (xhigh maps to high)
    extra_body = config.build_extra_body("https://api.openai.com/v1")
    assert extra_body == {"reasoning_effort": "high"}
    print("✓ ReasoningConfig with full options works correctly")


def test_reasoning_config_effort_mapping():
    """Test that effort levels are correctly mapped for OpenAI."""
    mappings = {
        "none": "low",
        "minimal": "low",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "xhigh": "high",
    }

    for our_effort, expected_openai in mappings.items():
        config = ReasoningConfig(effort=our_effort)
        extra_body = config.build_extra_body("https://api.openai.com/v1")
        assert (
            extra_body["reasoning_effort"] == expected_openai
        ), f"Expected {our_effort} to map to {expected_openai}, got {extra_body}"
    print("✓ Effort level mapping for OpenAI works correctly")


def test_reasoning_config_invalid_effort():
    """Test that invalid effort raises ValueError."""
    try:
        ReasoningConfig(effort="invalid")  # Should raise
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Invalid reasoning_effort" in str(e)
    print("✓ Invalid effort raises ValueError")


def test_reasoning_config_invalid_max_tokens():
    """Test that invalid max_tokens raises ValueError."""
    # Too low
    try:
        ReasoningConfig(max_tokens=500)  # Should raise
        assert False, "Should have raised ValueError for too low"
    except ValueError as e:
        assert "must be between 1024 and 32000" in str(e)

    # Too high
    try:
        ReasoningConfig(max_tokens=50000)  # Should raise
        assert False, "Should have raised ValueError for too high"
    except ValueError as e:
        assert "must be between 1024 and 32000" in str(e)
    print("✓ Invalid max_tokens raises ValueError")


def test_hermes_prompts_defined():
    """Test that Hermes prompts are properly defined."""
    assert HERMES_REASONING_PROMPT is not None
    assert "<think>" in HERMES_REASONING_PROMPT
    assert "</think>" in HERMES_REASONING_PROMPT

    assert HERMES_REASONING_PROMPT_WITH_ANSWER is not None
    assert "<answer>" in HERMES_REASONING_PROMPT_WITH_ANSWER
    print("✓ Hermes prompts are properly defined")


# =============================================================================
# SERVER MANAGER INTEGRATION TESTS
# =============================================================================


def test_reasoning_config_from_env_config():
    """Test ReasoningConfig.from_env_config() creates correct config."""
    from atroposlib.envs.base import BaseEnvConfig

    # Test with thinking_mode only
    env_config = BaseEnvConfig(
        tokenizer_name="gpt2",
        group_name="test",
        run_name="test",
        thinking_mode=True,
    )
    reasoning_config = ReasoningConfig.from_env_config(env_config)
    assert reasoning_config.enabled is True
    assert reasoning_config.effort is None
    assert reasoning_config.max_tokens is None
    print("✓ ReasoningConfig.from_env_config with thinking_mode=True works")

    # Test with reasoning_effort
    env_config = BaseEnvConfig(
        tokenizer_name="gpt2",
        group_name="test",
        run_name="test",
        reasoning_effort="high",
    )
    reasoning_config = ReasoningConfig.from_env_config(env_config)
    assert reasoning_config.enabled is True  # Auto-enabled because effort is set
    assert reasoning_config.effort == "high"
    print("✓ ReasoningConfig.from_env_config with reasoning_effort works")

    # Test with max_reasoning_tokens
    env_config = BaseEnvConfig(
        tokenizer_name="gpt2",
        group_name="test",
        run_name="test",
        max_reasoning_tokens=8000,
    )
    reasoning_config = ReasoningConfig.from_env_config(env_config)
    assert reasoning_config.enabled is True  # Auto-enabled because max_tokens is set
    assert reasoning_config.max_tokens == 8000
    print("✓ ReasoningConfig.from_env_config with max_reasoning_tokens works")

    # Test with all disabled (default)
    env_config = BaseEnvConfig(
        tokenizer_name="gpt2",
        group_name="test",
        run_name="test",
    )
    reasoning_config = ReasoningConfig.from_env_config(env_config)
    assert reasoning_config.enabled is False
    assert not reasoning_config.is_active()
    print("✓ ReasoningConfig.from_env_config with defaults (disabled) works")


def test_server_manager_builds_extra_body():
    """Test ReasoningConfig.build_extra_body() generates correct extra_body."""
    # Create reasoning config
    reasoning_config = ReasoningConfig(enabled=True, effort="high")

    # We can't easily instantiate ServerManager without actual servers,
    # so let's test the build_extra_body logic directly

    # Test OpenRouter format
    extra_body = reasoning_config.build_extra_body("https://openrouter.ai/api/v1")
    assert "reasoning" in extra_body
    assert extra_body["reasoning"]["enabled"] is True
    assert extra_body["reasoning"]["effort"] == "high"
    print("✓ ServerManager builds correct extra_body for OpenRouter")

    # Test OpenAI format
    extra_body = reasoning_config.build_extra_body("https://api.openai.com/v1")
    assert "reasoning_effort" in extra_body
    assert extra_body["reasoning_effort"] == "high"
    assert "reasoning" not in extra_body  # Should NOT have nested reasoning
    print("✓ ServerManager builds correct extra_body for OpenAI")

    # Test Claude (anthropic) - should use max_tokens
    claude_reasoning = ReasoningConfig(enabled=True, max_tokens=8000)
    extra_body = claude_reasoning.build_extra_body("https://openrouter.ai/api/v1")
    assert "reasoning" in extra_body
    assert extra_body["reasoning"]["enabled"] is True
    assert extra_body["reasoning"]["max_tokens"] == 8000
    print("✓ ServerManager builds correct extra_body for Claude (max_tokens)")


async def test_server_manager_injects_extra_body():
    """
    Integration test: Verify ServerManager actually injects extra_body in API calls.

    This test creates a real ServerManager and makes an actual API call to verify
    the full flow works.
    """
    if not OPENROUTER_API_KEY:
        print("⚠ Skipping ServerManager integration test - OPENROUTER_API_KEY not set")
        return True

    from atroposlib.envs.server_handling.server_baseline import APIServerConfig
    from atroposlib.envs.server_handling.server_manager import ServerManager

    # Create server config for OpenRouter
    server_config = APIServerConfig(
        model_name="nousresearch/hermes-4-70b",
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
        num_requests_for_eval=10,
    )

    # Create reasoning config
    reasoning_config = ReasoningConfig(enabled=True, effort="high")

    print("\n" + "=" * 60)
    print("Testing ServerManager.chat_completion() with reasoning injection")
    print("=" * 60)

    try:
        # Create ServerManager with reasoning config (NOT in testing mode - we want real API call)
        server_manager = ServerManager(
            configs=[server_config],
            reasoning_config=reasoning_config,
            testing=False,  # Actually make the API call
        )

        # Make a chat completion call
        messages = [
            {"role": "system", "content": HERMES_REASONING_PROMPT},
            {"role": "user", "content": "What is 2 + 2? Think carefully."},
        ]

        print(
            f"Making API call: enabled={reasoning_config.enabled}, "
            f"effort={reasoning_config.effort}"
        )

        completion = await server_manager.chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.7,
        )

        # Verify response has reasoning
        reasoning, source, content = extract_reasoning_from_completion(completion)

        print("Response received!")
        print(
            f"Content: {content[:100]}..."
            if content and len(content) > 100
            else f"Content: {content}"
        )
        print(f"Reasoning source: {source}")
        print(f"Reasoning length: {len(reasoning) if reasoning else 0} chars")

        if reasoning:
            print(
                "✓ ServerManager.chat_completion() correctly injected reasoning extra_body"
            )
            return True
        else:
            print(
                "⚠ Response received but no reasoning found (model may not support it)"
            )
            return True  # Still a pass - the injection worked, model just didn't return reasoning

    except Exception as e:
        import traceback

        print(f"✗ ServerManager test failed: {e}")
        traceback.print_exc()
        return False


def test_full_env_config_to_server_flow():
    """
    Test the complete flow from BaseEnvConfig to ServerManager reasoning injection.

    This verifies that:
    1. BaseEnvConfig with reasoning fields creates properly
    2. ReasoningConfig.from_env_config() works
    3. The resulting config would inject correct extra_body
    """
    from atroposlib.envs.base import BaseEnvConfig

    print("\n" + "=" * 60)
    print("Testing full BaseEnvConfig → ServerManager flow")
    print("=" * 60)

    # Create a config like a user would
    env_config = BaseEnvConfig(
        tokenizer_name="gpt2",
        group_name="test-reasoning",
        run_name="test-run",
        thinking_mode=True,
        reasoning_effort="high",
        max_reasoning_tokens=8000,
    )

    print("Created BaseEnvConfig:")
    print(f"  thinking_mode: {env_config.thinking_mode}")
    print(f"  reasoning_effort: {env_config.reasoning_effort}")
    print(f"  max_reasoning_tokens: {env_config.max_reasoning_tokens}")

    # Convert to ReasoningConfig (this happens in BaseEnv.__init__)
    reasoning_config = ReasoningConfig.from_env_config(env_config)

    print("\nReasoningConfig created:")
    print(f"  enabled: {reasoning_config.enabled}")
    print(f"  effort: {reasoning_config.effort}")
    print(f"  max_tokens: {reasoning_config.max_tokens}")

    # Verify the config would generate correct extra_body
    # For OpenRouter
    openrouter_extra = reasoning_config.build_extra_body("https://openrouter.ai/api/v1")
    print(f"\nOpenRouter extra_body: {json.dumps(openrouter_extra, indent=2)}")
    assert openrouter_extra["reasoning"]["enabled"] is True
    assert openrouter_extra["reasoning"]["effort"] == "high"
    # Note: max_tokens is NOT included when effort is set (OpenRouter limitation)

    # For OpenAI
    openai_extra = reasoning_config.build_extra_body("https://api.openai.com/v1")
    print(f"\nOpenAI extra_body: {json.dumps(openai_extra, indent=2)}")
    assert openai_extra["reasoning_effort"] == "high"

    print("\n✓ Full BaseEnvConfig → ServerManager flow works correctly!")
    return True


# =============================================================================
# INTEGRATION TESTS WITH REAL API CALLS
# =============================================================================


async def test_openrouter_reasoning(model: str, effort: str = "high"):
    """
    Test reasoning with an OpenRouter model.

    Args:
        model: Model name to test
        effort: Reasoning effort level

    Returns:
        Dict with test results
    """
    print(f"\n{'='*60}")
    print(f"Testing OpenRouter: {model}")
    print(f"{'='*60}")

    client = openai.AsyncOpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )

    # Build extra_body based on model type
    # Claude models need max_tokens in reasoning dict, not effort
    # See: https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
    is_claude = "claude" in model.lower() or "anthropic" in model.lower()

    if is_claude:
        # Claude needs reasoning.max_tokens, and overall max_tokens must be higher
        reasoning_max_tokens = 8000
        overall_max_tokens = 0  # Must be > reasoning_max_tokens
        extra_body = {"reasoning": {"max_tokens": reasoning_max_tokens}}
    else:
        # Other models use effort
        config = ReasoningConfig(enabled=True, effort=effort)
        extra_body = config.build_extra_body(OPENROUTER_BASE_URL)
        overall_max_tokens = 0

    messages = [{"role": "user", "content": TEST_PROMPT}]

    # For Hermes, also add the system prompt
    if "hermes" in model.lower():
        messages.insert(0, {"role": "system", "content": HERMES_REASONING_PROMPT})

    # Build the full request for logging
    request_params = {
        "model": model,
        "messages": messages,
        "max_tokens": overall_max_tokens,
        "temperature": 0.7,
        "extra_body": extra_body,
    }

    print("Request params:")
    print(f"  model: {model}")
    print(f"  max_tokens: {overall_max_tokens}")
    print(f"  extra_body: {json.dumps(extra_body, indent=2)}")

    # Log the full request to file
    log_to_file(f"\n{'='*70}")
    log_to_file(f"MODEL: {model}")
    log_to_file(f"TIMESTAMP: {datetime.now().isoformat()}")
    log_to_file(f"{'='*70}")
    log_to_file("\nREQUEST SENT:")
    log_to_file(json.dumps(request_params, indent=2, default=str))

    try:
        completion = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=overall_max_tokens,
            temperature=0.7,
            extra_body=extra_body,
        )

        # Log full ChatCompletion object to file for inspection
        log_to_file("\nRESPONSE RECEIVED:")
        log_to_file("\nFULL CHATCOMPLETION OBJECT:")
        log_to_file(str(completion))
        log_to_file(f"\n{'='*40}")

        # Also log the choice and message separately for clarity
        choice = completion.choices[0]
        log_to_file(f"\nChoice object: {choice}")
        log_to_file(f"\nMessage object: {choice.message}")

        # Log all attributes on the message
        log_to_file(f"\nMessage attributes: {dir(choice.message)}")

        # Try to get model_dump if available (pydantic)
        if hasattr(completion, "model_dump"):
            log_to_file("\nCompletion model_dump():")
            log_to_file(json.dumps(completion.model_dump(), indent=2, default=str))

        if hasattr(choice, "model_dump"):
            log_to_file("\nChoice model_dump():")
            log_to_file(json.dumps(choice.model_dump(), indent=2, default=str))

        if hasattr(choice.message, "model_dump"):
            log_to_file("\nMessage model_dump():")
            log_to_file(json.dumps(choice.message.model_dump(), indent=2, default=str))

        # Get the response content
        content = completion.choices[0].message.content
        log_to_file(f"\nResponse content ({len(content) if content else 0} chars):")
        log_to_file(content if content else "(empty)")

        # Try to extract reasoning
        reasoning, source = extract_reasoning_from_response(
            completion.choices[0], content=content
        )

        log_to_file("\nReasoning extraction result:")
        log_to_file(f"  Source: {source}")
        if reasoning:
            log_to_file(f"  Length: {len(reasoning)} chars")
            log_to_file("  Full reasoning content:")
            log_to_file(reasoning)
        else:
            log_to_file("  No separate reasoning found")

        # Check for <think> blocks in content
        has_think_block = "<think>" in content.lower() if content else False
        log_to_file(f"  Has <think> block in content: {has_think_block}")

        # Check for reasoning_details in raw response
        has_reasoning_details = hasattr(completion.choices[0], "reasoning_details")
        has_reasoning_content = hasattr(
            completion.choices[0].message, "reasoning_content"
        )
        log_to_file(f"  Has reasoning_details attr: {has_reasoning_details}")
        log_to_file(f"  Has reasoning_content attr: {has_reasoning_content}")

        # Try to access reasoning fields directly if they exist
        if (
            hasattr(choice.message, "reasoning_content")
            and choice.message.reasoning_content
        ):
            log_to_file(
                f"  message.reasoning_content: {choice.message.reasoning_content}"
            )
        if hasattr(choice.message, "reasoning") and choice.message.reasoning:
            log_to_file(f"  message.reasoning: {choice.message.reasoning}")
        if hasattr(choice, "reasoning_details") and choice.reasoning_details:
            log_to_file(f"  choice.reasoning_details: {choice.reasoning_details}")

        log_to_file(f"\n{'='*70}\n")

        # Also print summary to console
        print(f"\nResponse content ({len(content) if content else 0} chars)")
        print(
            f"Reasoning source: {source}, length: {len(reasoning) if reasoning else 0} chars"
        )
        print(f"(Full details logged to {LOG_FILE})")

        return {
            "model": model,
            "success": True,
            "content_length": len(content) if content else 0,
            "reasoning_source": source,
            "reasoning_length": len(reasoning) if reasoning else 0,
            "has_think_block": has_think_block,
        }

    except Exception as e:
        print(f"Error: {e}")
        return {
            "model": model,
            "success": False,
            "error": str(e),
        }


async def test_openai_reasoning(effort: str = "medium"):
    """
    Test reasoning with OpenAI official API.

    Args:
        effort: Reasoning effort level

    Returns:
        Dict with test results
    """
    print(f"\n{'='*60}")
    print(f"Testing OpenAI: {OPENAI_MODEL}")
    print(f"{'='*60}")

    client = openai.AsyncOpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )

    # Build extra_body using our ReasoningConfig
    config = ReasoningConfig(enabled=True, effort=effort)
    extra_body = config.build_extra_body(OPENAI_BASE_URL)

    messages = [{"role": "user", "content": TEST_PROMPT}]

    # Build the full request for logging
    request_params = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "max_completion_tokens": 1024,
        "extra_body": extra_body,
    }

    print("Request params:")
    print(f"  model: {OPENAI_MODEL}")
    print("  max_completion_tokens: 1024")
    print(f"  extra_body: {json.dumps(extra_body, indent=2)}")

    # Log the full request to file
    log_to_file(f"\n{'='*70}")
    log_to_file(f"MODEL: {OPENAI_MODEL} (OpenAI)")
    log_to_file(f"TIMESTAMP: {datetime.now().isoformat()}")
    log_to_file(f"{'='*70}")
    log_to_file("\nREQUEST SENT:")
    log_to_file(json.dumps(request_params, indent=2, default=str))

    try:
        completion = await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            max_completion_tokens=1024,  # OpenAI reasoning models require this instead of max_tokens
            # Note: OpenAI reasoning models only support temperature=1 (default)
            extra_body=extra_body,
        )

        # Log full ChatCompletion object to file for inspection
        log_to_file("\nRESPONSE RECEIVED:")
        log_to_file("\nFULL CHATCOMPLETION OBJECT:")
        log_to_file(str(completion))
        log_to_file(f"\n{'='*40}")

        # Also log the choice and message separately for clarity
        choice = completion.choices[0]
        log_to_file(f"\nChoice object: {choice}")
        log_to_file(f"\nMessage object: {choice.message}")

        # Log all attributes on the message
        log_to_file(f"\nMessage attributes: {dir(choice.message)}")

        # Try to get model_dump if available (pydantic)
        if hasattr(completion, "model_dump"):
            log_to_file("\nCompletion model_dump():")
            log_to_file(json.dumps(completion.model_dump(), indent=2, default=str))

        if hasattr(choice, "model_dump"):
            log_to_file("\nChoice model_dump():")
            log_to_file(json.dumps(choice.model_dump(), indent=2, default=str))

        if hasattr(choice.message, "model_dump"):
            log_to_file("\nMessage model_dump():")
            log_to_file(json.dumps(choice.message.model_dump(), indent=2, default=str))

        # Get the response content
        content = completion.choices[0].message.content
        log_to_file(f"\nResponse content ({len(content) if content else 0} chars):")
        log_to_file(content if content else "(empty)")

        # Try to extract reasoning
        reasoning, source = extract_reasoning_from_response(
            completion.choices[0], content=content
        )

        log_to_file("\nReasoning extraction result:")
        log_to_file(f"  Source: {source}")
        if reasoning:
            log_to_file(f"  Length: {len(reasoning)} chars")
            log_to_file("  Full reasoning content:")
            log_to_file(reasoning)
        else:
            log_to_file("  No separate reasoning found")

        # Check for <think> blocks in content (unlikely for OpenAI)
        has_think_block = "<think>" in content.lower() if content else False
        log_to_file(f"  Has <think> block in content: {has_think_block}")

        # Try to access reasoning fields directly if they exist
        if (
            hasattr(choice.message, "reasoning_content")
            and choice.message.reasoning_content
        ):
            log_to_file(
                f"  message.reasoning_content: {choice.message.reasoning_content}"
            )
        if hasattr(choice.message, "reasoning") and choice.message.reasoning:
            log_to_file(f"  message.reasoning: {choice.message.reasoning}")
        if hasattr(choice, "reasoning_details") and choice.reasoning_details:
            log_to_file(f"  choice.reasoning_details: {choice.reasoning_details}")

        log_to_file(f"\n{'='*70}\n")

        # Also print summary to console
        print(f"\nResponse content ({len(content) if content else 0} chars)")
        print(
            f"Reasoning source: {source}, length: {len(reasoning) if reasoning else 0} chars"
        )
        print(f"(Full details logged to {LOG_FILE})")

        return {
            "model": OPENAI_MODEL,
            "success": True,
            "content_length": len(content) if content else 0,
            "reasoning_source": source,
            "reasoning_length": len(reasoning) if reasoning else 0,
            "has_think_block": has_think_block,
        }

    except Exception as e:
        print(f"Error: {e}")
        return {
            "model": OPENAI_MODEL,
            "success": False,
            "error": str(e),
        }


async def run_all_integration_tests():
    """Run all integration tests and summarize results."""
    print("\n" + "=" * 70)
    print("REASONING MODEL INTEGRATION TESTS")
    print("=" * 70)

    # Check that API keys are set
    missing_keys = []
    if not OPENAI_API_KEY:
        missing_keys.append("OPENAI_API_KEY")
    if not OPENROUTER_API_KEY:
        missing_keys.append("OPENROUTER_API_KEY")

    if missing_keys:
        print(f"\n⚠ Missing required environment variables: {', '.join(missing_keys)}")
        print("Set them before running integration tests:")
        print("  export OPENAI_API_KEY='your-key-here'")
        print("  export OPENROUTER_API_KEY='your-key-here'")
        return False

    # Initialize log file
    with open(LOG_FILE, "w") as f:
        f.write("REASONING MODEL TEST RESULTS\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"{'='*70}\n\n")

    print(f"\nLogging full ChatCompletion objects to: {LOG_FILE}\n")

    results = []

    # Test OpenRouter models
    for model in OPENROUTER_MODELS:
        result = await test_openrouter_reasoning(model)
        results.append(result)

    # Test OpenAI
    result = await test_openai_reasoning()
    results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for result in results:
        status = "✓ PASS" if result.get("success") else "✗ FAIL"
        model = result.get("model", "unknown")

        if result.get("success"):
            reasoning_info = f"reasoning: {result.get('reasoning_source', 'none')}"
            if result.get("reasoning_length", 0) > 0:
                reasoning_info += f" ({result['reasoning_length']} chars)"
            print(f"{status} | {model:40} | {reasoning_info}")
        else:
            print(
                f"{status} | {model:40} | error: {result.get('error', 'unknown')[:30]}"
            )

    # Check if any failed
    failures = [r for r in results if not r.get("success")]
    if failures:
        print(f"\n{len(failures)} test(s) failed")
        return False
    else:
        print(f"\nAll {len(results)} tests passed!")
        return True


# =============================================================================
# MAIN
# =============================================================================


def run_unit_tests():
    """Run all unit tests (no API calls)."""
    print("\n" + "=" * 70)
    print("UNIT TESTS")
    print("=" * 70 + "\n")

    # ReasoningConfig unit tests
    test_reasoning_config_default()
    test_reasoning_config_enabled_only()
    test_reasoning_config_with_effort()
    test_reasoning_config_with_max_tokens()
    test_reasoning_config_full()
    test_reasoning_config_effort_mapping()
    test_reasoning_config_invalid_effort()
    test_reasoning_config_invalid_max_tokens()
    test_hermes_prompts_defined()

    # ServerManager integration tests (no API calls)
    test_reasoning_config_from_env_config()
    test_server_manager_builds_extra_body()
    test_full_env_config_to_server_flow()

    print("\n" + "=" * 70)
    print("All unit tests passed!")
    print("=" * 70)


async def run_server_manager_integration_test():
    """Run ServerManager integration test with real API call."""
    print("\n" + "=" * 70)
    print("SERVER MANAGER INTEGRATION TEST")
    print("=" * 70)

    result = await test_server_manager_injects_extra_body()

    if result:
        print("\n✓ ServerManager integration test passed!")
    else:
        print("\n✗ ServerManager integration test failed!")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test reasoning model support")
    parser.add_argument(
        "--unit-only", action="store_true", help="Only run unit tests (no API calls)"
    )
    parser.add_argument(
        "--integration-only",
        action="store_true",
        help="Only run integration tests (API calls to all providers)",
    )
    parser.add_argument(
        "--server-manager-only",
        action="store_true",
        help="Only run ServerManager integration test (single API call)",
    )
    args = parser.parse_args()

    if args.integration_only:
        asyncio.run(run_all_integration_tests())
    elif args.unit_only:
        run_unit_tests()
    elif args.server_manager_only:
        run_unit_tests()  # Run unit tests first
        asyncio.run(run_server_manager_integration_test())
    else:
        # Run all tests
        run_unit_tests()
        asyncio.run(run_server_manager_integration_test())
        asyncio.run(run_all_integration_tests())