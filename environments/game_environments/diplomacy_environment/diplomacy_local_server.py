#!/usr/bin/env python
"""
Local test server for Diplomacy minimal environment

This script:
1. Starts a mock Atropos policy server
2. Runs a quick test game
3. Useful for development and testing
"""

import asyncio
import json
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockPolicyServer:
    """Simple mock server that returns random but valid Diplomacy responses."""

    def __init__(self, port=8000):
        self.port = port
        self.app = None
        self.runner = None

    async def start(self):
        """Start the mock server."""
        from aiohttp import web

        self.app = web.Application()
        self.app.router.add_post("/v1/chat/completions", self.handle_chat_completion)
        self.app.router.add_post("/v1/completions", self.handle_completion)

        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "localhost", self.port)
        await site.start()
        logger.info(f"Mock policy server started on port {self.port}")

    async def stop(self):
        """Stop the mock server."""
        if self.runner:
            await self.runner.cleanup()

    async def handle_chat_completion(self, request):
        """Handle chat completion requests."""
        data = await request.json()

        # Extract the last user message
        messages = data.get("messages", [])
        last_message = messages[-1]["content"] if messages else ""

        # Generate a simple response based on the request type
        response_text = self._generate_response(last_message)

        # Return OpenAI-compatible response
        from aiohttp import web

        return web.json_response(
            {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": data.get("model", "mock-model"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": response_text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                },
            }
        )

    async def handle_completion(self, request):
        """Handle completion requests."""
        data = await request.json()
        prompt = data.get("prompt", "")

        response_text = self._generate_response(prompt)

        from aiohttp import web

        return web.json_response(
            {
                "id": f"cmpl-{int(time.time())}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": data.get("model", "mock-model"),
                "choices": [
                    {
                        "text": response_text,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                },
            }
        )

    def _generate_response(self, prompt: str) -> str:
        """Generate a mock response based on the prompt."""
        prompt_lower = prompt.lower()

        if "orders" in prompt_lower:
            # Return empty orders (AI_Diplomacy will use defaults)
            return json.dumps(
                {
                    "orders": {},
                    "explanations": {"general": "Mock server - using default orders"},
                }
            )
        elif "message" in prompt_lower or "negotiate" in prompt_lower:
            # Return no messages
            return json.dumps(
                {
                    "messages": [],
                    "explanations": {"general": "Mock server - no negotiations"},
                }
            )
        else:
            return "Mock response from test server"


async def run_test_game():
    """Run a quick test game with the mock server."""
    # Start mock server
    server = MockPolicyServer(port=8000)
    await server.start()

    try:
        # Import and run the environment
        from diplomacy_env_minimal import DiplomacyEnvMinimal, DiplomacyEnvMinimalConfig

        from atroposlib.envs.base import APIServerConfig

        # Create minimal config
        config = DiplomacyEnvMinimalConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
            group_size=2,  # Just 2 parallel games for testing
            max_game_turns=3,  # Very short games
            start_diplomacy_server=False,  # We'll start it manually
            save_game_logs=True,
        )

        server_configs = [
            APIServerConfig(
                model_name="training-policy",
                base_url="http://localhost:8000/v1",
                api_key="x",
            )
        ]

        # Create environment
        env = DiplomacyEnvMinimal(config, server_configs)
        await env.setup()

        # Run one trajectory collection
        item = await env.get_next_item()
        result = await env.collect_trajectory(item)

        if result[0]:
            logger.info(
                f"Successfully collected trajectory with score: {result[0]['scores']}"
            )
        else:
            logger.error("Failed to collect trajectory")

    finally:
        await server.stop()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Diplomacy minimal environment test server"
    )
    parser.add_argument("--quick", action="store_true", help="Run a quick test game")
    parser.add_argument(
        "--server-only", action="store_true", help="Just run the mock server"
    )
    parser.add_argument("--port", type=int, default=8000, help="Port for mock server")

    args = parser.parse_args()

    if args.server_only:
        # Just run the server
        async def run_server():
            server = MockPolicyServer(port=args.port)
            await server.start()
            logger.info(
                f"Mock server running on port {args.port}. Press Ctrl+C to stop."
            )
            try:
                await asyncio.Event().wait()
            except KeyboardInterrupt:
                await server.stop()

        asyncio.run(run_server())
    else:
        # Run test game
        asyncio.run(run_test_game())


if __name__ == "__main__":
    main()
