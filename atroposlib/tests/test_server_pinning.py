import asyncio
import os
import sys
import unittest
import warnings
from unittest.mock import AsyncMock, MagicMock, patch

# Add atropos to path relative to tests dir
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from atroposlib.envs.server_handling.server_manager import ServerManager, APIServerConfig
from atroposlib.envs.server_handling.server_baseline import APIServer

class MockConfig(APIServerConfig):
    base_url: str = "http://localhost:1111"
    model_name: str = "test-model"
    server_type: str = "sglang" # Use sglang to bypass OpenAI restrictions

class MockServer(APIServer):
    def __init__(self, config, reasoning_config=None):
        super().__init__(config, reasoning_config=reasoning_config)
        self.server_healthy = True
        self.sem = asyncio.Semaphore(10)
        self.eval_sem = asyncio.Semaphore(10)

class DummyTokenizer:
    def encode(self, *args, **kwargs):
        return [1, 2, 3]
    def decode(self, *args, **kwargs):
        return "dummy"


class TestServerPinning(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        os.environ["ATROPOS_ALLOW_DUMMY_MANAGED_SERVER"] = "1"
        self.config1 = MockConfig(base_url="http://worker-1:8000", server_type="openai")
        self.config2 = MockConfig(base_url="http://worker-2:8000", server_type="openai")
        self.config3 = MockConfig(base_url="http://worker-3:8000", server_type="openai")
        
        # Test ServerManager initialization
        self.manager = ServerManager(
            configs=[self.config1, self.config2, self.config3],
            server_class=MockServer,
            testing=False,
            slurm=False
        )

    async def test_managed_server_pinning_respects_base_url(self):
        """Verify that managed_server strictly follows a base_url pin."""
        # Make worker-2 very busy (fewest slots available)
        self.manager.servers[0].sem = asyncio.Semaphore(10)
        self.manager.servers[1].sem = asyncio.Semaphore(1)  # worker-2
        self.manager.servers[2].sem = asyncio.Semaphore(10)

        # Attempt to pin to worker-2
        target_url = "http://worker-2:8000"
        async with self.manager.managed_server(base_url=target_url, tokenizer=DummyTokenizer()) as managed:
            self.assertEqual(
                managed.server.config.base_url, 
                target_url, 
                "Pinning failed: ServerManager returned the wrong server."
            )

    async def test_managed_server_pinning_fallback(self):
        """Verify fallback behavior when pinned base_url is unhealthy/invalid."""
        self.manager.servers[0].sem = asyncio.Semaphore(5)
        self.manager.servers[1].sem = asyncio.Semaphore(10) # worker-2 (most available)
        self.manager.servers[2].sem = asyncio.Semaphore(2)

        fake_url = "http://worker-fake:8000"
        
        # Should fallback to worker-2 because it's most available
        async with self.manager.managed_server(base_url=fake_url, tokenizer=DummyTokenizer()) as managed:
            self.assertEqual(
                managed.server.config.base_url, 
                "http://worker-2:8000",
                "Fallback failed: Did not select most available valid server."
            )

    async def test_managed_server_session_id_mapping(self):
        """Verify that session_id accurately maps to a specific server over most available."""
        # Make all servers equally available so we can trust the hash determinism
        self.manager.servers[0].sem = asyncio.Semaphore(10)
        self.manager.servers[1].sem = asyncio.Semaphore(10)
        self.manager.servers[2].sem = asyncio.Semaphore(10)
        
        # 'demo_session_1_for_hashing' parses to a specific index. Let's rely on that hash being stable.
        target_session = "demo_session_1_for_hashing"
        
        # Test routing by session id
        async with self.manager.managed_server(session_id=target_session) as managed:
            url_1 = managed.server.config.base_url
            
        # Do it again to ensure stability
        async with self.manager.managed_server(session_id=target_session) as managed2:
            url_2 = managed2.server.config.base_url
            
        self.assertEqual(url_1, url_2, "Session ID mapping should be deterministic.")

    async def test_managed_server_unhealthy_fallback(self):
        """Verify fallback behavior when pinned base_url is unhealthy."""
        # Make worker-1 targetted but unhealthy
        self.manager.servers[0].server_healthy = False
        
        # Make worker-3 most available
        self.manager.servers[1].sem = asyncio.Semaphore(2) 
        self.manager.servers[2].sem = asyncio.Semaphore(10) # worker-3

        target_url = "http://worker-1:8000"
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            async with self.manager.managed_server(base_url=target_url, tokenizer=DummyTokenizer()) as managed:
                self.assertEqual(
                    managed.server.config.base_url, 
                    "http://worker-3:8000",
                    "Unhealthy node bypass failed."
                )

if __name__ == '__main__':
    unittest.main()
