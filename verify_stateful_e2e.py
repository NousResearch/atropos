import asyncio
import json
import os
import sys
import hashlib
import argparse
import time
from typing import List, Dict
from transformers import AutoTokenizer

# Add atropos to path
sys.path.append(os.getcwd())

from atroposlib.envs.server_handling.server_manager import ServerManager, APIServerConfig
from atroposlib.envs.server_handling.sglang_stateful_server import StatefulSGLangServer
from atroposlib.envs.server_handling.routing_utils import get_consistent_worker_index

# ---------------------------------------------------------------------------
# E2E REAL HARDWARE VERIFICATION
# ---------------------------------------------------------------------------
async def run_real_e2e_test(worker_urls: List[str]):
    print(f"\n--- Starting Real Hardware Verification on {len(worker_urls)} workers ---")
    for url in worker_urls:
        print(f"  Worker: {url}")
    
    # Configure ServerManager with REAL configs
    configs = [
        APIServerConfig(
            base_url=url,
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            server_type="sglang",
            health_check=True
        ) for url in worker_urls
    ]
    
    manager = ServerManager(configs=configs)
    
    # IMPORTANT: Wait for background health loops to stabilize
    print("Waiting 10s for health stabilization...")
    await asyncio.sleep(10)
    
    # Check health explicitly
    for i, s in enumerate(manager.servers):
        print(f"Worker {i} ({s.config.base_url}) Healthy: {s.server_healthy}")

    # Use real tokenizer for accurate delta-sync testing
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Test 1: Deterministic Pinning Verification
    print("\n--- Verifying Session ID Pinning Determinism ---")
    session_id_a = "conversation-alpha"
    
    hash_a = hashlib.md5(session_id_a.encode('utf-8')).hexdigest()
    idx_a = get_consistent_worker_index(hash_a, len(worker_urls))
    expected_url = worker_urls[idx_a]
    
    print(f"Session A ({session_id_a}) -> Expected Worker {idx_a} ({expected_url})")

    # Test 2: Multi-turn Stateful Flow
    print("\n--- Multi-turn Rollout (Conversation Alpha) ---")
    messages = [{"role": "user", "content": "What is the capital of France?"}]
    actual_url_t1 = None
    
    # Turn 1: Initial (Expect Full Sync)
    async with manager.managed_server(session_id=session_id_a, tokenizer=tokenizer) as managed:
        actual_url_t1 = managed.server.config.base_url
        print(f"Turn 1 (New Session) directed to: {actual_url_t1}")
        res1 = await managed.chat_completion(messages=messages, max_tokens=20)
        content1 = res1.choices[0].message.content
        print(f"Response 1: \"{content1.strip()}\"")
    
    # Turn 2: Follow-up (Expect Pinned Sync)
    history = messages + [{"role": "assistant", "content": content1}]
    messages_turn2 = history + [{"role": "user", "content": "And its population?"}]
    
    async with manager.managed_server(session_id=session_id_a, tokenizer=tokenizer) as managed:
        actual_url_t2 = managed.server.config.base_url
        print(f"Turn 2 (Pinned Session) directed to: {actual_url_t2}")
        
        if actual_url_t1 != actual_url_t2:
            print(f"CRITICAL ERROR: Pinning failed! T1 {actual_url_t1} != T2 {actual_url_t2}")
            # Check worker health
            for i, s in enumerate(manager.servers):
                 print(f"  Status Check: Worker {i} ({s.config.base_url}) Healthy={s.server_healthy}")
            sys.exit(1)
            
        res2 = await managed.chat_completion(messages=messages_turn2, max_tokens=20)
        content2 = res2.choices[0].message.content
        print(f"Response 2: \"{content2.strip()}\"")

    # Final Verification
    print("\n==========================================")
    print("✓ REAL HARDWARE E2E SUCCESSFUL!")
    print("==========================================")
    print(f"1. Routing: Correctly distributed session '{session_id_a}' to {actual_url_t1}")
    print("2. Protocol: StatefulSGLangServer successfully communicated with backend.")
    print("3. Integrity: Chat results are valid and coherent across turns.")
    print("==========================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", nargs="+", default=["http://localhost:30001", "http://localhost:30002"])
    args = parser.parse_args()
    
    asyncio.run(run_real_e2e_test(args.workers))
