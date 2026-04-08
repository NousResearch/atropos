import asyncio
import os
import sys
import hashlib
import argparse
import time
from typing import List
from transformers import AutoTokenizer

sys.path.append(os.getcwd())

from atroposlib.envs.server_handling.server_manager import ServerManager, APIServerConfig
from atroposlib.envs.server_handling.routing_utils import get_consistent_worker_index

async def run_real_e2e_test(worker_urls: List[str]):
    print(f"Hardware Verification on {len(worker_urls)} workers")
    
    configs = [
        APIServerConfig(
            base_url=url,
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            server_type="sglang",
            health_check=True
        ) for url in worker_urls
    ]
    
    manager = ServerManager(configs=configs)
    await asyncio.sleep(8)
    
    for i, s in enumerate(manager.servers):
        print(f"Worker {i} ({s.config.base_url}) Healthy: {s.server_healthy}")

    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    session_id = "conversation-alpha"
    idx = get_consistent_worker_index(hashlib.md5(session_id.encode()).hexdigest(), len(worker_urls))
    expected_url = worker_urls[idx]
    
    print(f"Session {session_id} -> Expected {expected_url}")

    messages = [{"role": "user", "content": "What is the capital of France?"}]
    
    # Turn 1
    async with manager.managed_server(session_id=session_id, tokenizer=tokenizer) as managed:
        url_t1 = managed.server.config.base_url
        print(f"Turn 1 directed to: {url_t1}")
        res1 = await managed.chat_completion(messages=messages, max_tokens=20)
        content1 = res1.choices[0].message.content.strip()
        print(f"Response 1: {content1}")
    
    # Turn 2
    history = messages + [{"role": "assistant", "content": content1}]
    messages_t2 = history + [{"role": "user", "content": "And its population?"}]
    
    async with manager.managed_server(session_id=session_id, tokenizer=tokenizer) as managed:
        url_t2 = managed.server.config.base_url
        print(f"Turn 2 directed to: {url_t2}")
        
        if url_t1 != url_t2:
            print(f"FAIL: Pinning failed ({url_t1} != {url_t2})")
            sys.exit(1)
            
        res2 = await managed.chat_completion(messages=messages_t2, max_tokens=20)
        print(f"Response 2: {res2.choices[0].message.content.strip()}")

    print("\nE2E VERIFICATION SUCCESSFUL")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", nargs="+", default=["http://localhost:30001", "http://localhost:30002"])
    args = parser.parse_args()
    asyncio.run(run_real_e2e_test(args.workers))
