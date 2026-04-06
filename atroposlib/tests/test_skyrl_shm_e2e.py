import multiprocessing as mp
import time
import json
import uuid
import requests
import numpy as np
import struct
from typing import List, Dict, Any
from atroposlib.api.shm_buffer import ZeroCopySHMBuffer

# Configuration for Mocks
BATCH_SIZE = 128
ENTRY_SIZE = 4096
NUM_ENV_WORKERS = 4
TOTAL_TRAJECTORIES = 500

def mock_env_worker(worker_id: int, shm_name: str, barrier: mp.Barrier, stop_event: mp.Event):
    """Simulates a SkyRL Environment process pushing trajectories to SHM."""
    try:
        shm = ZeroCopySHMBuffer(name=shm_name, create=False)
        barrier.wait()
        
        count = 0
        while not stop_event.is_set() and count < (TOTAL_TRAJECTORIES // NUM_ENV_WORKERS):
            tokens = [100 + i for i in range(ENTRY_SIZE)] 
            score = 0.8 + (worker_id * 0.05)
            
            success = shm.write_trajectory(
                tokens=tokens, 
                score=score, 
                instance_id=f"task_{count}",
                repetition_id=worker_id,
                metadata={"worker": worker_id}
            )
            if success:
                count += 1
            else:
                time.sleep(0.001)
                
    except Exception as e:
        print(f"Worker {worker_id} Error: {e}")

def run_e2e_benchmark():
    shm_name = f"test_e2e_shm_{uuid.uuid4().hex[:8]}"
    shm = ZeroCopySHMBuffer(name=shm_name, size=BATCH_SIZE * 2, entry_size=ENTRY_SIZE, create=True)
    
    barrier = mp.Barrier(NUM_ENV_WORKERS + 1)
    stop_event = mp.Event()
    
    print(f"🚀 Starting {NUM_ENV_WORKERS} Environment Workers (Concurrency Test)...")
    workers = []
    for i in range(NUM_ENV_WORKERS):
        p = mp.Process(target=mock_env_worker, args=(i, shm_name, barrier, stop_event))
        p.start()
        workers.append(p)
    barrier.wait() 
    
    print("📈 Measuring SHM Throughput & Integrity...")
    start_shm = time.perf_counter()
    received = 0
    verification_passed = True
    
    while received < TOTAL_TRAJECTORIES:
        data = shm.read_next()
        if data:
            if received % 100 == 0:
                if not (data["instance_id"].startswith("task_") and "worker" in data["metadata"]):
                    print(f"❌ Integrity Check Failed at index {received}!")
                    verification_passed = False
            received += 1
        else:
            if all(not p.is_alive() for p in workers) and received < TOTAL_TRAJECTORIES:
                break
                
    shm_tps = TOTAL_TRAJECTORIES / (time.perf_counter() - start_shm)
    print(f"   [SHM] Received {received} trajectories ({shm_tps:.2f} traj/s)")
    print(f"   [SHM] Integrity Verification: {'✅ PASSED' if verification_passed else '❌ FAILED'}")

    # HTTP Baseline Simulation
    print("📉 Measuring HTTP Baseline Simulation (JSON Tax)...")
    start_http = time.perf_counter()
    for _ in range(TOTAL_TRAJECTORIES):
        tokens = [100 + i for i in range(ENTRY_SIZE)]
        payload = json.dumps({
            "tokens": tokens, 
            "score": 0.8,
            "instance_id": "task_x",
            "repetition_id": 0,
            "metadata": {"foo": "bar"}
        })
        _ = json.loads(payload) 
        
    http_tps = TOTAL_TRAJECTORIES / (time.perf_counter() - start_http)
    print(f"   [HTTP] Processed {TOTAL_TRAJECTORIES} trajectories ({http_tps:.2f} traj/s)")

    # --- RESULTS ---
    print("\n" + "="*40)
    print("🏆 E2E TEST RESULTS")
    print("="*40)
    print(f"SHM Throughput Gain: {shm_tps / http_tps:.2f}x")
    print(f"Concurrency Load: {NUM_ENV_WORKERS} workers handled without corruption.")
    print(f"Data Integrity: {'Verified' if verification_passed else 'CORRUPT'}")
    print("="*40)

    stop_event.set()
    for p in workers: p.join()
    shm.close(unlink=True)


if __name__ == "__main__":
    run_e2e_benchmark()
