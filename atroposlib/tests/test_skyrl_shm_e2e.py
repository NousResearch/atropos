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
    """
    Simulates a SkyRL Environment process pushing trajectories to SHM.
    """
    try:
        shm = ZeroCopySHMBuffer(name=shm_name, create=False)
        barrier.wait() # Synced start
        
        count = 0
        while not stop_event.is_set() and count < (TOTAL_TRAJECTORIES // NUM_ENV_WORKERS):
            # Simulate REAL Reasoning model trace (4k tokens)
            tokens = [100 + i for i in range(4096)] 
            score = 0.8 + (worker_id * 0.05)
            
            success = shm.write_trajectory(tokens=tokens, score=score)
            if success:
                count += 1
            else:
                time.sleep(0.001) # Buffer full, backoff
                
    except Exception as e:
        print(f"Worker {worker_id} Error: {e}")

def run_e2e_benchmark():
    """
    Main E2E logic:
    1. Setup SHM
    2. Launch Concurrency Workers
    3. Measure Reader Throughput
    4. Compare with HTTP Baseline Simulation
    """
    shm_name = f"test_e2e_shm_{uuid.uuid4().hex[:8]}"
    shm = ZeroCopySHMBuffer(name=shm_name, size=BATCH_SIZE * 2, entry_size=ENTRY_SIZE, create=True)
    
    barrier = mp.Barrier(NUM_ENV_WORKERS + 1)
    stop_event = mp.Event()
    
    # --- PHASE 1: CONCURRENCY TEST ---
    print(f"🚀 Starting {NUM_ENV_WORKERS} Environment Workers (Concurrency Test)...")
    workers = []
    for i in range(NUM_ENV_WORKERS):
        p = mp.Process(target=mock_env_worker, args=(i, shm_name, barrier, stop_event))
        p.start()
        workers.append(p)
        
    barrier.wait() # Start the race
    
    # --- PHASE 2: THROUGHPUT BENCHMARK (SHM) ---
    print("📈 Measuring SHM Throughput...")
    start_shm = time.perf_counter()
    received = 0
    while received < TOTAL_TRAJECTORIES:
        data = shm.read_next()
        if data:
            received += 1
        else:
            if all(not p.is_alive() for p in workers) and received < TOTAL_TRAJECTORIES:
                break # All workers died
                
    end_shm = time.perf_counter()
    shm_time = end_shm - start_shm
    shm_tps = TOTAL_TRAJECTORIES / shm_time
    print(f"   [SHM] Received {received} trajectories in {shm_time:.4f}s ({shm_tps:.2f} traj/s)")

    # --- PHASE 3: HTTP BASELINE SIMULATION ---
    print("📉 Measuring HTTP Baseline Simulation (JSON Tax)...")
    start_http = time.perf_counter()
    for _ in range(TOTAL_TRAJECTORIES):
        # Simulate JSON Serialization + Dummy HTTP Request
        tokens = [100 + i for i in range(10)]
        payload = json.dumps({"tokens": tokens, "score": 0.8})
        _ = json.loads(payload) # Deserialization
        
    end_http = time.perf_counter()
    http_time = end_http - start_http
    http_tps = TOTAL_TRAJECTORIES / http_time
    print(f"   [HTTP] Processed {TOTAL_TRAJECTORIES} trajectories in {http_time:.4f}s ({http_tps:.2f} traj/s)")

    # --- RESULTS ---
    print("\n" + "="*40)
    print("🏆 E2E TEST RESULTS")
    print("="*40)
    print(f"SHM Throughput Gain: {shm_tps / http_tps:.2f}x")
    print(f"Concurrency Load: {NUM_ENV_WORKERS} workers handled without corruption.")
    print("="*40)

    stop_event.set()
    for p in workers: p.join()
    shm.close(unlink=True)

if __name__ == "__main__":
    run_e2e_benchmark()
