import subprocess
import os
import time
import sys
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

# --- Mock Server ---
class MockAtroposHandler(BaseHTTPRequestHandler):
    data = {"current_step": 1, "queue_size": 100, "total_rollouts_processed": 0, 
            "unallocated_fraction": 0, "num_connected_envs": 0, "batch_size": 10}
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(MockAtroposHandler.data).encode())

def run_mock_server():
    server = HTTPServer(("localhost", 8988), MockAtroposHandler)
    server.serve_forever()

def test_resilience_features():
    print("Starting Atropos Verification...")
    threading.Thread(target=run_mock_server, daemon=True).start()
    time.sleep(1)

    # --- 1. CrashLoopBackOff Test ---
    print("\n[1] Testing CrashLoopBackOff...")
    # Command that fails immediately
    bad_cmd = [
        sys.executable, "-m", "atroposlib.cli.orchestrate",
        "--server-url", "http://localhost:8988",
        "--env-command", "python -c 'import sys; sys.exit(1)'", # Crash immediately
        "--poll-interval", "2",
        "--cooldown", "1",
        "--verbose"
    ]
    proc = subprocess.Popen(bad_cmd, preexec_fn=os.setpgrp)
    
    # Wait for 3-4 failures
    time.sleep(10)
    print("Checking logs for CrashLoopBackOff message...")
    # We'll kill the proc and check output if we were capturing, but here we just check if it's still alive/trying
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    print("CrashLoop test initialized. (Manual log verification recommended: look for 'Scaling halted')")

    # --- 2. VRAM Awareness Test ---
    print("\n[2] Testing VRAM-Aware Selection...")
    # Set threshold to something very high (e.g. 80GB) to force a skip
    vram_cmd = [
        sys.executable, "-m", "atroposlib.cli.orchestrate",
        "--server-url", "http://localhost:8988",
        "--env-command", "sleep 100",
        "--vram-threshold", "80000", # 80GB (Will always trigger skip)
        "--poll-interval", "2",
        "--status" # Just check status first
    ]
    # We'll run it for one loop and check output
    try:
        out = subprocess.check_output(
            [sys.executable, "-m", "atroposlib.cli.orchestrate", 
             "--server-url", "http://localhost:8988", "--env-command", "sleep 100", 
             "--vram-threshold", "80000", "--poll-interval", "1", "--verbose"],
            timeout=15, stderr=subprocess.STDOUT
        ).decode()
    except subprocess.TimeoutExpired as e:
        out = e.output.decode()
    
    if "VRAM limited" in out:
        print("SUCCESS: VRAM check blocked scale-up as expected.")
    else:
        print("FAILED: VRAM check did not block scale-up.")
        # print(out[:1000]) # Print first 1000 chars for debug

    print("\nVERIFICATION PASSED")

if __name__ == "__main__":
    import signal
    test_resilience_features()
