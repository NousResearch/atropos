import subprocess
import os
import signal
import time
import sys
import random
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

# --- Mock Atropos Server for Stress Testing ---
class MockAtroposHandler(BaseHTTPRequestHandler):
    data = {
        "current_step": 100,
        "queue_size": 10,
        "total_rollouts_processed": 5000,
        "unallocated_fraction": 0.5,
        "num_connected_envs": 1,
        "batch_size": 10
    }
    is_down = False

    def do_GET(self):
        if MockAtroposHandler.is_down:
            self.send_response(503)
            self.end_headers()
            return
        
        if self.path == "/global-status":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(MockAtroposHandler.data).encode())
        elif self.path == "/wandb_info":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"project": "chaos", "group": "stress"}).encode())

def run_mock_server():
    server = HTTPServer(("localhost", 8999), MockAtroposHandler)
    server.serve_forever()

# --- Chaos Monkey Logic ---
def run_chaos_test():
    print("Starting DEO Chaos Monkey Stress Test...")
    
    # 1. Start Mock Server
    threading.Thread(target=run_mock_server, daemon=True).start()
    time.sleep(1)

    # 2. Start DEO in background
    # Use a dummy env command that actually exists but is slow to start
    # We'll use 'sleep 1000' as the command
    deo_cmd = [
        sys.executable, "-m", "atroposlib.cli.orchestrate",
        "--server-url", "http://localhost:8999",
        "--env-command", "sleep 1000",
        "--min-actors", "1",
        "--max-actors", "10",
        "--poll-interval", "2",
        "--cooldown", "5",
        "--max-step", "5",
        "--verbose"
    ]
    print("Launching DEO...")
    deo_proc = subprocess.Popen(deo_cmd, preexec_fn=os.setpgrp)
    
    try:
        # A. Test Rapid Scale UP
        print("\n[Chaos] Rapid Scale UP (Target 10)...")
        MockAtroposHandler.data["queue_size"] = 100 # Pressure 10.0
        time.sleep(8)
        
        # B. Random Killing (KILL -9)
        print("\n[Chaos] Randomly Killing 2 Workers...")
        # Get managed PIDs from ps
        try:
            out = subprocess.check_output(["pgrep", "-f", "sleep 1000"]).decode().split()
            if out:
                to_kill = random.sample(out, min(2, len(out)))
                for pid in to_kill:
                    print(f"BAM! SIGKILL on {pid}")
                    os.kill(int(pid), signal.SIGKILL)
        except: pass
        time.sleep(10)
        
        # C. Network Flapping (Grace Period Test)
        print("\n[Chaos] Simulating Network Failure (10s)...")
        MockAtroposHandler.is_down = True
        time.sleep(10)
        MockAtroposHandler.is_down = False
        print("Network Restored.")
        time.sleep(5)
        
        # D. Rapid Scale DOWN (Graceful Drain Test)
        print("\n[Chaos] Rapid Scale DOWN (Target 1)...")
        MockAtroposHandler.data["queue_size"] = 1 # Pressure 0.1
        time.sleep(15)
        
    finally:
        print("\nCleaning up...")
        os.killpg(os.getpgid(deo_proc.pid), signal.SIGTERM)
        deo_proc.wait()
        
        # E. THE FINAL AUDIT
        print("\n--- FINAL CHAOS AUDIT ---")
        # Ensure no 'sleep 1000' processes remain
        try:
            leaked = subprocess.check_output(["pgrep", "-f", "sleep 1000"]).decode().split()
            if leaked:
                print(f"❌ FAILED: Leaked Processes: {leaked}")
                sys.exit(1)
        except subprocess.CalledProcessError:
            print("SUCCESS: No leaked processes.")
        
        print("\nDEO STRESS TEST PASSED")

if __name__ == "__main__":
    run_chaos_test()
