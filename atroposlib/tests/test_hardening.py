import os
import signal
import subprocess
import sys
import time

from atroposlib.orchestration.strategy import LocalActor


def test_maintainer_standard():
    print("Running hardening verification...")

    # --- 1. Adopt Existing Processes ---
    print("\n[1] Testing process adoption...")
    orphan = subprocess.Popen(["sleep", "300"], preexec_fn=os.setpgrp)
    actor = LocalActor(["sleep", "300"])

    pids = [p.pid for p in actor.processes]
    print(f"Associated PIDs: {pids}")
    success_adopt = orphan.pid in pids

    # --- 2. Graceful Drain ---
    print("\n[2] Testing graceful drain...")
    script = "/tmp/drain_worker.py"
    with open(script, "w") as f:
        f.write("""
import signal, time, sys
def handler(sig, frame):
    time.sleep(3)
    sys.exit(0)
signal.signal(signal.SIGUSR1, handler)
while True: time.sleep(1)
""")

    worker_actor = LocalActor([sys.executable, script])
    worker_actor.set_instance_count(1)

    # Wait for startup
    time.sleep(1)

    start = time.time()
    worker_actor.set_instance_count(0)  # Triggers SIGUSR1 + Loop
    duration = time.time() - start
    print(f"Drain duration: {duration:.2f}s")

    success_drain = 2.0 < duration < 8.0

    # Cleanup
    actor.cleanup()
    worker_actor.cleanup()

    if success_adopt and success_drain:
        print("\nHARDENING VERIFIED")
        sys.exit(0)
    else:
        print("\nVERIFICATION FAILED")
        print(f"Adopt: {success_adopt}, Drain: {success_drain}")
        sys.exit(1)


if __name__ == "__main__":
    test_maintainer_standard()
