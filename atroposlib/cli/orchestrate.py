import argparse
import time
import logging
import signal
import sys
from atroposlib.orchestration.metrics import MetricsCollector
from atroposlib.orchestration.controller import ScalingController
from atroposlib.orchestration.strategy import LocalActor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("DEO")

def main():
    parser = argparse.ArgumentParser(description="Atropos Elastic Orchestrator (DEO)")
    parser.add_argument("--server-url", type=str, default="http://localhost:8000", help="Atropos server URL")
    parser.add_argument("--env-command", type=str, required=True, help="Command to launch environment server")
    parser.add_argument("--min-actors", type=int, default=1, help="Min environment actors")
    parser.add_argument("--max-actors", type=int, default=20, help="Max environment actors")
    parser.add_argument("--target-pressure", type=float, default=1.0, help="Target Rollout Pressure (Queue/BatchSize)")
    parser.add_argument("--poll-interval", type=int, default=10, help="Poll interval in seconds")
    parser.add_argument("--cooldown", type=int, default=60, help="Scaling cooldown in seconds")
    parser.add_argument("--max-step", type=int, default=4, help="Max actors to add/remove at once")
    
    args = parser.parse_args()
    
    # 1. Initialize metrics collector
    collector = MetricsCollector(args.server_url)
    
    # 2. Initialize Scaling Controller
    controller = ScalingController(
        min_actors=args.min_actors,
        max_actors=args.max_actors,
        target_pressure=args.target_pressure,
        cooldown_seconds=args.cooldown,
        max_step_change=args.max_step
    )
    
    # 3. Initialize Strategy (LocalActor)
    # Convert command string to list
    env_command_list = args.env_command.split()
    actor = LocalActor(env_command_list)
    
    logger.info(f"Starting DEO against {args.server_url}...")
    logger.info(f"Command: {args.env_command}")
    
    # Graceful shutdown handler
    def handle_shutdown(sig, frame):
        logger.info("Shutdown signal received. Cleaning up...")
        actor.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    # 4. Main Control Loop
    try:
        while True:
            metrics = collector.poll()
            if metrics:
                current_actors = actor.get_current_count()
                target_actors = controller.calculate_desired(metrics, current_actors)
                
                if target_actors != current_actors:
                    actor.set_instance_count(target_actors)
            else:
                logger.warning("Could not fetch metrics. Check if Atropos server is running.")
            
            time.sleep(args.poll_interval)
            
    except Exception as e:
        logger.error(f"DEO loop crashed: {e}")
        actor.cleanup()

if __name__ == "__main__":
    main()
