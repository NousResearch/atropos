import argparse
import time
import logging
import signal
import sys
import requests
import wandb
from atroposlib.orchestration.metrics import MetricsCollector
from atroposlib.orchestration.controller import ScalingController
from atroposlib.orchestration.strategy import LocalActor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("DEO")

def fetch_wandb_info(server_url: str):
    """Fetch wandb project/group info from Atropos server."""
    try:
        resp = requests.get(f"{server_url}/wandb_info", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.debug(f"Could not fetch wandb info from server: {e}")
    return {"group": None, "project": None}

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
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("atroposlib.orchestration").setLevel(logging.DEBUG)

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
    env_command_list = args.env_command.split()
    actor = LocalActor(env_command_list)
    
    # 4. Initialize WandB if requested
    if args.wandb:
        wb_info = fetch_wandb_info(args.server_url)
        if wb_info.get("project") and wb_info.get("group"):
            wandb.init(
                project=wb_info["project"],
                group=wb_info["group"],
                name=f"deo-{int(time.time())}",
                job_type="orchestration",
                config=vars(args)
            )
            logger.info(f"WandB initialized: {wb_info['project']}/{wb_info['group']}")
        else:
            logger.warning("WandB enabled but server returned no project/group. Logging disabled.")

    logger.info(f"Starting DEO against {args.server_url}...")
    
    def handle_shutdown(sig, frame):
        logger.info("Shutdown signal received. Cleaning up...")
        actor.cleanup()
        if args.wandb:
            wandb.finish()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    try:
        while True:
            metrics = collector.poll()
            if metrics:
                current_actors = actor.get_current_count()
                target_actors = controller.calculate_desired(metrics, current_actors)
                
                if target_actors != current_actors:
                    actor.set_instance_count(target_actors)
                
                if args.wandb and wandb.run:
                    wandb.log({
                        "deo/rollout_pressure": metrics.rollout_pressure,
                        "deo/num_actors": current_actors,
                        "deo/queue_size": metrics.queue_size,
                        "deo/target_actors": target_actors,
                        "deo/total_rollouts": metrics.total_rollouts
                    })
            else:
                logger.warning("Could not fetch metrics. Check if Atropos server is running.")
            
            time.sleep(args.poll_interval)
            
    except Exception as e:
        logger.error(f"DEO loop crashed: {e}")
        actor.cleanup()
        if args.wandb:
            wandb.finish()

if __name__ == "__main__":
    main()
