"""
Atropos API communication utilities.

Handles communication with the Atropos API server for:
- Server health checks
- Trainer registration
- Batch retrieval
"""

import time as _time

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import TrainingConfig


def check_atropos_api(url: str = "http://localhost:8000", timeout: float = 30.0) -> bool:
    """
    Check if the Atropos API server is reachable.

    Args:
        url: Base URL of the Atropos API server
        timeout: Maximum time to wait for the server

    Returns:
        True if server is reachable
    """
    start = _time.time()
    while _time.time() - start < timeout:
        try:
            response = requests.get(f"{url}/info", timeout=2)
            if response.status_code == 200:
                print(f"[Trainer] ✓ Atropos API server is reachable at {url}")
                return True
        except requests.exceptions.ConnectionError:
            pass
        except Exception as e:
            print(f"[Trainer] Waiting for Atropos API at {url}... ({e})")
        _time.sleep(1)

    print(f"[Trainer] ⚠ Warning: Atropos API server not reachable at {url}")
    return False


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30))
def register_trainer(config: TrainingConfig):
    """
    Register the trainer with the Atropos API.

    Verifies registration succeeded before returning.
    """
    url = config.atropos_url
    response = requests.post(
        f"{url}/register",
        json={
            # wandb fields are required strings - use empty string if None
            "wandb_group": config.wandb_group or "",
            "wandb_project": config.wandb_project or "",
            "batch_size": config.batch_size * config.gradient_accumulation_steps,
            "max_token_len": config.seq_len,
            "starting_step": 0,
            "checkpoint_dir": config.save_path,
            "save_checkpoint_interval": config.training_steps,
            "num_steps": config.training_steps,
        },
        timeout=10,
    )

    # Check for HTTP errors
    response.raise_for_status()

    # Verify we got a valid response with UUID
    data = response.json()
    if "uuid" not in data:
        raise RuntimeError(f"Registration failed: {data}")

    print(f"[Trainer] ✓ Registered with Atropos API at {url} (uuid: {data['uuid']})")


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30))
def get_batch(url: str = "http://localhost:8000"):
    """
    Get a batch of training data from the Atropos API.
    
    Args:
        url: Base URL of the Atropos API server
        
    Returns:
        Batch data dictionary containing tokens, masks, scores, etc.
        
    Raises:
        RuntimeError: If trainer is not registered or other API error
    """
    data = requests.get(f"{url}/batch", timeout=10).json()

    # Check if there was an error (trainer not registered)
    if data.get("status") == "error":
        raise RuntimeError(f"Atropos API error: {data.get('message', 'Unknown error')}")

    return data

