import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from environments.cline_env.nomad_manager import NomadManager, NomadJobError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    load_dotenv()
    nomad_addr = os.getenv("NOMAD_ADDR", "http://127.0.0.1:4646")
    job_file = Path(__file__).with_name("nomad_job.hcl")
    manager = NomadManager(job_file, nomad_address=nomad_addr, job_name="cline-smoke")

    job_vars = {
        "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY", ""),
        "anthropic_model": os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4.5-20250929"),
    }

    if not job_vars["anthropic_api_key"]:
        raise RuntimeError("ANTHROPIC_API_KEY must be set")

    try:
        logger.info("Submitting Nomad job %s", job_file)
        manager.submit(job_vars)
        alloc_id = manager.wait_for_allocation()
        logger.info("Allocation %s is running", alloc_id)
        logger.info("Inspect logs via: nomad alloc logs %s", alloc_id)
    except (NomadJobError, TimeoutError) as exc:
        logger.error("Nomad run failed: %s", exc)
    finally:
        manager.stop_job()


if __name__ == "__main__":
    main()
