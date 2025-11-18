import logging
from pathlib import Path

from cline_core_launcher import ClineCoreConfig, ClineCoreProcess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    this_dir = Path(__file__).resolve().parent

    # In Atropos, Cline is vendored at environments/cline_env/cline
    cline_root = this_dir / "cline"
    if not cline_root.exists():
        raise FileNotFoundError(
            f"Expected Cline repo at {cline_root}. "
            "Ensure the Cline submodule is checked out at environments/cline_env/cline."
        )

    config = ClineCoreConfig(
        cline_root=cline_root,
        protobus_port=26040,
        hostbridge_port=26041,
        workspace_dir=None,
        use_coverage=False,
    )

    logger.info("Starting Cline core gRPC server for smoke test")
    proc = ClineCoreProcess(config)

    try:
        proc.start(timeout=60.0)
        logger.info(
            "Cline core is listening on 127.0.0.1:%d", config.protobus_port
        )
        logger.info("Smoke test succeeded")
    except Exception as exc:
        logger.exception("Cline core smoke test failed: %s", exc)
    finally:
        logger.info("Stopping Cline core process")
        proc.stop()


if __name__ == "__main__":
    main()

