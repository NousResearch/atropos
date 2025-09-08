"""Factorio Learning Environment integration for Atropos RL trainer."""

# Ensure FLE is in the path
import sys
from pathlib import Path

fle_path = Path(__file__).parent / "fle"
if str(fle_path) not in sys.path:
    sys.path.insert(0, str(fle_path))

# Import key components
try:
    import fle
    from fle.env import FactorioInstance
    from fle.env.gym_env.registry import list_available_environments

    FLE_AVAILABLE = True
except ImportError:
    FLE_AVAILABLE = False
    print("Warning: FLE not available. Install with: pip install -e ./fle")

__all__ = ["FLE_AVAILABLE", "FactorioInstance", "list_available_environments"]
