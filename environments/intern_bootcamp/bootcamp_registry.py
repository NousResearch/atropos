"""
Bootcamp Registry for InternBootcamp Environment

This module provides a registry system for dynamically discovering and managing
InternBootcamp tasks without having to manually import each one.
"""

import importlib
import inspect
import logging
from typing import Any, Dict, List, Type

logger = logging.getLogger(__name__)


class BootcampRegistry:
    """Registry for InternBootcamp tasks with dynamic discovery."""

    def __init__(self):
        self._registry: Dict[str, Type] = {}
        self._discovered = False

    def discover_bootcamps(self) -> None:
        """Dynamically discover all available bootcamp classes from InternBootcamp."""
        if self._discovered:
            return

        try:
            # Import the internbootcamp.bootcamp module
            bootcamp_module = importlib.import_module("internbootcamp.bootcamp")

            # Get all attributes from the module
            for name in dir(bootcamp_module):
                if name.endswith("bootcamp") and not name.startswith("_"):
                    try:
                        obj = getattr(bootcamp_module, name)
                        # Check if it's a class and has the required methods
                        if (
                            inspect.isclass(obj)
                            and hasattr(obj, "case_generator")
                            and hasattr(obj, "prompt_func")
                            and hasattr(obj, "verify_score")
                        ):
                            self._registry[name] = obj
                            logger.debug(f"Registered bootcamp: {name}")
                    except Exception as e:
                        logger.warning(f"Failed to register {name}: {e}")

            self._discovered = True
            logger.info(f"Discovered {len(self._registry)} bootcamp tasks")

        except ImportError as e:
            logger.error(f"Failed to import internbootcamp.bootcamp: {e}")
            raise

    def get_bootcamp_class(self, name: str) -> Type:
        """Get a bootcamp class by name."""
        if not self._discovered:
            self.discover_bootcamps()

        if name not in self._registry:
            available = self.list_available_bootcamps()
            raise ValueError(
                f"Unknown bootcamp: {name}. "
                f"Available bootcamps: {', '.join(available[:10])}..."
                f" ({len(available)} total)"
            )

        return self._registry[name]

    def create_bootcamp_instance(self, name: str, **params) -> Any:
        """Create an instance of a bootcamp with given parameters."""
        bootcamp_class = self.get_bootcamp_class(name)

        # Get the __init__ signature to see what parameters are accepted
        try:
            sig = inspect.signature(bootcamp_class.__init__)
            valid_params = {}

            # Filter out parameters that the bootcamp doesn't accept
            for param_name, param_value in params.items():
                if param_name in sig.parameters:
                    valid_params[param_name] = param_value
                else:
                    logger.warning(
                        f"Parameter '{param_name}' not accepted by {name}, ignoring"
                    )

            return bootcamp_class(**valid_params)

        except Exception as e:
            logger.error(f"Failed to create instance of {name}: {e}")
            # Try with no parameters as fallback
            try:
                return bootcamp_class()
            except Exception as e:
                raise e

    def list_available_bootcamps(self) -> List[str]:
        """List all available bootcamp names."""
        if not self._discovered:
            self.discover_bootcamps()
        return sorted(list(self._registry.keys()))

    def get_bootcamp_info(self, name: str) -> Dict[str, Any]:
        """Get information about a specific bootcamp."""
        bootcamp_class = self.get_bootcamp_class(name)

        info = {
            "name": name,
            "class": bootcamp_class,
            "docstring": inspect.getdoc(bootcamp_class) or "No documentation available",
            "parameters": {},
        }

        # Get __init__ parameters
        try:
            sig = inspect.signature(bootcamp_class.__init__)
            for param_name, param in sig.parameters.items():
                if param_name not in ["self"]:
                    param_info = {
                        "default": (
                            param.default
                            if param.default != inspect.Parameter.empty
                            else None
                        ),
                        "annotation": (
                            str(param.annotation)
                            if param.annotation != inspect.Parameter.empty
                            else None
                        ),
                    }
                    info["parameters"][param_name] = param_info
        except Exception as e:
            logger.warning(f"Could not inspect parameters for {name}: {e}")

        return info


# Global registry instance
bootcamp_registry = BootcampRegistry()


def get_available_bootcamps() -> List[str]:
    """Get a list of all available bootcamp names."""
    return bootcamp_registry.list_available_bootcamps()


def create_bootcamp(name: str, **params) -> Any:
    """Create a bootcamp instance by name with parameters."""
    return bootcamp_registry.create_bootcamp_instance(name, **params)
