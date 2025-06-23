"""
2,500-Agent LLM Society Simulation

A comprehensive framework for simulating large-scale LLM-driven agent societies
using Mesa-frames, FLAME GPU, Atropos, and 3D asset generation.
"""

__version__ = "0.1.0"
__author__ = "Research Team"
__email__ = "research@example.com"

from . import agents
from . import simulation  
from . import llm
from . import assets
from . import monitoring
from . import utils

__all__ = [
    "agents",
    "simulation", 
    "llm",
    "assets",
    "monitoring",
    "utils",
] 