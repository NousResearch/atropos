"""
FLAME GPU 2 Integration Module for LLM Society Simulation Phase β

This module provides GPU-accelerated simulation capabilities for handling
501-2,500 agents with complex social, economic, and cultural interactions.

Key Features:
- FLAME GPU 2 kernel implementations for agent behaviors
- GPU-accelerated spatial interactions and social networks
- Parallel economic transaction processing
- Cultural influence propagation on GPU
- Memory-efficient agent state management
"""

from .agent_kernels import (
    CulturalInfluenceKernel,
    EconomicTradeKernel,
    FamilyInteractionKernel,
    MovementKernel,
    ResourceManagementKernel,
    SocialInteractionKernel,
)
from .flame_gpu_simulation import FlameGPUSimulation
from .gpu_memory_manager import GPUMemoryManager
from .performance_profiler import FlameGPUProfiler

__all__ = [
    "FlameGPUSimulation",
    "SocialInteractionKernel",
    "EconomicTradeKernel",
    "CulturalInfluenceKernel",
    "MovementKernel",
    "FamilyInteractionKernel",
    "ResourceManagementKernel",
    "GPUMemoryManager",
    "FlameGPUProfiler",
]
