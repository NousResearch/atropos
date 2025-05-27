"""
AI Integration module for CloudVR-PerfGuard
Provides interfaces to AI research tools for performance analysis
"""

from .data_adapter import PerformanceDataAdapter
from .paper_generator import ResearchPaperGenerator
from .function_discovery import OptimizationDiscovery

__version__ = "0.1.0"
__all__ = ["PerformanceDataAdapter", "ResearchPaperGenerator", "OptimizationDiscovery"] 