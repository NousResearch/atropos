"""
AI Integration Module for CloudVR-PerfGuard
Provides FunSearch evolution and AI Scientist paper generation capabilities
"""

from .funsearch_integration import FunSearchIntegration
from .ai_scientist_integration import AIScientistIntegration
from .research_orchestrator import ResearchOrchestrator

__all__ = [
    'FunSearchIntegration',
    'AIScientistIntegration', 
    'ResearchOrchestrator'
] 