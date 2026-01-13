"""
Core module for Sentinel-2 TSS Pipeline.

Contains the main unified processor that coordinates all processing stages.
"""

from .unified_processor import UnifiedS2TSSProcessor

__all__ = [
    'UnifiedS2TSSProcessor',
]
