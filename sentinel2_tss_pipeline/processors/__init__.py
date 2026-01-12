"""
Processors module for Sentinel-2 TSS Pipeline.

Provides specialized processors for TSS, CHL, and water quality estimation.
"""

from .snap_calculator import SNAPTSMCHLCalculator, ProcessingResult

__all__ = [
    'SNAPTSMCHLCalculator',
    'ProcessingResult',
]
