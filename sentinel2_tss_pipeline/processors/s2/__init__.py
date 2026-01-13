"""
Sentinel-2 Processing Module.

Contains S2 SNAP GPT processing and TSM/CHL calculation.
"""

from .processor import S2Processor, ProcessingStatus
from .snap_calculator import TSMChlorophyllCalculator, ProcessingResult

# Backwards compatibility
SNAPTSMCHLCalculator = TSMChlorophyllCalculator

__all__ = [
    'S2Processor',
    'ProcessingStatus',
    'TSMChlorophyllCalculator',
    'ProcessingResult',
    # Backwards compatibility
    'SNAPTSMCHLCalculator',
]
