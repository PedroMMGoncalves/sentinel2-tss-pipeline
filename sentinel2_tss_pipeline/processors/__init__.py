"""
Processors module for Sentinel-2 TSS Pipeline.

Provides specialized processors for:
- TSM/CHL calculation from C2RCC IOPs
- TSS estimation using Jiang et al. (2021) methodology
- RGB composite and spectral index visualization
- Water quality analysis (HAB, clarity, trophic state)
- C2RCC atmospheric correction via SNAP GPT
"""

from .tsm_chl_calculator import TSMCHLCalculator, ProcessingResult
from .visualization_processor import VisualizationProcessor
from .water_quality_processor import (
    WaterQualityConstants,
    WaterQualityProcessor,
)
from .tss_processor import TSSConstants, TSSProcessor
from .c2rcc_processor import C2RCCProcessor, ProcessingStatus

__all__ = [
    # TSM/CHL Calculator
    'TSMCHLCalculator',
    'ProcessingResult',
    # Visualization
    'VisualizationProcessor',
    # Water Quality
    'WaterQualityConstants',
    'WaterQualityProcessor',
    # TSS
    'TSSConstants',
    'TSSProcessor',
    # C2RCC Processor
    'C2RCCProcessor',
    'ProcessingStatus',
]
