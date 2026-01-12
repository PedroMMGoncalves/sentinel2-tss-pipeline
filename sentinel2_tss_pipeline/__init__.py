"""
Sentinel-2 TSS Pipeline Package

A comprehensive pipeline for processing Sentinel-2 imagery and estimating
Total Suspended Solids (TSS) using the Jiang et al. (2021) methodology.

Reference:
    Jiang, D., Matsushita, B., Pahlevan, N., et al. (2021).
    "Remotely Estimating Total Suspended Solids Concentration in Clear to
    Extremely Turbid Waters Using a Novel Semi-Analytical Method."
    Remote Sensing of Environment, 258, 112386.
    DOI: https://doi.org/10.1016/j.rse.2021.112386
"""

__version__ = "2.0.0"
__author__ = "Pedro Gonçalves"

# Import main config classes for convenience
from .config import (
    ProcessingMode,
    ProductType,
    ResamplingConfig,
    SubsetConfig,
    C2RCCConfig,
    WaterQualityConfig,
    MarineVisualizationConfig,
    JiangTSSConfig,
    ProcessingConfig,
)

# Import utility classes
from .utils import (
    ColoredFormatter,
    setup_enhanced_logging,
    get_default_logger,
    SafeMathNumPy,
    MemoryManager,
    RasterIO,
    ProductDetector,
    SystemMonitor,
)

# Import processor classes
from .processors import (
    SNAPTSMCHLCalculator,
    ProcessingResult,
)

__all__ = [
    # Config classes
    'ProcessingMode',
    'ProductType',
    'ResamplingConfig',
    'SubsetConfig',
    'C2RCCConfig',
    'WaterQualityConfig',
    'MarineVisualizationConfig',
    'JiangTSSConfig',
    'ProcessingConfig',
    # Utility classes
    'ColoredFormatter',
    'setup_enhanced_logging',
    'get_default_logger',
    'SafeMathNumPy',
    'MemoryManager',
    'RasterIO',
    'ProductDetector',
    'SystemMonitor',
    # Processor classes
    'SNAPTSMCHLCalculator',
    'ProcessingResult',
]
