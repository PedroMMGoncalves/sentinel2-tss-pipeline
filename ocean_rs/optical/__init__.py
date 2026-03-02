"""
OceanRS Optical — Sentinel-2 Water Quality Pipeline

A comprehensive pipeline for processing Sentinel-2 imagery and estimating
Total Suspended Solids (TSS) using the Jiang et al. (2021) methodology.

Reference:
    Jiang, D., Matsushita, B., Pahlevan, N., et al. (2021).
    "Remotely Estimating Total Suspended Solids Concentration in Clear to
    Extremely Turbid Waters Using a Novel Semi-Analytical Method."
    Remote Sensing of Environment, 258, 112386.
    DOI: https://doi.org/10.1016/j.rse.2021.112386

Part of the OceanRS toolkit.
"""

__version__ = "3.0.0"
__author__ = "Pedro Gonçalves"

# Import main config classes
from .config import (
    ProcessingMode,
    ProductType,
    ResamplingConfig,
    SubsetConfig,
    C2RCCConfig,
    WaterQualityConfig,
    OutputCategoryConfig,
    TSSConfig,
    ProcessingConfig,
)

# Import shared utility classes
from ocean_rs.shared import (
    ColoredFormatter,
    setup_enhanced_logging,
    get_default_logger,
    SafeMathNumPy,  # Re-exported for public API convenience
    MemoryManager,
    RasterIO,
)

# Import optical-specific utility classes
from .utils import (
    ProductDetector,
    SystemMonitor,
)
from .utils.output_structure import OutputStructure

# Import processor classes
from .processors import (
    TSMCHLCalculator,
    ProcessingResult,
    VisualizationProcessor,
    WaterQualityConstants,
    WaterQualityProcessor,
    TSSConstants,
    TSSProcessor,
    C2RCCProcessor,
    ProcessingStatus,
)

# Import core classes
from .core import UnifiedS2TSSProcessor

# Import GUI classes
from .gui import UnifiedS2TSSGUI, bring_window_to_front

__all__ = [
    # Config classes
    'ProcessingMode',
    'ProductType',
    'ResamplingConfig',
    'SubsetConfig',
    'C2RCCConfig',
    'WaterQualityConfig',
    'OutputCategoryConfig',
    'TSSConfig',
    'ProcessingConfig',
    # Shared utility classes
    'ColoredFormatter',
    'setup_enhanced_logging',
    'get_default_logger',
    'SafeMathNumPy',
    'MemoryManager',
    'RasterIO',
    # Optical utility classes
    'ProductDetector',
    'SystemMonitor',
    'OutputStructure',
    # Processor classes
    'TSMCHLCalculator',
    'ProcessingResult',
    'VisualizationProcessor',
    'WaterQualityConstants',
    'WaterQualityProcessor',
    'TSSConstants',
    'TSSProcessor',
    'C2RCCProcessor',
    'ProcessingStatus',
    # Core classes
    'UnifiedS2TSSProcessor',
    # GUI classes
    'UnifiedS2TSSGUI',
    'bring_window_to_front',
]
