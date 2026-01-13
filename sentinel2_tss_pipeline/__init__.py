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

# Import processor classes (new names)
from .processors import (
    TSMChlorophyllCalculator,
    ProcessingResult,
    RGBCompositeDefinitions,
    VisualizationProcessor,
    WaterQualityConstants,
    WaterQualityProcessor,
    JiangTSSConstants,
    JiangTSSProcessor,
    S2Processor,
    ProcessingStatus,
    create_water_quality_processor,
    process_water_quality_from_c2rcc,
)

# Backwards compatibility aliases
from .processors import (
    SNAPTSMCHLCalculator,
    S2MarineRGBGenerator,
    S2MarineVisualizationProcessor,
    create_advanced_processor,
    integrate_with_existing_pipeline,
)

# Import core classes
from .core import UnifiedS2TSSProcessor

# Import GUI classes (transitional - imports from main module)
from .gui import UnifiedS2TSSGUI, bring_window_to_front

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
    # Processor classes (new names)
    'TSMChlorophyllCalculator',
    'ProcessingResult',
    'RGBCompositeDefinitions',
    'VisualizationProcessor',
    'WaterQualityConstants',
    'WaterQualityProcessor',
    'JiangTSSConstants',
    'JiangTSSProcessor',
    'S2Processor',
    'ProcessingStatus',
    'create_water_quality_processor',
    'process_water_quality_from_c2rcc',
    # Backwards compatibility aliases
    'SNAPTSMCHLCalculator',
    'S2MarineRGBGenerator',
    'S2MarineVisualizationProcessor',
    'create_advanced_processor',
    'integrate_with_existing_pipeline',
    # Core classes
    'UnifiedS2TSSProcessor',
    # GUI classes
    'UnifiedS2TSSGUI',
    'bring_window_to_front',
]
