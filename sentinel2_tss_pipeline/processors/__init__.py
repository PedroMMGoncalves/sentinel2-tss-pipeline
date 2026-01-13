"""
Processors module for Sentinel-2 TSS Pipeline.

Provides specialized processors for TSS, CHL, water quality estimation,
marine visualization (RGB composites and spectral indices), Jiang TSS methodology,
and S2 processing via SNAP GPT.
"""

from .snap_calculator import TSMChlorophyllCalculator, ProcessingResult
from .marine_viz import RGBCompositeDefinitions, VisualizationProcessor
from .water_quality_processor import (
    WaterQualityConstants,
    WaterQualityProcessor,
    create_water_quality_processor,
    process_water_quality_from_c2rcc,
)
from .jiang_processor import JiangTSSConstants, JiangTSSProcessor
from .s2_processor import S2Processor, ProcessingStatus

# Backwards compatibility aliases
SNAPTSMCHLCalculator = TSMChlorophyllCalculator
S2MarineRGBGenerator = RGBCompositeDefinitions
S2MarineVisualizationProcessor = VisualizationProcessor
create_advanced_processor = create_water_quality_processor
integrate_with_existing_pipeline = process_water_quality_from_c2rcc

__all__ = [
    # TSM/CHL Calculator
    'TSMChlorophyllCalculator',
    'ProcessingResult',
    # Visualization
    'RGBCompositeDefinitions',
    'VisualizationProcessor',
    # Water Quality
    'WaterQualityConstants',
    'WaterQualityProcessor',
    'create_water_quality_processor',
    'process_water_quality_from_c2rcc',
    # Jiang TSS
    'JiangTSSConstants',
    'JiangTSSProcessor',
    # S2 Processor
    'S2Processor',
    'ProcessingStatus',
    # Backwards compatibility
    'SNAPTSMCHLCalculator',
    'S2MarineRGBGenerator',
    'S2MarineVisualizationProcessor',
    'create_advanced_processor',
    'integrate_with_existing_pipeline',
]
