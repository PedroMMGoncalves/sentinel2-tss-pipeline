"""
Configuration module for Sentinel-2 TSS Pipeline

This module contains all configuration dataclasses used throughout the pipeline.
"""

from .enums import ProcessingMode, ProductType
from .s2_config import ResamplingConfig, SubsetConfig, C2RCCConfig
from .water_quality_config import WaterQualityConfig
from .marine_config import MarineVisualizationConfig
from .jiang_config import JiangTSSConfig
from .processing_config import ProcessingConfig

__all__ = [
    'ProcessingMode',
    'ProductType',
    'ResamplingConfig',
    'SubsetConfig',
    'C2RCCConfig',
    'WaterQualityConfig',
    'MarineVisualizationConfig',
    'JiangTSSConfig',
    'ProcessingConfig',
]
