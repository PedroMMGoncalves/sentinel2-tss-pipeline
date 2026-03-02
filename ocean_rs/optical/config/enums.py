"""
Enumeration types for Sentinel-2 TSS Pipeline

Part of the sentinel2_tss_pipeline package.
"""

from enum import Enum


class ProcessingMode(Enum):
    """Processing mode enumeration"""
    COMPLETE_PIPELINE = "complete_pipeline"
    S2_PROCESSING_ONLY = "s2_processing_only"
    TSS_PROCESSING_ONLY = "tss_processing_only"


class ProductType(Enum):
    """Product type enumeration"""
    L1C_ZIP = "l1c_zip"
    L1C_SAFE = "l1c_safe"
    GEOMETRIC_DIM = "geometric_dim"
    C2RCC_DIM = "c2rcc_dim"
    UNKNOWN = "unknown"
