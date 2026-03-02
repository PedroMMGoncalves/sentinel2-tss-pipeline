"""
Optical-specific utilities for Sentinel-2 TSS Pipeline.

Provides product detection and output folder structure.
Shared utilities (logging, raster I/O, math, etc.) are in ocean_rs.shared.
"""

from .product_detector import ProductDetector, SystemMonitor
from .output_structure import OutputStructure

__all__ = [
    'ProductDetector',
    'SystemMonitor',
    'OutputStructure',
]
