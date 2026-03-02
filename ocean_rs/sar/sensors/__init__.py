"""
Sensor adapters for OceanRS SAR Bathymetry Toolkit.

Each adapter preprocesses raw SAR data into the OceanImage contract.
"""

from .base import SensorAdapter
from .sentinel1 import Sentinel1Adapter

__all__ = [
    'SensorAdapter',
    'Sentinel1Adapter',
]
