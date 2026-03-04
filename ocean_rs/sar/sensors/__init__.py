"""
Sensor adapters for OceanRS SAR Toolkit.

Each adapter preprocesses raw SAR data into the OceanImage contract
(for bathymetry) or the SLCImage contract (for InSAR).
"""

from .base import SensorAdapter
from .sentinel1 import Sentinel1Adapter
from .nisar import NISARAdapter
from .alos2 import ALOS2Adapter

__all__ = [
    'SensorAdapter',
    'Sentinel1Adapter',
    'NISARAdapter',
    'ALOS2Adapter',
]
