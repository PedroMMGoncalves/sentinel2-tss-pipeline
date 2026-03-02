"""
Data models for SAR Bathymetry Toolkit.

Sensor-agnostic containers for SAR imagery, swell fields, and bathymetry results.
The OceanImage contract decouples sensor adapters from the bathymetry pipeline.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np


class ImageType(Enum):
    """SAR image type hierarchy. Higher value = better for bathymetry."""
    SIGMA0 = 1          # Intensity (any SAR sensor)
    PSEUDO_ALPHA = 2    # Dual-pol decomposition (Sentinel-1, ALOS-2)
    ALPHA = 3           # Quad-pol decomposition (RADARSAT-2, NISAR)


@dataclass
class GeoTransform:
    """Geospatial reference for raster data."""
    origin_x: float
    origin_y: float
    pixel_size_x: float
    pixel_size_y: float
    crs_wkt: str
    rows: int = 0
    cols: int = 0


@dataclass
class OceanImage:
    """Sensor-agnostic SAR image container."""
    data: np.ndarray
    image_type: ImageType
    geo: GeoTransform
    metadata: dict = field(default_factory=dict)
    pixel_spacing_m: float = 10.0


@dataclass
class SwellField:
    """FFT-derived swell parameters per tile."""
    wavelength: np.ndarray
    direction: np.ndarray
    confidence: np.ndarray
    tile_centers_x: np.ndarray
    tile_centers_y: np.ndarray
    tile_size_m: float = 512.0
    geo: Optional[GeoTransform] = None


@dataclass
class BathymetryResult:
    """Inverted depth map with uncertainty. Depth is positive downward."""
    depth: np.ndarray
    uncertainty: np.ndarray
    method: str = "linear_dispersion"
    wave_period: float = 0.0
    wave_period_source: str = "wavewatch3"
    geo: Optional[GeoTransform] = None
    metadata: dict = field(default_factory=dict)
