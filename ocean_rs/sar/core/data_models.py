"""
Data models for SAR Toolkit.

Sensor-agnostic containers for SAR imagery, swell fields, bathymetry results,
InSAR interferograms, and displacement fields.

The OceanImage contract decouples sensor adapters from the bathymetry pipeline.
The SLCImage contract decouples sensor adapters from the InSAR pipeline.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
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


# ---------------------------------------------------------------------------
# InSAR and Displacement data models
# ---------------------------------------------------------------------------

@dataclass
class OrbitStateVector:
    """Single orbit state vector (position + velocity at a time)."""
    time_utc: str               # ISO 8601 timestamp
    x: float                    # ECEF position X (m)
    y: float                    # ECEF position Y (m)
    z: float                    # ECEF position Z (m)
    vx: float                   # ECEF velocity X (m/s)
    vy: float                   # ECEF velocity Y (m/s)
    vz: float                   # ECEF velocity Z (m/s)


@dataclass
class SLCImage:
    """Complex SAR Single Look Complex image.

    Metadata dict must include:
        - 'orbit_state_vectors': List[OrbitStateVector]
        - 'acquisition_time': str (ISO 8601)
        - 'sensor': str (e.g. 'Sentinel-1', 'NISAR', 'ALOS-2')
        - 'beam_mode': str (e.g. 'IW', 'SM', 'FBS')
    """
    data: np.ndarray              # complex64 or complex128
    geo: GeoTransform
    metadata: dict
    wavelength_m: float           # Read from product metadata
    pixel_spacing_range: float    # Range pixel spacing (m)
    pixel_spacing_azimuth: float  # Azimuth pixel spacing (m)
    is_debursted: bool = False    # True after TOPS deburst (S1 IW)


@dataclass
class InSARPair:
    """Paired SLC images for interferometric processing."""
    primary: SLCImage
    secondary: SLCImage
    temporal_baseline_days: float
    perpendicular_baseline_m: float  # Computed from orbit state vectors


@dataclass
class Interferogram:
    """Interferometric phase and coherence.

    Phase convention: ifg = primary * conj(secondary).
    Positive unwrapped phase = increased sensor-to-target range.
    """
    phase: np.ndarray               # Wrapped phase [-pi, pi]
    coherence: np.ndarray           # [0, 1]
    unwrapped_phase: Optional[np.ndarray] = None
    geo: Optional[GeoTransform] = None
    wavelength_m: float = 0.0
    temporal_baseline_days: float = 0.0
    perpendicular_baseline_m: float = 0.0
    incidence_angle: Optional[np.ndarray] = None  # Radians, for LOS decomposition
    metadata: dict = field(default_factory=dict)


@dataclass
class DisplacementField:
    """Line-of-sight or decomposed displacement.

    Sign convention (for ifg = primary * conj(secondary)):
        Positive LOS = increased sensor-to-target distance
        (subsidence / motion away from sensor).

    Component 'quasi_vertical' assumes purely vertical motion
    (no horizontal component). This is an approximation.
    """
    displacement_m: np.ndarray
    uncertainty_m: np.ndarray
    component: str = "LOS"          # "LOS" | "quasi_vertical"
    reference_date: str = ""
    measurement_date: str = ""
    geo: Optional[GeoTransform] = None
    metadata: dict = field(default_factory=dict)
