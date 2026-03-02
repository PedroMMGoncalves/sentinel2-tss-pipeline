"""
OceanRS SAR — SAR Bathymetry Toolkit

SAR-based nearshore bathymetry estimation using swell wave analysis.
"""

__version__ = "0.1.0"
__author__ = "Pedro Goncalves"

from .config import (
    SARProcessingConfig,
    SearchConfig,
    DownloadConfig,
    FFTConfig,
    DepthInversionConfig,
    CompositingConfig,
)
from .core import (
    ImageType,
    GeoTransform,
    OceanImage,
    SwellField,
    BathymetryResult,
    BathymetryPipeline,
)
from .download import (
    CredentialManager,
    CredentialError,
    SceneMetadata,
    search_scenes,
    BatchDownloader,
)
from .sensors import (
    SensorAdapter,
    Sentinel1Adapter,
)
from .bathymetry import (
    extract_swell,
    get_wave_period,
    invert_depth,
    composite_bathymetry,
)

__all__ = [
    # Config
    'SARProcessingConfig', 'SearchConfig', 'DownloadConfig',
    'FFTConfig', 'DepthInversionConfig', 'CompositingConfig',
    # Core
    'ImageType', 'GeoTransform', 'OceanImage', 'SwellField',
    'BathymetryResult', 'BathymetryPipeline',
    # Download
    'CredentialManager', 'CredentialError', 'SceneMetadata',
    'search_scenes', 'BatchDownloader',
    # Sensors
    'SensorAdapter', 'Sentinel1Adapter',
    # Bathymetry
    'extract_swell', 'get_wave_period', 'invert_depth', 'composite_bathymetry',
]
