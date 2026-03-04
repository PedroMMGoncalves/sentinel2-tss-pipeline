"""
OceanRS SAR — SAR Toolkit

SAR-based nearshore bathymetry, InSAR interferometry, and displacement analysis.

Capabilities:
    - Bathymetry: FFT swell extraction -> wave dispersion -> depth inversion
    - InSAR: SLC co-registration -> interferogram -> unwrapping -> geocoding
    - Displacement: DInSAR (single-pair) + SBAS (time-series)

Supported sensors: Sentinel-1, NISAR, ALOS-2 PALSAR-2
"""

__version__ = "1.0.0"
__author__ = "Pedro Goncalves"

__all__ = [
    # Config
    'SARProcessingConfig', 'SearchConfig', 'DownloadConfig',
    'FFTConfig', 'DepthInversionConfig', 'CompositingConfig',
    'InSARConfig', 'DisplacementConfig',
    # Core data models
    'ImageType', 'GeoTransform', 'OceanImage', 'SwellField',
    'BathymetryResult', 'OrbitStateVector', 'SLCImage', 'InSARPair',
    'Interferogram', 'DisplacementField',
    # Pipelines
    'BathymetryPipeline', 'InSARPipeline', 'DisplacementPipeline',
    # Download
    'CredentialManager', 'CredentialError', 'SceneMetadata',
    'search_scenes', 'BatchDownloader',
    # Sensors
    'SensorAdapter', 'Sentinel1Adapter', 'NISARAdapter', 'ALOS2Adapter',
    # Bathymetry algorithms
    'extract_swell', 'get_wave_period', 'invert_depth', 'composite_bathymetry',
]

# Lazy import mapping: attribute name -> (module, name)
_LAZY_IMPORTS = {
    # Config
    'SARProcessingConfig': ('.config', 'SARProcessingConfig'),
    'SearchConfig': ('.config', 'SearchConfig'),
    'DownloadConfig': ('.config', 'DownloadConfig'),
    'FFTConfig': ('.config', 'FFTConfig'),
    'DepthInversionConfig': ('.config', 'DepthInversionConfig'),
    'CompositingConfig': ('.config', 'CompositingConfig'),
    'InSARConfig': ('.config', 'InSARConfig'),
    'DisplacementConfig': ('.config', 'DisplacementConfig'),
    # Core data models
    'ImageType': ('.core', 'ImageType'),
    'GeoTransform': ('.core', 'GeoTransform'),
    'OceanImage': ('.core', 'OceanImage'),
    'SwellField': ('.core', 'SwellField'),
    'BathymetryResult': ('.core', 'BathymetryResult'),
    'OrbitStateVector': ('.core', 'OrbitStateVector'),
    'SLCImage': ('.core', 'SLCImage'),
    'InSARPair': ('.core', 'InSARPair'),
    'Interferogram': ('.core', 'Interferogram'),
    'DisplacementField': ('.core', 'DisplacementField'),
    'BathymetryPipeline': ('.core', 'BathymetryPipeline'),
    # Download
    'CredentialManager': ('.download', 'CredentialManager'),
    'CredentialError': ('.download', 'CredentialError'),
    'SceneMetadata': ('.download', 'SceneMetadata'),
    'search_scenes': ('.download', 'search_scenes'),
    'BatchDownloader': ('.download', 'BatchDownloader'),
    # Sensors
    'SensorAdapter': ('.sensors', 'SensorAdapter'),
    'Sentinel1Adapter': ('.sensors', 'Sentinel1Adapter'),
    'NISARAdapter': ('.sensors', 'NISARAdapter'),
    'ALOS2Adapter': ('.sensors', 'ALOS2Adapter'),
    # Bathymetry algorithms
    'extract_swell': ('.bathymetry', 'extract_swell'),
    'get_wave_period': ('.bathymetry', 'get_wave_period'),
    'invert_depth': ('.bathymetry', 'invert_depth'),
    'composite_bathymetry': ('.bathymetry', 'composite_bathymetry'),
    # Pipelines
    'InSARPipeline': ('.insar', 'InSARPipeline'),
    'DisplacementPipeline': ('.displacement', 'DisplacementPipeline'),
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path, __name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module 'ocean_rs.sar' has no attribute {name!r}")
