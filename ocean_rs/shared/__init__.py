"""
OceanRS Shared Utilities.

Common utilities used by both optical and SAR processing pipelines:
- Logging: ColoredFormatter, setup_enhanced_logging, StepTracker
- Math: SafeMathNumPy
- Memory: MemoryManager
- Raster I/O: RasterIO (GDAL wrapper)
- Geometry: load_geometry, validate_wkt, etc.
- PROJ: configure_proj_environment
"""

from .logging_utils import (
    ColoredFormatter,
    setup_enhanced_logging,
    get_default_logger,
    StepTracker,
    parse_scene_metadata,
)
from .math_utils import SafeMathNumPy
from .memory_manager import MemoryManager
from .raster_io import RasterIO
from .proj_fix import (
    configure_proj_environment,
    ensure_proj_configured,
    find_proj_data_paths,
    find_proj_lib_paths,
    test_proj_configuration,
)
from .geometry_utils import (
    load_geometry,
    load_shapefile,
    load_kml,
    load_geojson,
    validate_wkt,
    generate_area_name,
    create_bbox_wkt,
    HAS_SHAPELY,
    HAS_FIONA,
    HAS_OGR,
)

__all__ = [
    # Logging
    'ColoredFormatter',
    'setup_enhanced_logging',
    'get_default_logger',
    'StepTracker',
    'parse_scene_metadata',
    # Math
    'SafeMathNumPy',
    # Memory
    'MemoryManager',
    # Raster I/O
    'RasterIO',
    # PROJ configuration
    'configure_proj_environment',
    'ensure_proj_configured',
    'find_proj_data_paths',
    'find_proj_lib_paths',
    'test_proj_configuration',
    # Geometry utilities
    'load_geometry',
    'load_shapefile',
    'load_kml',
    'load_geojson',
    'validate_wkt',
    'generate_area_name',
    'create_bbox_wkt',
    'HAS_SHAPELY',
    'HAS_FIONA',
    'HAS_OGR',
]
