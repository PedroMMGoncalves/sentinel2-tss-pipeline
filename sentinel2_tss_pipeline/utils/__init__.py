"""
Utilities module for Sentinel-2 TSS Pipeline.

Provides logging, math, memory management, raster I/O, geometry, and PROJ utilities.
"""

from .logging_utils import ColoredFormatter, setup_enhanced_logging, get_default_logger
from .math_utils import SafeMathNumPy
from .memory_manager import MemoryManager
from .raster_io import RasterIO
from .product_detector import ProductDetector, SystemMonitor
from .proj_fix import (
    configure_proj_environment,
    ensure_proj_configured,
    find_proj_data_paths,
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
    # Backwards compatibility aliases
    load_geometry_from_file,
    validate_wkt_geometry,
    get_area_name,
    HAS_SHAPELY,
    HAS_FIONA,
    HAS_OGR,
)
from .output_structure import OutputStructure

__all__ = [
    # Logging
    'ColoredFormatter',
    'setup_enhanced_logging',
    'get_default_logger',
    # Math
    'SafeMathNumPy',
    # Memory
    'MemoryManager',
    # Raster I/O
    'RasterIO',
    # Product detection
    'ProductDetector',
    'SystemMonitor',
    # PROJ configuration
    'configure_proj_environment',
    'ensure_proj_configured',
    'find_proj_data_paths',
    'test_proj_configuration',
    # Geometry utilities
    'load_geometry',
    'load_shapefile',
    'load_kml',
    'load_geojson',
    'validate_wkt',
    'generate_area_name',
    'create_bbox_wkt',
    'load_geometry_from_file',
    'validate_wkt_geometry',
    'get_area_name',
    'HAS_SHAPELY',
    'HAS_FIONA',
    'HAS_OGR',
    # Output structure
    'OutputStructure',
]
