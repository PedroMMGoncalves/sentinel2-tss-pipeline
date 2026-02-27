"""
PROJ Database Configuration Utilities.

This module provides functions to configure PROJ environment variables
and resolve common PROJ database issues that occur with GDAL.

The main issue addressed is the "no database context specified" error
that can occur when PROJ cannot find its database files.
"""

import os
import sys
import logging

logger = logging.getLogger('sentinel2_tss_pipeline')


def find_proj_data_paths() -> list:
    """
    Find valid PROJ data directory paths.

    Searches common installation locations for PROJ data files,
    including Conda environments, system installations, and
    Windows-specific paths.

    Returns:
        list: List of valid PROJ data directory paths found.
    """
    proj_data_paths = []

    # Check conda environment first (highest priority)
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        conda_proj = os.path.join(conda_prefix, 'share', 'proj')
        if os.path.exists(conda_proj):
            proj_data_paths.append(conda_proj)

    # Check common PROJ installation paths
    common_paths = [
        # Conda-forge typical locations
        os.path.join(sys.prefix, 'share', 'proj'),
        os.path.join(sys.prefix, 'Library', 'share', 'proj'),  # Windows conda
        # System installations (Linux/macOS)
        '/usr/share/proj',
        '/usr/local/share/proj',
        # OSGeo4W (Windows)
        'C:/OSGeo4W64/share/proj',
        'C:/OSGeo4W/share/proj',
    ]

    # Dynamically find any QGIS installation (Windows)
    import glob as _glob
    for qgis_proj in _glob.glob('C:/Program Files/QGIS */share/proj'):
        if qgis_proj not in common_paths:
            common_paths.append(qgis_proj)

    for path in common_paths:
        if os.path.exists(path) and path not in proj_data_paths:
            proj_data_paths.append(path)

    return proj_data_paths


def find_proj_lib_paths() -> list:
    """
    Find valid PROJ library paths.

    Searches common installation locations for PROJ library files.

    Returns:
        list: List of valid PROJ library paths found.
    """
    proj_lib_paths = []

    # Check conda environment first
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        conda_lib = os.path.join(conda_prefix, 'lib')
        if os.path.exists(conda_lib):
            proj_lib_paths.append(conda_lib)

    # Common library paths
    lib_paths = [
        os.path.join(sys.prefix, 'lib'),
        os.path.join(sys.prefix, 'Library', 'lib'),  # Windows conda
        '/usr/lib',
        '/usr/local/lib',
        'C:/OSGeo4W64/lib',
        'C:/OSGeo4W/lib',
    ]

    for path in lib_paths:
        if os.path.exists(path) and path not in proj_lib_paths:
            proj_lib_paths.append(path)

    return proj_lib_paths


def test_proj_configuration() -> bool:
    """
    Test PROJ functionality by creating a spatial reference.

    Returns:
        bool: True if PROJ is working correctly, False otherwise.
    """
    try:
        from osgeo import osr

        # Test PROJ functionality with WGS84
        source_srs = osr.SpatialReference()
        source_srs.ImportFromEPSG(4326)

        return True

    except Exception as e:
        logger.warning(f"PROJ test failed: {e}")
        return False


def configure_proj_environment(verbose: bool = True) -> bool:
    """
    Configure PROJ environment variables for GDAL compatibility.

    This function resolves the "no database context specified" error
    by setting PROJ_DATA and PROJ_LIB environment variables to valid paths.

    Args:
        verbose: If True, print status messages during configuration.

    Returns:
        bool: True if PROJ was configured successfully, False otherwise.

    Example:
        >>> from sentinel2_tss_pipeline.utils.proj_fix import configure_proj_environment
        >>> success = configure_proj_environment()
        >>> if success:
        ...     from osgeo import gdal  # Now safe to import
    """
    if verbose:
        print("Configuring PROJ database...")

    # Step 1: Find and set PROJ_DATA
    proj_data_paths = find_proj_data_paths()

    if proj_data_paths:
        proj_data = proj_data_paths[0]
        os.environ['PROJ_DATA'] = proj_data
        if verbose:
            print(f"  PROJ_DATA set to: {proj_data}")
    else:
        if verbose:
            print("  WARNING: No PROJ data directory found")
        return False

    # Step 2: Find and set PROJ_LIB
    proj_lib_paths = find_proj_lib_paths()

    if proj_lib_paths:
        os.environ['PROJ_LIB'] = proj_lib_paths[0]
        if verbose:
            print(f"  PROJ_LIB set to: {proj_lib_paths[0]}")

    # Step 3: Test the configuration
    if test_proj_configuration():
        if verbose:
            print("  PROJ configuration successful")
        return True
    else:
        if verbose:
            print("  WARNING: PROJ configuration test failed")
        return False


# Module-level initialization
# This ensures PROJ is configured when the module is imported
_proj_configured = False


def ensure_proj_configured(verbose: bool = False) -> bool:
    """
    Ensure PROJ is configured (only runs once).

    This function can be called multiple times safely - it will only
    configure PROJ on the first call.

    Args:
        verbose: If True, print status messages.

    Returns:
        bool: True if PROJ is configured successfully.
    """
    global _proj_configured

    if not _proj_configured:
        _proj_configured = configure_proj_environment(verbose=verbose)

    return _proj_configured


__all__ = [
    'configure_proj_environment',
    'ensure_proj_configured',
    'find_proj_data_paths',
    'find_proj_lib_paths',
    'test_proj_configuration',
]
