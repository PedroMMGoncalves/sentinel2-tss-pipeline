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

logger = logging.getLogger('ocean_rs')


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
    Verify PROJ data files exist at the configured path.

    Does NOT import osgeo — this function must run BEFORE any osgeo
    import so that PROJ loads the correct proj.db on first use.

    Returns:
        bool: True if PROJ data files are found, False otherwise.
    """
    proj_data = os.environ.get('PROJ_DATA', '')
    if not proj_data:
        logger.warning("PROJ test failed: PROJ_DATA not set")
        return False

    proj_db = os.path.join(proj_data, 'proj.db')
    if not os.path.exists(proj_db):
        logger.warning(f"PROJ test failed: {proj_db} not found")
        return False

    return True


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
        >>> from ocean_rs.shared.proj_fix import configure_proj_environment
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
        os.environ['PROJ_LIB'] = proj_data  # Some GDAL versions use PROJ_LIB
        if verbose:
            print(f"  PROJ_DATA set to: {proj_data}")

        # Set PROJ_DB explicitly to prevent conflicts with other
        # PROJ installations (e.g. PostgreSQL/PostGIS, QGIS, OSGeo4W)
        proj_db = os.path.join(proj_data, 'proj.db')
        if os.path.exists(proj_db):
            os.environ['PROJ_DB'] = proj_db
            if verbose:
                print(f"  PROJ_DB set to: {proj_db}")

        # Ensure conda's PROJ library path is first on PATH to prevent
        # GDAL from loading an incompatible proj.dll from other software
        conda_prefix = os.environ.get('CONDA_PREFIX')
        if conda_prefix:
            conda_lib_dir = os.path.join(conda_prefix, 'Library', 'bin')
            if os.path.exists(conda_lib_dir):
                current_path = os.environ.get('PATH', '')
                if conda_lib_dir not in current_path:
                    os.environ['PATH'] = conda_lib_dir + os.pathsep + current_path
                    if verbose:
                        print(f"  Prepended to PATH: {conda_lib_dir}")
    else:
        if verbose:
            print("  WARNING: No PROJ data directory found")
        return False

    # Step 2: Find and set library paths
    proj_lib_paths = find_proj_lib_paths()

    if proj_lib_paths and verbose:
        print(f"  PROJ lib path: {proj_lib_paths[0]}")

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
# Benign race -- configure_proj_environment() is idempotent, no lock needed
# None = not checked yet, True = success, False = failed (won't retry)
_proj_configured = None


def ensure_proj_configured(verbose: bool = False) -> bool:
    """
    Ensure PROJ is configured (only runs once).

    This function can be called multiple times safely - it will only
    configure PROJ on the first call. If PROJ was not found on the first
    attempt, subsequent calls return False immediately without re-scanning.

    Args:
        verbose: If True, print status messages.

    Returns:
        bool: True if PROJ is configured successfully.
    """
    global _proj_configured

    if _proj_configured is None:
        _proj_configured = configure_proj_environment(verbose=verbose)

    return _proj_configured if _proj_configured is not None else False


__all__ = [
    'configure_proj_environment',
    'ensure_proj_configured',
    'find_proj_data_paths',
    'find_proj_lib_paths',
    'test_proj_configuration',
]
