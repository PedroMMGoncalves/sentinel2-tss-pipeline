"""
Unified GUI for Sentinel-2 TSS Pipeline.

Professional interface that combines S2 pre-processing with TSS estimation,
featuring automatic SNAP TSM/CHL generation and optional Jiang methodology.

NOTE: This module currently imports from the main sentinel2_tss_pipeline.py file
as a transitional step during refactoring. The GUI code will be fully migrated
here in a future iteration.

Reference:
    Jiang, D., Matsushita, B., Pahlevan, N., et al. (2021).
    "Remotely Estimating Total Suspended Solids Concentration in Clear to
    Extremely Turbid Waters Using a Novel Semi-Analytical Method."
    Remote Sensing of Environment, 258, 112386.
    DOI: https://doi.org/10.1016/j.rse.2021.112386
"""

import sys
import logging

logger = logging.getLogger('sentinel2_tss_pipeline')


def bring_window_to_front(window):
    """
    Enhanced window focus management.

    Brings a tkinter window to the front of other windows
    and ensures it receives focus.
    """
    try:
        window.lift()
        window.attributes('-topmost', True)
        window.focus_force()
        window.grab_set()
        window.update_idletasks()
        window.update()
        window.after(100, lambda: window.attributes('-topmost', False))

        if sys.platform.startswith('win'):
            try:
                import ctypes
                hwnd = ctypes.windll.user32.GetActiveWindow()
                ctypes.windll.user32.FlashWindow(hwnd, True)
            except:
                pass

    except Exception as e:
        logger.warning(f"Could not bring window to front: {e}")


# Transitional import: GUI class from main file
# This will be replaced with the full GUI code in a future iteration
def _get_gui_class():
    """
    Get the UnifiedS2TSSGUI class from the main module.

    This is a transitional function that imports the GUI class from
    the original sentinel2_tss_pipeline.py file. The GUI code will be
    fully migrated to this module in a future iteration.
    """
    try:
        # Import from the parent package's main module
        import importlib.util
        import os

        # Get the path to the main sentinel2_tss_pipeline.py file
        package_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        main_file = os.path.join(package_dir, 'sentinel2_tss_pipeline.py')

        if os.path.exists(main_file):
            spec = importlib.util.spec_from_file_location("main_module", main_file)
            main_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(main_module)
            return main_module.UnifiedS2TSSGUI
        else:
            raise ImportError(f"Main module not found at {main_file}")

    except Exception as e:
        logger.error(f"Failed to import GUI class: {e}")
        raise


# Create a placeholder class that will be populated from main module
class UnifiedS2TSSGUI:
    """
    Unified GUI for Complete S2 Processing and TSS Estimation Pipeline.

    This class provides a professional interface that combines S2 pre-processing
    with TSS estimation, featuring automatic SNAP TSM/CHL generation and optional
    Jiang methodology.

    NOTE: During the transitional period, this class delegates to the original
    implementation in sentinel2_tss_pipeline.py. The GUI code will be fully
    migrated here in a future iteration.

    Features:
        - Complete S2 processing pipeline (L1C -> C2RCC)
        - Automatic SNAP TSM/CHL calculation
        - Optional Jiang TSS methodology
        - Marine visualization products
        - Batch processing capabilities
        - Progress monitoring and logging
    """

    _impl_class = None

    def __new__(cls, *args, **kwargs):
        """Create a new instance by delegating to the original implementation."""
        if cls._impl_class is None:
            cls._impl_class = _get_gui_class()
        return cls._impl_class(*args, **kwargs)


# Geometry utility functions - kept here for GUI support
def load_geometry_from_file(file_path: str) -> tuple:
    """
    Load geometry from various file formats and convert to WKT.

    Supports Shapefile (.shp), KML (.kml), and GeoJSON (.geojson/.json) formats.
    Uses Fiona and OGR as fallbacks for robust file reading.

    Args:
        file_path: Path to the geometry file

    Returns:
        tuple: (wkt_string, info_message, success)
    """
    try:
        # Try to import from main module
        impl = _get_gui_class()
        if hasattr(impl, '__module__'):
            import importlib
            main_module = importlib.import_module(impl.__module__)
            if hasattr(main_module, 'load_geometry_from_file'):
                return main_module.load_geometry_from_file(file_path)
    except:
        pass

    # Fallback: return error
    return None, "Geometry loading not available in transitional module", False


def validate_wkt_geometry(wkt_string: str) -> tuple:
    """
    Validate WKT geometry string.

    Args:
        wkt_string: WKT geometry string to validate

    Returns:
        tuple: (is_valid, message)
    """
    try:
        from shapely.wkt import loads as wkt_loads
        test_geom = wkt_loads(wkt_string)

        if not test_geom.is_valid:
            return False, "WKT geometry is not valid"

        bounds = test_geom.bounds
        info_msg = f"Valid {test_geom.geom_type} geometry"
        info_msg += f"\nBounds: W={bounds[0]:.6f}, S={bounds[1]:.6f}, E={bounds[2]:.6f}, N={bounds[3]:.6f}"

        return True, info_msg

    except Exception as e:
        return False, f"Invalid WKT string: {str(e)}"


def get_area_name(wkt_string: str) -> str:
    """
    Generate a descriptive area name from a WKT string.

    Args:
        wkt_string: WKT geometry string

    Returns:
        str: Descriptive area name based on geometry bounds
    """
    if wkt_string:
        try:
            from shapely.wkt import loads as wkt_loads
            geom = wkt_loads(wkt_string)
            bounds = geom.bounds
            return f"CustomArea_{bounds[0]:.3f}_{bounds[1]:.3f}_{bounds[2]:.3f}_{bounds[3]:.3f}"
        except:
            return "CustomArea"
    return "FullScene"


__all__ = [
    'UnifiedS2TSSGUI',
    'bring_window_to_front',
    'load_geometry_from_file',
    'validate_wkt_geometry',
    'get_area_name',
]
