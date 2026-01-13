"""
Unified GUI for Sentinel-2 TSS Pipeline.

Professional interface that combines S2 pre-processing with TSS estimation,
featuring automatic SNAP TSM/CHL generation and optional Jiang methodology.

NOTE: The GUI class is currently loaded from legacy/sentinel2_tss_pipeline.py
as a transitional step. The geometry utilities are now in the modular package.

Reference:
    Jiang, D., Matsushita, B., Pahlevan, N., et al. (2021).
    "Remotely Estimating Total Suspended Solids Concentration in Clear to
    Extremely Turbid Waters Using a Novel Semi-Analytical Method."
    Remote Sensing of Environment, 258, 112386.
    DOI: https://doi.org/10.1016/j.rse.2021.112386
"""

import sys
import os
import logging

logger = logging.getLogger('sentinel2_tss_pipeline')

# Import geometry utilities from the modular package
from ..utils.geometry_utils import (
    load_geometry,
    validate_wkt,
    generate_area_name,
    # Backwards compatibility aliases
    load_geometry_from_file,
    validate_wkt_geometry,
    get_area_name,
)


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

    Search order:
        1. legacy/sentinel2_tss_pipeline.py (new structure)
        2. sentinel2_tss_pipeline.py in parent (old structure)
    """
    try:
        # Import from the parent package's main module
        import importlib.util
        import os

        # Get the repo root directory (parent of the package)
        package_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        # Search order: legacy folder first, then parent directory
        possible_paths = [
            os.path.join(package_dir, 'legacy', 'sentinel2_tss_pipeline.py'),
            os.path.join(package_dir, 'sentinel2_tss_pipeline.py'),
        ]

        main_file = None
        for path in possible_paths:
            if os.path.exists(path):
                main_file = path
                break

        if main_file:
            logger.debug(f"Loading GUI from: {main_file}")
            spec = importlib.util.spec_from_file_location("main_module", main_file)
            main_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(main_module)
            return main_module.UnifiedS2TSSGUI
        else:
            searched = '\n  - '.join(possible_paths)
            raise ImportError(f"Main module not found. Searched:\n  - {searched}")

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


__all__ = [
    'UnifiedS2TSSGUI',
    'bring_window_to_front',
    # Re-exported from geometry_utils for backwards compatibility
    'load_geometry_from_file',
    'validate_wkt_geometry',
    'get_area_name',
    # New names
    'load_geometry',
    'validate_wkt',
    'generate_area_name',
]
