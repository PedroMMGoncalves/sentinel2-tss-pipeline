"""
GUI module for Sentinel-2 TSS Pipeline.

Contains the unified GUI for processing and TSS estimation.

Modular structure:
    - tabs/ - Individual tab modules (processing, resampling, subset, c2rcc, tss, monitoring)
    - handlers.py - Event handlers and presets
    - config_io.py - Configuration save/load
    - processing_controller.py - Background processing management
    - unified_gui.py - Main GUI class (transitional)
"""

from .unified_gui import (
    UnifiedS2TSSGUI,
    bring_window_to_front,
    # Geometry utilities (re-exported for backwards compatibility)
    load_geometry_from_file,
    validate_wkt_geometry,
    get_area_name,
    load_geometry,
    validate_wkt,
    generate_area_name,
)

# Tab creation functions
from .tabs import (
    create_processing_tab,
    create_resampling_tab,
    create_subset_tab,
    create_c2rcc_tab,
    create_tss_tab,
    create_monitoring_tab,
)

# Event handlers
from .handlers import (
    on_mode_change,
    update_tab_visibility,
    update_subset_visibility,
    update_jiang_visibility,
    on_ecmwf_toggle,
    on_rhow_toggle,
    validate_input_directory,
    validate_geometry,
    browse_input_dir,
    browse_output_dir,
    apply_water_preset,
    apply_snap_defaults,
    apply_essential_outputs,
    apply_scientific_outputs,
    reset_all_outputs,
    on_closing,
)

# Configuration I/O
from .config_io import (
    save_config,
    load_config,
)

# Processing controller
from .processing_controller import (
    start_processing,
    stop_processing,
    start_gui_updates,
    update_system_info,
    update_processing_stats,
)

__all__ = [
    # Main GUI
    'UnifiedS2TSSGUI',
    'bring_window_to_front',
    # Geometry utilities
    'load_geometry_from_file',
    'validate_wkt_geometry',
    'get_area_name',
    'load_geometry',
    'validate_wkt',
    'generate_area_name',
    # Tab creators
    'create_processing_tab',
    'create_resampling_tab',
    'create_subset_tab',
    'create_c2rcc_tab',
    'create_tss_tab',
    'create_monitoring_tab',
    # Handlers
    'on_mode_change',
    'update_tab_visibility',
    'validate_input_directory',
    'browse_input_dir',
    'browse_output_dir',
    'apply_snap_defaults',
    'on_closing',
    # Config I/O
    'save_config',
    'load_config',
    # Processing controller
    'start_processing',
    'stop_processing',
    'start_gui_updates',
]
