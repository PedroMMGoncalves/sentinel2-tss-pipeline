"""
GUI module for Sentinel-2 TSS Pipeline.

5-tab interface:
    1. Processing - Mode, I/O, options
    2. Spatial - Resampling + subset + map preview
    3. C2RCC - Atmospheric correction parameters
    4. Outputs - 6 category toggles, Jiang config, water mask
    5. Monitor - Progress, system info, statistics
"""

from .unified_gui import UnifiedS2TSSGUI, bring_window_to_front

# Tab creation functions
from .tabs import (
    create_processing_tab,
    create_spatial_tab,
    create_c2rcc_tab,
    create_outputs_tab,
    create_monitoring_tab,
)

# Event handlers
from .handlers import (
    on_mode_change,
    update_tab_visibility,
    validate_input_directory,
    on_closing,
)

# Configuration I/O
from .config_io import (
    update_configurations,
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
    'UnifiedS2TSSGUI',
]
