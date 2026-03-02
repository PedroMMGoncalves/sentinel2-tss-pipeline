"""
GUI Tab Modules for Sentinel-2 TSS Pipeline.

Each tab is implemented as a separate module for maintainability.
Tabs are created as functions that receive the parent GUI instance.

5-tab layout:
    1. Processing - Mode, I/O, options
    2. Spatial - Resampling + subset + map preview
    3. C2RCC - Atmospheric correction parameters
    4. Outputs - 6 category toggles, Jiang config, water mask
    5. Monitor - Progress, system info, statistics
"""

from .processing_tab import create_processing_tab
from .spatial_tab import create_spatial_tab
from .c2rcc_tab import create_c2rcc_tab
from .outputs_tab import create_outputs_tab
from .monitoring_tab import create_monitoring_tab

__all__ = [
    'create_processing_tab',
    'create_spatial_tab',
    'create_c2rcc_tab',
    'create_outputs_tab',
    'create_monitoring_tab',
]
