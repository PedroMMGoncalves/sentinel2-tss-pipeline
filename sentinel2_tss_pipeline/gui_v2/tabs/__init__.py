"""
Tab modules for GUI v2.

Each tab is a separate module with improved layout,
collapsible sections, and explicit product lists.
"""

from .processing_tab import create_processing_tab
from .spatial_tab import create_spatial_tab
from .c2rcc_tab import create_c2rcc_tab
from .tss_outputs_tab import create_tss_outputs_tab
from .monitoring_tab import create_monitoring_tab

__all__ = [
    'create_processing_tab',
    'create_spatial_tab',
    'create_c2rcc_tab',
    'create_tss_outputs_tab',
    'create_monitoring_tab',
]
