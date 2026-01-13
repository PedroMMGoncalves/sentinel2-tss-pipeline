"""
GUI Tab Modules for Sentinel-2 TSS Pipeline.

Each tab is implemented as a separate module for maintainability.
Tabs are created as functions that receive the parent GUI instance.
"""

from .processing_tab import create_processing_tab
from .resampling_tab import create_resampling_tab
from .subset_tab import create_subset_tab
from .c2rcc_tab import create_c2rcc_tab
from .tss_tab import create_tss_tab
from .monitoring_tab import create_monitoring_tab

__all__ = [
    'create_processing_tab',
    'create_resampling_tab',
    'create_subset_tab',
    'create_c2rcc_tab',
    'create_tss_tab',
    'create_monitoring_tab',
]
