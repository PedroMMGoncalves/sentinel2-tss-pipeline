"""
GUI Tab Modules for SAR Bathymetry Toolkit.

4 tabs, each as a factory function receiving the parent GUI instance.
"""

from .search_tab import create_search_tab
from .download_tab import create_download_tab
from .processing_tab import create_processing_tab
from .results_tab import create_results_tab

__all__ = [
    'create_search_tab',
    'create_download_tab',
    'create_processing_tab',
    'create_results_tab',
]
