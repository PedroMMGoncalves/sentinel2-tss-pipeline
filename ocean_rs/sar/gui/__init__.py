"""
GUI module for OceanRS SAR Bathymetry Toolkit.

4-tab interface:
    1. Search & Select
    2. Download & Credentials
    3. Processing
    4. Results & Monitor
"""

from .unified_gui import UnifiedSARGUI, bring_window_to_front

__all__ = [
    'UnifiedSARGUI',
    'bring_window_to_front',
]
