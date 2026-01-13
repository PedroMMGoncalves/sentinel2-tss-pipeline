"""
GUI v2 module for Sentinel-2 TSS Pipeline.

Improved GUI with modern styling, collapsible sections,
tooltips, and better layout organization.

This is developed in parallel with gui/ and will replace
it when ready for production.
"""

from .unified_gui import UnifiedS2TSSGUI
from .theme import ThemeManager

__all__ = [
    'UnifiedS2TSSGUI',
    'ThemeManager',
]
