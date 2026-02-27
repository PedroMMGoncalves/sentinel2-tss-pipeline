"""
Custom widgets for the GUI.

Provides modern, reusable widgets:
- CollapsibleFrame: Expandable/collapsible sections
- CheckboxGroup: Compact multi-column checkbox layouts
- Tooltip: Hover tooltips with delay
"""

from .collapsible_frame import CollapsibleFrame
from .checkbox_group import CheckboxGroup
from .tooltip import Tooltip, create_tooltip

__all__ = [
    'CollapsibleFrame',
    'CheckboxGroup',
    'Tooltip',
    'create_tooltip',
]
