"""
Custom widgets for GUI v2.

Provides modern, reusable widgets:
- CollapsibleFrame: Expandable/collapsible sections
- CheckboxGroup: Compact multi-column checkbox layouts
- Tooltip: Hover tooltips with delay
- RecentDirectories: Directory dropdown with history
- ValidationEntry: Entry with visual validation feedback
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
