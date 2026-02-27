"""
Checkbox Group Widget.

A compact multi-column checkbox layout for organizing multiple options.
"""

import tkinter as tk
from tkinter import ttk
from typing import List, Tuple, Optional


class CheckboxGroup(ttk.LabelFrame):
    """
    A compact multi-column checkbox group.

    Replaces vertical checkbox lists with a more compact grid layout.

    Features:
    - Configurable number of columns
    - Optional tooltips for each checkbox
    - Select all / Deselect all buttons
    - Consistent styling

    Usage:
        items = [
            ("Natural color", var1, "True color RGB composite"),
            ("False color", var2, "Infrared false color"),
            ("Water-specific", var3, "Optimized for water features"),
        ]
        group = CheckboxGroup(parent, title="RGB Composites", items=items, columns=2)
        group.pack(fill=tk.X, padx=5, pady=5)
    """

    def __init__(
        self,
        parent,
        title: str = "",
        items: Optional[List[Tuple]] = None,
        columns: int = 2,
        show_select_all: bool = False,
        **kwargs
    ):
        """
        Initialize checkbox group.

        Args:
            parent: Parent widget
            title: Group title (shown in LabelFrame)
            items: List of (label, variable, tooltip) tuples
            columns: Number of columns for checkbox grid
            show_select_all: Show Select All / Deselect All buttons
            **kwargs: Additional LabelFrame options
        """
        super().__init__(parent, text=title, padding="5", **kwargs)

        self._items = items or []
        self._columns = columns
        self._checkboxes = []

        self._create_widgets(show_select_all)

    def _create_widgets(self, show_select_all):
        """Create checkbox grid."""
        # Content frame for checkboxes
        self.checkbox_frame = ttk.Frame(self)
        self.checkbox_frame.pack(fill=tk.X, expand=True)

        # Create checkboxes in grid
        for i, item in enumerate(self._items):
            label = item[0]
            variable = item[1]
            tooltip = item[2] if len(item) > 2 else None

            row = i // self._columns
            col = i % self._columns

            cb = ttk.Checkbutton(
                self.checkbox_frame,
                text=label,
                variable=variable
            )
            cb.grid(row=row, column=col, sticky=tk.W, padx=5, pady=2)

            # Add tooltip if provided
            if tooltip:
                self._add_tooltip(cb, tooltip)

            self._checkboxes.append(cb)

        # Configure column weights for even distribution
        for col in range(self._columns):
            self.checkbox_frame.columnconfigure(col, weight=1)

        # Select All / Deselect All buttons
        if show_select_all:
            button_frame = ttk.Frame(self)
            button_frame.pack(fill=tk.X, pady=(5, 0))

            ttk.Button(
                button_frame,
                text="Select All",
                command=self.select_all,
                width=10
            ).pack(side=tk.LEFT, padx=2)

            ttk.Button(
                button_frame,
                text="Deselect All",
                command=self.deselect_all,
                width=10
            ).pack(side=tk.LEFT, padx=2)

    def _add_tooltip(self, widget, text):
        """Add simple tooltip to widget."""
        def show_tooltip(event):
            tooltip = tk.Toplevel(widget)
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root + 10}+{event.y_root + 10}")

            label = ttk.Label(
                tooltip,
                text=text,
                background="#ffffe0",
                relief="solid",
                borderwidth=1,
                padding=(5, 2)
            )
            label.pack()

            widget._tooltip = tooltip

            def hide_tooltip(e=None):
                if hasattr(widget, '_tooltip'):
                    widget._tooltip.destroy()
                    del widget._tooltip

            widget.bind('<Leave>', hide_tooltip)
            tooltip.after(3000, hide_tooltip)  # Auto-hide after 3s

        widget.bind('<Enter>', show_tooltip)

    def select_all(self):
        """Select all checkboxes."""
        for item in self._items:
            if len(item) > 1 and hasattr(item[1], 'set'):
                item[1].set(True)

    def deselect_all(self):
        """Deselect all checkboxes."""
        for item in self._items:
            if len(item) > 1 and hasattr(item[1], 'set'):
                item[1].set(False)

    def get_selected(self) -> List[str]:
        """Get list of selected item labels."""
        selected = []
        for item in self._items:
            if len(item) > 1 and hasattr(item[1], 'get') and item[1].get():
                selected.append(item[0])
        return selected

    def add_item(self, label: str, variable: tk.BooleanVar, tooltip: str = None):
        """
        Add a new checkbox item.

        Args:
            label: Checkbox label text
            variable: BooleanVar for checkbox state
            tooltip: Optional tooltip text
        """
        item = (label, variable, tooltip) if tooltip else (label, variable)
        self._items.append(item)

        i = len(self._checkboxes)
        row = i // self._columns
        col = i % self._columns

        cb = ttk.Checkbutton(
            self.checkbox_frame,
            text=label,
            variable=variable
        )
        cb.grid(row=row, column=col, sticky=tk.W, padx=5, pady=2)

        if tooltip:
            self._add_tooltip(cb, tooltip)

        self._checkboxes.append(cb)
