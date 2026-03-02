"""
Collapsible Frame Widget.

A frame that can be expanded or collapsed by clicking on its header.
Useful for organizing complex forms and reducing visual clutter.
"""

import tkinter as tk
from tkinter import ttk


class CollapsibleFrame(ttk.Frame):
    """
    A frame that can be expanded or collapsed.

    Features:
    - Click header to toggle expand/collapse
    - Animated arrow indicator (▶/▼)
    - Optional initial state (expanded/collapsed)
    - Smooth transition (optional)

    Usage:
        frame = CollapsibleFrame(parent, title="Advanced Options", expanded=False)
        frame.pack(fill=tk.X, padx=5, pady=5)

        # Add content to frame.content_frame
        ttk.Label(frame.content_frame, text="Option 1").pack()
        ttk.Checkbutton(frame.content_frame, text="Enable").pack()
    """

    def __init__(self, parent, title="", expanded=True, **kwargs):
        """
        Initialize collapsible frame.

        Args:
            parent: Parent widget
            title: Header title text
            expanded: Initial state (True=expanded, False=collapsed)
            **kwargs: Additional frame options
        """
        super().__init__(parent, **kwargs)

        self._expanded = tk.BooleanVar(value=expanded)
        self._title = title

        self._create_widgets()
        self._update_state()

    def _create_widgets(self):
        """Create header and content widgets."""
        # Header frame (clickable)
        self.header_frame = ttk.Frame(self, style='Collapsible.TFrame')
        self.header_frame.pack(fill=tk.X)

        # Arrow indicator
        self.arrow_label = ttk.Label(
            self.header_frame,
            text='\u25BC',  # ▼
            font=('Segoe UI', 10),
            cursor='hand2'
        )
        self.arrow_label.pack(side=tk.LEFT, padx=(5, 0))

        # Title label
        self.title_label = ttk.Label(
            self.header_frame,
            text=self._title,
            font=('Segoe UI', 11, 'bold'),
            cursor='hand2'
        )
        self.title_label.pack(side=tk.LEFT, padx=5, pady=5)

        # Bind click events to header elements
        for widget in [self.header_frame, self.arrow_label, self.title_label]:
            widget.bind('<Button-1>', self._toggle)
            widget.bind('<Enter>', self._on_enter)
            widget.bind('<Leave>', self._on_leave)

        # Separator line
        self.separator = ttk.Separator(self, orient=tk.HORIZONTAL)
        self.separator.pack(fill=tk.X)

        # Content frame (holds the actual content)
        self.content_frame = ttk.Frame(self)
        # Note: content_frame is packed/unpacked by _update_state

    def _toggle(self, event=None):
        """Toggle expanded/collapsed state."""
        self._expanded.set(not self._expanded.get())
        self._update_state()

    def _update_state(self):
        """Update widget visibility based on state."""
        if self._expanded.get():
            self.arrow_label.configure(text='\u25BC')  # ▼ Down
            self.content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        else:
            self.arrow_label.configure(text='\u25B6')  # ▶ Right
            self.content_frame.pack_forget()

    def _on_enter(self, event=None):
        """Mouse enter - highlight header."""
        self.header_frame.configure(style='Collapsible.TFrame')

    def _on_leave(self, event=None):
        """Mouse leave - reset header."""
        self.header_frame.configure(style='TFrame')

    def expand(self):
        """Expand the frame."""
        self._expanded.set(True)
        self._update_state()

    def collapse(self):
        """Collapse the frame."""
        self._expanded.set(False)
        self._update_state()

    def toggle(self):
        """Toggle the frame state."""
        self._toggle()

    @property
    def is_expanded(self):
        """Return True if frame is expanded."""
        return self._expanded.get()

    @property
    def title(self):
        """Get the title text."""
        return self._title

    @title.setter
    def title(self, value):
        """Set the title text."""
        self._title = value
        self.title_label.configure(text=value)
