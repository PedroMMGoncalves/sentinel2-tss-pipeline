"""
Tooltip Widget.

Provides hover tooltips with configurable delay and styling.
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional


class Tooltip:
    """
    Create a tooltip for a given widget.

    Features:
    - Configurable delay before showing
    - Auto-dismiss after timeout
    - Follows mouse position
    - Consistent styling

    Usage:
        button = ttk.Button(parent, text="Save")
        Tooltip(button, "Save the current configuration to a file")
    """

    def __init__(
        self,
        widget,
        text: str,
        delay: int = 500,
        duration: int = 5000,
        wrap_length: int = 300
    ):
        """
        Initialize tooltip.

        Args:
            widget: Widget to attach tooltip to
            text: Tooltip text
            delay: Delay in ms before showing tooltip
            duration: How long to show tooltip (0 = until mouse leaves)
            wrap_length: Text wrap length in pixels
        """
        self.widget = widget
        self.text = text
        self.delay = delay
        self.duration = duration
        self.wrap_length = wrap_length

        self._tooltip_window: Optional[tk.Toplevel] = None
        self._after_id: Optional[str] = None

        # Bind events
        self.widget.bind('<Enter>', self._on_enter)
        self.widget.bind('<Leave>', self._on_leave)
        self.widget.bind('<ButtonPress>', self._on_leave)

    def _on_enter(self, event=None):
        """Mouse entered widget - schedule tooltip."""
        self._cancel_scheduled()
        self._after_id = self.widget.after(self.delay, self._show)

    def _on_leave(self, event=None):
        """Mouse left widget - hide tooltip."""
        self._cancel_scheduled()
        self._hide()

    def _cancel_scheduled(self):
        """Cancel any scheduled tooltip."""
        if self._after_id:
            self.widget.after_cancel(self._after_id)
            self._after_id = None

    def _show(self):
        """Show the tooltip."""
        if self._tooltip_window:
            return

        # Get position
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5

        # Create tooltip window
        self._tooltip_window = tk.Toplevel(self.widget)
        self._tooltip_window.wm_overrideredirect(True)
        self._tooltip_window.wm_geometry(f"+{x}+{y}")

        # Tooltip content
        frame = ttk.Frame(
            self._tooltip_window,
            style='Tooltip.TFrame'
        )
        frame.pack()

        label = ttk.Label(
            frame,
            text=self.text,
            justify=tk.LEFT,
            wraplength=self.wrap_length,
            background="#ffffe0",
            foreground="#333333",
            relief="solid",
            borderwidth=1,
            padding=(8, 4)
        )
        label.pack()

        # Auto-hide after duration
        if self.duration > 0:
            self._after_id = self.widget.after(self.duration, self._hide)

    def _hide(self):
        """Hide the tooltip."""
        if self._tooltip_window:
            self._tooltip_window.destroy()
            self._tooltip_window = None

    def update_text(self, text: str):
        """Update tooltip text."""
        self.text = text
        if self._tooltip_window:
            self._hide()
            self._show()


def create_tooltip(widget, text: str, **kwargs) -> Tooltip:
    """
    Convenience function to create a tooltip.

    Args:
        widget: Widget to attach tooltip to
        text: Tooltip text
        **kwargs: Additional Tooltip options

    Returns:
        Tooltip instance
    """
    return Tooltip(widget, text, **kwargs)


# Scientific parameter tooltips (for reuse)
PARAMETER_TOOLTIPS = {
    # C2RCC Parameters
    'salinity': "Water salinity in PSU (Practical Salinity Units).\nTypical values: Ocean 35, Coastal 30-35, Estuary 0-30, Freshwater <0.5",
    'temperature': "Water temperature in Celsius.\nAffects water absorption properties.",
    'ozone': "Total ozone column in Dobson Units (DU).\nTypical range: 200-500 DU",
    'pressure': "Surface air pressure in hPa.\nStandard atmosphere: 1013.25 hPa",

    # Processing modes
    'complete_pipeline': "Full processing: L1C input → Resampling → C2RCC atmospheric correction → TSS estimation",
    's2_processing_only': "Sentinel-2 processing only: L1C → Resampling → C2RCC\nNo TSS calculation",
    'tss_processing_only': "TSS from existing C2RCC: Requires .dim files from previous C2RCC processing",

    # Output products
    'rhow': "Water-leaving reflectance (ρw)\nRequired for Jiang TSS calculation",
    'rrs': "Remote sensing reflectance (Rrs)\nRrs = rhow / π",
    'kd': "Diffuse attenuation coefficient\nIndicates water clarity",
    'uncertainties': "Uncertainty estimates for TSM and CHL products",

    # Jiang TSS
    'jiang_tss': "Semi-analytical TSS estimation using Jiang et al. (2021) methodology.\nClassifies water into 4 types based on turbidity.",

    # Marine visualization
    'natural_color': "True color composite (R=B4, G=B3, B=B2)\nSimilar to human vision",
    'false_color': "Infrared false color (R=B8, G=B4, B=B3)\nVegetation appears red",
    'water_specific': "Optimized for water features\nEnhances turbidity and chlorophyll patterns",

    # Spectral indices
    'ndwi': "Normalized Difference Water Index\nNDWI = (Green - NIR) / (Green + NIR)",
    'ndti': "Normalized Difference Turbidity Index\nSensitive to suspended sediments",
    'ndci': "Normalized Difference Chlorophyll Index\nDetects algal blooms",
}
