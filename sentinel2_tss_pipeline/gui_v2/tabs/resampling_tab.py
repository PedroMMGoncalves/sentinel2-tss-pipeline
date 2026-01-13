"""
Resampling Tab for GUI v2.

Compact layout with horizontal resolution selection.
"""

import tkinter as tk
from tkinter import ttk

from ..widgets import CollapsibleFrame, create_tooltip
from ..theme import ThemeManager


def create_resampling_tab(gui, notebook):
    """
    Create the Resampling Configuration tab.

    Args:
        gui: Parent GUI instance
        notebook: ttk.Notebook to add tab to

    Returns:
        Tab index
    """
    frame = ttk.Frame(notebook, padding="10")
    tab_index = notebook.add(frame, text=" Resampling ")

    # Title
    ttk.Label(
        frame,
        text="S2 Resampling Configuration",
        style='Subtitle.TLabel'
    ).pack(pady=(0, 10))

    # === Target Resolution Section ===
    res_section = CollapsibleFrame(frame, title="Target Resolution", expanded=True)
    res_section.pack(fill=tk.X, pady=5)

    # Horizontal resolution selection with visual cards
    res_frame = ttk.Frame(res_section.content_frame)
    res_frame.pack(fill=tk.X, pady=10)

    resolutions = [
        ("10", "10 meters", "Best spatial detail", "Larger files, slower processing"),
        ("20", "20 meters", "Balanced", "Good detail with reasonable file size"),
        ("60", "60 meters", "Fastest", "Smallest files, quickest processing"),
    ]

    for i, (value, title, desc1, desc2) in enumerate(resolutions):
        # Card-like frame for each resolution
        card = ttk.Frame(res_frame, relief="groove", borderwidth=1)
        card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        inner = ttk.Frame(card, padding="10")
        inner.pack(fill=tk.BOTH, expand=True)

        rb = ttk.Radiobutton(
            inner,
            text=title,
            variable=gui.resolution_var,
            value=value,
            style='TRadiobutton'
        )
        rb.pack(anchor=tk.W)

        ttk.Label(
            inner,
            text=desc1,
            style='Muted.TLabel',
            font=('Segoe UI', 9)
        ).pack(anchor=tk.W)

        ttk.Label(
            inner,
            text=desc2,
            style='Muted.TLabel',
            font=('Segoe UI', 8)
        ).pack(anchor=tk.W)

    # === Advanced Resampling Options ===
    advanced_section = CollapsibleFrame(frame, title="Advanced Resampling Options", expanded=False)
    advanced_section.pack(fill=tk.X, pady=5)

    # Options in a compact grid
    opts_frame = ttk.Frame(advanced_section.content_frame)
    opts_frame.pack(fill=tk.X, pady=5)

    # Row 1: Upsampling and Downsampling
    ttk.Label(opts_frame, text="Upsampling:").grid(row=0, column=0, sticky=tk.E, padx=5, pady=3)
    up_combo = ttk.Combobox(
        opts_frame,
        textvariable=gui.upsampling_var,
        values=["Nearest", "Bilinear", "Bicubic"],
        state="readonly",
        width=12
    )
    up_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=3)
    create_tooltip(up_combo, "Method for increasing resolution\nBilinear is recommended for smooth results")

    ttk.Label(opts_frame, text="Downsampling:").grid(row=0, column=2, sticky=tk.E, padx=5, pady=3)
    down_combo = ttk.Combobox(
        opts_frame,
        textvariable=gui.downsampling_var,
        values=["First", "Min", "Max", "Mean", "Median"],
        state="readonly",
        width=12
    )
    down_combo.grid(row=0, column=3, sticky=tk.W, padx=5, pady=3)
    create_tooltip(down_combo, "Method for decreasing resolution\nMean preserves spectral accuracy")

    # Row 2: Flag downsampling
    ttk.Label(opts_frame, text="Flag Downsampling:").grid(row=1, column=0, sticky=tk.E, padx=5, pady=3)
    flag_combo = ttk.Combobox(
        opts_frame,
        textvariable=gui.flag_downsampling_var,
        values=["First", "FlagAnd", "FlagOr", "FlagMedianAnd", "FlagMedianOr"],
        state="readonly",
        width=12
    )
    flag_combo.grid(row=1, column=1, sticky=tk.W, padx=5, pady=3)
    create_tooltip(flag_combo, "Method for combining quality flags\nFirst keeps original flag values")

    # Pyramid checkbox
    pyramid_cb = ttk.Checkbutton(
        opts_frame,
        text="Resample on pyramid levels",
        variable=gui.pyramid_var
    )
    pyramid_cb.grid(row=1, column=2, columnspan=2, sticky=tk.W, padx=5, pady=3)
    create_tooltip(pyramid_cb, "Use image pyramids for faster resampling\nRecommended for large images")

    # Info note
    info_frame = ttk.Frame(advanced_section.content_frame)
    info_frame.pack(fill=tk.X, pady=10)

    ttk.Label(
        info_frame,
        text="ℹ Default settings are optimized for water quality analysis",
        style='Info.TLabel',
        font=('Segoe UI', 9)
    ).pack(anchor=tk.W)

    return tab_index
