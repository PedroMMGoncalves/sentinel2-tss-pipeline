"""
Resampling Tab for Sentinel-2 TSS Pipeline GUI.

Provides the S2 resampling configuration interface.
"""

import tkinter as tk
from tkinter import ttk
import logging

logger = logging.getLogger('sentinel2_tss_pipeline')


def create_resampling_tab(gui, notebook):
    """
    Create the Resampling configuration tab.

    Args:
        gui: Parent GUI instance with all tk variables.
        notebook: ttk.Notebook to add the tab to.

    Returns:
        int: Tab index.
    """
    frame = ttk.Frame(notebook)
    tab_index = notebook.add(frame, text="Resampling")

    # Title
    ttk.Label(
        frame, text="S2 Resampling Configuration",
        font=("Arial", 14, "bold")
    ).pack(pady=10)

    # Resolution selection
    res_frame = ttk.LabelFrame(frame, text="Target Resolution", padding="10")
    res_frame.pack(fill=tk.X, padx=10, pady=5)

    resolution_options = [
        ("10", "10 meters (Default - Best spatial detail)",
         "Highest resolution, larger file sizes"),
        ("20", "20 meters (Balanced resolution)",
         "Good balance of detail and file size"),
        ("60", "60 meters (Fastest processing)",
         "Lowest resolution, smallest files")
    ]

    for value, text, description in resolution_options:
        radio_frame = ttk.Frame(res_frame)
        radio_frame.pack(fill=tk.X, pady=2)

        ttk.Radiobutton(
            radio_frame, text=text,
            variable=gui.resolution_var, value=value
        ).pack(anchor=tk.W)

        ttk.Label(
            radio_frame, text=description,
            font=("Arial", 8), foreground="gray"
        ).pack(anchor=tk.W, padx=(20, 0))

    # Advanced resampling options
    advanced_frame = ttk.LabelFrame(
        frame, text="Advanced Resampling Options", padding="10"
    )
    advanced_frame.pack(fill=tk.X, padx=10, pady=5)

    options_grid = ttk.Frame(advanced_frame)
    options_grid.pack(fill=tk.X)

    # Upsampling method
    ttk.Label(options_grid, text="Upsampling Method:").grid(
        row=0, column=0, sticky=tk.W, padx=5, pady=2
    )
    ttk.Combobox(
        options_grid, textvariable=gui.upsampling_var,
        values=["Bilinear", "Bicubic", "Nearest"],
        state="readonly", width=15
    ).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)

    # Downsampling method
    ttk.Label(options_grid, text="Downsampling Method:").grid(
        row=1, column=0, sticky=tk.W, padx=5, pady=2
    )
    ttk.Combobox(
        options_grid, textvariable=gui.downsampling_var,
        values=["Mean", "Median", "Min", "Max", "First", "Last"],
        state="readonly", width=15
    ).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

    # Flag downsampling
    ttk.Label(options_grid, text="Flag Downsampling:").grid(
        row=2, column=0, sticky=tk.W, padx=5, pady=2
    )
    ttk.Combobox(
        options_grid, textvariable=gui.flag_downsampling_var,
        values=["First", "FlagAnd", "FlagOr", "FlagMedianAnd", "FlagMedianOr"],
        state="readonly", width=15
    ).grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)

    # Pyramid levels checkbox
    ttk.Checkbutton(
        options_grid, text="Resample on pyramid levels (recommended)",
        variable=gui.pyramid_var
    ).grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)

    return tab_index
