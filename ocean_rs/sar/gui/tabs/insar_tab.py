"""InSAR Tab for SAR Toolkit GUI.

Co-registration, interferogram, filtering, unwrapping, and topo removal parameters.
"""

import tkinter as tk
from tkinter import ttk, filedialog
import logging

from .processing_tab import create_param_row

logger = logging.getLogger('ocean_rs')


def create_insar_tab(gui, notebook):
    """Create the InSAR tab."""
    frame = ttk.Frame(notebook)
    tab_index = notebook.add(frame, text="InSAR")

    # Scrollable content
    canvas = tk.Canvas(frame, highlightthickness=0)
    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
    scrollable = ttk.Frame(canvas)

    scrollable.bind("<Configure>",
                    lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=scrollable, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Mousewheel scrolling (canvas-specific, not global bind_all)
    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    def _bind_mousewheel(event):
        canvas.bind("<MouseWheel>", _on_mousewheel)
    def _unbind_mousewheel(event):
        canvas.unbind("<MouseWheel>")
    canvas.bind("<Enter>", _bind_mousewheel)
    canvas.bind("<Leave>", _unbind_mousewheel)

    # --- Input SLC Scenes ---
    input_frame = ttk.LabelFrame(scrollable, text="Input SLC Scenes", padding="10")
    input_frame.pack(fill=tk.X, padx=10, pady=(10, 5))

    primary_row = ttk.Frame(input_frame)
    primary_row.pack(fill=tk.X, pady=2)
    ttk.Label(primary_row, text="Primary SLC:", width=22).pack(side=tk.LEFT)
    ttk.Entry(primary_row, textvariable=gui.insar_primary_var, width=50).pack(
        side=tk.LEFT, padx=5, fill=tk.X, expand=True)
    ttk.Button(primary_row, text="Browse...",
               command=lambda: _browse_file(gui, gui.insar_primary_var,
                                            "Select Primary SLC")).pack(side=tk.LEFT)

    secondary_row = ttk.Frame(input_frame)
    secondary_row.pack(fill=tk.X, pady=2)
    ttk.Label(secondary_row, text="Secondary SLC:", width=22).pack(side=tk.LEFT)
    ttk.Entry(secondary_row, textvariable=gui.insar_secondary_var, width=50).pack(
        side=tk.LEFT, padx=5, fill=tk.X, expand=True)
    ttk.Button(secondary_row, text="Browse...",
               command=lambda: _browse_file(gui, gui.insar_secondary_var,
                                            "Select Secondary SLC")).pack(side=tk.LEFT)

    # --- Co-registration ---
    coreg_frame = ttk.LabelFrame(scrollable, text="Co-registration", padding="10")
    coreg_frame.pack(fill=tk.X, padx=10, pady=5)

    method_row = ttk.Frame(coreg_frame)
    method_row.pack(fill=tk.X, pady=2)
    ttk.Label(method_row, text="Method:", width=22).pack(side=tk.LEFT)
    ttk.Combobox(method_row, textvariable=gui.insar_coreg_method_var, width=18,
                 values=["auto", "esd", "coherence"],
                 state="readonly").pack(side=tk.LEFT, padx=5)

    create_param_row(coreg_frame, "Patch Size:", gui.insar_coreg_patch_var, 32, 512)

    # --- Interferogram ---
    ifg_frame = ttk.LabelFrame(scrollable, text="Interferogram", padding="10")
    ifg_frame.pack(fill=tk.X, padx=10, pady=5)

    create_param_row(ifg_frame, "Coherence Window Range (pixels):", gui.insar_coh_range_var, 1, 63)
    create_param_row(ifg_frame, "Coherence Window Azimuth (pixels):", gui.insar_coh_azimuth_var, 1, 63)

    # --- Phase Filtering ---
    filter_frame = ttk.LabelFrame(scrollable, text="Phase Filtering", padding="10")
    filter_frame.pack(fill=tk.X, padx=10, pady=5)

    create_param_row(filter_frame, "Filter Alpha:", gui.insar_filter_alpha_var,
                      0.0, 1.0, increment=0.05)
    ttk.Label(filter_frame, text="0=no filtering, 1=maximum filtering",
              style='Status.TLabel').pack(anchor=tk.W, pady=(0, 2))
    create_param_row(filter_frame, "Patch Size:", gui.insar_filter_patch_var, 8, 128)

    # --- Phase Unwrapping ---
    unwrap_frame = ttk.LabelFrame(scrollable, text="Phase Unwrapping", padding="10")
    unwrap_frame.pack(fill=tk.X, padx=10, pady=5)

    unwrap_row = ttk.Frame(unwrap_frame)
    unwrap_row.pack(fill=tk.X, pady=2)
    ttk.Label(unwrap_row, text="Method:", width=22).pack(side=tk.LEFT)
    ttk.Combobox(unwrap_row, textvariable=gui.insar_unwrap_method_var, width=18,
                 values=["auto", "snaphu", "quality_guided"],
                 state="readonly").pack(side=tk.LEFT, padx=5)

    # --- Topographic Removal ---
    topo_frame = ttk.LabelFrame(scrollable, text="Topographic Removal", padding="10")
    topo_frame.pack(fill=tk.X, padx=10, pady=5)

    ttk.Checkbutton(topo_frame, text="Remove Topographic Phase",
                    variable=gui.insar_remove_topo_var).pack(anchor=tk.W, pady=2)

    dem_row = ttk.Frame(topo_frame)
    dem_row.pack(fill=tk.X, pady=2)
    ttk.Label(dem_row, text="DEM Path:", width=22).pack(side=tk.LEFT)
    ttk.Entry(dem_row, textvariable=gui.insar_dem_path_var, width=50).pack(
        side=tk.LEFT, padx=5, fill=tk.X, expand=True)
    ttk.Button(dem_row, text="Browse...",
               command=lambda: _browse_file(gui, gui.insar_dem_path_var,
                                            "Select DEM File")).pack(side=tk.LEFT)
    ttk.Label(topo_frame, text="Leave empty for SRTM auto-download",
              style='Status.TLabel').pack(anchor=tk.W)

    # --- Output Products ---
    output_frame = ttk.LabelFrame(scrollable, text="Output Products", padding="10")
    output_frame.pack(fill=tk.X, padx=10, pady=5)

    ttk.Checkbutton(output_frame, text="Coherence Map",
                    variable=gui.insar_output_coh_var).pack(anchor=tk.W, pady=2)
    ttk.Checkbutton(output_frame, text="Interferogram",
                    variable=gui.insar_output_ifg_var).pack(anchor=tk.W, pady=2)
    ttk.Checkbutton(output_frame, text="Unwrapped Phase",
                    variable=gui.insar_output_unwrap_var).pack(anchor=tk.W, pady=2)

    return tab_index


def _browse_file(gui, target_var, title="Select File"):
    """Browse for a SAR file (SLC, BEAM-DIMAP, HDF5, etc.)."""
    filepath = filedialog.askopenfilename(
        title=title,
        filetypes=[
            ("SAR Products", "*.zip *.SAFE *.h5 *.dim"),
            ("ZIP Archives", "*.zip"),
            ("SAFE Directories", "*.SAFE"),
            ("HDF5 Files", "*.h5"),
            ("BEAM-DIMAP", "*.dim"),
            ("All Files", "*.*"),
        ]
    )
    if filepath:
        target_var.set(filepath)
