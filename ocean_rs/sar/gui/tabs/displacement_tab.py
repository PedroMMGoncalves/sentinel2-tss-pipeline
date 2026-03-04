"""Displacement Tab for SAR Toolkit GUI.

DInSAR and SBAS displacement analysis parameters.
"""

import tkinter as tk
from tkinter import ttk
import logging

from .processing_tab import create_param_row

logger = logging.getLogger('ocean_rs')


def create_displacement_tab(gui, notebook) -> int:
    """Create the Displacement tab."""
    frame = ttk.Frame(notebook)
    tab_index = notebook.add(frame, text="Displacement")

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

    # --- Analysis Mode ---
    mode_frame = ttk.LabelFrame(scrollable, text="Analysis Mode", padding="10")
    mode_frame.pack(fill=tk.X, padx=10, pady=(10, 5))

    ttk.Radiobutton(mode_frame, text="DInSAR (Single Pair)",
                    variable=gui.disp_mode_var,
                    value="dinsar").pack(anchor=tk.W, pady=2)
    ttk.Radiobutton(mode_frame, text="SBAS (Time Series)",
                    variable=gui.disp_mode_var,
                    value="sbas").pack(anchor=tk.W, pady=2)

    # --- Network Parameters (SBAS) ---
    network_frame = ttk.LabelFrame(scrollable, text="Network Parameters (SBAS)",
                                   padding="10")
    network_frame.pack(fill=tk.X, padx=10, pady=5)

    create_param_row(network_frame, "Max Temporal Baseline (days):",
                      gui.disp_max_temporal_var, 1, 730)
    create_param_row(network_frame, "Max Perpendicular Baseline (m):",
                      gui.disp_max_perp_var, 10, 1000, increment=10.0)
    ttk.Label(network_frame, text="Only used in SBAS mode",
              style='Status.TLabel').pack(anchor=tk.W, pady=(2, 0))

    # --- Quality Filters ---
    quality_frame = ttk.LabelFrame(scrollable, text="Quality Filters", padding="10")
    quality_frame.pack(fill=tk.X, padx=10, pady=5)

    create_param_row(quality_frame, "Temporal Coherence Threshold:",
                      gui.disp_temp_coh_var, 0.0, 1.0, increment=0.05)
    ttk.Checkbutton(quality_frame, text="Atmospheric Filter",
                    variable=gui.disp_atm_filter_var).pack(anchor=tk.W, pady=2)

    # --- Reference Point ---
    ref_frame = ttk.LabelFrame(scrollable, text="Reference Point", padding="10")
    ref_frame.pack(fill=tk.X, padx=10, pady=5)

    ttk.Checkbutton(ref_frame, text="Use Reference Point",
                    variable=gui.disp_use_ref_point_var).pack(anchor=tk.W, pady=2)

    lon_row = ttk.Frame(ref_frame)
    lon_row.pack(fill=tk.X, pady=2)
    ttk.Label(lon_row, text="Longitude:", width=22).pack(side=tk.LEFT)
    ttk.Entry(lon_row, textvariable=gui.disp_ref_lon_var, width=12).pack(
        side=tk.LEFT, padx=5)

    lat_row = ttk.Frame(ref_frame)
    lat_row.pack(fill=tk.X, pady=2)
    ttk.Label(lat_row, text="Latitude:", width=22).pack(side=tk.LEFT)
    ttk.Entry(lat_row, textvariable=gui.disp_ref_lat_var, width=12).pack(
        side=tk.LEFT, padx=5)

    # --- Output Products ---
    output_frame = ttk.LabelFrame(scrollable, text="Output Products", padding="10")
    output_frame.pack(fill=tk.X, padx=10, pady=5)

    ttk.Checkbutton(output_frame, text="Quasi-Vertical Displacement",
                    variable=gui.disp_output_vertical_var).pack(anchor=tk.W, pady=2)
    ttk.Checkbutton(output_frame, text="LOS Displacement",
                    variable=gui.disp_output_los_var).pack(anchor=tk.W, pady=2)

    # Sign convention note
    sign_frame = ttk.Frame(scrollable)
    sign_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
    ttk.Label(
        sign_frame,
        text="Sign convention: positive LOS displacement = away from sensor (subsidence). "
             "Quasi-vertical assumes purely vertical motion.",
        style='Status.TLabel',
        wraplength=600
    ).pack(anchor=tk.W)

    return tab_index
