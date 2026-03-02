"""
Results & Monitor Tab for SAR Bathymetry Toolkit GUI.
"""

import tkinter as tk
from tkinter import ttk
import logging

logger = logging.getLogger('ocean_rs')


def create_results_tab(gui, notebook):
    """Create the Results & Monitor tab."""
    frame = ttk.Frame(notebook)
    tab_index = notebook.add(frame, text="Results & Monitor")

    # --- System Info ---
    sys_frame = ttk.LabelFrame(frame, text="System Information", padding="10")
    sys_frame.pack(fill=tk.X, padx=10, pady=(10, 5))

    gui.sys_info_labels = {}
    for key, label in [("cpu", "CPU:"), ("ram", "RAM:"), ("disk", "Disk:")]:
        row = ttk.Frame(sys_frame)
        row.pack(fill=tk.X, pady=1)
        ttk.Label(row, text=label, width=6).pack(side=tk.LEFT)
        lbl = ttk.Label(row, text="--", style='Status.TLabel')
        lbl.pack(side=tk.LEFT)
        gui.sys_info_labels[key] = lbl

    # --- Processing Log ---
    log_frame = ttk.LabelFrame(frame, text="Processing Log", padding="5")
    log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    gui.processing_log = tk.Text(log_frame, height=15, font=('Consolas', 9),
                                  state=tk.DISABLED, wrap=tk.WORD)
    log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL,
                                   command=gui.processing_log.yview)
    gui.processing_log.configure(yscrollcommand=log_scrollbar.set)
    gui.processing_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # --- Results Summary ---
    results_frame = ttk.LabelFrame(frame, text="Results Summary", padding="10")
    results_frame.pack(fill=tk.X, padx=10, pady=5)

    gui.results_labels = {}
    for key, label in [("scenes", "Scenes Processed:"),
                       ("depth_range", "Depth Range:"),
                       ("mean_depth", "Mean Depth:"),
                       ("uncertainty", "Mean Uncertainty:")]:
        row = ttk.Frame(results_frame)
        row.pack(fill=tk.X, pady=1)
        ttk.Label(row, text=label, width=20).pack(side=tk.LEFT)
        lbl = ttk.Label(row, text="--", style='Status.TLabel')
        lbl.pack(side=tk.LEFT)
        gui.results_labels[key] = lbl

    # --- Export ---
    export_frame = ttk.Frame(frame)
    export_frame.pack(fill=tk.X, padx=10, pady=5)
    ttk.Button(export_frame, text="Open Output Folder",
               command=lambda: _open_output(gui)).pack(side=tk.LEFT, padx=2)

    return tab_index


def _open_output(gui):
    """Open output folder in file explorer."""
    import subprocess
    import sys
    import os
    from tkinter import messagebox
    output = gui.output_dir_var.get()
    if output:
        if not os.path.isdir(output):
            messagebox.showwarning("Warning", f"Output directory not found:\n{output}",
                                   parent=gui.root)
            return
        try:
            if sys.platform.startswith('win'):
                subprocess.Popen(['explorer', output])
            elif sys.platform.startswith('darwin'):
                subprocess.Popen(['open', output])
            else:
                subprocess.Popen(['xdg-open', output])
        except Exception as e:
            messagebox.showerror("Error", f"Cannot open directory: {e}",
                                parent=gui.root)
