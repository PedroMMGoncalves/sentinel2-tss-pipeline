"""
Processing Tab for SAR Bathymetry Toolkit GUI.

SNAP GPT config, FFT parameters, wave period, depth inversion, output.
"""

import tkinter as tk
from tkinter import ttk, filedialog
import logging

logger = logging.getLogger('ocean_rs')


def create_processing_tab(gui, notebook):
    """Create the Processing tab."""
    frame = ttk.Frame(notebook)
    tab_index = notebook.add(frame, text="Processing")

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

    # Mousewheel scrolling
    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
    canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

    # --- SNAP GPT ---
    snap_frame = ttk.LabelFrame(scrollable, text="SNAP GPT", padding="10")
    snap_frame.pack(fill=tk.X, padx=10, pady=(10, 5))

    snap_row = ttk.Frame(snap_frame)
    snap_row.pack(fill=tk.X)
    ttk.Label(snap_row, text="GPT Path:").pack(side=tk.LEFT)
    ttk.Entry(snap_row, textvariable=gui.snap_gpt_var, width=50).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
    ttk.Button(snap_row, text="Browse...",
               command=lambda: _browse_gpt(gui)).pack(side=tk.LEFT)
    ttk.Label(snap_frame, text="Leave empty for auto-detection (SNAP_HOME)",
              style='Status.TLabel').pack(anchor=tk.W)

    # --- FFT Parameters ---
    fft_frame = ttk.LabelFrame(scrollable, text="FFT Swell Extraction", padding="10")
    fft_frame.pack(fill=tk.X, padx=10, pady=5)

    _create_param_row(fft_frame, "Tile Size (m):", gui.tile_size_var, 128, 2048)
    _create_param_row(fft_frame, "Overlap:", gui.overlap_var, 0.0, 0.9, increment=0.1)
    _create_param_row(fft_frame, "Min Wavelength (m):", gui.min_wavelength_var, 10, 200)
    _create_param_row(fft_frame, "Max Wavelength (m):", gui.max_wavelength_var, 100, 1000)
    _create_param_row(fft_frame, "Confidence Threshold:", gui.confidence_var, 0.0, 1.0, increment=0.05)

    # --- Wave Period ---
    wave_frame = ttk.LabelFrame(scrollable, text="Wave Period", padding="10")
    wave_frame.pack(fill=tk.X, padx=10, pady=5)

    ttk.Radiobutton(wave_frame, text="Auto (WaveWatch III - ERDDAP)",
                    variable=gui.wave_source_var,
                    value="wavewatch3").pack(anchor=tk.W, pady=2)

    manual_row = ttk.Frame(wave_frame)
    manual_row.pack(fill=tk.X, pady=2)
    ttk.Radiobutton(manual_row, text="Manual:",
                    variable=gui.wave_source_var,
                    value="manual").pack(side=tk.LEFT)
    ttk.Spinbox(manual_row, textvariable=gui.manual_period_var,
                from_=1.0, to=30.0, increment=0.5, width=8).pack(side=tk.LEFT, padx=5)
    ttk.Label(manual_row, text="seconds").pack(side=tk.LEFT)

    # --- Depth Inversion ---
    depth_frame = ttk.LabelFrame(scrollable, text="Depth Inversion", padding="10")
    depth_frame.pack(fill=tk.X, padx=10, pady=5)

    _create_param_row(depth_frame, "Max Depth (m):", gui.max_depth_var, 1, 500)

    comp_row = ttk.Frame(depth_frame)
    comp_row.pack(fill=tk.X, pady=2)
    ttk.Checkbutton(comp_row, text="Multi-temporal Compositing",
                    variable=gui.compositing_var).pack(side=tk.LEFT)
    ttk.Combobox(comp_row, textvariable=gui.compositing_method_var, width=18,
                 values=["weighted_median", "weighted_mean"],
                 state="readonly").pack(side=tk.LEFT, padx=10)

    # --- Output ---
    out_frame = ttk.LabelFrame(scrollable, text="Output", padding="10")
    out_frame.pack(fill=tk.X, padx=10, pady=5)

    out_row = ttk.Frame(out_frame)
    out_row.pack(fill=tk.X)
    ttk.Label(out_row, text="Output Directory:").pack(side=tk.LEFT)
    ttk.Entry(out_row, textvariable=gui.output_dir_var, width=50).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
    ttk.Button(out_row, text="Browse...",
               command=lambda: _browse_output(gui)).pack(side=tk.LEFT)

    # --- Process Button ---
    btn_frame = ttk.Frame(scrollable)
    btn_frame.pack(fill=tk.X, padx=10, pady=10)
    gui.process_start_btn = ttk.Button(btn_frame, text="Run Processing",
                                        style='Primary.TButton',
                                        command=lambda: _start_processing(gui))
    gui.process_start_btn.pack(side=tk.LEFT, padx=2)
    gui.process_stop_btn = ttk.Button(btn_frame, text="Stop",
                                       style='Danger.TButton',
                                       state=tk.DISABLED,
                                       command=lambda: _stop_processing(gui))
    gui.process_stop_btn.pack(side=tk.LEFT, padx=2)

    return tab_index


def _create_param_row(parent, label, variable, from_val, to_val, increment=1.0):
    """Create a labeled Spinbox row."""
    row = ttk.Frame(parent)
    row.pack(fill=tk.X, pady=2)
    ttk.Label(row, text=label, width=22).pack(side=tk.LEFT)
    ttk.Spinbox(row, textvariable=variable, from_=from_val, to=to_val,
                increment=increment, width=10).pack(side=tk.LEFT, padx=5)


def _browse_gpt(gui):
    filepath = filedialog.askopenfilename(
        title="Select SNAP GPT",
        filetypes=[("Executable", "*.exe"), ("All", "*.*")]
    )
    if filepath:
        gui.snap_gpt_var.set(filepath)


def _browse_output(gui):
    directory = filedialog.askdirectory(title="Select Output Directory")
    if directory:
        gui.output_dir_var.set(directory)


def _start_processing(gui):
    """Start bathymetry processing in background thread."""
    from ocean_rs.sar.gui.processing_controller import start_processing
    start_processing(gui)


def _stop_processing(gui):
    """Stop processing."""
    from ocean_rs.sar.gui.processing_controller import stop_processing
    stop_processing(gui)
