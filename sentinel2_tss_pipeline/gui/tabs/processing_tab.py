"""
Processing Mode Tab for Sentinel-2 TSS Pipeline GUI.

Provides the processing mode selection and I/O configuration interface.
"""

import tkinter as tk
from tkinter import ttk, filedialog
import logging

logger = logging.getLogger('sentinel2_tss_pipeline')


def _validate_numeric(value_if_allowed):
    """Validate spinbox input is numeric."""
    if value_if_allowed == "":
        return True
    try:
        int(value_if_allowed)
        return True
    except ValueError:
        return False


def create_processing_tab(gui, notebook):
    """
    Create the Processing Mode tab.

    Args:
        gui: Parent GUI instance with all tk variables.
        notebook: ttk.Notebook to add the tab to.

    Returns:
        int: Tab index.
    """
    frame = ttk.Frame(notebook)
    tab_index = notebook.add(frame, text="Processing Mode")

    # Processing mode selection
    mode_frame = ttk.LabelFrame(frame, text="Processing Mode Selection", padding="10")
    mode_frame.pack(fill=tk.X, padx=10, pady=10)

    mode_descriptions = {
        "complete_pipeline": (
            "Complete Pipeline: L1C -> S2 Processing -> C2RCC -> Optional Jiang TSS\n"
            "  Input: Raw Sentinel-2 L1C products (.zip/.SAFE)\n"
            "  Output: C2RCC with SNAP TSM/CHL + optional Jiang TSS"
        ),
        "s2_processing_only": (
            "S2 Processing Only: L1C -> S2 Processing -> C2RCC\n"
            "  Input: Raw Sentinel-2 L1C products (.zip/.SAFE)\n"
            "  Output: C2RCC with automatic SNAP TSM/CHL generation"
        ),
        "tss_processing_only": (
            "TSS Processing Only: C2RCC -> Jiang TSS\n"
            "  Input: C2RCC products (.dim files)\n"
            "  Output: Jiang TSS products only"
        )
    }

    for mode, description in mode_descriptions.items():
        radio_frame = ttk.Frame(mode_frame)
        radio_frame.pack(fill=tk.X, pady=2)

        ttk.Radiobutton(
            radio_frame, text="", variable=gui.processing_mode,
            value=mode, command=gui.on_mode_change
        ).pack(side=tk.LEFT)

        ttk.Label(
            radio_frame, text=description, font=("Arial", 9),
            wraplength=700, justify=tk.LEFT
        ).pack(side=tk.LEFT, padx=(5, 0))

    # Input/Output configuration
    io_frame = ttk.LabelFrame(frame, text="Input/Output Configuration", padding="10")
    io_frame.pack(fill=tk.X, padx=10, pady=10)

    # Input directory
    _create_directory_selector(
        io_frame, "Input Directory:", gui.input_dir_var,
        lambda: _browse_directory(gui, "input")
    )

    # Input validation display
    gui.input_validation_frame = ttk.Frame(io_frame)
    gui.input_validation_frame.pack(fill=tk.X, pady=2)
    gui.input_validation_label = ttk.Label(
        gui.input_validation_frame, text="",
        foreground="gray", font=("Arial", 9)
    )
    gui.input_validation_label.pack(anchor=tk.W)

    # Output directory
    _create_directory_selector(
        io_frame, "Output Directory:", gui.output_dir_var,
        lambda: _browse_directory(gui, "output")
    )

    # Processing options
    options_frame = ttk.LabelFrame(frame, text="Processing Options", padding="10")
    options_frame.pack(fill=tk.X, padx=10, pady=10)

    options_grid = ttk.Frame(options_frame)
    options_grid.pack(fill=tk.X)

    # Left column - checkboxes
    left_options = ttk.Frame(options_grid)
    left_options.pack(side=tk.LEFT, fill=tk.X, expand=True)

    ttk.Checkbutton(
        left_options, text="Skip existing output files",
        variable=gui.skip_existing_var
    ).pack(anchor=tk.W, pady=2)

    ttk.Checkbutton(
        left_options, text="Test mode (process only 5 files)",
        variable=gui.test_mode_var
    ).pack(anchor=tk.W, pady=2)

    ttk.Checkbutton(
        left_options, text="Delete intermediate files (.dim/.data) after processing",
        variable=gui.delete_intermediate_var
    ).pack(anchor=tk.W, pady=2)

    # Right column - performance settings
    right_options = ttk.Frame(options_grid)
    right_options.pack(side=tk.RIGHT, fill=tk.X, expand=True)

    perf_frame = ttk.Frame(right_options)
    perf_frame.pack(anchor=tk.W)

    # Numeric validation for spinboxes
    vcmd = (gui.root.register(_validate_numeric), '%P')

    ttk.Label(perf_frame, text="Memory Limit (GB):").pack(side=tk.LEFT)
    ttk.Spinbox(
        perf_frame, from_=4, to=256, width=5,
        textvariable=gui.memory_limit_var,
        validate='key', validatecommand=vcmd
    ).pack(side=tk.LEFT, padx=(5, 20))

    ttk.Label(perf_frame, text="Thread Count:").pack(side=tk.LEFT)
    ttk.Spinbox(
        perf_frame, from_=1, to=64, width=5,
        textvariable=gui.thread_count_var,
        validate='key', validatecommand=vcmd
    ).pack(side=tk.LEFT, padx=(5, 0))

    # Bind input directory change to validation
    gui.input_dir_var.trace("w", gui.validate_input_directory)

    return tab_index


def _create_directory_selector(parent, label_text, variable, browse_command):
    """Create a directory selector with label, entry and browse button."""
    frame = ttk.Frame(parent)
    frame.pack(fill=tk.X, pady=5)

    ttk.Label(frame, text=label_text).pack(anchor=tk.W)

    path_frame = ttk.Frame(frame)
    path_frame.pack(fill=tk.X, pady=2)

    ttk.Entry(path_frame, textvariable=variable).pack(
        side=tk.LEFT, fill=tk.X, expand=True
    )
    ttk.Button(
        path_frame, text="Browse...", command=browse_command
    ).pack(side=tk.RIGHT, padx=(5, 0))


def _browse_directory(gui, dir_type):
    """Browse for a directory."""
    title = "Select Input Directory" if dir_type == "input" else "Select Output Directory"
    directory = filedialog.askdirectory(title=title, parent=gui.root)

    if directory:
        if dir_type == "input":
            gui.input_dir_var.set(directory)
        else:
            gui.output_dir_var.set(directory)
            # Setup logging to output folder
            from ...utils.logging_utils import setup_enhanced_logging
            setup_enhanced_logging(
                log_level=logging.INFO, output_folder=directory
            )
