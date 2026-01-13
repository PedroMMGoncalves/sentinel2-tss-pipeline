"""
Processing Tab for GUI v2.

Improved layout with collapsible sections and validation feedback.
"""

import tkinter as tk
from tkinter import ttk, filedialog
import os
import logging

from ..widgets import CollapsibleFrame, create_tooltip
from ..theme import ThemeManager

logger = logging.getLogger('sentinel2_tss_pipeline')


def create_processing_tab(gui, notebook):
    """
    Create the Processing Mode tab.

    Args:
        gui: Parent GUI instance
        notebook: ttk.Notebook to add tab to

    Returns:
        Tab index
    """
    frame = ttk.Frame(notebook, padding="5")
    tab_index = notebook.add(frame, text=" Processing ")

    # Scrollable canvas
    canvas = tk.Canvas(frame, highlightthickness=0, bg=ThemeManager.COLORS['bg_main'])
    scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
    content = ttk.Frame(canvas)

    content.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas_window = canvas.create_window((0, 0), window=content, anchor="nw")

    # Make canvas resize with window
    def on_canvas_configure(event):
        canvas.itemconfig(canvas_window, width=event.width)
    canvas.bind("<Configure>", on_canvas_configure)

    # Mouse wheel scrolling
    def on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    canvas.bind_all("<MouseWheel>", on_mousewheel)

    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # === Processing Mode Section ===
    mode_section = CollapsibleFrame(content, title="Processing Mode Selection", expanded=True)
    mode_section.pack(fill=tk.X, padx=5, pady=5)

    # Mode descriptions with icons
    modes = [
        ("complete_pipeline", "Complete Pipeline",
         "L1C → Resampling → C2RCC → TSS + Visualization",
         "Full processing chain with all outputs"),
        ("s2_processing_only", "S2 Processing Only",
         "L1C → Resampling → C2RCC",
         "Atmospheric correction without TSS calculation"),
        ("tss_processing_only", "TSS Processing Only",
         "C2RCC (.dim) → TSS + Visualization",
         "Process existing C2RCC products"),
    ]

    for value, title, subtitle, tooltip in modes:
        mode_frame = ttk.Frame(mode_section.content_frame)
        mode_frame.pack(fill=tk.X, pady=4)

        rb = ttk.Radiobutton(
            mode_frame,
            text="",
            variable=gui.processing_mode,
            value=value,
            command=lambda: _on_mode_change(gui)
        )
        rb.pack(side=tk.LEFT)

        text_frame = ttk.Frame(mode_frame)
        text_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Label(
            text_frame,
            text=title,
            font=('Segoe UI', 10, 'bold')
        ).pack(anchor=tk.W)

        ttk.Label(
            text_frame,
            text=subtitle,
            style='Muted.TLabel'
        ).pack(anchor=tk.W)

        create_tooltip(mode_frame, tooltip)

    # === Input/Output Section ===
    io_section = CollapsibleFrame(content, title="Input/Output Configuration", expanded=True)
    io_section.pack(fill=tk.X, padx=5, pady=5)

    # Input directory
    input_frame = ttk.Frame(io_section.content_frame)
    input_frame.pack(fill=tk.X, pady=3)

    ttk.Label(input_frame, text="Input:", width=8).pack(side=tk.LEFT)

    gui.input_entry = ttk.Entry(input_frame, textvariable=gui.input_dir_var)
    gui.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

    ttk.Button(
        input_frame,
        text="Browse...",
        command=lambda: _browse_directory(gui, "input")
    ).pack(side=tk.RIGHT)

    # Input validation label
    gui.input_validation_label = ttk.Label(
        io_section.content_frame,
        text="",
        style='Muted.TLabel'
    )
    gui.input_validation_label.pack(fill=tk.X, padx=5)

    # Bind validation
    gui.input_dir_var.trace_add('write', lambda *args: _validate_input(gui))

    # Output directory
    output_frame = ttk.Frame(io_section.content_frame)
    output_frame.pack(fill=tk.X, pady=3)

    ttk.Label(output_frame, text="Output:", width=8).pack(side=tk.LEFT)

    ttk.Entry(output_frame, textvariable=gui.output_dir_var).pack(
        side=tk.LEFT, fill=tk.X, expand=True, padx=5
    )

    ttk.Button(
        output_frame,
        text="Browse...",
        command=lambda: _browse_directory(gui, "output")
    ).pack(side=tk.RIGHT)

    # === Processing Options Section ===
    options_section = CollapsibleFrame(content, title="Processing Options", expanded=False)
    options_section.pack(fill=tk.X, padx=5, pady=5)

    # Options in two columns
    opts_frame = ttk.Frame(options_section.content_frame)
    opts_frame.pack(fill=tk.X, pady=5)

    # Left column
    left_col = ttk.Frame(opts_frame)
    left_col.pack(side=tk.LEFT, fill=tk.X, expand=True)

    skip_cb = ttk.Checkbutton(
        left_col,
        text="Skip existing output files",
        variable=gui.skip_existing_var
    )
    skip_cb.pack(anchor=tk.W, pady=2)
    create_tooltip(skip_cb, "Skip products that already have output files")

    test_cb = ttk.Checkbutton(
        left_col,
        text="Test mode (process 1 file only)",
        variable=gui.test_mode_var
    )
    test_cb.pack(anchor=tk.W, pady=2)
    create_tooltip(test_cb, "Process only the first product for testing")

    # Right column - Performance
    right_col = ttk.Frame(opts_frame)
    right_col.pack(side=tk.RIGHT, padx=20)

    perf_frame = ttk.LabelFrame(right_col, text="Performance", padding="5")
    perf_frame.pack()

    # Memory limit
    mem_frame = ttk.Frame(perf_frame)
    mem_frame.pack(fill=tk.X, pady=2)
    ttk.Label(mem_frame, text="Memory (GB):").pack(side=tk.LEFT)
    mem_spin = ttk.Spinbox(
        mem_frame,
        textvariable=gui.memory_limit_var,
        from_=1, to=256, width=6
    )
    mem_spin.pack(side=tk.RIGHT, padx=5)
    create_tooltip(mem_spin, "Maximum memory allocation for processing")

    # Thread count
    thread_frame = ttk.Frame(perf_frame)
    thread_frame.pack(fill=tk.X, pady=2)
    ttk.Label(thread_frame, text="Threads:").pack(side=tk.LEFT)
    thread_spin = ttk.Spinbox(
        thread_frame,
        textvariable=gui.thread_count_var,
        from_=1, to=64, width=6
    )
    thread_spin.pack(side=tk.RIGHT, padx=5)
    create_tooltip(thread_spin, "Number of processing threads")

    return tab_index


def _browse_directory(gui, dir_type):
    """Browse for directory."""
    title = "Select Input Directory" if dir_type == "input" else "Select Output Directory"
    initial = gui.input_dir_var.get() if dir_type == "input" else gui.output_dir_var.get()

    directory = filedialog.askdirectory(title=title, initialdir=initial or os.path.expanduser("~"))

    if directory:
        if dir_type == "input":
            gui.input_dir_var.set(directory)
        else:
            gui.output_dir_var.set(directory)


def _validate_input(gui):
    """Validate input directory and show feedback."""
    input_dir = gui.input_dir_var.get()

    if not input_dir:
        gui.input_validation_label.configure(
            text="",
            style='Muted.TLabel'
        )
        gui.input_validation_result = {"valid": False, "products": []}
        return

    if not os.path.isdir(input_dir):
        gui.input_validation_label.configure(
            text="⚠ Directory does not exist",
            style='Warning.TLabel'
        )
        gui.input_validation_result = {"valid": False, "products": []}
        return

    # Count products based on mode
    mode = gui.processing_mode.get()
    products = []

    try:
        if mode == "tss_processing_only":
            # Look for .dim files
            for f in os.listdir(input_dir):
                if f.endswith('.dim'):
                    products.append(f)
            product_type = "C2RCC products"
        else:
            # Look for L1C products
            for f in os.listdir(input_dir):
                if f.endswith('.zip') or (f.endswith('.SAFE') and os.path.isdir(os.path.join(input_dir, f))):
                    if 'L1C' in f or 'MSIL1C' in f:
                        products.append(f)
            product_type = "L1C products"

        if products:
            gui.input_validation_label.configure(
                text=f"✓ Found {len(products)} {product_type}",
                style='Success.TLabel'
            )
            gui.input_validation_result = {"valid": True, "products": products}
        else:
            gui.input_validation_label.configure(
                text=f"⚠ No {product_type} found",
                style='Warning.TLabel'
            )
            gui.input_validation_result = {"valid": False, "products": []}

    except Exception as e:
        gui.input_validation_label.configure(
            text=f"⚠ Error reading directory: {e}",
            style='Error.TLabel'
        )
        gui.input_validation_result = {"valid": False, "products": []}


def _on_mode_change(gui):
    """Handle processing mode change."""
    _validate_input(gui)
    # Could also update tab visibility here if needed
