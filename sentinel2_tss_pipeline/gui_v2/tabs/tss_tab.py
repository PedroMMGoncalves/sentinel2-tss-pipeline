"""
TSS & Visualization Tab for GUI v2.

Compact layout with checkbox groups for marine visualization options.
"""

import tkinter as tk
from tkinter import ttk

from ..widgets import CollapsibleFrame, CheckboxGroup, create_tooltip
from ..theme import ThemeManager


def create_tss_tab(gui, notebook):
    """
    Create the TSS & Visualization tab.

    Args:
        gui: Parent GUI instance
        notebook: ttk.Notebook to add tab to

    Returns:
        Tab index
    """
    frame = ttk.Frame(notebook, padding="5")
    tab_index = notebook.add(frame, text=" TSS & Visualization ")

    # Scrollable canvas
    canvas = tk.Canvas(frame, highlightthickness=0)
    scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
    content = ttk.Frame(canvas)

    content.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas_window = canvas.create_window((0, 0), window=content, anchor="nw")

    def on_canvas_configure(event):
        canvas.itemconfig(canvas_window, width=event.width)
    canvas.bind("<Configure>", on_canvas_configure)

    def on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    canvas.bind_all("<MouseWheel>", on_mousewheel)

    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Info note about SNAP products
    info_frame = ttk.Frame(content)
    info_frame.pack(fill=tk.X, padx=10, pady=10)

    info_text = (
        "SNAP TSM/CHL Products:\n"
        "✓ TSM and CHL concentrations are automatically calculated during C2RCC\n"
        "✓ Uncertainty maps included when enabled in C2RCC parameters\n"
        "✓ No additional configuration required"
    )
    ttk.Label(
        info_frame,
        text=info_text,
        style='Info.TLabel',
        font=('Segoe UI', 9),
        justify=tk.LEFT
    ).pack(anchor=tk.W)

    # === Jiang TSS Section ===
    jiang_section = CollapsibleFrame(content, title="Jiang et al. 2021 TSS Methodology", expanded=True)
    jiang_section.pack(fill=tk.X, padx=5, pady=5)

    # Enable checkbox
    enable_frame = ttk.Frame(jiang_section.content_frame)
    enable_frame.pack(fill=tk.X, pady=5)

    jiang_cb = ttk.Checkbutton(
        enable_frame,
        text="Enable Jiang TSS processing",
        variable=gui.enable_jiang_var,
        command=lambda: _update_jiang_visibility(gui)
    )
    jiang_cb.pack(side=tk.LEFT)
    create_tooltip(jiang_cb, "Semi-analytical TSS estimation\nClassifies water into 4 turbidity types")

    # Jiang options frame
    gui.jiang_options_frame = ttk.Frame(jiang_section.content_frame)
    gui.jiang_options_frame.pack(fill=tk.X, pady=5)

    ttk.Checkbutton(
        gui.jiang_options_frame,
        text="Output intermediate products (absorption, backscattering, water types)",
        variable=gui.jiang_intermediates_var
    ).pack(anchor=tk.W, pady=2)

    if hasattr(gui, 'jiang_comparison_var'):
        ttk.Checkbutton(
            gui.jiang_options_frame,
            text="Generate comparison statistics with SNAP TSM",
            variable=gui.jiang_comparison_var
        ).pack(anchor=tk.W, pady=2)

    # Reference
    ttk.Label(
        jiang_section.content_frame,
        text="Reference: Jiang et al. (2021) Remote Sensing of Environment, 258, 112386",
        style='Muted.TLabel',
        font=('Segoe UI', 8, 'italic')
    ).pack(anchor=tk.W, pady=5)

    # === Marine Visualization Section ===
    marine_section = CollapsibleFrame(content, title="Marine Visualization Products", expanded=True)
    marine_section.pack(fill=tk.X, padx=5, pady=5)

    # Enable checkbox
    marine_cb = ttk.Checkbutton(
        marine_section.content_frame,
        text="Generate RGB composites and spectral indices",
        variable=gui.enable_marine_viz_var,
        command=lambda: _update_marine_visibility(gui)
    )
    marine_cb.pack(anchor=tk.W, pady=5)

    # Marine options frame
    gui.marine_options_frame = ttk.Frame(marine_section.content_frame)
    gui.marine_options_frame.pack(fill=tk.X)

    # RGB Composites - compact 2x2 grid
    rgb_items = [
        ("Natural color", gui.natural_color_var, "True color (B4, B3, B2)"),
        ("False color", gui.false_color_var, "Infrared (B8, B4, B3)"),
        ("Water-specific", gui.water_specific_var, "Turbidity & chlorophyll enhanced"),
        ("Research", gui.research_rgb_var, "Advanced combinations"),
    ]

    rgb_group = CheckboxGroup(
        gui.marine_options_frame,
        title="RGB Composites",
        items=rgb_items,
        columns=2,
        show_select_all=False
    )
    rgb_group.pack(fill=tk.X, pady=5)

    # Spectral Indices - compact 2x2 grid
    indices_items = [
        ("Water quality", gui.water_quality_indices_var, "NDWI, MNDWI, AWEI"),
        ("Chlorophyll", gui.chlorophyll_indices_var, "NDCI, GNDVI, CHL"),
        ("Turbidity", gui.turbidity_indices_var, "NDTI, TSI, sediment"),
        ("Advanced", gui.advanced_indices_var, "FUI, SDD, CDOM"),
    ]

    indices_group = CheckboxGroup(
        gui.marine_options_frame,
        title="Spectral Indices",
        items=indices_items,
        columns=2,
        show_select_all=False
    )
    indices_group.pack(fill=tk.X, pady=5)

    # Quick preset buttons
    presets_frame = ttk.Frame(gui.marine_options_frame)
    presets_frame.pack(fill=tk.X, pady=10)

    ttk.Label(presets_frame, text="Quick presets:").pack(side=tk.LEFT, padx=5)

    ttk.Button(
        presets_frame,
        text="Essential",
        width=10,
        command=lambda: _apply_essential_preset(gui)
    ).pack(side=tk.LEFT, padx=2)

    ttk.Button(
        presets_frame,
        text="Complete",
        width=10,
        command=lambda: _apply_complete_preset(gui)
    ).pack(side=tk.LEFT, padx=2)

    ttk.Button(
        presets_frame,
        text="Research",
        width=10,
        command=lambda: _apply_research_preset(gui)
    ).pack(side=tk.LEFT, padx=2)

    # === Advanced Algorithms Section ===
    advanced_section = CollapsibleFrame(content, title="Advanced Aquatic Algorithms", expanded=False)
    advanced_section.pack(fill=tk.X, padx=5, pady=5)

    if hasattr(gui, 'enable_advanced_var'):
        ttk.Checkbutton(
            advanced_section.content_frame,
            text="Enable advanced algorithms",
            variable=gui.enable_advanced_var
        ).pack(anchor=tk.W, pady=5)

    # Advanced algorithm options
    advanced_opts = ttk.Frame(advanced_section.content_frame)
    advanced_opts.pack(fill=tk.X, pady=5)

    algorithms = [
        ("water_clarity_var", "Water Clarity Indices", "Secchi depth, Kd estimation"),
        ("hab_detection_var", "Harmful Algal Bloom Detection", "Cyanobacteria indicators"),
        ("trophic_state_var", "Trophic State Index", "Lake eutrophication level"),
    ]

    for var_name, label, tooltip in algorithms:
        if hasattr(gui, var_name):
            cb = ttk.Checkbutton(
                advanced_opts,
                text=label,
                variable=getattr(gui, var_name)
            )
            cb.pack(anchor=tk.W, pady=2, padx=10)
            create_tooltip(cb, tooltip)

    # Initialize visibility
    _update_jiang_visibility(gui)
    _update_marine_visibility(gui)

    return tab_index


def _update_jiang_visibility(gui):
    """Update Jiang options visibility."""
    if hasattr(gui, 'jiang_options_frame'):
        if gui.enable_jiang_var.get():
            gui.jiang_options_frame.pack(fill=tk.X, pady=5)
        else:
            gui.jiang_options_frame.pack_forget()


def _update_marine_visibility(gui):
    """Update marine options visibility."""
    if hasattr(gui, 'marine_options_frame'):
        if gui.enable_marine_viz_var.get():
            gui.marine_options_frame.pack(fill=tk.X)
        else:
            gui.marine_options_frame.pack_forget()


def _apply_essential_preset(gui):
    """Apply essential visualization preset."""
    gui.natural_color_var.set(True)
    gui.false_color_var.set(True)
    gui.water_specific_var.set(False)
    gui.research_rgb_var.set(False)
    gui.water_quality_indices_var.set(True)
    gui.chlorophyll_indices_var.set(False)
    gui.turbidity_indices_var.set(True)
    gui.advanced_indices_var.set(False)


def _apply_complete_preset(gui):
    """Apply complete visualization preset."""
    gui.natural_color_var.set(True)
    gui.false_color_var.set(True)
    gui.water_specific_var.set(True)
    gui.research_rgb_var.set(False)
    gui.water_quality_indices_var.set(True)
    gui.chlorophyll_indices_var.set(True)
    gui.turbidity_indices_var.set(True)
    gui.advanced_indices_var.set(False)


def _apply_research_preset(gui):
    """Apply research visualization preset - all outputs."""
    gui.natural_color_var.set(True)
    gui.false_color_var.set(True)
    gui.water_specific_var.set(True)
    gui.research_rgb_var.set(True)
    gui.water_quality_indices_var.set(True)
    gui.chlorophyll_indices_var.set(True)
    gui.turbidity_indices_var.set(True)
    gui.advanced_indices_var.set(True)
