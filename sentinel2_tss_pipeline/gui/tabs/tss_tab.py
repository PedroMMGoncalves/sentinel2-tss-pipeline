"""
TSS & Visualization Tab for Sentinel-2 TSS Pipeline GUI.

Provides the Jiang TSS methodology, marine visualization, and
advanced aquatic algorithms configuration interface.
"""

import tkinter as tk
from tkinter import ttk
import logging

logger = logging.getLogger('sentinel2_tss_pipeline')


def create_tss_tab(gui, notebook):
    """
    Create the TSS & Visualization configuration tab.

    Args:
        gui: Parent GUI instance with all tk variables.
        notebook: ttk.Notebook to add the tab to.

    Returns:
        int: Tab index.
    """
    frame = ttk.Frame(notebook)
    tab_index = notebook.add(frame, text="TSS & Visualization")

    # Create scrollable frame
    canvas = tk.Canvas(frame)
    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Title
    ttk.Label(
        scrollable_frame, text="TSS & Visualization Configuration",
        font=("Arial", 14, "bold")
    ).pack(pady=10)

    # SNAP TSM/CHL products note
    _create_snap_note(scrollable_frame)

    # Jiang TSS Configuration
    _create_jiang_section(scrollable_frame, gui)

    # Marine Visualization Configuration
    _create_marine_viz_section(scrollable_frame, gui)

    # Advanced Aquatic Algorithms
    _create_advanced_algorithms_section(scrollable_frame, gui)

    # Update initial visibility
    _update_jiang_visibility(gui)
    _update_marine_viz_visibility(gui)
    _update_advanced_visibility(gui)

    return tab_index


def _create_snap_note(parent):
    """Create SNAP TSM/CHL products information note."""
    note_frame = ttk.Frame(parent, relief=tk.SUNKEN, borderwidth=1)
    note_frame.pack(fill=tk.X, padx=10, pady=5)

    note_text = (
        "SNAP TSM/CHL Products:\n"
        "  SNAP TSM and CHL concentrations are automatically generated during C2RCC processing\n"
        "  These products include uncertainty maps when uncertainties are enabled\n"
        "  No additional configuration needed - always included in C2RCC output"
    )

    ttk.Label(
        note_frame, text=note_text,
        font=("Arial", 9), foreground="darkblue",
        wraplength=600, justify=tk.LEFT, padding="5"
    ).pack()


def _create_jiang_section(parent, gui):
    """Create Jiang TSS methodology configuration section."""
    jiang_frame = ttk.LabelFrame(
        parent, text="Jiang et al. 2021 TSS Methodology (Optional)", padding="10"
    )
    jiang_frame.pack(fill=tk.X, padx=10, pady=5)

    ttk.Checkbutton(
        jiang_frame, text="Enable Jiang TSS processing",
        variable=gui.enable_jiang_var,
        command=lambda g=gui: _update_jiang_visibility(g)
    ).pack(anchor=tk.W, pady=5)

    # Jiang options frame (toggleable)
    gui.jiang_options_frame = ttk.Frame(jiang_frame)

    ttk.Checkbutton(
        gui.jiang_options_frame, text="Output intermediate products",
        variable=gui.jiang_intermediates_var
    ).pack(anchor=tk.W, pady=2)

    ttk.Checkbutton(
        gui.jiang_options_frame, text="Generate comparison statistics",
        variable=gui.jiang_comparison_var
    ).pack(anchor=tk.W, pady=2)

    # Water Mask Section
    _create_water_mask_section(gui.jiang_options_frame, gui)


def _create_water_mask_section(parent, gui):
    """Create water mask configuration section."""
    mask_frame = ttk.LabelFrame(parent, text="Water Mask (Optional)", padding="5")
    mask_frame.pack(fill=tk.X, pady=(10, 5))

    # Description
    ttk.Label(
        mask_frame,
        text="Mask land pixels in TSS output. Choose shapefile mask OR NIR threshold (not both).",
        font=("Arial", 9), foreground="gray", wraplength=500
    ).pack(anchor=tk.W, pady=(0, 5))

    # Option 1: Shapefile mask
    shp_frame = ttk.Frame(mask_frame)
    shp_frame.pack(fill=tk.X, pady=2)

    ttk.Label(shp_frame, text="Water Shapefile:").pack(side=tk.LEFT)
    ttk.Entry(shp_frame, textvariable=gui.water_mask_shapefile_var, width=40).pack(side=tk.LEFT, padx=5)
    ttk.Button(
        shp_frame, text="Browse...", width=10,
        command=lambda: _browse_water_mask_shapefile(gui)
    ).pack(side=tk.LEFT)

    # Option 2: NIR threshold
    nir_frame = ttk.Frame(mask_frame)
    nir_frame.pack(fill=tk.X, pady=2)

    ttk.Checkbutton(
        nir_frame, text="Apply NIR threshold mask",
        variable=gui.apply_nir_water_mask_var
    ).pack(side=tk.LEFT)

    ttk.Label(nir_frame, text="Threshold:").pack(side=tk.LEFT, padx=(20, 5))
    threshold_spin = ttk.Spinbox(
        nir_frame,
        textvariable=gui.water_mask_threshold_var,
        from_=0.001, to=0.5, increment=0.005, width=8, format="%.3f"
    )
    threshold_spin.pack(side=tk.LEFT)

    ttk.Label(
        mask_frame,
        text="NIR threshold: pixels with Rrs(865nm) < threshold are classified as water",
        font=("Arial", 8), foreground="gray"
    ).pack(anchor=tk.W, pady=(2, 0))


def _browse_water_mask_shapefile(gui):
    """Browse for water mask shapefile."""
    from tkinter import filedialog
    filepath = filedialog.askopenfilename(
        title="Select Water Mask Shapefile",
        filetypes=[("Shapefiles", "*.shp"), ("All files", "*.*")]
    )
    if filepath:
        gui.water_mask_shapefile_var.set(filepath)


def _create_marine_viz_section(parent, gui):
    """Create marine visualization configuration section."""
    marine_frame = ttk.LabelFrame(
        parent, text="Marine Visualization (RGB + Indices)", padding="10"
    )
    marine_frame.pack(fill=tk.X, padx=10, pady=10)

    # Enable marine visualization
    ttk.Checkbutton(
        marine_frame,
        text="Generate RGB composites and spectral indices (ENABLED BY DEFAULT)",
        variable=gui.enable_marine_viz_var,
        command=lambda g=gui: _update_marine_viz_visibility(g)
    ).pack(anchor=tk.W, pady=5)

    # Description
    desc_text = (
        "Automatically generates comprehensive marine visualization products:\n"
        "  RGB Composites: Natural color, false color, turbidity-enhanced, chlorophyll-enhanced\n"
        "  Spectral Indices: NDTI, NDWI, NDCI, TSI, water quality indices\n"
        "  Uses actual SNAP C2RCC output files (rho_toa_*.img)\n"
        "  Export formats: GeoTIFF with contrast enhancement for optimal visualization"
    )

    ttk.Label(
        marine_frame, text=desc_text, font=("Arial", 9),
        foreground="darkgreen", wraplength=600, justify=tk.LEFT
    ).pack(anchor=tk.W, padx=(20, 0), pady=5)

    # Marine visualization options frame (toggleable)
    gui.marine_viz_options_frame = ttk.Frame(marine_frame)

    # RGB options
    _create_rgb_options(gui.marine_viz_options_frame, gui)

    # Spectral indices options
    _create_indices_options(gui.marine_viz_options_frame, gui)

    # Quick preset buttons
    _create_preset_buttons(gui.marine_viz_options_frame, gui)


def _create_rgb_options(parent, gui):
    """Create RGB composite options section."""
    rgb_frame = ttk.LabelFrame(parent, text="RGB Composites", padding="5")
    rgb_frame.pack(fill=tk.X, pady=5)

    ttk.Checkbutton(
        rgb_frame, text="Natural color combinations (true color, enhanced contrast)",
        variable=gui.natural_color_var
    ).pack(anchor=tk.W, pady=1)

    ttk.Checkbutton(
        rgb_frame, text="False color combinations (infrared, NIR)",
        variable=gui.false_color_var
    ).pack(anchor=tk.W, pady=1)

    ttk.Checkbutton(
        rgb_frame, text="Water-specific combinations (turbidity, chlorophyll, sediment)",
        variable=gui.water_specific_var
    ).pack(anchor=tk.W, pady=1)

    ttk.Checkbutton(
        rgb_frame, text="Research combinations (advanced users)",
        variable=gui.research_rgb_var
    ).pack(anchor=tk.W, pady=1)


def _create_indices_options(parent, gui):
    """Create spectral indices options section."""
    indices_frame = ttk.LabelFrame(parent, text="Spectral Indices", padding="5")
    indices_frame.pack(fill=tk.X, pady=5)

    ttk.Checkbutton(
        indices_frame, text="Water quality indices (NDWI, MNDWI, water delineation)",
        variable=gui.water_quality_indices_var
    ).pack(anchor=tk.W, pady=1)

    ttk.Checkbutton(
        indices_frame, text="Chlorophyll indices (NDCI, GNDVI, red edge)",
        variable=gui.chlorophyll_indices_var
    ).pack(anchor=tk.W, pady=1)

    ttk.Checkbutton(
        indices_frame, text="Turbidity indices (NDTI, TSI, sediment)",
        variable=gui.turbidity_indices_var
    ).pack(anchor=tk.W, pady=1)

    ttk.Checkbutton(
        indices_frame, text="Advanced indices (FUI, SDD, CDOM)",
        variable=gui.advanced_indices_var
    ).pack(anchor=tk.W, pady=1)


def _create_preset_buttons(parent, gui):
    """Create quick preset buttons for marine visualization."""
    preset_frame = ttk.Frame(parent)
    preset_frame.pack(fill=tk.X, pady=(10, 0))

    ttk.Label(preset_frame, text="Quick presets:").pack(side=tk.LEFT)

    btn_frame = ttk.Frame(preset_frame)
    btn_frame.pack(side=tk.LEFT, padx=(10, 0))

    ttk.Button(
        btn_frame, text="Essential", width=10,
        command=lambda g=gui: _apply_essential_preset(g)
    ).pack(side=tk.LEFT, padx=2)

    ttk.Button(
        btn_frame, text="Complete", width=10,
        command=lambda g=gui: _apply_complete_preset(g)
    ).pack(side=tk.LEFT, padx=2)

    ttk.Button(
        btn_frame, text="Research", width=10,
        command=lambda g=gui: _apply_research_preset(g)
    ).pack(side=tk.LEFT, padx=2)


def _create_advanced_algorithms_section(parent, gui):
    """Create advanced aquatic algorithms configuration section."""
    advanced_frame = ttk.LabelFrame(
        parent, text="Advanced Aquatic Algorithms", padding="10"
    )
    advanced_frame.pack(fill=tk.X, padx=10, pady=10)

    gui.enable_advanced_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(
        advanced_frame, text="Enable advanced aquatic algorithms",
        variable=gui.enable_advanced_var,
        command=lambda g=gui: _update_advanced_visibility(g)
    ).pack(anchor=tk.W, pady=5)

    # Advanced options frame (toggleable)
    gui.advanced_options_frame = ttk.Frame(advanced_frame)

    working_frame = ttk.LabelFrame(
        gui.advanced_options_frame, text="Available Algorithms", padding="5"
    )
    working_frame.pack(fill=tk.X, pady=5)

    gui.water_clarity_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(
        working_frame, text="Water Clarity Indices",
        variable=gui.water_clarity_var
    ).pack(anchor=tk.W, pady=2)

    gui.hab_detection_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(
        working_frame, text="Harmful Algal Bloom Detection",
        variable=gui.hab_detection_var
    ).pack(anchor=tk.W, pady=2)


def _update_jiang_visibility(gui):
    """Update Jiang TSS options visibility."""
    if hasattr(gui, 'jiang_options_frame'):
        if gui.enable_jiang_var.get():
            gui.jiang_options_frame.pack(fill=tk.X, pady=(10, 0))
        else:
            gui.jiang_options_frame.pack_forget()


def _update_marine_viz_visibility(gui):
    """Update marine visualization options visibility."""
    if hasattr(gui, 'marine_viz_options_frame'):
        if gui.enable_marine_viz_var.get():
            gui.marine_viz_options_frame.pack(fill=tk.X, pady=(10, 0))
        else:
            gui.marine_viz_options_frame.pack_forget()


def _update_advanced_visibility(gui):
    """Update advanced algorithms options visibility."""
    if hasattr(gui, 'advanced_options_frame'):
        if gui.enable_advanced_var.get():
            gui.advanced_options_frame.pack(fill=tk.X, pady=(10, 0))
        else:
            gui.advanced_options_frame.pack_forget()


def _apply_essential_preset(gui):
    """Apply essential marine visualization preset."""
    # RGB
    gui.natural_color_var.set(True)
    gui.false_color_var.set(False)
    gui.water_specific_var.set(True)
    gui.research_rgb_var.set(False)

    # Indices
    gui.water_quality_indices_var.set(True)
    gui.chlorophyll_indices_var.set(True)
    gui.turbidity_indices_var.set(True)
    gui.advanced_indices_var.set(False)

    gui.status_var.set("Essential marine visualization preset applied")


def _apply_complete_preset(gui):
    """Apply complete marine visualization preset."""
    # RGB
    gui.natural_color_var.set(True)
    gui.false_color_var.set(True)
    gui.water_specific_var.set(True)
    gui.research_rgb_var.set(False)

    # Indices
    gui.water_quality_indices_var.set(True)
    gui.chlorophyll_indices_var.set(True)
    gui.turbidity_indices_var.set(True)
    gui.advanced_indices_var.set(True)

    gui.status_var.set("Complete marine visualization preset applied")


def _apply_research_preset(gui):
    """Apply research marine visualization preset."""
    # RGB - everything
    gui.natural_color_var.set(True)
    gui.false_color_var.set(True)
    gui.water_specific_var.set(True)
    gui.research_rgb_var.set(True)

    # Indices - everything
    gui.water_quality_indices_var.set(True)
    gui.chlorophyll_indices_var.set(True)
    gui.turbidity_indices_var.set(True)
    gui.advanced_indices_var.set(True)

    gui.status_var.set("Research marine visualization preset applied")
