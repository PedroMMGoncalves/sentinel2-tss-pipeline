"""
Outputs Tab for Sentinel-2 TSS Pipeline GUI.

Provides the Jiang TSS configuration, water masking, and
output category selection (6 ON/OFF toggles).
"""

import tkinter as tk
from tkinter import ttk, filedialog
import logging

logger = logging.getLogger('sentinel2_tss_pipeline')


def create_outputs_tab(gui, notebook):
    """
    Create the Outputs configuration tab.

    Args:
        gui: Parent GUI instance with all tk variables.
        notebook: ttk.Notebook to add the tab to.

    Returns:
        int: Tab index.
    """
    frame = ttk.Frame(notebook)
    tab_index = notebook.add(frame, text="Outputs")

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

    # Enable mousewheel scrolling (scoped to this tab)
    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    scrollable_frame.bind('<Enter>', lambda e: canvas.bind_all('<MouseWheel>', _on_mousewheel))
    scrollable_frame.bind('<Leave>', lambda e: canvas.unbind_all('<MouseWheel>'))

    # SNAP products note
    _create_snap_note(scrollable_frame)

    # Jiang TSS Algorithm section
    _create_jiang_section(scrollable_frame, gui)

    # Water Mask section
    _create_water_mask_section(scrollable_frame, gui)

    # Output Categories section
    _create_output_categories_section(scrollable_frame, gui)

    # Set initial visibility
    _update_tss_visibility(gui)

    return tab_index


def _create_snap_note(parent):
    """Create SNAP products information note."""
    note_frame = ttk.Frame(parent, relief=tk.SUNKEN, borderwidth=1)
    note_frame.pack(fill=tk.X, padx=10, pady=5)

    note_text = (
        "SNAP TSM/CHL Products:\n"
        "  TSM and CHL concentrations are automatically generated during C2RCC processing.\n"
        "  No additional configuration needed - always included in C2RCC output."
    )

    ttk.Label(
        note_frame, text=note_text,
        font=("Arial", 9), foreground="darkblue",
        wraplength=600, justify=tk.LEFT, padding="5"
    ).pack()


def _create_jiang_section(parent, gui):
    """Create Jiang TSS methodology configuration section."""
    jiang_frame = ttk.LabelFrame(
        parent, text="Jiang TSS Algorithm", padding="10"
    )
    jiang_frame.pack(fill=tk.X, padx=10, pady=5)

    ttk.Checkbutton(
        jiang_frame, text="Enable Jiang TSS processing",
        variable=gui.enable_jiang_var,
        command=lambda: _update_tss_visibility(gui)
    ).pack(anchor=tk.W, pady=2)

    # Jiang options (toggleable visibility)
    gui.jiang_options_frame = ttk.Frame(jiang_frame)

    ttk.Checkbutton(
        gui.jiang_options_frame, text="Output intermediate products",
        variable=gui.jiang_intermediates_var
    ).pack(anchor=tk.W, pady=2)

    ttk.Checkbutton(
        gui.jiang_options_frame, text="Generate comparison statistics",
        variable=gui.jiang_comparison_var
    ).pack(anchor=tk.W, pady=2)


def _create_water_mask_section(parent, gui):
    """Create water mask configuration section."""
    mask_frame = ttk.LabelFrame(parent, text="Water Mask", padding="10")
    mask_frame.pack(fill=tk.X, padx=10, pady=5)

    # Auto water mask (NDWI + NIR)
    ttk.Checkbutton(
        mask_frame,
        text="Auto-detect water (NDWI + NIR threshold)",
        variable=gui.auto_water_mask_var
    ).pack(anchor=tk.W, pady=2)

    ttk.Label(
        mask_frame,
        text="Masks land pixels before TSS calculation using NDWI > 0 AND NIR(865nm) < 0.03.\n"
             "Applied to TSS, Water Clarity, HAB, and Trophic State only. "
             "RGB composites and indices always cover the full scene.",
        font=("Arial", 8), foreground="gray", wraplength=600, justify=tk.LEFT
    ).pack(anchor=tk.W, padx=(20, 0), pady=(0, 5))

    # Shapefile override
    shp_frame = ttk.Frame(mask_frame)
    shp_frame.pack(fill=tk.X, pady=2)

    ttk.Label(shp_frame, text="Override shapefile:").pack(side=tk.LEFT)
    ttk.Entry(shp_frame, textvariable=gui.water_mask_shapefile_var, width=40).pack(
        side=tk.LEFT, padx=5
    )
    ttk.Button(
        shp_frame, text="Browse...", width=10,
        command=lambda: _browse_water_mask_shapefile(gui)
    ).pack(side=tk.LEFT)

    ttk.Label(
        mask_frame,
        text="If provided, the shapefile mask takes priority over auto-detection.",
        font=("Arial", 8), foreground="gray"
    ).pack(anchor=tk.W, padx=(20, 0), pady=(2, 0))


def _create_output_categories_section(parent, gui):
    """Create output categories section with 6 ON/OFF toggles."""
    cat_frame = ttk.LabelFrame(parent, text="Output Categories", padding="10")
    cat_frame.pack(fill=tk.X, padx=10, pady=5)

    ttk.Label(
        cat_frame,
        text="Select which product categories to generate for each scene.",
        font=("Arial", 9), foreground="gray"
    ).pack(anchor=tk.W, pady=(0, 10))

    # Category definitions: (label, variable_name, product_count, description, default)
    categories = [
        (
            "TSS Products",
            gui.enable_tss_var,
            "7 products",
            "TSS, Absorption, Backscattering, ReferenceBand, WaterTypes, ValidMask, Legend"
        ),
        (
            "RGB Composites",
            gui.enable_rgb_var,
            "15 composites",
            "Natural color, false color, water-specific, turbidity, chlorophyll, bathymetric"
        ),
        (
            "Spectral Indices",
            gui.enable_indices_var,
            "13 indices",
            "NDWI, MNDWI, NDTI, NDCI, GNDVI, TSI, CDOM, pSDB, and more"
        ),
        (
            "Water Clarity",
            gui.enable_water_clarity_var,
            "6 products",
            "Secchi Depth, Kd, Clarity Index, Euphotic Depth, Beam Attenuation, Relative Turbidity"
        ),
        (
            "HAB Detection",
            gui.enable_hab_var,
            "9 products",
            "NDCI/MCI bloom detection, probability, risk level, potential bloom, biomass alerts"
        ),
        (
            "Trophic State",
            gui.enable_trophic_state_var,
            "3 products",
            "TSI-Chlorophyll, TSI-Secchi, Trophic Classification (Carlson 1977)"
        ),
    ]

    for label, variable, count, description in categories:
        cat_row = ttk.Frame(cat_frame)
        cat_row.pack(fill=tk.X, pady=3)

        ttk.Checkbutton(
            cat_row, text=label, variable=variable
        ).pack(side=tk.LEFT)

        ttk.Label(
            cat_row, text=f"({count})",
            font=("Arial", 9, "bold"), foreground="#2563eb"
        ).pack(side=tk.LEFT, padx=(5, 0))

        ttk.Label(
            cat_frame, text=description,
            font=("Arial", 8), foreground="gray", wraplength=600
        ).pack(anchor=tk.W, padx=(24, 0), pady=(0, 2))


def _update_tss_visibility(gui):
    """Update Jiang TSS options visibility."""
    if hasattr(gui, 'jiang_options_frame'):
        if gui.enable_jiang_var.get():
            gui.jiang_options_frame.pack(fill=tk.X, pady=(5, 0))
        else:
            gui.jiang_options_frame.pack_forget()


def _browse_water_mask_shapefile(gui):
    """Browse for water mask shapefile."""
    filepath = filedialog.askopenfilename(
        title="Select Water Mask Shapefile",
        filetypes=[("Shapefiles", "*.shp"), ("All files", "*.*")],
        parent=gui.root
    )
    if filepath:
        gui.water_mask_shapefile_var.set(filepath)
