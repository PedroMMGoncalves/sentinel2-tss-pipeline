"""
TSS & Outputs Tab for GUI v2.

Explicit product lists with clear descriptions instead of vague categories.
Replaces the old tss_tab.py with complete visibility of all outputs.
"""

import tkinter as tk
from tkinter import ttk

from ..widgets import CollapsibleFrame, create_tooltip
from ..theme import ThemeManager


def create_tss_outputs_tab(gui, notebook):
    """
    Create the TSS & Outputs tab with explicit product lists.

    Args:
        gui: Parent GUI instance
        notebook: ttk.Notebook to add tab to

    Returns:
        Tab index
    """
    frame = ttk.Frame(notebook, padding="5")
    tab_index = notebook.add(frame, text=" TSS & Outputs ")

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

    # Title
    ttk.Label(
        content,
        text="TSS Calculation & Output Products",
        style='Subtitle.TLabel'
    ).pack(pady=(5, 10), padx=10, anchor=tk.W)

    # === SECTION 1: Jiang TSS ===
    _create_jiang_tss_section(gui, content)

    # === SECTION 2: RGB Composites ===
    _create_rgb_composites_section(gui, content)

    # === SECTION 3: Spectral Indices ===
    _create_spectral_indices_section(gui, content)

    # === SECTION 4: Advanced Aquatic Algorithms ===
    _create_advanced_algorithms_section(gui, content)

    return tab_index


def _create_jiang_tss_section(gui, parent):
    """Create Jiang TSS configuration section."""
    section = CollapsibleFrame(parent, title="Jiang TSS Algorithm (2021)", expanded=True)
    section.pack(fill=tk.X, padx=5, pady=5)

    # Description
    info_frame = ttk.Frame(section.content_frame)
    info_frame.pack(fill=tk.X, padx=5, pady=5)

    info_text = (
        "Semi-analytical algorithm for Total Suspended Solids estimation.\n"
        "Uses 4 water types based on reflectance ratios:\n"
        "  Type I (Clear): Rrs(490) > Rrs(560) - uses 560nm\n"
        "  Type II (Moderate): Rrs(490) > Rrs(620) - uses 665nm\n"
        "  Type III (Turbid): Default - uses 740nm\n"
        "  Type IV (Extreme): Rrs(740) > 0.010 - uses 865nm"
    )
    ttk.Label(
        info_frame,
        text=info_text,
        font=('Consolas', 9),
        justify=tk.LEFT
    ).pack(anchor=tk.W)

    # Enable toggle
    enable_frame = ttk.Frame(section.content_frame)
    enable_frame.pack(fill=tk.X, padx=5, pady=5)

    ttk.Checkbutton(
        enable_frame,
        text="Calculate Jiang TSS (recommended for water quality studies)",
        variable=gui.enable_jiang_tss_var
    ).pack(anchor=tk.W)

    # Output options
    output_frame = ttk.LabelFrame(section.content_frame, text="Jiang TSS Outputs", padding="5")
    output_frame.pack(fill=tk.X, padx=5, pady=5)

    ttk.Checkbutton(
        output_frame,
        text="TSS.tif - Total Suspended Solids concentration (g/m3)",
        variable=gui.output_tss_var
    ).pack(anchor=tk.W, pady=2)

    ttk.Checkbutton(
        output_frame,
        text="WaterTypes.tif - Water type classification map (1-4)",
        variable=gui.output_water_types_var
    ).pack(anchor=tk.W, pady=2)

    ttk.Checkbutton(
        output_frame,
        text="Absorption.tif - Total absorption coefficient",
        variable=gui.output_absorption_var
    ).pack(anchor=tk.W, pady=2)

    ttk.Checkbutton(
        output_frame,
        text="Backscattering.tif - Particle backscattering coefficient",
        variable=gui.output_backscattering_var
    ).pack(anchor=tk.W, pady=2)

    # Comparison option
    ttk.Separator(output_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

    ttk.Checkbutton(
        output_frame,
        text="Compare with SNAP TSM (generates comparison statistics)",
        variable=gui.compare_snap_tsm_var
    ).pack(anchor=tk.W, pady=2)


def _create_rgb_composites_section(gui, parent):
    """Create RGB composites section with explicit product list."""
    section = CollapsibleFrame(parent, title="RGB Composite Images", expanded=True)
    section.pack(fill=tk.X, padx=5, pady=5)

    # Preset buttons
    presets_frame = ttk.Frame(section.content_frame)
    presets_frame.pack(fill=tk.X, padx=5, pady=5)

    ttk.Label(presets_frame, text="Quick Select:", font=('Segoe UI', 10)).pack(side=tk.LEFT, padx=(0, 10))

    ttk.Button(
        presets_frame,
        text="Essential",
        width=10,
        command=lambda: _apply_rgb_preset(gui, "essential")
    ).pack(side=tk.LEFT, padx=2)

    ttk.Button(
        presets_frame,
        text="Complete",
        width=10,
        command=lambda: _apply_rgb_preset(gui, "complete")
    ).pack(side=tk.LEFT, padx=2)

    ttk.Button(
        presets_frame,
        text="Research",
        width=10,
        command=lambda: _apply_rgb_preset(gui, "research")
    ).pack(side=tk.LEFT, padx=2)

    ttk.Button(
        presets_frame,
        text="None",
        width=10,
        command=lambda: _apply_rgb_preset(gui, "none")
    ).pack(side=tk.LEFT, padx=2)

    # Standard Visualization
    std_frame = ttk.LabelFrame(section.content_frame, text="Standard Visualization", padding="5")
    std_frame.pack(fill=tk.X, padx=5, pady=5)

    _create_rgb_checkbox(std_frame, gui.rgb_true_color_var,
        "True Color (B4, B3, B2)", "Natural appearance, how human eye would see")
    _create_rgb_checkbox(std_frame, gui.rgb_false_color_infrared_var,
        "False Color Infrared (B8, B4, B3)", "Vegetation red, water dark, land patterns")
    _create_rgb_checkbox(std_frame, gui.rgb_enhanced_contrast_var,
        "Enhanced Contrast (B4, B3, B1)", "Improved visibility of subtle features")
    _create_rgb_checkbox(std_frame, gui.rgb_natural_color_var,
        "Natural Color Enhanced (B12, B8A, B4)", "Enhanced natural appearance")

    # Water Quality Visualization
    wq_frame = ttk.LabelFrame(section.content_frame, text="Water Quality Visualization", padding="5")
    wq_frame.pack(fill=tk.X, padx=5, pady=5)

    _create_rgb_checkbox(wq_frame, gui.rgb_turbidity_enhanced_var,
        "Turbidity Enhanced (B8A, B5, B3)", "Highlights sediment plumes and turbid water")
    _create_rgb_checkbox(wq_frame, gui.rgb_chlorophyll_enhanced_var,
        "Chlorophyll Enhanced (B5, B4, B3)", "Algae blooms, chlorophyll concentrations")
    _create_rgb_checkbox(wq_frame, gui.rgb_coastal_aerosol_var,
        "Coastal Aerosol (B8A, B4, B1)", "Coastal features with aerosol band")
    _create_rgb_checkbox(wq_frame, gui.rgb_cyanobacteria_var,
        "Cyanobacteria Detection (B5, B4, B2)", "Highlights blue-green algae")

    # Specialized Composites
    spec_frame = ttk.LabelFrame(section.content_frame, text="Specialized Composites", padding="5")
    spec_frame.pack(fill=tk.X, padx=5, pady=5)

    _create_rgb_checkbox(spec_frame, gui.rgb_sediment_transport_var,
        "Sediment Transport (B12, B8A, B4)", "River plumes, sediment movement")
    _create_rgb_checkbox(spec_frame, gui.rgb_bathymetric_var,
        "Bathymetric (B3, B2, B1)", "Shallow water depth estimation")
    _create_rgb_checkbox(spec_frame, gui.rgb_ocean_color_var,
        "Ocean Color Standard (B2, B3, B1)", "NASA/ESA ocean color standard")
    _create_rgb_checkbox(spec_frame, gui.rgb_atmospheric_penetration_var,
        "Atmospheric Penetration (B12, B11, B8A)", "Minimizes atmospheric effects")
    _create_rgb_checkbox(spec_frame, gui.rgb_swir_nir_var,
        "SWIR-NIR (B12, B8A, B5)", "Water/land contrast, moisture content")
    _create_rgb_checkbox(spec_frame, gui.rgb_geology_var,
        "Geology (B12, B11, B2)", "Geological features, mineral content")


def _create_rgb_checkbox(parent, variable, label, description):
    """Create an RGB checkbox with description."""
    frame = ttk.Frame(parent)
    frame.pack(fill=tk.X, pady=2)

    cb = ttk.Checkbutton(frame, text=label, variable=variable)
    cb.pack(side=tk.LEFT)

    ttk.Label(
        frame,
        text=f"- {description}",
        style='Muted.TLabel',
        font=('Segoe UI', 9)
    ).pack(side=tk.LEFT, padx=10)


def _create_spectral_indices_section(gui, parent):
    """Create spectral indices section with explicit list."""
    section = CollapsibleFrame(parent, title="Spectral Indices", expanded=True)
    section.pack(fill=tk.X, padx=5, pady=5)

    # Preset buttons
    presets_frame = ttk.Frame(section.content_frame)
    presets_frame.pack(fill=tk.X, padx=5, pady=5)

    ttk.Label(presets_frame, text="Quick Select:", font=('Segoe UI', 10)).pack(side=tk.LEFT, padx=(0, 10))

    ttk.Button(
        presets_frame,
        text="Essential",
        width=10,
        command=lambda: _apply_index_preset(gui, "essential")
    ).pack(side=tk.LEFT, padx=2)

    ttk.Button(
        presets_frame,
        text="Complete",
        width=10,
        command=lambda: _apply_index_preset(gui, "complete")
    ).pack(side=tk.LEFT, padx=2)

    ttk.Button(
        presets_frame,
        text="None",
        width=10,
        command=lambda: _apply_index_preset(gui, "none")
    ).pack(side=tk.LEFT, padx=2)

    # Water Detection Indices
    water_frame = ttk.LabelFrame(section.content_frame, text="Water Detection", padding="5")
    water_frame.pack(fill=tk.X, padx=5, pady=5)

    _create_index_checkbox(water_frame, gui.idx_ndwi_var,
        "NDWI", "(B3-B8)/(B3+B8)", "Normalized Difference Water Index")
    _create_index_checkbox(water_frame, gui.idx_mndwi_var,
        "MNDWI", "(B3-B11)/(B3+B11)", "Modified NDWI - better urban water detection")
    _create_index_checkbox(water_frame, gui.idx_awei_var,
        "AWEI", "4*(B3-B11)-(0.25*B8+2.75*B11)", "Automated Water Extraction Index")
    _create_index_checkbox(water_frame, gui.idx_wri_var,
        "WRI", "(B3+B4)/(B8+B11)", "Water Ratio Index")

    # Chlorophyll & Algae Indices
    chl_frame = ttk.LabelFrame(section.content_frame, text="Chlorophyll & Algae", padding="5")
    chl_frame.pack(fill=tk.X, padx=5, pady=5)

    _create_index_checkbox(chl_frame, gui.idx_ndci_var,
        "NDCI", "(B5-B4)/(B5+B4)", "Normalized Difference Chlorophyll Index")
    _create_index_checkbox(chl_frame, gui.idx_gndvi_var,
        "GNDVI", "(B8-B3)/(B8+B3)", "Green NDVI - aquatic vegetation")
    _create_index_checkbox(chl_frame, gui.idx_fai_var,
        "FAI", "B8-(B4+0.5*(B11-B4))", "Floating Algae Index")
    _create_index_checkbox(chl_frame, gui.idx_flh_var,
        "FLH", "B5-0.5*(B4+B6)", "Fluorescence Line Height (chlorophyll fluorescence)")
    _create_index_checkbox(chl_frame, gui.idx_mci_var,
        "MCI", "B5-0.5*(B4+B6)", "Maximum Chlorophyll Index")

    # Turbidity & Sediment Indices
    turb_frame = ttk.LabelFrame(section.content_frame, text="Turbidity & Sediment", padding="5")
    turb_frame.pack(fill=tk.X, padx=5, pady=5)

    _create_index_checkbox(turb_frame, gui.idx_ndti_var,
        "NDTI", "(B4-B3)/(B4+B3)", "Normalized Difference Turbidity Index")
    _create_index_checkbox(turb_frame, gui.idx_ngrdi_var,
        "NGRDI", "(B3-B4)/(B3+B4)", "Normalized Green-Red Difference Index")

    # Advanced Properties
    adv_frame = ttk.LabelFrame(section.content_frame, text="Advanced Water Properties", padding="5")
    adv_frame.pack(fill=tk.X, padx=5, pady=5)

    _create_index_checkbox(adv_frame, gui.idx_fui_var,
        "FUI", "Forel-Ule Index", "Water color classification (1-21 scale, 360 hue)")
    _create_index_checkbox(adv_frame, gui.idx_sdd_var,
        "SDD", "Secchi Disk Depth", "Water transparency proxy (Gordon 1989)")
    _create_index_checkbox(adv_frame, gui.idx_cdom_var,
        "CDOM", "(B3-B4)/(B2+B4)", "Colored Dissolved Organic Matter index")


def _create_index_checkbox(parent, variable, name, formula, description):
    """Create an index checkbox with formula and description."""
    frame = ttk.Frame(parent)
    frame.pack(fill=tk.X, pady=2)

    cb = ttk.Checkbutton(frame, text=name, variable=variable, width=8)
    cb.pack(side=tk.LEFT)

    ttk.Label(
        frame,
        text=formula,
        font=('Consolas', 9),
        width=30
    ).pack(side=tk.LEFT, padx=5)

    ttk.Label(
        frame,
        text=description,
        style='Muted.TLabel',
        font=('Segoe UI', 9)
    ).pack(side=tk.LEFT, padx=5)


def _create_advanced_algorithms_section(gui, parent):
    """Create advanced aquatic algorithms section with explicit options."""
    section = CollapsibleFrame(parent, title="Advanced Aquatic Algorithms", expanded=False)
    section.pack(fill=tk.X, padx=5, pady=5)

    # Description
    ttk.Label(
        section.content_frame,
        text="Specialized algorithms for water quality assessment and ecological monitoring",
        style='Muted.TLabel',
        font=('Segoe UI', 9)
    ).pack(anchor=tk.W, padx=5, pady=(0, 10))

    # Water Clarity Analysis
    clarity_frame = ttk.LabelFrame(section.content_frame, text="Water Clarity Analysis", padding="5")
    clarity_frame.pack(fill=tk.X, padx=5, pady=5)

    ttk.Checkbutton(
        clarity_frame,
        text="Secchi Depth Estimation (Gordon 1989)",
        variable=gui.enable_secchi_depth_var
    ).pack(anchor=tk.W, pady=2)
    ttk.Label(
        clarity_frame,
        text="Estimates water transparency from optical properties",
        style='Muted.TLabel',
        font=('Segoe UI', 9)
    ).pack(anchor=tk.W, padx=20)

    ttk.Checkbutton(
        clarity_frame,
        text="Euphotic Depth Calculation",
        variable=gui.enable_euphotic_depth_var
    ).pack(anchor=tk.W, pady=2)
    ttk.Label(
        clarity_frame,
        text="Depth of 1% surface light penetration",
        style='Muted.TLabel',
        font=('Segoe UI', 9)
    ).pack(anchor=tk.W, padx=20)

    ttk.Checkbutton(
        clarity_frame,
        text="Diffuse Attenuation (Kd) Mapping",
        variable=gui.enable_kd_mapping_var
    ).pack(anchor=tk.W, pady=2)

    # HAB Detection
    hab_frame = ttk.LabelFrame(section.content_frame, text="Harmful Algal Bloom (HAB) Detection", padding="5")
    hab_frame.pack(fill=tk.X, padx=5, pady=5)

    ttk.Checkbutton(
        hab_frame,
        text="Enable HAB Detection Module",
        variable=gui.enable_hab_detection_var
    ).pack(anchor=tk.W, pady=2)

    hab_options = ttk.Frame(hab_frame)
    hab_options.pack(fill=tk.X, padx=20, pady=5)

    ttk.Checkbutton(
        hab_options,
        text="Cyanobacteria Probability Map (NDCI-based)",
        variable=gui.hab_cyano_map_var
    ).pack(anchor=tk.W, pady=2)

    ttk.Checkbutton(
        hab_options,
        text="Bloom Risk Classification (0-3 scale)",
        variable=gui.hab_risk_class_var
    ).pack(anchor=tk.W, pady=2)

    ttk.Checkbutton(
        hab_options,
        text="Biomass Alert Flags",
        variable=gui.hab_alert_flags_var
    ).pack(anchor=tk.W, pady=2)

    # Thresholds
    thresh_frame = ttk.Frame(hab_options)
    thresh_frame.pack(fill=tk.X, pady=5)

    ttk.Label(thresh_frame, text="High Biomass Threshold:").pack(side=tk.LEFT)
    ttk.Spinbox(
        thresh_frame,
        textvariable=gui.hab_biomass_threshold_var,
        from_=5, to=100, increment=5, width=8
    ).pack(side=tk.LEFT, padx=5)
    ttk.Label(thresh_frame, text="ug/L", style='Muted.TLabel').pack(side=tk.LEFT)

    ttk.Label(thresh_frame, text="   Extreme:").pack(side=tk.LEFT, padx=(20, 0))
    ttk.Spinbox(
        thresh_frame,
        textvariable=gui.hab_extreme_threshold_var,
        from_=50, to=500, increment=10, width=8
    ).pack(side=tk.LEFT, padx=5)
    ttk.Label(thresh_frame, text="ug/L", style='Muted.TLabel').pack(side=tk.LEFT)

    # Trophic State Index
    tsi_frame = ttk.LabelFrame(section.content_frame, text="Trophic State Index (Carlson 1977)", padding="5")
    tsi_frame.pack(fill=tk.X, padx=5, pady=5)

    ttk.Checkbutton(
        tsi_frame,
        text="Calculate TSI from Chlorophyll-a",
        variable=gui.enable_tsi_var
    ).pack(anchor=tk.W, pady=2)

    ttk.Checkbutton(
        tsi_frame,
        text="Include Secchi depth component in TSI",
        variable=gui.tsi_include_secchi_var
    ).pack(anchor=tk.W, pady=2, padx=20)

    ttk.Label(
        tsi_frame,
        text="Scale: <40 Oligotrophic | 40-50 Mesotrophic | 50-70 Eutrophic | >70 Hypereutrophic",
        font=('Consolas', 9)
    ).pack(anchor=tk.W, padx=5, pady=5)

    # Oceanographic Features
    ocean_frame = ttk.LabelFrame(section.content_frame, text="Oceanographic Features", padding="5")
    ocean_frame.pack(fill=tk.X, padx=5, pady=5)

    # Upwelling
    up_frame = ttk.Frame(ocean_frame)
    up_frame.pack(fill=tk.X, pady=2)

    ttk.Checkbutton(
        up_frame,
        text="Upwelling Detection",
        variable=gui.enable_upwelling_var
    ).pack(side=tk.LEFT)

    ttk.Label(up_frame, text="CHL Threshold:", style='Muted.TLabel').pack(side=tk.LEFT, padx=(20, 5))
    ttk.Spinbox(
        up_frame,
        textvariable=gui.upwelling_chl_threshold_var,
        from_=1, to=50, increment=1, width=6
    ).pack(side=tk.LEFT)
    ttk.Label(up_frame, text="ug/L", style='Muted.TLabel').pack(side=tk.LEFT, padx=5)

    # River Plumes
    plume_frame = ttk.Frame(ocean_frame)
    plume_frame.pack(fill=tk.X, pady=2)

    ttk.Checkbutton(
        plume_frame,
        text="River Plume Tracking",
        variable=gui.enable_river_plumes_var
    ).pack(side=tk.LEFT)

    ttk.Label(plume_frame, text="TSS Threshold:", style='Muted.TLabel').pack(side=tk.LEFT, padx=(20, 5))
    ttk.Spinbox(
        plume_frame,
        textvariable=gui.river_plume_tss_threshold_var,
        from_=5, to=100, increment=5, width=6
    ).pack(side=tk.LEFT)
    ttk.Label(plume_frame, text="g/m3", style='Muted.TLabel').pack(side=tk.LEFT, padx=5)

    # Particle Size
    ttk.Checkbutton(
        ocean_frame,
        text="Particle Size Estimation (bbp spectral slope)",
        variable=gui.enable_particle_size_var
    ).pack(anchor=tk.W, pady=2)


def _apply_rgb_preset(gui, preset):
    """Apply RGB composite preset."""
    # All RGB variables
    rgb_vars = [
        gui.rgb_true_color_var,
        gui.rgb_false_color_infrared_var,
        gui.rgb_enhanced_contrast_var,
        gui.rgb_natural_color_var,
        gui.rgb_turbidity_enhanced_var,
        gui.rgb_chlorophyll_enhanced_var,
        gui.rgb_coastal_aerosol_var,
        gui.rgb_cyanobacteria_var,
        gui.rgb_sediment_transport_var,
        gui.rgb_bathymetric_var,
        gui.rgb_ocean_color_var,
        gui.rgb_atmospheric_penetration_var,
        gui.rgb_swir_nir_var,
        gui.rgb_geology_var,
    ]

    # Essential RGB composites
    essential = [
        gui.rgb_true_color_var,
        gui.rgb_false_color_infrared_var,
        gui.rgb_turbidity_enhanced_var,
        gui.rgb_chlorophyll_enhanced_var,
    ]

    if preset == "none":
        for var in rgb_vars:
            var.set(False)
    elif preset == "essential":
        for var in rgb_vars:
            var.set(var in essential)
    elif preset == "complete":
        for var in rgb_vars:
            var.set(True)
    elif preset == "research":
        # Essential plus scientific
        research = essential + [
            gui.rgb_sediment_transport_var,
            gui.rgb_bathymetric_var,
            gui.rgb_ocean_color_var,
            gui.rgb_cyanobacteria_var,
        ]
        for var in rgb_vars:
            var.set(var in research)


def _apply_index_preset(gui, preset):
    """Apply spectral index preset."""
    # All index variables
    idx_vars = [
        gui.idx_ndwi_var,
        gui.idx_mndwi_var,
        gui.idx_awei_var,
        gui.idx_wri_var,
        gui.idx_ndci_var,
        gui.idx_gndvi_var,
        gui.idx_fai_var,
        gui.idx_flh_var,
        gui.idx_mci_var,
        gui.idx_ndti_var,
        gui.idx_ngrdi_var,
        gui.idx_fui_var,
        gui.idx_sdd_var,
        gui.idx_cdom_var,
    ]

    # Essential indices
    essential = [
        gui.idx_ndwi_var,
        gui.idx_ndci_var,
        gui.idx_ndti_var,
        gui.idx_fui_var,
    ]

    if preset == "none":
        for var in idx_vars:
            var.set(False)
    elif preset == "essential":
        for var in idx_vars:
            var.set(var in essential)
    elif preset == "complete":
        for var in idx_vars:
            var.set(True)
