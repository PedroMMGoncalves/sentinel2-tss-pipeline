"""
C2RCC Tab for Sentinel-2 TSS Pipeline GUI.

Provides the C2RCC atmospheric correction configuration interface.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import logging

logger = logging.getLogger('sentinel2_tss_pipeline')


def create_c2rcc_tab(gui, notebook):
    """
    Create the C2RCC configuration tab.

    Args:
        gui: Parent GUI instance with all tk variables.
        notebook: ttk.Notebook to add the tab to.

    Returns:
        int: Tab index.
    """
    frame = ttk.Frame(notebook)
    tab_index = notebook.add(frame, text="C2RCC Parameters")

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

    # Title
    ttk.Label(
        scrollable_frame, text="C2RCC Atmospheric Correction Parameters",
        font=("Arial", 14, "bold")
    ).pack(pady=10)

    # Info note
    _create_info_note(scrollable_frame)

    # ECMWF Configuration
    _create_ecmwf_section(scrollable_frame, gui)

    # Neural Network and DEM Configuration
    _create_nn_dem_section(scrollable_frame, gui)

    # Water Properties
    _create_water_properties_section(scrollable_frame, gui)

    # Atmospheric Parameters
    _create_atmospheric_section(scrollable_frame, gui)

    # Output Products
    _create_output_products_section(scrollable_frame, gui)

    return tab_index


def _create_info_note(parent):
    """Create informational note about SNAP products."""
    note_frame = ttk.Frame(parent, relief=tk.SUNKEN, borderwidth=1)
    note_frame.pack(fill=tk.X, padx=10, pady=5)

    note_text = (
        "Automatic SNAP Products:\n"
        "  TSM and CHL concentrations are automatically calculated during C2RCC processing\n"
        "  Uncertainty maps (unc_tsm.img, unc_chl.img) are generated when uncertainties are enabled\n"
        "  Water leaving reflectance (rhow) bands are generated for optional Jiang TSS processing"
    )

    ttk.Label(
        note_frame, text=note_text,
        font=("Arial", 9), foreground="darkblue",
        wraplength=600, justify=tk.LEFT, padding="5"
    ).pack()


def _create_ecmwf_section(parent, gui):
    """Create ECMWF configuration section."""
    ecmwf_frame = ttk.LabelFrame(
        parent, text="ECMWF Auxiliary Data (Enhanced Accuracy)", padding="10"
    )
    ecmwf_frame.pack(fill=tk.X, padx=10, pady=5)

    ttk.Checkbutton(
        ecmwf_frame, text="Use ECMWF auxiliary data (ENABLED BY DEFAULT)",
        variable=gui.use_ecmwf_var,
        command=lambda g=gui: _on_ecmwf_toggle(g)
    ).pack(anchor=tk.W, pady=2)

    ttk.Label(
        ecmwf_frame,
        text="Uses real atmospheric conditions at acquisition time for superior accuracy",
        font=("Arial", 9), foreground="darkgreen"
    ).pack(anchor=tk.W, pady=2)


def _create_nn_dem_section(parent, gui):
    """Create neural network and DEM configuration section."""
    nn_frame = ttk.LabelFrame(
        parent, text="Neural Network & Terrain Correction", padding="10"
    )
    nn_frame.pack(fill=tk.X, padx=10, pady=5)

    nn_grid = ttk.Frame(nn_frame)
    nn_grid.pack(fill=tk.X)

    # Neural Network selection
    ttk.Label(nn_grid, text="Neural Network:").grid(
        row=0, column=0, sticky=tk.W, padx=5, pady=5
    )
    nn_combo = ttk.Combobox(
        nn_grid, textvariable=gui.net_set_var, width=25, state="readonly",
        values=["C2RCC-Nets", "C2X-Nets", "C2X-COMPLEX-Nets"]
    )
    nn_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

    ttk.Label(
        nn_grid, text="C2RCC-Nets: Standard | C2X-Nets: Extended range | C2X-COMPLEX: Turbid waters",
        font=("Arial", 8), foreground="gray"
    ).grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)

    # DEM selection
    ttk.Label(nn_grid, text="DEM Source:").grid(
        row=1, column=0, sticky=tk.W, padx=5, pady=5
    )
    dem_combo = ttk.Combobox(
        nn_grid, textvariable=gui.dem_name_var, width=25, state="readonly",
        values=["Copernicus 30m Global DEM", "SRTM 3Sec", "GETASSE30"]
    )
    dem_combo.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)

    ttk.Label(
        nn_grid, text="Digital Elevation Model for terrain correction",
        font=("Arial", 8), foreground="gray"
    ).grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)

    # Elevation override
    ttk.Label(nn_grid, text="Elevation (m):").grid(
        row=2, column=0, sticky=tk.W, padx=5, pady=5
    )
    ttk.Spinbox(
        nn_grid, from_=0, to=5000, width=10,
        textvariable=gui.elevation_var, increment=10
    ).grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)

    ttk.Label(
        nn_grid, text="Override elevation (0 = use DEM automatically)",
        font=("Arial", 8), foreground="gray"
    ).grid(row=2, column=2, sticky=tk.W, padx=5, pady=5)


def _create_water_properties_section(parent, gui):
    """Create water properties section."""
    water_frame = ttk.LabelFrame(parent, text="Water Properties", padding="10")
    water_frame.pack(fill=tk.X, padx=10, pady=5)

    # Quick presets
    preset_frame = ttk.Frame(water_frame)
    preset_frame.pack(fill=tk.X, pady=(0, 10))

    ttk.Label(preset_frame, text="Quick Presets:").pack(side=tk.LEFT)

    preset_btn_frame = ttk.Frame(preset_frame)
    preset_btn_frame.pack(side=tk.LEFT, padx=(10, 0))

    presets = [
        ("Coastal", {"salinity": 35.0, "temperature": 15.0}),
        ("Inland", {"salinity": 0.1, "temperature": 20.0}),
        ("Estuary", {"salinity": 15.0, "temperature": 18.0})
    ]

    for name, values in presets:
        ttk.Button(
            preset_btn_frame, text=name, width=8,
            command=lambda v=values, g=gui: _apply_water_preset(g, v)
        ).pack(side=tk.LEFT, padx=2)

    # Water parameters grid
    water_grid = ttk.Frame(water_frame)
    water_grid.pack(fill=tk.X)

    # Salinity
    ttk.Label(water_grid, text="Salinity (PSU):").grid(
        row=0, column=0, sticky=tk.W, padx=5, pady=2
    )
    ttk.Spinbox(
        water_grid, from_=0.1, to=42, width=10,
        textvariable=gui.salinity_var, increment=0.5
    ).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)

    # Temperature
    ttk.Label(water_grid, text="Temperature (C):").grid(
        row=0, column=2, sticky=tk.W, padx=5, pady=2
    )
    ttk.Spinbox(
        water_grid, from_=0.1, to=35, width=10,
        textvariable=gui.temperature_var, increment=0.5
    ).grid(row=0, column=3, sticky=tk.W, padx=5, pady=2)


def _create_atmospheric_section(parent, gui):
    """Create atmospheric parameters section."""
    atmos_frame = ttk.LabelFrame(
        parent, text="Atmospheric Parameters", padding="10"
    )
    atmos_frame.pack(fill=tk.X, padx=10, pady=5)

    atmos_grid = ttk.Frame(atmos_frame)
    atmos_grid.pack(fill=tk.X)

    # Ozone
    ttk.Label(atmos_grid, text="Ozone (DU):").grid(
        row=0, column=0, sticky=tk.W, padx=5, pady=2
    )
    ttk.Spinbox(
        atmos_grid, from_=100, to=800, width=10,
        textvariable=gui.ozone_var, increment=10
    ).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)

    # Pressure
    ttk.Label(atmos_grid, text="Pressure (hPa):").grid(
        row=0, column=2, sticky=tk.W, padx=5, pady=2
    )
    ttk.Spinbox(
        atmos_grid, from_=850, to=1030, width=10,
        textvariable=gui.pressure_var, increment=5
    ).grid(row=0, column=3, sticky=tk.W, padx=5, pady=2)


def _create_output_products_section(parent, gui):
    """Create output products configuration section."""
    output_frame = ttk.LabelFrame(
        parent, text="Output Products Configuration", padding="10"
    )
    output_frame.pack(fill=tk.X, padx=10, pady=5)

    # Essential outputs
    essential_frame = ttk.LabelFrame(
        output_frame, text="Essential Outputs", padding="5"
    )
    essential_frame.pack(fill=tk.X, pady=5)

    ttk.Checkbutton(
        essential_frame,
        text="Water leaving reflectance (rhow) - Required for TSS",
        variable=gui.output_rhow_var,
        command=lambda g=gui: _on_rhow_toggle(g)
    ).pack(anchor=tk.W, pady=2)

    ttk.Checkbutton(
        essential_frame, text="Diffuse attenuation coefficient (Kd)",
        variable=gui.output_kd_var
    ).pack(anchor=tk.W, pady=2)

    ttk.Checkbutton(
        essential_frame,
        text="Uncertainty estimates (enables unc_tsm.img & unc_chl.img)",
        variable=gui.output_uncertainties_var
    ).pack(anchor=tk.W, pady=2)

    # Reflectance products
    reflectance_frame = ttk.LabelFrame(
        output_frame, text="Reflectance Products", padding="5"
    )
    reflectance_frame.pack(fill=tk.X, pady=5)

    ttk.Checkbutton(
        reflectance_frame, text="Atmospherically corrected reflectance",
        variable=gui.output_ac_reflectance_var
    ).pack(anchor=tk.W, pady=2)

    ttk.Checkbutton(
        reflectance_frame, text="Top-of-atmosphere reflectance (rtoa)",
        variable=gui.output_rtoa_var
    ).pack(anchor=tk.W, pady=2)

    ttk.Checkbutton(
        reflectance_frame, text="Remote sensing reflectance (Rrs)",
        variable=gui.output_rrs_var
    ).pack(anchor=tk.W, pady=2)


def _on_ecmwf_toggle(gui):
    """Handle ECMWF toggle with confirmation."""
    if not gui.use_ecmwf_var.get():
        result = messagebox.askyesno(
            "ECMWF Disabled",
            "Disabling ECMWF auxiliary data will reduce atmospheric correction accuracy.\n\n"
            "ECMWF provides real-time ozone and pressure data at acquisition time.\n\n"
            "Continue anyway?",
            parent=gui.root
        )
        if not result:
            gui.use_ecmwf_var.set(True)


def _on_rhow_toggle(gui):
    """Handle rhow toggle with warning."""
    if not gui.output_rhow_var.get():
        result = messagebox.askyesno(
            "Warning",
            "Disabling water leaving reflectance (rhow) will prevent Jiang TSS processing.\n\n"
            "This output is required for advanced TSS analysis.\n\n"
            "Continue anyway?",
            parent=gui.root
        )
        if not result:
            gui.output_rhow_var.set(True)


def _apply_water_preset(gui, values):
    """Apply water parameter preset."""
    gui.salinity_var.set(values["salinity"])
    gui.temperature_var.set(values["temperature"])
    gui.status_var.set("Water preset applied")
