"""
C2RCC Parameters Tab for GUI v2.

Expanded layout with neural network selection, DEM options, and clear output products.
"""

import tkinter as tk
from tkinter import ttk

from ..widgets import CollapsibleFrame, CheckboxGroup, create_tooltip
from ..theme import ThemeManager


def create_c2rcc_tab(gui, notebook):
    """
    Create the C2RCC Parameters tab.

    Args:
        gui: Parent GUI instance
        notebook: ttk.Notebook to add tab to

    Returns:
        Tab index
    """
    frame = ttk.Frame(notebook, padding="5")
    tab_index = notebook.add(frame, text=" C2RCC ")

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
        text="C2RCC Atmospheric Correction",
        style='Subtitle.TLabel'
    ).pack(pady=(5, 10), padx=10, anchor=tk.W)

    # Info note
    info_frame = ttk.Frame(content)
    info_frame.pack(fill=tk.X, padx=10, pady=5)

    info_text = (
        "C2RCC (Case-2 Regional Coast Colour) performs atmospheric correction\n"
        "and derives water quality products from Sentinel-2 imagery."
    )
    ttk.Label(
        info_frame,
        text=info_text,
        style='Muted.TLabel',
        font=('Segoe UI', 9),
        justify=tk.LEFT
    ).pack(anchor=tk.W)

    # === SECTION 1: Neural Network & DEM ===
    _create_nn_dem_section(gui, content)

    # === SECTION 2: Auxiliary Data (ECMWF) ===
    _create_ecmwf_section(gui, content)

    # === SECTION 3: Water Properties ===
    _create_water_properties_section(gui, content)

    # === SECTION 4: Atmospheric Parameters ===
    _create_atmospheric_section(gui, content)

    # === SECTION 5: Output Products ===
    _create_output_products_section(gui, content)

    return tab_index


def _create_nn_dem_section(gui, parent):
    """Create Neural Network and DEM configuration section."""
    section = CollapsibleFrame(parent, title="Neural Network & Terrain", expanded=True)
    section.pack(fill=tk.X, padx=5, pady=5)

    grid_frame = ttk.Frame(section.content_frame)
    grid_frame.pack(fill=tk.X, pady=5)

    # Neural Network selection
    ttk.Label(
        grid_frame,
        text="Neural Network:",
        font=('Segoe UI', 10)
    ).grid(row=0, column=0, sticky=tk.E, padx=5, pady=5)

    nn_combo = ttk.Combobox(
        grid_frame,
        textvariable=gui.neural_network_var,
        values=[
            "C2RCC-Nets",
            "C2X-Nets",
            "C2X-COMPLEX-Nets"
        ],
        state="readonly",
        width=20
    )
    nn_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
    create_tooltip(nn_combo,
        "C2RCC-Nets: Standard networks (faster)\n"
        "C2X-Nets: Extended range, better for turbid waters\n"
        "C2X-COMPLEX-Nets: Complex waters with extreme conditions"
    )

    # Neural network description
    nn_desc = ttk.Label(
        grid_frame,
        text="Standard networks for typical coastal/inland waters",
        style='Muted.TLabel',
        font=('Segoe UI', 9)
    )
    nn_desc.grid(row=0, column=2, sticky=tk.W, padx=10, pady=5)

    def update_nn_desc(*args):
        nn = gui.neural_network_var.get()
        descriptions = {
            "C2RCC-Nets": "Standard networks for typical coastal/inland waters",
            "C2X-Nets": "Extended range for turbid and high-TSM waters",
            "C2X-COMPLEX-Nets": "For extreme conditions (very turbid, algal blooms)"
        }
        nn_desc.configure(text=descriptions.get(nn, ""))

    gui.neural_network_var.trace_add("write", update_nn_desc)

    # DEM selection
    ttk.Label(
        grid_frame,
        text="Digital Elevation Model:",
        font=('Segoe UI', 10)
    ).grid(row=1, column=0, sticky=tk.E, padx=5, pady=5)

    dem_combo = ttk.Combobox(
        grid_frame,
        textvariable=gui.dem_var,
        values=[
            "Copernicus 30m Global DEM",
            "SRTM 3Sec",
            "GETASSE30"
        ],
        state="readonly",
        width=25
    )
    dem_combo.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
    create_tooltip(dem_combo,
        "DEM used for terrain correction\n"
        "Copernicus 30m: Most accurate, global coverage\n"
        "SRTM 3Sec: ~90m resolution, good alternative\n"
        "GETASSE30: Combined GLOBE/ACE30, for compatibility"
    )

    # Elevation override
    ttk.Label(
        grid_frame,
        text="Elevation Override (m):",
        font=('Segoe UI', 10)
    ).grid(row=2, column=0, sticky=tk.E, padx=5, pady=5)

    elev_spin = ttk.Spinbox(
        grid_frame,
        textvariable=gui.elevation_var,
        from_=-500,
        to=9000,
        increment=10,
        width=10
    )
    elev_spin.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
    create_tooltip(elev_spin,
        "Manual elevation in meters\n"
        "Set to 0 to use DEM values\n"
        "Useful for water bodies at known elevations"
    )


def _create_ecmwf_section(gui, parent):
    """Create ECMWF auxiliary data section."""
    section = CollapsibleFrame(parent, title="Auxiliary Atmospheric Data", expanded=True)
    section.pack(fill=tk.X, padx=5, pady=5)

    ecmwf_cb = ttk.Checkbutton(
        section.content_frame,
        text="Use ECMWF auxiliary data (ERA5 reanalysis)",
        variable=gui.use_ecmwf_var,
        command=lambda: _on_ecmwf_toggle(gui)
    )
    ecmwf_cb.pack(anchor=tk.W, pady=5, padx=5)

    info_text = (
        "RECOMMENDED: Uses ERA5 reanalysis data for accurate atmospheric\n"
        "conditions (ozone, pressure, water vapor) at acquisition time.\n"
        "Requires internet connection for data download."
    )
    ttk.Label(
        section.content_frame,
        text=info_text,
        style='Success.TLabel',
        font=('Segoe UI', 9),
        justify=tk.LEFT
    ).pack(anchor=tk.W, padx=25)


def _create_water_properties_section(gui, parent):
    """Create water properties configuration section."""
    section = CollapsibleFrame(parent, title="Water Properties", expanded=True)
    section.pack(fill=tk.X, padx=5, pady=5)

    # Presets row
    presets_frame = ttk.Frame(section.content_frame)
    presets_frame.pack(fill=tk.X, pady=5)

    ttk.Label(presets_frame, text="Quick Presets:", font=('Segoe UI', 10)).pack(side=tk.LEFT, padx=5)

    presets = [
        ("Coastal Ocean", 35.0, 15.0, "Open ocean, high salinity"),
        ("Inland Lake", 0.5, 18.0, "Freshwater, temperate"),
        ("Estuary", 15.0, 16.0, "Mixed fresh/salt water"),
        ("Tropical", 35.0, 28.0, "Warm tropical waters"),
        ("Cold Seas", 35.0, 5.0, "Arctic/sub-arctic waters"),
    ]

    for name, sal, temp, desc in presets:
        btn = ttk.Button(
            presets_frame,
            text=name,
            width=12,
            command=lambda s=sal, t=temp: _apply_preset(gui, s, t)
        )
        btn.pack(side=tk.LEFT, padx=2)
        create_tooltip(btn, f"{desc}\nSalinity: {sal} PSU, Temperature: {temp} C")

    # Parameters grid
    params_frame = ttk.Frame(section.content_frame)
    params_frame.pack(fill=tk.X, pady=10)

    # Salinity
    ttk.Label(params_frame, text="Salinity (PSU):", font=('Segoe UI', 10)).grid(row=0, column=0, sticky=tk.E, padx=5, pady=5)
    sal_spin = ttk.Spinbox(
        params_frame,
        textvariable=gui.salinity_var,
        from_=0, to=50, increment=0.5, width=10
    )
    sal_spin.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
    create_tooltip(sal_spin, "Practical Salinity Units\nOcean: ~35, Brackish: 5-30, Fresh: <0.5")

    # Temperature
    ttk.Label(params_frame, text="Temperature (C):", font=('Segoe UI', 10)).grid(row=0, column=2, sticky=tk.E, padx=5, pady=5)
    temp_spin = ttk.Spinbox(
        params_frame,
        textvariable=gui.temperature_var,
        from_=-2, to=40, increment=0.5, width=10
    )
    temp_spin.grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)
    create_tooltip(temp_spin, "Water surface temperature\nAffects water absorption properties")


def _create_atmospheric_section(gui, parent):
    """Create atmospheric parameters section."""
    section = CollapsibleFrame(parent, title="Manual Atmospheric Parameters", expanded=False)
    section.pack(fill=tk.X, padx=5, pady=5)

    ttk.Label(
        section.content_frame,
        text="These values are used only when ECMWF data is disabled",
        style='Warning.TLabel',
        font=('Segoe UI', 9)
    ).pack(anchor=tk.W, padx=5, pady=(0, 10))

    atmos_frame = ttk.Frame(section.content_frame)
    atmos_frame.pack(fill=tk.X, pady=5)

    # Ozone
    ttk.Label(atmos_frame, text="Ozone (DU):").grid(row=0, column=0, sticky=tk.E, padx=5, pady=5)
    oz_spin = ttk.Spinbox(
        atmos_frame,
        textvariable=gui.ozone_var,
        from_=200, to=500, increment=5, width=10
    )
    oz_spin.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
    create_tooltip(oz_spin, "Total ozone column in Dobson Units\nTypical range: 250-400 DU")

    # Pressure
    ttk.Label(atmos_frame, text="Pressure (hPa):").grid(row=0, column=2, sticky=tk.E, padx=5, pady=5)
    press_spin = ttk.Spinbox(
        atmos_frame,
        textvariable=gui.pressure_var,
        from_=900, to=1100, increment=5, width=10
    )
    press_spin.grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)
    create_tooltip(press_spin, "Surface air pressure\nStandard sea level: 1013.25 hPa")


def _create_output_products_section(gui, parent):
    """Create output products configuration section with clear product names."""
    section = CollapsibleFrame(parent, title="C2RCC Output Products", expanded=True)
    section.pack(fill=tk.X, padx=5, pady=5)

    # Info about automatic products
    auto_frame = ttk.LabelFrame(section.content_frame, text="Always Produced (Required)", padding="5")
    auto_frame.pack(fill=tk.X, padx=5, pady=5)

    auto_products = [
        "conc_tsm.img - Total Suspended Matter concentration (g/m3)",
        "conc_chl.img - Chlorophyll-a concentration (mg/m3)",
        "iop_bpart.img - Particle backscattering coefficient",
        "iop_bwit.img - White particle scattering coefficient",
        "iop_apig.img - Phytoplankton absorption coefficient",
    ]

    for product in auto_products:
        ttk.Label(
            auto_frame,
            text=f"  {product}",
            font=('Consolas', 9)
        ).pack(anchor=tk.W)

    # Reflectance products
    refl_frame = ttk.LabelFrame(section.content_frame, text="Reflectance Products", padding="5")
    refl_frame.pack(fill=tk.X, padx=5, pady=5)

    ttk.Checkbutton(
        refl_frame,
        text="rhow_B1-B8A - Water-leaving reflectance (REQUIRED for Jiang TSS)",
        variable=gui.output_rhow_var
    ).pack(anchor=tk.W, pady=2)
    create_tooltip(refl_frame, "Water-leaving reflectance is essential for TSS calculation")

    ttk.Checkbutton(
        refl_frame,
        text="rrs_B1-B8A - Remote sensing reflectance (Rrs = rhow / pi)",
        variable=gui.output_rrs_var
    ).pack(anchor=tk.W, pady=2)

    ttk.Checkbutton(
        refl_frame,
        text="rtoa_B1-B12 - Top-of-atmosphere reflectance",
        variable=gui.output_rtoa_var
    ).pack(anchor=tk.W, pady=2)

    ttk.Checkbutton(
        refl_frame,
        text="rtosa_gc_B* - Gas-corrected TOA reflectance",
        variable=gui.output_ac_reflectance_var
    ).pack(anchor=tk.W, pady=2)

    # Optical properties
    iop_frame = ttk.LabelFrame(section.content_frame, text="Inherent Optical Properties (IOPs)", padding="5")
    iop_frame.pack(fill=tk.X, padx=5, pady=5)

    ttk.Checkbutton(
        iop_frame,
        text="kd_B1-B8A - Diffuse attenuation coefficient (water clarity)",
        variable=gui.output_kd_var
    ).pack(anchor=tk.W, pady=2)

    ttk.Checkbutton(
        iop_frame,
        text="iop_adet - Detritus absorption coefficient",
        variable=gui.output_iop_var
    ).pack(anchor=tk.W, pady=2)

    ttk.Checkbutton(
        iop_frame,
        text="iop_agelb - Gelbstoff (CDOM) absorption coefficient",
        variable=gui.output_agelb_var
    ).pack(anchor=tk.W, pady=2)

    # Uncertainty products
    unc_frame = ttk.LabelFrame(section.content_frame, text="Uncertainty Products", padding="5")
    unc_frame.pack(fill=tk.X, padx=5, pady=5)

    ttk.Checkbutton(
        unc_frame,
        text="unc_tsm.img, unc_chl.img - TSM/CHL uncertainty maps",
        variable=gui.output_uncertainties_var
    ).pack(anchor=tk.W, pady=2)

    ttk.Label(
        unc_frame,
        text="Provides per-pixel uncertainty estimates for quality assessment",
        style='Muted.TLabel',
        font=('Segoe UI', 9)
    ).pack(anchor=tk.W, padx=20)

    # Advanced atmospheric products (collapsed)
    adv_frame = ttk.LabelFrame(section.content_frame, text="Advanced Atmospheric Products", padding="5")
    adv_frame.pack(fill=tk.X, padx=5, pady=5)

    ttk.Checkbutton(
        adv_frame,
        text="rpath_B* - Path radiance",
        variable=gui.output_rpath_var
    ).pack(anchor=tk.W, pady=2)

    trans_frame = ttk.Frame(adv_frame)
    trans_frame.pack(anchor=tk.W, pady=2)

    ttk.Checkbutton(
        trans_frame,
        text="tdown_B* - Downwelling transmittance",
        variable=gui.output_tdown_var
    ).pack(side=tk.LEFT)

    ttk.Checkbutton(
        trans_frame,
        text="tup_B* - Upwelling transmittance",
        variable=gui.output_tup_var
    ).pack(side=tk.LEFT, padx=20)

    ttk.Checkbutton(
        adv_frame,
        text="c2rcc_flags - Quality and out-of-scope flags",
        variable=gui.output_flags_var
    ).pack(anchor=tk.W, pady=2)


def _apply_preset(gui, salinity, temperature):
    """Apply water property preset."""
    gui.salinity_var.set(salinity)
    gui.temperature_var.set(temperature)


def _on_ecmwf_toggle(gui):
    """Handle ECMWF toggle."""
    pass  # Visual feedback could be added here
