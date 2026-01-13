"""
Spatial Tab for GUI v2.

Combined tab for Resampling + Spatial Subset + Processing Mask.
"""

import tkinter as tk
from tkinter import ttk, filedialog

from ..widgets import CollapsibleFrame, create_tooltip
from ..theme import ThemeManager


def create_spatial_tab(gui, notebook):
    """
    Create the Spatial Configuration tab.

    Combines:
    - Resampling settings
    - Spatial subset options
    - Processing mask configuration

    Args:
        gui: Parent GUI instance
        notebook: ttk.Notebook to add tab to

    Returns:
        Tab index
    """
    frame = ttk.Frame(notebook, padding="5")
    tab_index = notebook.add(frame, text=" Spatial ")

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
        text="Spatial Configuration",
        style='Subtitle.TLabel'
    ).pack(pady=(5, 10), padx=10, anchor=tk.W)

    # === SECTION A: RESAMPLING ===
    _create_resampling_section(gui, content)

    # === SECTION B: SPATIAL SUBSET ===
    _create_subset_section(gui, content)

    # === SECTION C: PROCESSING MASK ===
    _create_mask_section(gui, content)

    return tab_index


def _create_resampling_section(gui, parent):
    """Create the Resampling configuration section."""
    section = CollapsibleFrame(parent, title="Resampling Configuration", expanded=True)
    section.pack(fill=tk.X, padx=5, pady=5)

    # Description
    ttk.Label(
        section.content_frame,
        text="Configure target resolution and resampling methods for band alignment",
        style='Muted.TLabel',
        font=('Segoe UI', 9)
    ).pack(anchor=tk.W, padx=5, pady=(0, 10))

    # Resolution selection with visual cards
    res_frame = ttk.Frame(section.content_frame)
    res_frame.pack(fill=tk.X, pady=5)

    ttk.Label(res_frame, text="Target Resolution:", font=('Segoe UI', 10, 'bold')).pack(anchor=tk.W, padx=5, pady=(0, 5))

    cards_frame = ttk.Frame(res_frame)
    cards_frame.pack(fill=tk.X, padx=5)

    resolutions = [
        ("10", "10 meters", "Highest spatial detail", "Larger files (~2GB)"),
        ("20", "20 meters", "Balanced option", "Medium files (~500MB)"),
        ("60", "60 meters", "Fastest processing", "Smallest files (~50MB)"),
    ]

    for value, title, desc1, desc2 in resolutions:
        card = ttk.Frame(cards_frame, relief="groove", borderwidth=1)
        card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=3)

        inner = ttk.Frame(card, padding="8")
        inner.pack(fill=tk.BOTH, expand=True)

        rb = ttk.Radiobutton(
            inner,
            text=title,
            variable=gui.resolution_var,
            value=value,
            style='TRadiobutton'
        )
        rb.pack(anchor=tk.W)

        ttk.Label(inner, text=desc1, style='Muted.TLabel', font=('Segoe UI', 9)).pack(anchor=tk.W)
        ttk.Label(inner, text=desc2, style='Muted.TLabel', font=('Segoe UI', 8)).pack(anchor=tk.W)

    # Advanced options (collapsed by default)
    advanced_frame = ttk.LabelFrame(section.content_frame, text="Advanced Resampling Options", padding="5")
    advanced_frame.pack(fill=tk.X, padx=5, pady=10)

    opts_frame = ttk.Frame(advanced_frame)
    opts_frame.pack(fill=tk.X, pady=5)

    # Row 1: Upsampling and Downsampling
    ttk.Label(opts_frame, text="Upsampling:").grid(row=0, column=0, sticky=tk.E, padx=5, pady=3)
    up_combo = ttk.Combobox(
        opts_frame,
        textvariable=gui.upsampling_var,
        values=["Nearest", "Bilinear", "Bicubic"],
        state="readonly",
        width=12
    )
    up_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=3)
    create_tooltip(up_combo, "Method for increasing resolution\nBilinear recommended for smooth results")

    ttk.Label(opts_frame, text="Downsampling:").grid(row=0, column=2, sticky=tk.E, padx=5, pady=3)
    down_combo = ttk.Combobox(
        opts_frame,
        textvariable=gui.downsampling_var,
        values=["First", "Min", "Max", "Mean", "Median"],
        state="readonly",
        width=12
    )
    down_combo.grid(row=0, column=3, sticky=tk.W, padx=5, pady=3)
    create_tooltip(down_combo, "Method for decreasing resolution\nMean preserves spectral accuracy")

    # Row 2: Flag downsampling and Pyramid
    ttk.Label(opts_frame, text="Flag Downsampling:").grid(row=1, column=0, sticky=tk.E, padx=5, pady=3)
    flag_combo = ttk.Combobox(
        opts_frame,
        textvariable=gui.flag_downsampling_var,
        values=["First", "FlagAnd", "FlagOr", "FlagMedianAnd", "FlagMedianOr"],
        state="readonly",
        width=12
    )
    flag_combo.grid(row=1, column=1, sticky=tk.W, padx=5, pady=3)
    create_tooltip(flag_combo, "Method for combining quality flags\nFirst keeps original flag values")

    pyramid_cb = ttk.Checkbutton(
        opts_frame,
        text="Use pyramid levels",
        variable=gui.pyramid_var
    )
    pyramid_cb.grid(row=1, column=2, columnspan=2, sticky=tk.W, padx=5, pady=3)
    create_tooltip(pyramid_cb, "Use image pyramids for faster resampling\nRecommended for large images")


def _create_subset_section(gui, parent):
    """Create the Spatial Subset configuration section."""
    section = CollapsibleFrame(parent, title="Spatial Subset", expanded=False)
    section.pack(fill=tk.X, padx=5, pady=5)

    # Enable toggle
    enable_frame = ttk.Frame(section.content_frame)
    enable_frame.pack(fill=tk.X, pady=5)

    enable_cb = ttk.Checkbutton(
        enable_frame,
        text="Enable spatial subset (process only a region of interest)",
        variable=gui.enable_subset_var,
        command=lambda: _toggle_subset_options(gui)
    )
    enable_cb.pack(anchor=tk.W, padx=5)

    # Subset options container
    gui.subset_options_frame = ttk.Frame(section.content_frame)
    gui.subset_options_frame.pack(fill=tk.X, pady=5)

    # Method selection
    method_frame = ttk.LabelFrame(gui.subset_options_frame, text="Subset Method", padding="5")
    method_frame.pack(fill=tk.X, padx=5, pady=5)

    methods = [
        ("none", "Full Scene", "Process entire image without subsetting"),
        ("geometry", "Geometry File", "Use shapefile, KML, or GeoJSON boundary"),
        ("wkt", "WKT Geometry", "Enter Well-Known Text geometry string"),
        ("bbox", "Bounding Box", "Define region using coordinates"),
    ]

    for value, label, tooltip_text in methods:
        rb = ttk.Radiobutton(
            method_frame,
            text=label,
            variable=gui.subset_method_var,
            value=value,
            command=lambda: _update_subset_ui(gui)
        )
        rb.pack(anchor=tk.W, padx=5, pady=2)
        create_tooltip(rb, tooltip_text)

    # Geometry file browser
    gui.geometry_file_frame = ttk.LabelFrame(gui.subset_options_frame, text="Geometry File", padding="5")
    gui.geometry_file_frame.pack(fill=tk.X, padx=5, pady=5)

    file_row = ttk.Frame(gui.geometry_file_frame)
    file_row.pack(fill=tk.X)

    gui.geometry_entry = ttk.Entry(file_row, textvariable=gui.geometry_file_var, width=50)
    gui.geometry_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

    browse_btn = ttk.Button(
        file_row,
        text="Browse...",
        command=lambda: _browse_geometry_file(gui)
    )
    browse_btn.pack(side=tk.LEFT)

    ttk.Label(
        gui.geometry_file_frame,
        text="Supported: Shapefile (.shp), KML (.kml), GeoJSON (.geojson)",
        style='Muted.TLabel',
        font=('Segoe UI', 9)
    ).pack(anchor=tk.W, pady=(5, 0))

    # WKT entry
    gui.wkt_frame = ttk.LabelFrame(gui.subset_options_frame, text="WKT Geometry", padding="5")
    gui.wkt_frame.pack(fill=tk.X, padx=5, pady=5)

    gui.wkt_text = tk.Text(gui.wkt_frame, height=4, font=('Consolas', 9))
    gui.wkt_text.pack(fill=tk.X)
    gui.wkt_text.insert('1.0', "POLYGON((-9.5 38.5, -9.5 39.0, -9.0 39.0, -9.0 38.5, -9.5 38.5))")

    ttk.Label(
        gui.wkt_frame,
        text="Enter WKT geometry (POLYGON, MULTIPOLYGON, or POINT)",
        style='Muted.TLabel',
        font=('Segoe UI', 9)
    ).pack(anchor=tk.W, pady=(5, 0))

    # Bounding box
    gui.bbox_frame = ttk.LabelFrame(gui.subset_options_frame, text="Bounding Box (WGS84)", padding="5")
    gui.bbox_frame.pack(fill=tk.X, padx=5, pady=5)

    bbox_grid = ttk.Frame(gui.bbox_frame)
    bbox_grid.pack()

    # North
    ttk.Label(bbox_grid, text="North:").grid(row=0, column=1, padx=5, pady=2)
    north_spin = ttk.Spinbox(bbox_grid, textvariable=gui.bbox_north_var, from_=-90, to=90, increment=0.1, width=10)
    north_spin.grid(row=0, column=2, padx=5, pady=2)

    # West and East
    ttk.Label(bbox_grid, text="West:").grid(row=1, column=0, padx=5, pady=2)
    west_spin = ttk.Spinbox(bbox_grid, textvariable=gui.bbox_west_var, from_=-180, to=180, increment=0.1, width=10)
    west_spin.grid(row=1, column=1, padx=5, pady=2)

    ttk.Label(bbox_grid, text="East:").grid(row=1, column=2, padx=5, pady=2)
    east_spin = ttk.Spinbox(bbox_grid, textvariable=gui.bbox_east_var, from_=-180, to=180, increment=0.1, width=10)
    east_spin.grid(row=1, column=3, padx=5, pady=2)

    # South
    ttk.Label(bbox_grid, text="South:").grid(row=2, column=1, padx=5, pady=2)
    south_spin = ttk.Spinbox(bbox_grid, textvariable=gui.bbox_south_var, from_=-90, to=90, increment=0.1, width=10)
    south_spin.grid(row=2, column=2, padx=5, pady=2)

    # Initialize state
    _toggle_subset_options(gui)


def _create_mask_section(gui, parent):
    """Create the Processing Mask configuration section."""
    section = CollapsibleFrame(parent, title="Processing Mask", expanded=False)
    section.pack(fill=tk.X, padx=5, pady=5)

    # Description
    ttk.Label(
        section.content_frame,
        text="Define pixel filtering expressions for processing",
        style='Muted.TLabel',
        font=('Segoe UI', 9)
    ).pack(anchor=tk.W, padx=5, pady=(0, 10))

    # Valid pixel expression
    expr_frame = ttk.LabelFrame(section.content_frame, text="Valid Pixel Expression", padding="5")
    expr_frame.pack(fill=tk.X, padx=5, pady=5)

    gui.valid_pixel_entry = ttk.Entry(
        expr_frame,
        textvariable=gui.valid_pixel_expression_var,
        font=('Consolas', 10),
        width=50
    )
    gui.valid_pixel_entry.pack(fill=tk.X, pady=(0, 5))

    ttk.Label(
        expr_frame,
        text="SNAP band math expression (e.g., 'B8 > 0 && B8 < 0.1')\nOnly pixels matching this expression will be processed",
        style='Muted.TLabel',
        font=('Segoe UI', 9),
        justify=tk.LEFT
    ).pack(anchor=tk.W)

    # Preset expressions
    presets_frame = ttk.Frame(expr_frame)
    presets_frame.pack(fill=tk.X, pady=5)

    ttk.Label(presets_frame, text="Presets:").pack(side=tk.LEFT, padx=(0, 5))

    presets = [
        ("Water", "B8 > 0 && B8 < 0.1"),
        ("Land", "B8 >= 0.1"),
        ("Clear Sky", "quality_scene_classification != 3"),
        ("All Valid", "true"),
    ]

    for name, expr in presets:
        btn = ttk.Button(
            presets_frame,
            text=name,
            width=10,
            command=lambda e=expr: gui.valid_pixel_expression_var.set(e)
        )
        btn.pack(side=tk.LEFT, padx=2)

    # Water mask threshold
    threshold_frame = ttk.LabelFrame(section.content_frame, text="Water Mask Threshold", padding="5")
    threshold_frame.pack(fill=tk.X, padx=5, pady=5)

    thresh_row = ttk.Frame(threshold_frame)
    thresh_row.pack(fill=tk.X)

    ttk.Label(thresh_row, text="NIR Threshold:").pack(side=tk.LEFT, padx=5)

    thresh_spin = ttk.Spinbox(
        thresh_row,
        textvariable=gui.water_mask_threshold_var,
        from_=0.001,
        to=0.5,
        increment=0.005,
        width=10,
        format="%.3f"
    )
    thresh_spin.pack(side=tk.LEFT, padx=5)
    create_tooltip(thresh_spin, "Pixels with NIR < threshold are classified as water\nTypical values: 0.01-0.05")

    ttk.Label(
        threshold_frame,
        text="Pixels with NIR reflectance below this value are classified as water",
        style='Muted.TLabel',
        font=('Segoe UI', 9)
    ).pack(anchor=tk.W, padx=5, pady=(5, 0))


def _toggle_subset_options(gui):
    """Toggle subset options visibility based on enable checkbox."""
    if gui.enable_subset_var.get():
        gui.subset_options_frame.pack(fill=tk.X, pady=5)
        _update_subset_ui(gui)
    else:
        gui.subset_options_frame.pack_forget()


def _update_subset_ui(gui):
    """Update subset UI based on selected method."""
    method = gui.subset_method_var.get()

    # Hide all method-specific frames
    gui.geometry_file_frame.pack_forget()
    gui.wkt_frame.pack_forget()
    gui.bbox_frame.pack_forget()

    # Show relevant frame
    if method == "geometry":
        gui.geometry_file_frame.pack(fill=tk.X, padx=5, pady=5)
    elif method == "wkt":
        gui.wkt_frame.pack(fill=tk.X, padx=5, pady=5)
    elif method == "bbox":
        gui.bbox_frame.pack(fill=tk.X, padx=5, pady=5)


def _browse_geometry_file(gui):
    """Open file browser for geometry file."""
    filepath = filedialog.askopenfilename(
        title="Select Geometry File",
        filetypes=[
            ("All Geometry Files", "*.shp *.kml *.geojson *.json"),
            ("Shapefiles", "*.shp"),
            ("KML Files", "*.kml"),
            ("GeoJSON Files", "*.geojson *.json"),
            ("All Files", "*.*"),
        ]
    )
    if filepath:
        gui.geometry_file_var.set(filepath)
