"""
Spatial Processing Tab for Sentinel-2 TSS Pipeline GUI.

Merges resampling configuration and spatial subset into one tab.
Includes interactive map preview when tkintermapview is available.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import logging

logger = logging.getLogger('sentinel2_tss_pipeline')

# Optional map widget
try:
    from tkintermapview import TkinterMapView
    HAS_MAP_WIDGET = True
except ImportError:
    HAS_MAP_WIDGET = False


def create_spatial_tab(gui, notebook):
    """
    Create the Spatial Processing tab (Resampling + Subset + Map Preview).

    Args:
        gui: Parent GUI instance with all tk variables.
        notebook: ttk.Notebook to add the tab to.

    Returns:
        int: Tab index.
    """
    frame = ttk.Frame(notebook)
    tab_index = notebook.add(frame, text="Spatial")

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

    # --- Resampling Section ---
    _create_resampling_section(scrollable_frame, gui)

    # --- Subset Section ---
    _create_subset_section(scrollable_frame, gui)

    # --- Map Preview Section ---
    _create_map_preview_section(scrollable_frame, gui)

    # Set initial visibility
    _update_subset_visibility(gui)

    return tab_index


# ===== RESAMPLING =====

def _create_resampling_section(parent, gui):
    """Create resampling configuration section."""
    res_frame = ttk.LabelFrame(parent, text="Resampling", padding="10")
    res_frame.pack(fill=tk.X, padx=10, pady=5)

    # Resolution selection (horizontal radio buttons)
    res_row = ttk.Frame(res_frame)
    res_row.pack(fill=tk.X, pady=2)

    ttk.Label(res_row, text="Resolution:").pack(side=tk.LEFT, padx=(0, 10))

    for value, label in [("10", "10m"), ("20", "20m"), ("60", "60m")]:
        ttk.Radiobutton(
            res_row, text=label,
            variable=gui.resolution_var, value=value
        ).pack(side=tk.LEFT, padx=5)

    # Methods grid
    methods_grid = ttk.Frame(res_frame)
    methods_grid.pack(fill=tk.X, pady=5)

    # Upsampling
    ttk.Label(methods_grid, text="Upsampling:").grid(
        row=0, column=0, sticky=tk.W, padx=5, pady=2
    )
    ttk.Combobox(
        methods_grid, textvariable=gui.upsampling_var,
        values=["Bilinear", "Bicubic", "Nearest"],
        state="readonly", width=12
    ).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)

    # Downsampling
    ttk.Label(methods_grid, text="Downsampling:").grid(
        row=0, column=2, sticky=tk.W, padx=(20, 5), pady=2
    )
    ttk.Combobox(
        methods_grid, textvariable=gui.downsampling_var,
        values=["Mean", "Median", "Min", "Max", "First", "Last"],
        state="readonly", width=12
    ).grid(row=0, column=3, sticky=tk.W, padx=5, pady=2)

    # Flag downsampling
    ttk.Label(methods_grid, text="Flag Downsampling:").grid(
        row=1, column=0, sticky=tk.W, padx=5, pady=2
    )
    ttk.Combobox(
        methods_grid, textvariable=gui.flag_downsampling_var,
        values=["First", "FlagAnd", "FlagOr", "FlagMedianAnd", "FlagMedianOr"],
        state="readonly", width=12
    ).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

    # Pyramid levels
    ttk.Checkbutton(
        methods_grid, text="Resample on pyramid levels",
        variable=gui.pyramid_var
    ).grid(row=1, column=2, columnspan=2, sticky=tk.W, padx=(20, 5), pady=2)


# ===== SUBSET =====

def _create_subset_section(parent, gui):
    """Create spatial subset configuration section."""
    subset_frame = ttk.LabelFrame(parent, text="Spatial Subset", padding="10")
    subset_frame.pack(fill=tk.X, padx=10, pady=5)

    # Method selection (horizontal radio buttons)
    method_row = ttk.Frame(subset_frame)
    method_row.pack(fill=tk.X, pady=2)

    ttk.Label(method_row, text="Method:").pack(side=tk.LEFT, padx=(0, 10))

    for value, label in [("none", "Full Scene"), ("geometry", "Geometry"), ("pixel", "Pixel Coords")]:
        ttk.Radiobutton(
            method_row, text=label,
            variable=gui.subset_method_var, value=value,
            command=lambda: _update_subset_visibility(gui)
        ).pack(side=tk.LEFT, padx=5)

    # --- Geometry subset frame ---
    gui.geometry_frame = ttk.Frame(subset_frame)

    ttk.Label(gui.geometry_frame, text="WKT Geometry:").pack(anchor=tk.W, pady=2)

    # Geometry text area
    geometry_text_frame = ttk.Frame(gui.geometry_frame)
    geometry_text_frame.pack(fill=tk.X, pady=2)

    gui.geometry_text = tk.Text(
        geometry_text_frame, height=3, wrap=tk.WORD, font=("Consolas", 9)
    )
    geometry_scrollbar = ttk.Scrollbar(
        geometry_text_frame, orient=tk.VERTICAL,
        command=gui.geometry_text.yview
    )
    gui.geometry_text.configure(yscrollcommand=geometry_scrollbar.set)

    gui.geometry_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    geometry_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Geometry buttons
    geometry_btn_frame = ttk.Frame(gui.geometry_frame)
    geometry_btn_frame.pack(fill=tk.X, pady=5)

    ttk.Button(
        geometry_btn_frame, text="Load Geometry...",
        command=lambda: _load_geometry(gui)
    ).pack(side=tk.LEFT, padx=(0, 5))

    ttk.Button(
        geometry_btn_frame, text="Clear",
        command=lambda: gui.geometry_text.delete(1.0, tk.END)
    ).pack(side=tk.LEFT, padx=5)

    ttk.Button(
        geometry_btn_frame, text="Validate",
        command=lambda: _validate_geometry(gui)
    ).pack(side=tk.LEFT, padx=5)

    # --- Pixel subset frame ---
    gui.pixel_frame = ttk.Frame(subset_frame)

    pixel_grid = ttk.Frame(gui.pixel_frame)
    pixel_grid.pack(pady=5)

    for label, var_name, row, col in [
        ("Start X:", "pixel_start_x_var", 0, 0),
        ("Start Y:", "pixel_start_y_var", 0, 2),
        ("Width:", "pixel_width_var", 1, 0),
        ("Height:", "pixel_height_var", 1, 2),
    ]:
        ttk.Label(pixel_grid, text=label).grid(
            row=row, column=col, sticky=tk.W, padx=5, pady=2
        )
        ttk.Entry(pixel_grid, textvariable=getattr(gui, var_name), width=10).grid(
            row=row, column=col + 1, padx=5, pady=2
        )


# ===== MAP PREVIEW =====

def _create_map_preview_section(parent, gui):
    """Create map preview section with tkintermapview or text fallback."""
    map_frame = ttk.LabelFrame(parent, text="Subset Preview", padding="5")
    map_frame.pack(fill=tk.BOTH, padx=10, pady=5, expand=True)

    if HAS_MAP_WIDGET:
        # Interactive map widget
        gui.map_widget = TkinterMapView(map_frame, width=600, height=300, corner_radius=0)
        gui.map_widget.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # Default view (Portugal coast)
        gui.map_widget.set_position(38.7, -9.1)
        gui.map_widget.set_zoom(8)

        # Store polygon reference for updates
        gui._map_polygon = None

        # Bind geometry text changes to map updates
        gui.geometry_text.bind('<KeyRelease>', lambda e: _update_map_preview(gui))

        # Map controls
        ctrl_frame = ttk.Frame(map_frame)
        ctrl_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Button(
            ctrl_frame, text="Zoom to Geometry",
            command=lambda: _zoom_to_geometry(gui)
        ).pack(side=tk.LEFT, padx=2)

        gui.map_coords_label = ttk.Label(
            ctrl_frame, text="", font=("Consolas", 8), foreground="gray"
        )
        gui.map_coords_label.pack(side=tk.RIGHT, padx=5)
    else:
        # Fallback: text bounds display
        ttk.Label(
            map_frame,
            text="Map preview requires tkintermapview.\n"
                 "Install with: pip install tkintermapview",
            font=("Arial", 9), foreground="gray",
            justify=tk.CENTER
        ).pack(expand=True, pady=20)

        gui.bounds_label = ttk.Label(
            map_frame, text="Load a geometry to see bounds here.",
            font=("Consolas", 9), foreground="gray"
        )
        gui.bounds_label.pack(pady=5)

        gui.map_widget = None
        gui._map_polygon = None


# ===== HELPER FUNCTIONS =====

def _update_subset_visibility(gui):
    """Update subset frame visibility based on selected method."""
    method = gui.subset_method_var.get()

    if method == "geometry":
        gui.geometry_frame.pack(fill=tk.X, pady=(5, 0))
        gui.pixel_frame.pack_forget()
    elif method == "pixel":
        gui.pixel_frame.pack(fill=tk.X, pady=(5, 0))
        gui.geometry_frame.pack_forget()
    else:
        gui.geometry_frame.pack_forget()
        gui.pixel_frame.pack_forget()


def _load_geometry(gui):
    """Load geometry from file and update map preview."""
    from ...utils.geometry_utils import load_geometry

    filetypes = [
        ("All supported", "*.shp *.kml *.geojson *.json"),
        ("Shapefile", "*.shp"),
        ("KML", "*.kml"),
        ("GeoJSON", "*.geojson *.json"),
        ("All files", "*.*"),
    ]

    filepath = filedialog.askopenfilename(
        title="Select Geometry File",
        filetypes=filetypes,
        parent=gui.root
    )

    if not filepath:
        return

    wkt, info, success = load_geometry(filepath)

    if success:
        gui.geometry_text.delete(1.0, tk.END)
        gui.geometry_text.insert(1.0, wkt)
        gui.subset_method_var.set("geometry")
        _update_subset_visibility(gui)
        _update_map_preview(gui)
        _zoom_to_geometry(gui)
        gui.status_var.set("Geometry loaded")
        logger.info(info)
    else:
        messagebox.showerror("Error", f"Failed to load geometry:\n{info}", parent=gui.root)


def _validate_geometry(gui):
    """Validate WKT geometry."""
    from ...utils.geometry_utils import validate_wkt

    wkt_text = gui.geometry_text.get(1.0, tk.END).strip()

    if not wkt_text:
        messagebox.showwarning("Warning", "No geometry to validate", parent=gui.root)
        return

    is_valid, message = validate_wkt(wkt_text)

    if is_valid:
        messagebox.showinfo("Validation", f"Geometry is valid\n{message}", parent=gui.root)
        gui.status_var.set("Geometry validated")
        _update_map_preview(gui)
    else:
        messagebox.showerror("Validation Error", message, parent=gui.root)


def _update_map_preview(gui):
    """Update the map preview with current geometry."""
    if gui.map_widget is None:
        _update_bounds_label(gui)
        return

    wkt_text = gui.geometry_text.get(1.0, tk.END).strip()
    if not wkt_text:
        # Clear polygon
        if gui._map_polygon is not None:
            gui._map_polygon.delete()
            gui._map_polygon = None
        return

    try:
        coords = _parse_wkt_coords(wkt_text)
        if not coords:
            return

        # Remove old polygon
        if gui._map_polygon is not None:
            gui._map_polygon.delete()
            gui._map_polygon = None

        # Draw new polygon (tkintermapview uses (lat, lon) tuples)
        gui._map_polygon = gui.map_widget.set_polygon(
            coords,
            fill_color="#93b5f5",
            outline_color="#2563eb",
            border_width=2
        )

        # Update coordinates label
        lats = [c[0] for c in coords]
        lons = [c[1] for c in coords]
        if hasattr(gui, 'map_coords_label'):
            gui.map_coords_label.config(
                text=f"Bounds: {min(lats):.4f}N - {max(lats):.4f}N, "
                     f"{min(lons):.4f}E - {max(lons):.4f}E"
            )

    except Exception as e:
        logger.warning(f"Map preview update error: {e}")


def _zoom_to_geometry(gui):
    """Zoom map to fit the current geometry."""
    if gui.map_widget is None:
        return

    wkt_text = gui.geometry_text.get(1.0, tk.END).strip()
    if not wkt_text:
        return

    try:
        coords = _parse_wkt_coords(wkt_text)
        if not coords:
            return

        lats = [c[0] for c in coords]
        lons = [c[1] for c in coords]

        gui.map_widget.fit_bounding_box(
            (max(lats), min(lons)),
            (min(lats), max(lons))
        )
    except Exception as e:
        logger.debug(f"Zoom to geometry error: {e}")


def _update_bounds_label(gui):
    """Update text bounds label (fallback when no map widget)."""
    if not hasattr(gui, 'bounds_label'):
        return

    wkt_text = gui.geometry_text.get(1.0, tk.END).strip()
    if not wkt_text:
        gui.bounds_label.config(text="Load a geometry to see bounds here.")
        return

    try:
        coords = _parse_wkt_coords(wkt_text)
        if coords:
            lats = [c[0] for c in coords]
            lons = [c[1] for c in coords]
            gui.bounds_label.config(
                text=f"Bounds: {min(lats):.4f}N - {max(lats):.4f}N, "
                     f"{min(lons):.4f}E - {max(lons):.4f}E"
            )
    except Exception:
        gui.bounds_label.config(text="Could not parse geometry bounds.")


def _parse_wkt_coords(wkt_text):
    """
    Parse WKT POLYGON coordinates into list of (lat, lon) tuples.

    Returns list of (lat, lon) tuples, or empty list on failure.
    """
    import re

    # Extract coordinates from POLYGON((lon lat, lon lat, ...))
    match = re.search(r'POLYGON\s*\(\((.+?)\)\)', wkt_text, re.IGNORECASE | re.DOTALL)
    if not match:
        # Try MULTIPOLYGON
        match = re.search(r'MULTIPOLYGON\s*\(\(\((.+?)\)\)\)', wkt_text, re.IGNORECASE | re.DOTALL)

    if not match:
        return []

    coord_string = match.group(1)
    coords = []
    for pair in coord_string.split(','):
        pair = pair.strip()
        parts = pair.split()
        if len(parts) >= 2:
            try:
                lon = float(parts[0])
                lat = float(parts[1])
                coords.append((lat, lon))  # tkintermapview uses (lat, lon)
            except ValueError:
                continue

    return coords
