"""
Subset Tab for Sentinel-2 TSS Pipeline GUI.

Provides the spatial subset configuration interface.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import logging

logger = logging.getLogger('sentinel2_tss_pipeline')


def create_subset_tab(gui, notebook):
    """
    Create the Spatial Subset configuration tab.

    Args:
        gui: Parent GUI instance with all tk variables.
        notebook: ttk.Notebook to add the tab to.

    Returns:
        int: Tab index.
    """
    frame = ttk.Frame(notebook)
    tab_index = notebook.add(frame, text="Spatial Subset")

    # Title
    ttk.Label(
        frame, text="Spatial Subset Configuration",
        font=("Arial", 14, "bold")
    ).pack(pady=10)

    # Subset method selection
    method_frame = ttk.LabelFrame(frame, text="Subset Method", padding="10")
    method_frame.pack(fill=tk.X, padx=10, pady=5)

    subset_options = [
        ("none", "No spatial subset (process full scene)",
         "Process entire Sentinel-2 tile"),
        ("geometry", "Use geometry (WKT/Shapefile/KML)",
         "Define area using spatial geometry"),
        ("pixel", "Use pixel coordinates",
         "Define rectangular area using pixel coordinates")
    ]

    for value, text, description in subset_options:
        radio_frame = ttk.Frame(method_frame)
        radio_frame.pack(fill=tk.X, pady=2)

        ttk.Radiobutton(
            radio_frame, text=text,
            variable=gui.subset_method_var, value=value,
            command=lambda g=gui: _update_subset_visibility(g)
        ).pack(anchor=tk.W)

        ttk.Label(
            radio_frame, text=description,
            font=("Arial", 8), foreground="gray"
        ).pack(anchor=tk.W, padx=(20, 0))

    # Geometry subset frame
    gui.geometry_frame = ttk.LabelFrame(
        frame, text="Geometry Subset", padding="10"
    )
    gui.geometry_frame.pack(fill=tk.X, padx=10, pady=5)

    ttk.Label(gui.geometry_frame, text="WKT Geometry:").pack(anchor=tk.W, pady=2)

    # Geometry text area with scrollbar
    geometry_text_frame = ttk.Frame(gui.geometry_frame)
    geometry_text_frame.pack(fill=tk.X, pady=2)

    gui.geometry_text = tk.Text(
        geometry_text_frame, height=4, wrap=tk.WORD, font=("Consolas", 9)
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
        command=lambda g=gui: _load_geometry(g)
    ).pack(side=tk.LEFT, padx=(0, 5))

    ttk.Button(
        geometry_btn_frame, text="Clear",
        command=lambda g=gui: g.geometry_text.delete(1.0, tk.END)
    ).pack(side=tk.LEFT, padx=5)

    ttk.Button(
        geometry_btn_frame, text="Validate",
        command=lambda g=gui: _validate_geometry(g)
    ).pack(side=tk.LEFT, padx=5)

    # Pixel subset frame
    gui.pixel_frame = ttk.LabelFrame(frame, text="Pixel Subset", padding="10")
    gui.pixel_frame.pack(fill=tk.X, padx=10, pady=5)

    pixel_grid = ttk.Frame(gui.pixel_frame)
    pixel_grid.pack(pady=5)

    # Pixel coordinate entries
    coordinates = [
        ("Start X:", "pixel_start_x_var", 0, 0),
        ("Start Y:", "pixel_start_y_var", 0, 2),
        ("Width:", "pixel_width_var", 1, 0),
        ("Height:", "pixel_height_var", 1, 2)
    ]

    for label, var_name, row, col in coordinates:
        ttk.Label(pixel_grid, text=label).grid(
            row=row, column=col, sticky=tk.W, padx=5, pady=2
        )
        var = getattr(gui, var_name)
        ttk.Entry(pixel_grid, textvariable=var, width=10).grid(
            row=row, column=col+1, padx=5, pady=2
        )

    # Set initial visibility
    _update_subset_visibility(gui)

    return tab_index


def _update_subset_visibility(gui):
    """Update subset frame visibility based on selected method."""
    method = gui.subset_method_var.get()

    if method == "geometry":
        gui.geometry_frame.pack(fill=tk.X, padx=10, pady=5)
        gui.pixel_frame.pack_forget()
    elif method == "pixel":
        gui.pixel_frame.pack(fill=tk.X, padx=10, pady=5)
        gui.geometry_frame.pack_forget()
    else:
        gui.geometry_frame.pack_forget()
        gui.pixel_frame.pack_forget()


def _load_geometry(gui):
    """Load geometry from file."""
    from ...utils.geometry_utils import load_geometry

    filetypes = [
        ("All supported", "*.shp *.kml *.geojson *.json"),
        ("Shapefile", "*.shp"),
        ("KML", "*.kml"),
        ("GeoJSON", "*.geojson *.json"),
        ("All files", "*.*")
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
        messagebox.showinfo("Success", "Geometry loaded successfully!", parent=gui.root)
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
    else:
        messagebox.showerror("Validation Error", message, parent=gui.root)
