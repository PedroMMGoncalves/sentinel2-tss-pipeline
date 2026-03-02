"""
Search & Select Tab for SAR Bathymetry Toolkit GUI.

AOI input, date range, sensor filters, search button, results Treeview table.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import logging

logger = logging.getLogger('ocean_rs')


def create_search_tab(gui, notebook):
    """Create the Search & Select tab.

    Args:
        gui: Parent GUI instance with tk variables.
        notebook: ttk.Notebook to add tab to.

    Returns:
        int: Tab index.
    """
    frame = ttk.Frame(notebook)
    tab_index = notebook.add(frame, text="Search & Select")

    # --- AOI Section ---
    aoi_frame = ttk.LabelFrame(frame, text="Area of Interest", padding="10")
    aoi_frame.pack(fill=tk.X, padx=10, pady=(10, 5))

    ttk.Label(aoi_frame, text="WKT Polygon:").pack(anchor=tk.W)
    aoi_text_frame = ttk.Frame(aoi_frame)
    aoi_text_frame.pack(fill=tk.X, pady=2)
    gui.aoi_text = tk.Text(aoi_text_frame, height=3, width=60, font=('Consolas', 10))
    gui.aoi_text.pack(side=tk.LEFT, fill=tk.X, expand=True)

    aoi_btn_frame = ttk.Frame(aoi_frame)
    aoi_btn_frame.pack(fill=tk.X, pady=2)
    ttk.Button(aoi_btn_frame, text="Load from File...",
               command=lambda: _load_aoi_file(gui)).pack(side=tk.LEFT, padx=2)
    ttk.Button(aoi_btn_frame, text="Clear",
               command=lambda: gui.aoi_text.delete(1.0, tk.END)).pack(side=tk.LEFT, padx=2)

    # --- Date Range ---
    date_frame = ttk.LabelFrame(frame, text="Date Range", padding="10")
    date_frame.pack(fill=tk.X, padx=10, pady=5)

    date_row = ttk.Frame(date_frame)
    date_row.pack(fill=tk.X)
    ttk.Label(date_row, text="Start (YYYY-MM-DD):").pack(side=tk.LEFT)
    ttk.Entry(date_row, textvariable=gui.start_date_var, width=15).pack(side=tk.LEFT, padx=5)
    ttk.Label(date_row, text="End (YYYY-MM-DD):").pack(side=tk.LEFT, padx=(20, 0))
    ttk.Entry(date_row, textvariable=gui.end_date_var, width=15).pack(side=tk.LEFT, padx=5)

    # --- Sensor Filters ---
    filter_frame = ttk.LabelFrame(frame, text="Sensor Filters", padding="10")
    filter_frame.pack(fill=tk.X, padx=10, pady=5)

    filter_row1 = ttk.Frame(filter_frame)
    filter_row1.pack(fill=tk.X, pady=2)
    ttk.Label(filter_row1, text="Platform:").pack(side=tk.LEFT)
    ttk.Combobox(filter_row1, textvariable=gui.platform_var, width=15,
                 values=["Sentinel-1"], state="readonly").pack(side=tk.LEFT, padx=5)
    ttk.Label(filter_row1, text="Beam Mode:").pack(side=tk.LEFT, padx=(20, 0))
    ttk.Combobox(filter_row1, textvariable=gui.beam_mode_var, width=10,
                 values=["IW", "EW", "SM"], state="readonly").pack(side=tk.LEFT, padx=5)

    filter_row2 = ttk.Frame(filter_frame)
    filter_row2.pack(fill=tk.X, pady=2)
    ttk.Label(filter_row2, text="Polarization:").pack(side=tk.LEFT)
    ttk.Combobox(filter_row2, textvariable=gui.polarization_var, width=10,
                 values=["VV+VH", "VV", "HH+HV", "HH"], state="readonly").pack(side=tk.LEFT, padx=5)
    ttk.Label(filter_row2, text="Orbit:").pack(side=tk.LEFT, padx=(20, 0))
    ttk.Combobox(filter_row2, textvariable=gui.orbit_dir_var, width=15,
                 values=["", "ASCENDING", "DESCENDING"]).pack(side=tk.LEFT, padx=5)

    # --- Search Button ---
    search_btn_frame = ttk.Frame(frame)
    search_btn_frame.pack(fill=tk.X, padx=10, pady=5)
    gui.search_button = ttk.Button(search_btn_frame, text="Search ASF",
                                    style='Primary.TButton',
                                    command=lambda: _do_search(gui))
    gui.search_button.pack(side=tk.LEFT)
    gui.search_status = ttk.Label(search_btn_frame, text="", style='Status.TLabel')
    gui.search_status.pack(side=tk.LEFT, padx=10)

    # --- Results Table ---
    results_frame = ttk.LabelFrame(frame, text="Search Results", padding="5")
    results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    columns = ("scene_id", "date", "orbit", "pol", "pass_dir", "size_mb")
    gui.results_tree = ttk.Treeview(results_frame, columns=columns,
                                     show="headings", selectmode="extended",
                                     height=10)
    gui.results_tree.heading("scene_id", text="Scene ID")
    gui.results_tree.heading("date", text="Date")
    gui.results_tree.heading("orbit", text="Orbit")
    gui.results_tree.heading("pol", text="Pol")
    gui.results_tree.heading("pass_dir", text="Pass")
    gui.results_tree.heading("size_mb", text="Size (MB)")

    gui.results_tree.column("scene_id", width=300)
    gui.results_tree.column("date", width=120)
    gui.results_tree.column("orbit", width=60)
    gui.results_tree.column("pol", width=70)
    gui.results_tree.column("pass_dir", width=90)
    gui.results_tree.column("size_mb", width=80)

    scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL,
                               command=gui.results_tree.yview)
    gui.results_tree.configure(yscrollcommand=scrollbar.set)
    gui.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # --- Selection Controls ---
    sel_frame = ttk.Frame(frame)
    sel_frame.pack(fill=tk.X, padx=10, pady=5)
    ttk.Button(sel_frame, text="Select All",
               command=lambda: _select_all(gui)).pack(side=tk.LEFT, padx=2)
    ttk.Button(sel_frame, text="Deselect All",
               command=lambda: _deselect_all(gui)).pack(side=tk.LEFT, padx=2)
    gui.scene_count_label = ttk.Label(sel_frame, text="0 scenes selected (0 MB)",
                                       style='Status.TLabel')
    gui.scene_count_label.pack(side=tk.RIGHT)

    # Bind selection change event
    gui.results_tree.bind("<<TreeviewSelect>>", lambda e: _update_selection_count(gui))

    return tab_index


def _load_aoi_file(gui):
    """Load AOI from shapefile, GeoJSON, or KML."""
    filepath = filedialog.askopenfilename(
        title="Select AOI File",
        filetypes=[
            ("Shapefiles", "*.shp"),
            ("GeoJSON", "*.geojson *.json"),
            ("KML", "*.kml"),
            ("All files", "*.*"),
        ]
    )
    if filepath:
        try:
            from ocean_rs.shared import load_geometry
            wkt, info, success = load_geometry(filepath)
            if success and wkt:
                gui.aoi_text.delete(1.0, tk.END)
                gui.aoi_text.insert(1.0, wkt)
                logger.info(f"Loaded AOI: {info}")
            else:
                messagebox.showerror("Error", f"Failed to load geometry: {info}",
                                    parent=gui.root)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load geometry: {e}",
                               parent=gui.root)


def _do_search(gui):
    """Execute ASF search."""
    aoi = gui.aoi_text.get(1.0, tk.END).strip()
    start = gui.start_date_var.get().strip()
    end = gui.end_date_var.get().strip()

    if not aoi:
        messagebox.showerror("Error", "Please enter an AOI (WKT polygon).",
                            parent=gui.root)
        return
    if not start or not end:
        messagebox.showerror("Error", "Please enter start and end dates.",
                            parent=gui.root)
        return

    gui.search_button.config(state=tk.DISABLED)
    gui.search_status.config(text="Searching...")
    gui.root.update_idletasks()

    try:
        from ocean_rs.sar.download import search_scenes
        scenes = search_scenes(
            aoi_wkt=aoi,
            start_date=start,
            end_date=end,
            platform=gui.platform_var.get(),
            beam_mode=gui.beam_mode_var.get(),
            orbit_direction=gui.orbit_dir_var.get() or None,
        )

        gui.search_results = scenes

        # Clear existing results
        for item in gui.results_tree.get_children():
            gui.results_tree.delete(item)

        # Populate table
        for scene in scenes:
            gui.results_tree.insert("", tk.END, values=(
                scene.granule_id,
                scene.acquisition_date[:10] if scene.acquisition_date else "",
                scene.path_number,
                scene.polarization,
                scene.orbit_direction,
                f"{scene.size_mb:.0f}",
            ))

        gui.search_status.config(text=f"Found {len(scenes)} scenes")
        # Select all by default
        _select_all(gui)

    except Exception as e:
        messagebox.showerror("Search Error", str(e), parent=gui.root)
        gui.search_status.config(text="Search failed")
    finally:
        gui.search_button.config(state=tk.NORMAL)


def _select_all(gui):
    """Select all items in results tree."""
    for item in gui.results_tree.get_children():
        gui.results_tree.selection_add(item)
    _update_selection_count(gui)


def _deselect_all(gui):
    """Deselect all items in results tree."""
    children = gui.results_tree.get_children()
    if children:
        gui.results_tree.selection_remove(*children)
    _update_selection_count(gui)


def _update_selection_count(gui):
    """Update scene count label."""
    selected = gui.results_tree.selection()
    n = len(selected)
    total_mb = 0.0
    for item in selected:
        idx = gui.results_tree.index(item)
        if idx < len(gui.search_results):
            total_mb += gui.search_results[idx].size_mb
    gui.scene_count_label.config(text=f"{n} scenes selected ({total_mb:.0f} MB)")
