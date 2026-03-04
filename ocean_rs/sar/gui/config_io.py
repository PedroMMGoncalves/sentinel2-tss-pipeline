"""
Configuration I/O for SAR Toolkit GUI.

Saves/loads processing parameters as JSON. NEVER saves credentials.
"""

import json
import tkinter as tk
from tkinter import filedialog, messagebox
from dataclasses import asdict
import logging

logger = logging.getLogger('ocean_rs')


def update_configurations(gui):
    """Update config objects from GUI variables.

    Returns:
        bool: True if successful.
    """
    try:
        cfg = gui.config

        cfg.search_config.platform = gui.platform_var.get()
        cfg.search_config.beam_mode = gui.beam_mode_var.get()
        cfg.search_config.polarization = gui.polarization_var.get()
        cfg.search_config.orbit_direction = gui.orbit_dir_var.get()
        cfg.search_config.aoi_wkt = gui.aoi_text.get(1.0, tk.END).strip() if hasattr(gui, 'aoi_text') else ""
        cfg.search_config.start_date = gui.start_date_var.get()
        cfg.search_config.end_date = gui.end_date_var.get()

        cfg.snap_gpt_path = gui.snap_gpt_var.get()

        cfg.fft_config.tile_size_m = gui.tile_size_var.get()
        cfg.fft_config.overlap = gui.overlap_var.get()
        cfg.fft_config.min_wavelength_m = gui.min_wavelength_var.get()
        cfg.fft_config.max_wavelength_m = gui.max_wavelength_var.get()
        cfg.fft_config.confidence_threshold = gui.confidence_var.get()

        cfg.depth_config.wave_period_source = gui.wave_source_var.get()
        cfg.depth_config.manual_wave_period = gui.manual_period_var.get()
        cfg.depth_config.max_depth_m = gui.max_depth_var.get()

        cfg.compositing_config.enabled = gui.compositing_var.get()
        cfg.compositing_config.method = gui.compositing_method_var.get()

        cfg.output_directory = gui.output_dir_var.get()
        cfg.processing_mode = gui.processing_mode_var.get()

        # InSAR config
        cfg.insar_config.coregistration_method = gui.insar_coreg_method_var.get()
        cfg.insar_config.coregistration_patch_size = gui.insar_coreg_patch_var.get()
        cfg.insar_config.coherence_window_range = gui.insar_coh_range_var.get()
        cfg.insar_config.coherence_window_azimuth = gui.insar_coh_azimuth_var.get()
        cfg.insar_config.phase_filter_alpha = gui.insar_filter_alpha_var.get()
        cfg.insar_config.phase_filter_patch_size = gui.insar_filter_patch_var.get()
        cfg.insar_config.unwrapping_method = gui.insar_unwrap_method_var.get()
        cfg.insar_config.remove_topography = gui.insar_remove_topo_var.get()
        cfg.insar_config.dem_path = gui.insar_dem_path_var.get()
        cfg.insar_config.output_coherence = gui.insar_output_coh_var.get()
        cfg.insar_config.output_interferogram = gui.insar_output_ifg_var.get()
        cfg.insar_config.output_unwrapped = gui.insar_output_unwrap_var.get()

        # Displacement config
        cfg.displacement_config.mode = gui.disp_mode_var.get()
        cfg.displacement_config.max_temporal_baseline_days = gui.disp_max_temporal_var.get()
        cfg.displacement_config.max_perpendicular_baseline_m = gui.disp_max_perp_var.get()
        cfg.displacement_config.atmospheric_filter = gui.disp_atm_filter_var.get()
        cfg.displacement_config.temporal_coherence_threshold = gui.disp_temp_coh_var.get()
        cfg.displacement_config.output_quasi_vertical = gui.disp_output_vertical_var.get()
        cfg.displacement_config.output_los = gui.disp_output_los_var.get()
        if gui.disp_use_ref_point_var.get():
            cfg.displacement_config.reference_point = (
                gui.disp_ref_lon_var.get(),
                gui.disp_ref_lat_var.get(),
            )
        else:
            cfg.displacement_config.reference_point = None

        return True
    except Exception as e:
        logger.error(f"Failed to update config: {e}")
        messagebox.showerror("Config Error", str(e), parent=gui.root)
        return False


def save_config(gui):
    """Save config to JSON file."""
    filename = filedialog.asksaveasfilename(
        defaultextension=".json",
        filetypes=[("JSON", "*.json")],
        title="Save Configuration"
    )
    if not filename:
        return

    if not update_configurations(gui):
        return

    config_dict = {
        "version": "1.1",
        "processing_mode": gui.config.processing_mode,
        # Note: SearchConfig contains no credential fields — safe to serialize fully
        "search": asdict(gui.config.search_config),
        "processing": {
            "snap_gpt_path": gui.config.snap_gpt_path,
            "fft": asdict(gui.config.fft_config),
            "depth": asdict(gui.config.depth_config),
            "compositing": asdict(gui.config.compositing_config),
        },
        "insar": asdict(gui.config.insar_config),
        "displacement": asdict(gui.config.displacement_config),
        "output": {
            "output_directory": gui.config.output_directory,
            "export_geotiff": gui.config.export_geotiff,
            "export_png": gui.config.export_png,
        }
    }

    try:
        with open(filename, 'w') as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Config saved to: {filename}")
        messagebox.showinfo("Saved", f"Configuration saved to:\n{filename}",
                           parent=gui.root)
    except Exception as e:
        messagebox.showerror("Save Error", str(e), parent=gui.root)


def load_config(gui):
    """Load config from JSON file."""
    filename = filedialog.askopenfilename(
        filetypes=[("JSON", "*.json")],
        title="Load Configuration"
    )
    if not filename:
        return

    try:
        with open(filename, 'r') as f:
            data = json.load(f)

        version = data.get("version", "unknown")
        if version not in ("1.0", "1.1"):
            messagebox.showwarning(
                "Config Version",
                f"Config file version '{version}' may not be fully compatible "
                f"with current SAR toolkit (expected 1.0 or 1.1).",
                parent=gui.root
            )

        search = data.get("search", {})
        gui.platform_var.set(search.get("platform", "Sentinel-1"))
        gui.beam_mode_var.set(search.get("beam_mode", "IW"))
        gui.polarization_var.set(search.get("polarization", "VV+VH"))
        gui.orbit_dir_var.set(search.get("orbit_direction", ""))
        gui.start_date_var.set(search.get("start_date", ""))
        gui.end_date_var.set(search.get("end_date", ""))
        aoi = search.get("aoi_wkt", "")
        if aoi and hasattr(gui, 'aoi_text'):
            gui.aoi_text.delete(1.0, tk.END)
            gui.aoi_text.insert(1.0, aoi)

        proc = data.get("processing", {})
        gui.snap_gpt_var.set(proc.get("snap_gpt_path", ""))

        fft = proc.get("fft", {})
        gui.tile_size_var.set(fft.get("tile_size_m", 512.0))
        gui.overlap_var.set(fft.get("overlap", 0.5))
        gui.min_wavelength_var.set(fft.get("min_wavelength_m", 50.0))
        gui.max_wavelength_var.set(fft.get("max_wavelength_m", 600.0))
        gui.confidence_var.set(fft.get("confidence_threshold", 0.3))

        depth = proc.get("depth", {})
        gui.wave_source_var.set(depth.get("wave_period_source", "wavewatch3"))
        gui.manual_period_var.set(depth.get("manual_wave_period", 10.0))
        gui.max_depth_var.set(depth.get("max_depth_m", 100.0))

        comp = proc.get("compositing", {})
        gui.compositing_var.set(comp.get("enabled", True))
        gui.compositing_method_var.set(comp.get("method", "weighted_median"))

        output = data.get("output", {})
        gui.output_dir_var.set(output.get("output_directory", ""))

        gui.processing_mode_var.set(data.get("processing_mode", "bathymetry"))

        # InSAR config
        insar = data.get("insar", {})
        gui.insar_coreg_method_var.set(insar.get("coregistration_method", "auto"))
        gui.insar_coreg_patch_var.set(insar.get("coregistration_patch_size", 128))
        gui.insar_coh_range_var.set(insar.get("coherence_window_range", 15))
        gui.insar_coh_azimuth_var.set(insar.get("coherence_window_azimuth", 3))
        gui.insar_filter_alpha_var.set(insar.get("phase_filter_alpha", 0.5))
        gui.insar_filter_patch_var.set(insar.get("phase_filter_patch_size", 32))
        gui.insar_unwrap_method_var.set(insar.get("unwrapping_method", "auto"))
        gui.insar_remove_topo_var.set(insar.get("remove_topography", True))
        gui.insar_dem_path_var.set(insar.get("dem_path", ""))
        gui.insar_output_coh_var.set(insar.get("output_coherence", True))
        gui.insar_output_ifg_var.set(insar.get("output_interferogram", True))
        gui.insar_output_unwrap_var.set(insar.get("output_unwrapped", True))

        # Displacement config
        disp = data.get("displacement", {})
        gui.disp_mode_var.set(disp.get("mode", "dinsar"))
        gui.disp_max_temporal_var.set(disp.get("max_temporal_baseline_days", 180))
        gui.disp_max_perp_var.set(disp.get("max_perpendicular_baseline_m", 200.0))
        gui.disp_atm_filter_var.set(disp.get("atmospheric_filter", False))
        gui.disp_temp_coh_var.set(disp.get("temporal_coherence_threshold", 0.7))
        gui.disp_output_vertical_var.set(disp.get("output_quasi_vertical", True))
        gui.disp_output_los_var.set(disp.get("output_los", True))
        ref = disp.get("reference_point")
        if ref and isinstance(ref, (list, tuple)) and len(ref) == 2:
            gui.disp_use_ref_point_var.set(True)
            gui.disp_ref_lon_var.set(ref[0])
            gui.disp_ref_lat_var.set(ref[1])
        else:
            gui.disp_use_ref_point_var.set(False)

        logger.info(f"Config loaded from: {filename}")
        messagebox.showinfo("Loaded", "Configuration loaded successfully.",
                           parent=gui.root)
    except Exception as e:
        messagebox.showerror("Load Error", str(e), parent=gui.root)
