"""
Configuration I/O for SAR Bathymetry Toolkit GUI.

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
        "version": "1.0",
        "search": asdict(gui.config.search_config),
        "processing": {
            "snap_gpt_path": gui.config.snap_gpt_path,
            "fft": asdict(gui.config.fft_config),
            "depth": asdict(gui.config.depth_config),
            "compositing": asdict(gui.config.compositing_config),
        },
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

        logger.info(f"Config loaded from: {filename}")
        messagebox.showinfo("Loaded", "Configuration loaded successfully.",
                           parent=gui.root)
    except Exception as e:
        messagebox.showerror("Load Error", str(e), parent=gui.root)
