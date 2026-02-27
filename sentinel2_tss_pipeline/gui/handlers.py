"""
Event Handlers for Sentinel-2 TSS Pipeline GUI.

Provides event handling, validation, browsing, and preset functions.
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox
import logging

from ..utils.logging_utils import setup_enhanced_logging
from ..utils.product_detector import ProductDetector
from ..config.enums import ProcessingMode

logger = logging.getLogger('sentinel2_tss_pipeline')


# ===== MODE AND VISIBILITY HANDLERS =====

def on_mode_change(gui):
    """Handle processing mode change."""
    update_tab_visibility(gui)
    validate_input_directory(gui)
    gui.status_var.set(f"Mode changed to: {gui.processing_mode.get()}")


def update_tab_visibility(gui):
    """Update tab visibility based on processing mode."""
    try:
        mode = gui.processing_mode.get()

        # Show/hide tabs based on mode
        if mode in ["complete_pipeline", "s2_processing_only"]:
            # Show S2 processing tabs
            for tab_name in ['spatial', 'c2rcc']:
                if tab_name in gui.tab_indices:
                    try:
                        gui.notebook.tab(gui.tab_indices[tab_name], state="normal")
                    except tk.TclError:
                        pass
        else:
            # Hide S2 processing tabs for TSS-only mode
            for tab_name in ['spatial', 'c2rcc']:
                if tab_name in gui.tab_indices:
                    try:
                        gui.notebook.tab(gui.tab_indices[tab_name], state="hidden")
                    except tk.TclError:
                        pass

        # Outputs tab visibility
        if mode in ["complete_pipeline", "tss_processing_only"]:
            if 'outputs' in gui.tab_indices:
                try:
                    gui.notebook.tab(gui.tab_indices['outputs'], state="normal")
                except tk.TclError:
                    pass
        else:
            if 'outputs' in gui.tab_indices:
                try:
                    gui.notebook.tab(gui.tab_indices['outputs'], state="hidden")
                except tk.TclError:
                    pass

    except Exception as e:
        logger.error(f"Error updating tab visibility: {e}")


def update_subset_visibility(gui):
    """Update subset frame visibility."""
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


def update_jiang_visibility(gui):
    """Update Jiang options visibility."""
    if gui.enable_jiang_var.get():
        gui.jiang_options_frame.pack(fill=tk.X, pady=(10, 0))
    else:
        gui.jiang_options_frame.pack_forget()


# ===== TOGGLE HANDLERS =====

def on_ecmwf_toggle(gui):
    """Handle ECMWF toggle with confirmation dialog."""
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


def on_rhow_toggle(gui):
    """Handle rhow toggle with warning dialog."""
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


# ===== VALIDATION HANDLERS =====

def validate_input_directory(gui, *args):
    """Validate input directory and update display."""
    input_dir = gui.input_dir_var.get()
    if not input_dir or not os.path.exists(input_dir):
        gui.input_validation_result = {
            "valid": False,
            "message": "Please select a valid input directory",
            "products": []
        }
        gui.input_validation_label.config(
            text=gui.input_validation_result["message"],
            foreground="red"
        )
        return

    # Scan directory for products
    products = ProductDetector.scan_input_folder(input_dir)
    mode = ProcessingMode(gui.processing_mode.get())

    # Validate products for current mode
    valid, message, product_list = ProductDetector.validate_processing_mode(products, mode)

    gui.input_validation_result = {
        "valid": valid,
        "message": message,
        "products": product_list
    }

    # Update display
    color = "darkgreen" if valid else "red"
    gui.input_validation_label.config(text=message, foreground=color)

    # Update status
    if valid:
        gui.status_var.set(f"Ready: {len(product_list)} products found")
    else:
        gui.status_var.set("Input validation failed")


def validate_geometry(gui):
    """Validate WKT geometry."""
    wkt_text = gui.geometry_text.get(1.0, tk.END).strip()
    if not wkt_text:
        messagebox.showwarning("Warning", "No geometry to validate", parent=gui.root)
        return

    try:
        # Basic WKT validation
        valid_prefixes = ['POLYGON', 'POINT', 'LINESTRING', 'MULTIPOLYGON']
        if not any(wkt_text.upper().startswith(geom) for geom in valid_prefixes):
            raise ValueError("Invalid WKT format")

        messagebox.showinfo("Validation", "Geometry appears valid", parent=gui.root)
        gui.status_var.set("Geometry validated")
    except Exception as e:
        messagebox.showerror("Validation Error", f"Invalid geometry: {str(e)}", parent=gui.root)


# ===== BROWSE HANDLERS =====

def browse_input_dir(gui):
    """Browse for input directory."""
    directory = filedialog.askdirectory(
        title="Select Input Directory",
        parent=gui.root
    )
    if directory:
        gui.input_dir_var.set(directory)


def browse_output_dir(gui):
    """Browse for output directory and setup logging."""
    directory = filedialog.askdirectory(
        title="Select Output Directory",
        parent=gui.root
    )
    if directory:
        gui.output_dir_var.set(directory)

        # Setup logging to output folder
        global logger
        logger, log_file = setup_enhanced_logging(
            log_level=logging.INFO,
            output_folder=directory
        )
        logger.info(f"Output directory selected: {directory}")
        logger.info(f"Log file: {log_file}")
        gui.status_var.set(f"Output: {directory} | Log: {os.path.basename(log_file)}")


# ===== PRESET HANDLERS =====

def apply_water_preset(gui, values):
    """Apply water parameter preset."""
    gui.salinity_var.set(values["salinity"])
    gui.temperature_var.set(values["temperature"])
    gui.status_var.set("Water preset applied")


def apply_snap_defaults(gui):
    """Apply SNAP default values to all parameters."""
    # Basic parameters
    gui.salinity_var.set(35.0)
    gui.temperature_var.set(15.0)
    gui.ozone_var.set(330.0)
    gui.pressure_var.set(1000.0)
    gui.elevation_var.set(0.0)

    # Neural network and DEM
    gui.net_set_var.set("C2RCC-Nets")
    gui.dem_name_var.set("Copernicus 30m Global DEM")
    gui.use_ecmwf_var.set(True)

    # Output products (SNAP defaults)
    gui.output_rrs_var.set(True)
    gui.output_rhow_var.set(True)
    gui.output_kd_var.set(True)
    gui.output_uncertainties_var.set(True)
    gui.output_ac_reflectance_var.set(True)
    gui.output_rtoa_var.set(True)
    gui.output_rtosa_gc_var.set(False)
    gui.output_rtosa_gc_aann_var.set(False)
    gui.output_rpath_var.set(False)
    gui.output_tdown_var.set(False)
    gui.output_tup_var.set(False)
    gui.output_oos_var.set(False)

    # Advanced parameters
    gui.valid_pixel_var.set("B8 > 0 && B8 < 0.1")
    gui.threshold_rtosa_oos_var.set(0.05)
    gui.threshold_ac_reflec_oos_var.set(0.1)
    gui.threshold_cloud_tdown865_var.set(0.955)

    # TSM and CHL parameters
    gui.tsm_fac_var.set(1.06)
    gui.tsm_exp_var.set(0.942)
    gui.chl_fac_var.set(21.0)
    gui.chl_exp_var.set(1.04)

    gui.status_var.set("SNAP default values applied")


def apply_essential_outputs(gui):
    """Apply essential outputs preset."""
    reset_all_outputs(gui)

    # Enable essential outputs
    gui.output_rrs_var.set(True)
    gui.output_rhow_var.set(True)
    gui.output_kd_var.set(True)
    gui.output_uncertainties_var.set(True)
    gui.output_ac_reflectance_var.set(True)

    gui.status_var.set("Essential outputs preset applied")


def apply_scientific_outputs(gui):
    """Apply scientific outputs preset."""
    reset_all_outputs(gui)

    # Enable scientific outputs
    gui.output_rrs_var.set(True)
    gui.output_rhow_var.set(True)
    gui.output_kd_var.set(True)
    gui.output_ac_reflectance_var.set(True)
    gui.output_rtoa_var.set(True)
    gui.output_uncertainties_var.set(True)
    gui.output_oos_var.set(True)

    gui.status_var.set("Scientific outputs preset applied")


def reset_all_outputs(gui):
    """Reset all output variables to False."""
    gui.output_rhow_var.set(False)
    gui.output_kd_var.set(False)
    gui.output_ac_reflectance_var.set(False)
    gui.output_rtoa_var.set(False)
    gui.output_rrs_var.set(False)
    gui.output_rtosa_gc_var.set(False)
    gui.output_rtosa_gc_aann_var.set(False)
    gui.output_rpath_var.set(False)
    gui.output_tdown_var.set(False)
    gui.output_tup_var.set(False)
    gui.output_uncertainties_var.set(False)
    gui.output_oos_var.set(False)


# ===== WINDOW HANDLERS =====

def on_closing(gui):
    """Handle application closing."""
    if gui.processing_active:
        result = messagebox.askyesno(
            "Confirm Exit",
            "Processing is active. Stop processing and exit?",
            parent=gui.root
        )
        if result:
            gui.stop_processing()
            import time
            time.sleep(1)  # Give time to stop
        else:
            return

    # Cleanup
    try:
        if hasattr(gui, 'system_monitor'):
            gui.system_monitor.stop_monitoring()
        if hasattr(gui, 'processor') and gui.processor:
            gui.processor.cleanup()
    except Exception:
        pass

    gui.root.destroy()
