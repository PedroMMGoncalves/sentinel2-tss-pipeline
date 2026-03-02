"""
Event Handlers for Sentinel-2 TSS Pipeline GUI.

Provides event handling, validation, browsing, and preset functions.
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox
import logging

from ..utils.product_detector import ProductDetector
from ..config.enums import ProcessingMode

logger = logging.getLogger('ocean_rs')


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
                        pass  # Expected when tab is already hidden/shown
        else:
            # Hide S2 processing tabs for TSS-only mode
            for tab_name in ['spatial', 'c2rcc']:
                if tab_name in gui.tab_indices:
                    try:
                        gui.notebook.tab(gui.tab_indices[tab_name], state="hidden")
                    except tk.TclError:
                        pass  # Expected when tab is already hidden/shown

        # Outputs tab visibility
        if mode in ["complete_pipeline", "tss_processing_only"]:
            if 'outputs' in gui.tab_indices:
                try:
                    gui.notebook.tab(gui.tab_indices['outputs'], state="normal")
                except tk.TclError:
                    pass  # Expected when tab is already hidden/shown
        else:
            if 'outputs' in gui.tab_indices:
                try:
                    gui.notebook.tab(gui.tab_indices['outputs'], state="hidden")
                except tk.TclError:
                    pass  # Expected when tab is already hidden/shown

    except Exception as e:
        logger.error(f"Error updating tab visibility: {e}")


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
        else:
            return

    # Cleanup
    try:
        if hasattr(gui, 'system_monitor'):
            gui.system_monitor.stop_monitoring()
        if hasattr(gui, 'processor') and gui.processor:
            gui.processor.cleanup()
    except Exception as e:
        logger.debug(f"Cleanup during exit: {e}")

    def _check_and_destroy():
        if hasattr(gui, 'processing_thread') and gui.processing_thread and gui.processing_thread.is_alive():
            gui.root.after(100, _check_and_destroy)
        else:
            gui.root.destroy()

    gui.root.after(100, _check_and_destroy)
