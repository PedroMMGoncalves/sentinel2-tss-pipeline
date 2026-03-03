"""
Processing Controller for SAR Bathymetry Toolkit GUI.

Manages background processing thread and GUI updates.
"""

import tkinter as tk
from tkinter import messagebox
import threading
import logging
from pathlib import Path

from .config_io import update_configurations

logger = logging.getLogger('ocean_rs')


def start_processing(gui):
    """Start bathymetry processing in background thread."""
    if gui.processing_active:
        return

    if not update_configurations(gui):
        return

    output_dir = gui.output_dir_var.get().strip()
    if not output_dir:
        messagebox.showerror("Error", "Select an output directory.", parent=gui.root)
        return

    # Get downloaded scene paths
    scene_paths = getattr(gui, 'downloaded_paths', [])
    if not scene_paths:
        messagebox.showerror("Error",
                            "No downloaded scenes. Download scenes first.",
                            parent=gui.root)
        return

    gui.config.output_directory = output_dir

    # Set processing_active AFTER all validation passes, just before thread start
    gui.processing_active = True

    gui.process_start_btn.config(state=tk.DISABLED)
    gui.process_stop_btn.config(state=tk.NORMAL)
    gui.status_var.set("Processing...")

    def progress_callback(step, total, msg):
        pct = (step / total * 100) if total > 0 else 0
        gui.root.after(0, lambda p=pct, m=msg: (gui.progress_var.set(p), gui.status_var.set(m)))
        _log_processing(gui, msg)

    def processing_thread():
        try:
            from ocean_rs.sar.core import BathymetryPipeline

            pipeline = BathymetryPipeline(gui.config)
            gui.pipeline = pipeline

            result = pipeline.process_scenes(
                [Path(p) for p in scene_paths],
                progress_callback=progress_callback,
            )

            if result is not None:
                _update_results(gui, result, pipeline)
                _log_processing(gui, "Processing complete!")
            else:
                _log_processing(gui, "Processing produced no results.")

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            _log_processing(gui, f"Processing failed: {e}")
        finally:
            gui.root.after(0, lambda: setattr(gui, 'processing_active', False))
            gui.root.after(0, lambda: gui.process_start_btn.config(state=tk.NORMAL))
            gui.root.after(0, lambda: gui.process_stop_btn.config(state=tk.DISABLED))
            gui.root.after(0, lambda: gui.status_var.set("Ready"))

    gui.processing_thread = threading.Thread(target=processing_thread, daemon=True)
    gui.processing_thread.start()


def stop_processing(gui):
    """Stop processing."""
    if hasattr(gui, 'pipeline') and gui.pipeline:
        gui.pipeline.cancel()
    gui.status_var.set("Cancelling...")


def _log_processing(gui, msg):
    """Append message to processing log."""
    def _append():
        if hasattr(gui, 'processing_log'):
            gui.processing_log.config(state=tk.NORMAL)
            gui.processing_log.insert(tk.END, msg + "\n")
            gui.processing_log.see(tk.END)
            gui.processing_log.config(state=tk.DISABLED)
    gui.root.after(0, _append)


def _update_results(gui, result, pipeline):
    """Update results summary labels."""
    def _update():
        if hasattr(gui, 'results_labels'):
            gui.results_labels['scenes'].config(
                text=str(pipeline.processed_count))
            gui.results_labels['depth_range'].config(
                text=f"{result.depth.min():.1f} - {result.depth.max():.1f} m")
            gui.results_labels['mean_depth'].config(
                text=f"{result.depth.mean():.1f} m")
            gui.results_labels['uncertainty'].config(
                text=f"{result.uncertainty.mean():.1f} m")
    gui.root.after(0, _update)
