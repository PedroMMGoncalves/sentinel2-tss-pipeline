"""
Processing Controller for SAR Toolkit GUI.

Manages background processing threads for bathymetry, InSAR, and displacement.
"""

import tkinter as tk
from tkinter import messagebox
import threading
import logging
from pathlib import Path

from .config_io import update_configurations

logger = logging.getLogger('ocean_rs')


def start_processing(gui):
    """Start processing in background thread based on selected mode."""
    if gui.processing_active:
        return

    if not update_configurations(gui):
        return

    output_dir = gui.output_dir_var.get().strip()
    if not output_dir:
        messagebox.showerror("Error", "Select an output directory.", parent=gui.root)
        return

    gui.config.output_directory = output_dir
    mode = gui.processing_mode_var.get()

    if mode == "bathymetry":
        _start_bathymetry(gui, output_dir)
    elif mode == "insar":
        _start_insar(gui, output_dir)
    elif mode == "displacement":
        _start_displacement(gui, output_dir)
    else:
        messagebox.showerror("Error", f"Unknown processing mode: {mode}",
                            parent=gui.root)


def _activate_processing(gui):
    """Set GUI state to processing-active.

    Thread safety: This function must only be called from the main thread
    (GUI button callbacks). The counterpart _deactivate_processing() is
    called from background threads but marshals all GUI state changes
    through gui.root.after() to ensure main-thread-only access.
    """
    gui.processing_active = True
    gui.process_start_btn.config(state=tk.DISABLED)
    gui.process_stop_btn.config(state=tk.NORMAL)
    gui.status_var.set("Processing...")


def _deactivate_processing(gui):
    """Reset GUI state after processing."""
    gui.root.after(0, lambda: setattr(gui, 'processing_active', False))
    gui.root.after(0, lambda: gui.process_start_btn.config(state=tk.NORMAL))
    gui.root.after(0, lambda: gui.process_stop_btn.config(state=tk.DISABLED))
    gui.root.after(0, lambda: gui.status_var.set("Ready"))


def _progress_callback(gui):
    """Create a progress callback bound to the GUI."""
    def callback(step, total, msg):
        pct = (step / total * 100) if total > 0 else 0
        gui.root.after(0, lambda p=pct, m=msg: (gui.progress_var.set(p), gui.status_var.set(m)))
        _log_processing(gui, msg)
    return callback


def _start_bathymetry(gui, output_dir):
    """Start bathymetry processing."""
    scene_paths = getattr(gui, 'downloaded_paths', [])
    if not scene_paths:
        messagebox.showerror("Error",
                            "No downloaded scenes. Download scenes first.",
                            parent=gui.root)
        return

    _activate_processing(gui)

    def run():
        try:
            from ocean_rs.sar.core import BathymetryPipeline

            pipeline = BathymetryPipeline(gui.config)
            gui.root.after(0, lambda p=pipeline: setattr(gui, 'pipeline', p))

            result = pipeline.process_scenes(
                [Path(p) for p in scene_paths],
                progress_callback=_progress_callback(gui),
            )

            if result is not None:
                _update_results(gui, result, pipeline)
                _log_processing(gui, "Bathymetry processing complete!")
            else:
                _log_processing(gui, "Bathymetry produced no results.")

        except Exception as e:
            logger.error(f"Bathymetry processing failed: {e}")
            _log_processing(gui, f"Processing failed: {e}")
        finally:
            _deactivate_processing(gui)

    gui.processing_thread = threading.Thread(target=run, daemon=True)
    gui.processing_thread.start()


def _start_insar(gui, output_dir):
    """Start InSAR processing."""
    primary = gui.insar_primary_var.get().strip()
    secondary = gui.insar_secondary_var.get().strip()

    if not primary or not secondary:
        messagebox.showerror("Error",
                            "Select primary and secondary SLC scenes on the InSAR tab.",
                            parent=gui.root)
        return

    _activate_processing(gui)

    def run():
        try:
            from ocean_rs.sar.insar import InSARPipeline

            pipeline = InSARPipeline(gui.config)
            gui.root.after(0, lambda p=pipeline: setattr(gui, 'pipeline', p))

            result = pipeline.process(
                primary_path=primary,
                secondary_path=secondary,
                output_dir=output_dir,
                progress_callback=_progress_callback(gui),
            )

            if result is not None:
                _log_processing(gui, "InSAR processing complete!")
            else:
                _log_processing(gui, "InSAR processing cancelled or produced no results.")

        except Exception as e:
            logger.error(f"InSAR processing failed: {e}")
            _log_processing(gui, f"InSAR processing failed: {e}")
        finally:
            _deactivate_processing(gui)

    gui.processing_thread = threading.Thread(target=run, daemon=True)
    gui.processing_thread.start()


def _start_displacement(gui, output_dir):
    """Start displacement analysis on pre-computed interferograms.

    Looks for InSAR results in the output directory (primary/secondary SLC
    paths from the InSAR tab) and runs the displacement pipeline on them.
    """
    mode = gui.disp_mode_var.get()
    if mode == "sbas":
        messagebox.showinfo(
            "SBAS Not Available",
            "SBAS time-series analysis requires multiple pre-computed interferograms.\n\n"
            "Use InSAR mode to generate interferograms for multiple SLC pairs first, "
            "then use the SBAS API programmatically.\n\n"
            "For single-pair displacement, select DInSAR mode.",
            parent=gui.root
        )
        return

    primary = gui.insar_primary_var.get().strip()
    secondary = gui.insar_secondary_var.get().strip()

    if not primary or not secondary:
        messagebox.showerror(
            "Error",
            "Select primary and secondary SLC scenes on the InSAR tab.\n"
            "Displacement analysis requires pre-computed interferograms.",
            parent=gui.root,
        )
        return

    _activate_processing(gui)

    def run():
        try:
            from ocean_rs.sar.displacement import DisplacementPipeline
            from ocean_rs.sar.insar import InSARPipeline

            pipeline = DisplacementPipeline(gui.config)
            gui.root.after(0, lambda p=pipeline: setattr(gui, 'pipeline', p))

            # Generate interferogram via InSAR pipeline first
            _log_processing(gui, "Generating interferogram from SLC pair...")
            insar_pipeline = InSARPipeline(gui.config)
            # Store inner pipeline so stop_processing can cancel it too
            gui.root.after(0, lambda p=insar_pipeline: setattr(gui, '_insar_pipeline', p))
            ifg_result = insar_pipeline.process(
                primary_path=primary,
                secondary_path=secondary,
                output_dir=output_dir,
                progress_callback=_progress_callback(gui),
            )

            if ifg_result is None:
                _log_processing(gui, "Interferogram generation cancelled or failed.")
                return

            # Run displacement estimation
            _log_processing(gui, "Running displacement analysis...")
            displacement = pipeline.process_dinsar(
                interferogram=ifg_result,
                output_dir=output_dir,
                progress_callback=_progress_callback(gui),
            )

            if displacement is not None:
                _log_processing(gui, "Displacement analysis complete!")
            else:
                _log_processing(gui, "Displacement analysis cancelled or produced no results.")

        except Exception as e:
            logger.error(f"Displacement processing failed: {e}")
            _log_processing(gui, f"Displacement processing failed: {e}")
        finally:
            _deactivate_processing(gui)

    gui.processing_thread = threading.Thread(target=run, daemon=True)
    gui.processing_thread.start()


def stop_processing(gui):
    """Stop processing — cancels both outer and inner pipelines."""
    if hasattr(gui, 'pipeline') and gui.pipeline:
        gui.pipeline.cancel()
    if hasattr(gui, '_insar_pipeline') and gui._insar_pipeline:
        gui._insar_pipeline.cancel()
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
