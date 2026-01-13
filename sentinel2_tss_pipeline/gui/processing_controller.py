"""
Processing Controller for Sentinel-2 TSS Pipeline GUI.

Manages background processing threads and status updates.
"""

import os
import time
import threading
import tkinter as tk
from tkinter import messagebox
import logging

from ..utils.logging_utils import setup_enhanced_logging
from ..config.enums import ProcessingMode
from ..config.processing_config import ProcessingConfig
from ..core.unified_processor import UnifiedS2TSSProcessor
from .config_io import update_configurations

logger = logging.getLogger('sentinel2_tss_pipeline')


def start_processing(gui):
    """
    Start processing in background thread.

    Args:
        gui: GUI instance with all required variables and configurations.
    """
    if gui.processing_active:
        return

    # Setup logging to output folder
    output_folder = gui.output_dir_var.get()
    global logger
    logger, log_file = setup_enhanced_logging(
        log_level=logging.INFO,
        output_folder=output_folder
    )
    logger.info("Processing started - logs redirected to output folder")
    logger.info(f"Log file: {log_file}")

    try:
        # Ensure input_validation_result exists
        if not hasattr(gui, 'input_validation_result'):
            gui.input_validation_result = {"valid": False, "message": "", "products": []}

        # Validate configuration
        if not gui.input_validation_result.get("valid", False):
            messagebox.showerror(
                "Error",
                "Please fix input validation errors first",
                parent=gui.root
            )
            return

        if not gui.output_dir_var.get():
            messagebox.showerror(
                "Error",
                "Please select output directory",
                parent=gui.root
            )
            return

        if not update_configurations(gui):
            return

        # Create processing configuration
        processing_config = ProcessingConfig(
            processing_mode=ProcessingMode(gui.processing_mode.get()),
            input_folder=gui.input_dir_var.get(),
            output_folder=gui.output_dir_var.get(),
            resampling_config=gui.resampling_config,
            subset_config=gui.subset_config,
            c2rcc_config=gui.c2rcc_config,
            jiang_config=gui.jiang_config,
            skip_existing=gui.skip_existing_var.get(),
            test_mode=gui.test_mode_var.get(),
            memory_limit_gb=int(gui.memory_limit_var.get()),
            thread_count=int(gui.thread_count_var.get())
        )

        # Build and show confirmation dialog
        products = gui.input_validation_result.get("products", [])
        if not _confirm_processing(gui, processing_config, products):
            return

        # Start processing thread
        gui.processing_active = True
        gui.start_button.config(state=tk.DISABLED)
        gui.stop_button.config(state=tk.NORMAL)
        gui.status_var.set("Starting processing...")

        gui.processing_thread = threading.Thread(
            target=_run_processing_thread,
            args=(gui, processing_config, products),
            daemon=True
        )
        gui.processing_thread.start()

    except Exception as e:
        messagebox.showerror(
            "Error",
            f"Failed to start processing: {str(e)}",
            parent=gui.root
        )
        gui.processing_active = False
        gui.start_button.config(state=tk.NORMAL)
        gui.stop_button.config(state=tk.DISABLED)
        logger.error(f"Failed to start processing: {e}")


def _confirm_processing(gui, config, products):
    """
    Show confirmation dialog before processing.

    Returns:
        bool: True if user confirms, False otherwise.
    """
    process_count = len(products)
    if config.test_mode:
        process_count = min(1, process_count)

    mode_name = config.processing_mode.value.replace('_', ' ').title()

    confirm_msg = f"Start {mode_name} processing?\n\n"
    confirm_msg += f"Products found: {len(products)}\n"
    confirm_msg += f"Will process: {process_count} products\n"
    confirm_msg += f"Mode: {mode_name}\n"
    confirm_msg += f"Output: {config.output_folder}\n"
    confirm_msg += f"ECMWF: {'Enabled' if config.c2rcc_config.use_ecmwf_aux_data else 'Disabled'}\n"

    if config.processing_mode in [ProcessingMode.COMPLETE_PIPELINE, ProcessingMode.TSS_PROCESSING_ONLY]:
        if config.jiang_config.enable_jiang_tss:
            confirm_msg += "Jiang TSS: Enabled\n"
        else:
            confirm_msg += "Jiang TSS: Disabled (SNAP TSM/CHL only)\n"

    confirm_msg += "\nProceed?"

    return messagebox.askyesno("Confirm Processing", confirm_msg, parent=gui.root)


def _run_processing_thread(gui, config, products):
    """
    Run processing in background thread.

    Args:
        gui: GUI instance.
        config: ProcessingConfig object.
        products: List of product paths to process.
    """
    try:
        # Create processor
        gui.processor = UnifiedS2TSSProcessor(config)

        # Limit products for test mode
        if config.test_mode:
            products = products[:1]
            logger.info(f"TEST MODE: Processing only {len(products)} products")

        # Process products
        total_products = len(products)

        for i, product_path in enumerate(products):
            if not gui.processing_active:  # Check for stop signal
                break

            # Update status
            product_name = os.path.basename(product_path)
            gui.status_var.set(f"Processing {i+1}/{total_products}: {product_name}")

            # Process product
            try:
                gui.processor._process_single_product(product_path, i+1, total_products)
            except Exception as e:
                logger.error(f"Error processing {product_name}: {e}")

            # Update progress
            progress = ((i + 1) / total_products) * 100
            gui.progress_var.set(progress)

            # Update ETA
            status = gui.processor.get_processing_status()
            if status.eta_minutes > 0:
                gui.eta_var.set(
                    f"ETA: {status.eta_minutes:.1f} min | "
                    f"Speed: {status.processing_speed:.1f} products/min"
                )
            else:
                gui.eta_var.set("")

        # Processing completed
        _on_processing_complete(gui)

    except Exception as e:
        logger.error(f"Processing thread error: {e}")
        gui.status_var.set(f"Processing error: {str(e)}")
    finally:
        gui.processing_active = False
        gui.start_button.config(state=tk.NORMAL)
        gui.stop_button.config(state=tk.DISABLED)
        if gui.processor:
            gui.processor.cleanup()


def stop_processing(gui):
    """
    Stop active processing.

    Args:
        gui: GUI instance.
    """
    if gui.processing_active:
        gui.processing_active = False
        gui.status_var.set("Stopping processing...")
        gui.stop_button.config(state=tk.DISABLED)
        logger.info("Processing stop requested")


def _on_processing_complete(gui):
    """
    Handle processing completion.

    Args:
        gui: GUI instance.
    """
    if gui.processor:
        status = gui.processor.get_processing_status()

        # Final status
        gui.status_var.set("Processing completed!")
        gui.progress_var.set(100)

        # Show completion message
        total_time = (time.time() - gui.processor.start_time) / 60

        completion_msg = (
            f"Processing completed!\n\n"
            f"Successfully processed: {status.processed}\n"
            f"Failed: {status.failed}\n"
            f"Skipped: {status.skipped}\n"
            f"Total time: {total_time:.1f} minutes\n\n"
            f"Outputs saved to:\n{gui.output_dir_var.get()}\n\n"
            f"Check log file for details."
        )

        messagebox.showinfo("Processing Complete", completion_msg, parent=gui.root)
        logger.info(f"Processing complete: {status.processed} processed, {status.failed} failed")


def start_gui_updates(gui):
    """
    Start GUI update loop for system monitoring.

    Args:
        gui: GUI instance.
    """
    update_system_info(gui)
    update_processing_stats(gui)
    gui.root.after(2000, lambda: start_gui_updates(gui))  # Update every 2 seconds


def update_system_info(gui):
    """
    Update system information display.

    Args:
        gui: GUI instance with system labels.
    """
    try:
        if not hasattr(gui, 'system_monitor'):
            return

        info = gui.system_monitor.get_current_info()

        # Update system labels
        if hasattr(gui, 'cpu_label'):
            gui.cpu_label.config(text=f"CPU: {info['cpu_percent']:.1f}%")

        # Memory info
        if hasattr(gui, 'memory_label'):
            if info['memory_total_gb'] > 0:
                memory_percent = (info['memory_used_gb'] / info['memory_total_gb']) * 100
            else:
                memory_percent = 0.0

            gui.memory_label.config(
                text=f"Memory: {info['memory_used_gb']:.1f}/{info['memory_total_gb']:.1f} GB ({memory_percent:.1f}%)"
            )

            # Color coding for warnings
            if memory_percent > 90:
                gui.memory_label.config(foreground="red")
            elif memory_percent > 75:
                gui.memory_label.config(foreground="orange")
            else:
                gui.memory_label.config(foreground="black")

        # Disk info
        if hasattr(gui, 'disk_label'):
            gui.disk_label.config(text=f"Disk Free: {info['disk_free_gb']:.1f} GB")

            if info['disk_free_gb'] < 10:
                gui.disk_label.config(foreground="red")
            elif info['disk_free_gb'] < 50:
                gui.disk_label.config(foreground="orange")
            else:
                gui.disk_label.config(foreground="black")

        # SNAP status
        if hasattr(gui, 'snap_label'):
            snap_home = os.environ.get('SNAP_HOME', 'Not set')
            gui.snap_label.config(text=f"SNAP: {snap_home}")

    except Exception as e:
        logger.debug(f"GUI update error: {e}")
        # Set default values on error
        try:
            if hasattr(gui, 'cpu_label'):
                gui.cpu_label.config(text="CPU: --")
            if hasattr(gui, 'memory_label'):
                gui.memory_label.config(text="Memory: --")
            if hasattr(gui, 'disk_label'):
                gui.disk_label.config(text="Disk: --")
        except Exception:
            pass


def update_processing_stats(gui):
    """
    Update processing statistics display.

    Args:
        gui: GUI instance with stats_text widget.
    """
    if not hasattr(gui, 'processor') or not gui.processor or not gui.processing_active:
        return

    try:
        status = gui.processor.get_processing_status()

        # Update stats text
        stats_text = (
            f"Processing Statistics:\n"
            f"{'='*30}\n"
            f"Total Products: {status.total_products}\n"
            f"Processed: {status.processed}\n"
            f"Failed: {status.failed}\n"
            f"Skipped: {status.skipped}\n"
            f"Progress: {status.progress_percent:.1f}%\n"
            f"Current: {status.current_product}\n"
            f"Stage: {status.current_stage}\n"
            f"ETA: {status.eta_minutes:.1f} minutes\n"
            f"Speed: {status.processing_speed:.2f} products/min\n"
        )

        # System health
        if hasattr(gui, 'system_monitor'):
            healthy, warnings = gui.system_monitor.check_system_health()
            stats_text += f"\nSystem Health:\n{'='*15}\n"
            if healthy:
                stats_text += "System healthy\n"
            else:
                stats_text += "System warnings:\n"
                for warning in warnings:
                    stats_text += f"  - {warning}\n"

        # Update text widget
        if hasattr(gui, 'stats_text'):
            gui.stats_text.config(state=tk.NORMAL)
            gui.stats_text.delete(1.0, tk.END)
            gui.stats_text.insert(1.0, stats_text)
            gui.stats_text.config(state=tk.DISABLED)

    except Exception as e:
        logger.debug(f"Stats update error: {e}")
