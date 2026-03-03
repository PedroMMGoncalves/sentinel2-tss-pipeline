"""
Event Handlers for SAR Bathymetry Toolkit GUI.
"""

import os
import logging

logger = logging.getLogger('ocean_rs')


def on_closing(gui):
    """Handle window close."""
    from tkinter import messagebox
    if messagebox.askokcancel("Quit", "Are you sure you want to exit?"):
        # Cancel active processing/downloads
        if hasattr(gui, 'processing_active') and gui.processing_active:
            if hasattr(gui, 'pipeline') and gui.pipeline:
                gui.pipeline.cancel()
        if hasattr(gui, 'download_active') and gui.download_active:
            gui.download_active = False  # Signal download thread to stop

        def _delayed_destroy():
            gui.root.destroy()

        gui.root.after(200, _delayed_destroy)


def update_system_info(gui):
    """Update system info labels."""
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=0)
        ram = psutil.virtual_memory()
        disk = psutil.disk_usage(os.path.abspath('.'))

        gui.sys_info_labels['cpu'].config(text=f"{cpu:.0f}%")
        gui.sys_info_labels['ram'].config(
            text=f"{ram.used / 1e9:.1f} / {ram.total / 1e9:.1f} GB ({ram.percent}%)")
        gui.sys_info_labels['disk'].config(
            text=f"{disk.used / 1e9:.0f} / {disk.total / 1e9:.0f} GB ({disk.percent}%)")
    except ImportError:
        logger.debug("psutil not available -- system info monitoring disabled")
    except Exception as e:
        logger.debug(f"System info update failed: {e}")


def start_gui_updates(gui):
    """Start periodic GUI updates."""
    def _update():
        update_system_info(gui)
        gui.root.after(5000, _update)
    gui.root.after(1000, _update)
