"""
Monitoring Tab for Sentinel-2 TSS Pipeline GUI.

Provides system monitoring, processing status, and statistics interface.
"""

import tkinter as tk
from tkinter import ttk
import logging

logger = logging.getLogger('sentinel2_tss_pipeline')


def create_monitoring_tab(gui, notebook):
    """
    Create the System Monitoring tab.

    Args:
        gui: Parent GUI instance with all tk variables.
        notebook: ttk.Notebook to add the tab to.

    Returns:
        int: Tab index.
    """
    frame = ttk.Frame(notebook)
    tab_index = notebook.add(frame, text="System Monitor")

    # Title
    ttk.Label(
        frame, text="System Monitoring & Status",
        font=("Arial", 14, "bold")
    ).pack(pady=10)

    # System Information
    _create_system_info_section(frame, gui)

    # Processing Status
    _create_processing_status_section(frame, gui)

    # Processing Statistics
    _create_statistics_section(frame, gui)

    return tab_index


def _create_system_info_section(parent, gui):
    """Create system information display section."""
    sys_frame = ttk.LabelFrame(parent, text="System Information", padding="10")
    sys_frame.pack(fill=tk.X, padx=10, pady=5)

    # CPU label
    gui.cpu_label = ttk.Label(
        sys_frame, text="CPU: --", font=("Consolas", 10)
    )
    gui.cpu_label.pack(anchor=tk.W, pady=2)

    # Memory label
    gui.memory_label = ttk.Label(
        sys_frame, text="Memory: --", font=("Consolas", 10)
    )
    gui.memory_label.pack(anchor=tk.W, pady=2)

    # Disk label
    gui.disk_label = ttk.Label(
        sys_frame, text="Disk: --", font=("Consolas", 10)
    )
    gui.disk_label.pack(anchor=tk.W, pady=2)

    # SNAP label
    gui.snap_label = ttk.Label(
        sys_frame, text="SNAP: --", font=("Consolas", 10)
    )
    gui.snap_label.pack(anchor=tk.W, pady=2)


def _create_processing_status_section(parent, gui):
    """Create processing status display section."""
    status_frame = ttk.LabelFrame(parent, text="Processing Status", padding="10")
    status_frame.pack(fill=tk.X, padx=10, pady=5)

    # Progress bar
    gui.progress_bar = ttk.Progressbar(
        status_frame, variable=gui.progress_var,
        maximum=100, mode='determinate'
    )
    gui.progress_bar.pack(fill=tk.X, pady=5)

    # Current status label
    gui.current_status_label = ttk.Label(
        status_frame, textvariable=gui.status_var,
        font=("Arial", 10, "bold")
    )
    gui.current_status_label.pack(anchor=tk.W, pady=2)

    # ETA label
    gui.eta_label = ttk.Label(
        status_frame, textvariable=gui.eta_var,
        font=("Arial", 9), foreground="gray"
    )
    gui.eta_label.pack(anchor=tk.W, pady=2)


def _create_statistics_section(parent, gui):
    """Create processing statistics display section."""
    stats_frame = ttk.LabelFrame(parent, text="Processing Statistics", padding="10")
    stats_frame.pack(fill=tk.X, padx=10, pady=5)

    # Statistics text box
    gui.stats_text = tk.Text(
        stats_frame, height=8, font=("Consolas", 9),
        state=tk.DISABLED, wrap=tk.WORD
    )

    # Scrollbar for stats
    stats_scrollbar = ttk.Scrollbar(
        stats_frame, orient=tk.VERTICAL, command=gui.stats_text.yview
    )
    gui.stats_text.configure(yscrollcommand=stats_scrollbar.set)

    gui.stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    stats_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)


def update_system_info(gui):
    """
    Update system information labels with current values.

    Call this periodically to refresh system stats.
    """
    try:
        import psutil

        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        gui.cpu_label.config(text=f"CPU: {cpu_percent:.1f}%")

        # Memory usage
        memory = psutil.virtual_memory()
        memory_used_gb = memory.used / (1024 ** 3)
        memory_total_gb = memory.total / (1024 ** 3)
        gui.memory_label.config(
            text=f"Memory: {memory_used_gb:.1f} / {memory_total_gb:.1f} GB ({memory.percent:.1f}%)"
        )

        # Disk usage (for output directory if set)
        output_dir = gui.output_dir_var.get() if hasattr(gui, 'output_dir_var') else None
        if output_dir:
            try:
                disk = psutil.disk_usage(output_dir)
                disk_free_gb = disk.free / (1024 ** 3)
                gui.disk_label.config(text=f"Disk: {disk_free_gb:.1f} GB free")
            except Exception:
                gui.disk_label.config(text="Disk: --")
        else:
            gui.disk_label.config(text="Disk: (select output dir)")

        # SNAP status
        snap_status = _check_snap_status()
        gui.snap_label.config(text=f"SNAP: {snap_status}")

    except ImportError:
        gui.cpu_label.config(text="CPU: psutil not available")
        gui.memory_label.config(text="Memory: psutil not available")

    except Exception as e:
        logger.debug(f"Error updating system info: {e}")


def _check_snap_status():
    """Check SNAP installation status."""
    import os
    import shutil

    # Check common SNAP paths
    snap_paths = [
        os.environ.get('SNAP_HOME', ''),
        'C:\\Program Files\\esa-snap',
        'C:\\Program Files\\snap',
        '/opt/snap',
        os.path.expanduser('~/snap'),
    ]

    for path in snap_paths:
        if path and os.path.exists(path):
            gpt_path = os.path.join(path, 'bin', 'gpt')
            gpt_bat = os.path.join(path, 'bin', 'gpt.bat')
            if os.path.exists(gpt_path) or os.path.exists(gpt_bat):
                return f"Found at {path}"

    # Check if gpt is in PATH
    if shutil.which('gpt'):
        return "Available (in PATH)"

    return "Not found"


def update_statistics(gui, stats_dict):
    """
    Update the statistics text box with processing statistics.

    Args:
        gui: GUI instance with stats_text widget.
        stats_dict: Dictionary of statistics to display.
    """
    if not hasattr(gui, 'stats_text'):
        return

    try:
        gui.stats_text.config(state=tk.NORMAL)
        gui.stats_text.delete(1.0, tk.END)

        for key, value in stats_dict.items():
            gui.stats_text.insert(tk.END, f"{key}: {value}\n")

        gui.stats_text.config(state=tk.DISABLED)

    except Exception as e:
        logger.debug(f"Error updating statistics: {e}")


def clear_statistics(gui):
    """Clear the statistics text box."""
    if not hasattr(gui, 'stats_text'):
        return

    try:
        gui.stats_text.config(state=tk.NORMAL)
        gui.stats_text.delete(1.0, tk.END)
        gui.stats_text.config(state=tk.DISABLED)

    except Exception as e:
        logger.debug(f"Error clearing statistics: {e}")
