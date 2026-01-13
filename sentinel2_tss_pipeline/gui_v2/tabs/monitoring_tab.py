"""
Monitoring Tab for GUI v2.

System monitoring and processing statistics with improved layout.
"""

import tkinter as tk
from tkinter import ttk
import os
import logging

from ..widgets import CollapsibleFrame
from ..theme import ThemeManager

logger = logging.getLogger('sentinel2_tss_pipeline')

# Try to import psutil
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


def create_monitoring_tab(gui, notebook):
    """
    Create the System Monitoring tab.

    Args:
        gui: Parent GUI instance
        notebook: ttk.Notebook to add tab to

    Returns:
        Tab index
    """
    frame = ttk.Frame(notebook, padding="10")
    tab_index = notebook.add(frame, text=" Monitor ")

    # Title
    ttk.Label(
        frame,
        text="System Monitoring & Status",
        style='Subtitle.TLabel'
    ).pack(pady=(0, 10))

    # === System Information Section ===
    sys_section = CollapsibleFrame(frame, title="System Information", expanded=True)
    sys_section.pack(fill=tk.X, pady=5)

    # System info display
    sys_info_frame = ttk.Frame(sys_section.content_frame)
    sys_info_frame.pack(fill=tk.X, pady=5)

    # Create labels for system metrics
    gui.cpu_label = _create_metric_row(sys_info_frame, "CPU:", "Loading...", 0)
    gui.memory_label = _create_metric_row(sys_info_frame, "Memory:", "Loading...", 1)
    gui.disk_label = _create_metric_row(sys_info_frame, "Disk:", "Loading...", 2)
    gui.snap_label = _create_metric_row(sys_info_frame, "SNAP:", "Checking...", 3)

    # Start system monitoring
    _start_system_monitoring(gui)

    # === Processing Status Section ===
    status_section = CollapsibleFrame(frame, title="Processing Status", expanded=True)
    status_section.pack(fill=tk.X, pady=5)

    # Current product
    current_frame = ttk.Frame(status_section.content_frame)
    current_frame.pack(fill=tk.X, pady=5)

    ttk.Label(current_frame, text="Current:").pack(side=tk.LEFT)
    gui.current_product_label = ttk.Label(
        current_frame,
        text="No processing active",
        style='Muted.TLabel'
    )
    gui.current_product_label.pack(side=tk.LEFT, padx=10)

    # Progress details
    progress_frame = ttk.Frame(status_section.content_frame)
    progress_frame.pack(fill=tk.X, pady=5)

    # Stage progress bars
    stages = [
        ("Resampling", "resampling_progress"),
        ("C2RCC", "c2rcc_progress"),
        ("TSS", "tss_progress"),
    ]

    for stage_name, var_name in stages:
        stage_frame = ttk.Frame(progress_frame)
        stage_frame.pack(fill=tk.X, pady=2)

        ttk.Label(stage_frame, text=f"{stage_name}:", width=12).pack(side=tk.LEFT)

        progress_var = tk.DoubleVar(value=0)
        setattr(gui, var_name, progress_var)

        progress_bar = ttk.Progressbar(
            stage_frame,
            variable=progress_var,
            maximum=100,
            length=200
        )
        progress_bar.pack(side=tk.LEFT, padx=5)

        status_label = ttk.Label(stage_frame, text="Waiting", style='Muted.TLabel')
        status_label.pack(side=tk.LEFT, padx=5)
        setattr(gui, f"{var_name}_label", status_label)

    # === Processing Statistics Section ===
    stats_section = CollapsibleFrame(frame, title="Processing Statistics", expanded=True)
    stats_section.pack(fill=tk.BOTH, expand=True, pady=5)

    # Statistics text area
    stats_frame = ttk.Frame(stats_section.content_frame)
    stats_frame.pack(fill=tk.BOTH, expand=True, pady=5)

    gui.stats_text = tk.Text(
        stats_frame,
        height=12,
        font=('Consolas', 10),
        state=tk.DISABLED,
        bg='#f8f8f8',
        relief='flat'
    )
    gui.stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    stats_scrollbar = ttk.Scrollbar(stats_frame, orient=tk.VERTICAL, command=gui.stats_text.yview)
    stats_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    gui.stats_text.configure(yscrollcommand=stats_scrollbar.set)

    # Initialize stats
    _update_stats_display(gui, {
        'total': 0,
        'processed': 0,
        'failed': 0,
        'skipped': 0,
        'progress': 0.0,
        'current': ''
    })

    # Control buttons
    button_frame = ttk.Frame(stats_section.content_frame)
    button_frame.pack(fill=tk.X, pady=5)

    ttk.Button(
        button_frame,
        text="Clear Statistics",
        command=lambda: _clear_stats(gui)
    ).pack(side=tk.LEFT, padx=2)

    ttk.Button(
        button_frame,
        text="Export Log",
        command=lambda: _export_log(gui)
    ).pack(side=tk.LEFT, padx=2)

    return tab_index


def _create_metric_row(parent, label, initial_value, row):
    """Create a metric display row."""
    ttk.Label(parent, text=label, width=10).grid(row=row, column=0, sticky=tk.E, padx=5, pady=2)

    value_label = ttk.Label(parent, text=initial_value, font=('Consolas', 10))
    value_label.grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)

    return value_label


def _start_system_monitoring(gui):
    """Start background system monitoring."""
    def update():
        try:
            if PSUTIL_AVAILABLE:
                # CPU
                cpu_percent = psutil.cpu_percent(interval=None)
                gui.cpu_label.configure(text=f"{cpu_percent}%")

                # Memory
                mem = psutil.virtual_memory()
                mem_used = mem.used / (1024 ** 3)
                mem_total = mem.total / (1024 ** 3)
                gui.memory_label.configure(text=f"{mem_used:.1f} / {mem_total:.1f} GB ({mem.percent}%)")

                # Disk
                try:
                    disk = psutil.disk_usage(os.path.expanduser("~"))
                    disk_free = disk.free / (1024 ** 3)
                    gui.disk_label.configure(text=f"{disk_free:.1f} GB free")
                except Exception:
                    gui.disk_label.configure(text="N/A")
            else:
                gui.cpu_label.configure(text="psutil not available")
                gui.memory_label.configure(text="psutil not available")
                gui.disk_label.configure(text="psutil not available")

            # Check SNAP
            snap_path = _find_snap_installation()
            if snap_path:
                gui.snap_label.configure(text=snap_path, style='Success.TLabel')
            else:
                gui.snap_label.configure(text="Not found", style='Warning.TLabel')

        except Exception as e:
            logger.warning(f"System monitoring error: {e}")

        # Schedule next update
        gui.root.after(3000, update)

    # Initial update
    gui.root.after(100, update)


def _find_snap_installation():
    """Find SNAP installation path."""
    possible_paths = [
        r"C:\Program Files\esa-snap",
        r"C:\Program Files\snap",
        os.path.expanduser("~/esa-snap"),
        "/usr/local/snap",
        "/opt/snap",
    ]

    for path in possible_paths:
        if os.path.isdir(path):
            gpt_path = os.path.join(path, "bin", "gpt.exe" if os.name == 'nt' else "gpt")
            if os.path.isfile(gpt_path) or os.path.isfile(gpt_path.replace('.exe', '')):
                return path

    return None


def _update_stats_display(gui, stats):
    """Update statistics display."""
    gui.stats_text.configure(state=tk.NORMAL)
    gui.stats_text.delete('1.0', tk.END)

    text = f"""Processing Statistics:
{'='*30}
Total Products: {stats.get('total', 0)}
Processed: {stats.get('processed', 0)}
Failed: {stats.get('failed', 0)}
Skipped: {stats.get('skipped', 0)}
Progress: {stats.get('progress', 0):.1f}%
Current: {stats.get('current', '')}
"""

    gui.stats_text.insert('1.0', text)
    gui.stats_text.configure(state=tk.DISABLED)


def _clear_stats(gui):
    """Clear statistics display."""
    _update_stats_display(gui, {
        'total': 0,
        'processed': 0,
        'failed': 0,
        'skipped': 0,
        'progress': 0.0,
        'current': ''
    })


def _export_log(gui):
    """Export processing log to file."""
    from tkinter import filedialog

    filepath = filedialog.asksaveasfilename(
        defaultextension=".txt",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        title="Export Log"
    )

    if filepath:
        try:
            gui.stats_text.configure(state=tk.NORMAL)
            content = gui.stats_text.get('1.0', tk.END)
            gui.stats_text.configure(state=tk.DISABLED)

            with open(filepath, 'w') as f:
                f.write(content)

            logger.info(f"Log exported to: {filepath}")
        except Exception as e:
            logger.error(f"Failed to export log: {e}")
