"""
Unified GUI for SAR Toolkit.

6-tab layout:
    1. Search & Select
    2. Download & Credentials
    3. Processing
    4. InSAR
    5. Displacement
    6. Results & Monitor
"""

import sys
import logging
import tkinter as tk
from tkinter import ttk

from ..config import SARProcessingConfig
from .theme import ThemeManager
from .tabs import (create_search_tab, create_download_tab, create_processing_tab,
                    create_insar_tab, create_displacement_tab, create_results_tab)
from .handlers import on_closing, start_gui_updates
from .config_io import save_config, load_config

logger = logging.getLogger('ocean_rs')


def bring_window_to_front(window):
    """Bring window to front."""
    try:
        window.lift()
        window.attributes('-topmost', True)
        window.focus_force()
        window.update_idletasks()
        window.update()
        window.after(100, lambda: window.attributes('-topmost', False))
        if sys.platform.startswith('win'):
            try:
                import ctypes
                hwnd = ctypes.windll.user32.GetActiveWindow()
                ctypes.windll.user32.FlashWindow(hwnd, True)
            except Exception as e:
                logger.debug(f"Window flash not available: {e}")
    except Exception as e:
        logger.warning(f"Could not bring window to front: {e}")


class UnifiedSARGUI:
    """Unified GUI for SAR Toolkit."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("OceanRS \u2014 SAR Toolkit v1.0")

        self.theme = ThemeManager(self.root)

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = 1000
        window_height = max(800, int(screen_height * 0.8))
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")

        bring_window_to_front(self.root)

        self._init_configurations()
        self._init_state_variables()
        self._init_tk_variables()
        self._setup_gui()

        start_gui_updates(self)

    def _init_configurations(self):
        self.config = SARProcessingConfig()

    def _init_state_variables(self):
        self.processing_active = False
        self.download_active = False
        self.processing_thread = None
        self.download_thread = None
        self.search_results = []
        self.selected_scenes = []
        self.downloaded_paths = []

    def _init_tk_variables(self):
        # Search
        self.aoi_var = tk.StringVar()
        self.start_date_var = tk.StringVar()
        self.end_date_var = tk.StringVar()
        self.platform_var = tk.StringVar(value="Sentinel-1")
        self.beam_mode_var = tk.StringVar(value="IW")
        self.polarization_var = tk.StringVar(value="VV+VH")
        self.orbit_dir_var = tk.StringVar(value="")

        # Credentials
        self.username_var = tk.StringVar()
        self.password_var = tk.StringVar()
        self.download_dir_var = tk.StringVar()

        # Processing
        self.processing_mode_var = tk.StringVar(value="bathymetry")
        self.snap_gpt_var = tk.StringVar()
        self.tile_size_var = tk.DoubleVar(value=1024.0)
        self.overlap_var = tk.DoubleVar(value=0.5)
        self.min_wavelength_var = tk.DoubleVar(value=50.0)
        self.max_wavelength_var = tk.DoubleVar(value=600.0)
        self.confidence_var = tk.DoubleVar(value=0.3)
        self.wave_source_var = tk.StringVar(value="wavewatch3")
        self.manual_period_var = tk.DoubleVar(value=10.0)
        self.max_depth_var = tk.DoubleVar(value=100.0)
        self.compositing_var = tk.BooleanVar(value=True)
        self.compositing_method_var = tk.StringVar(value="weighted_median")
        self.output_dir_var = tk.StringVar()

        # InSAR
        self.insar_primary_var = tk.StringVar()
        self.insar_secondary_var = tk.StringVar()
        self.insar_coreg_method_var = tk.StringVar(value="auto")
        self.insar_coreg_patch_var = tk.IntVar(value=128)
        self.insar_coh_range_var = tk.IntVar(value=15)
        self.insar_coh_azimuth_var = tk.IntVar(value=3)
        self.insar_filter_alpha_var = tk.DoubleVar(value=0.5)
        self.insar_filter_patch_var = tk.IntVar(value=32)
        self.insar_unwrap_method_var = tk.StringVar(value="auto")
        self.insar_remove_topo_var = tk.BooleanVar(value=True)
        self.insar_dem_path_var = tk.StringVar()
        self.insar_output_coh_var = tk.BooleanVar(value=True)
        self.insar_output_ifg_var = tk.BooleanVar(value=True)
        self.insar_output_unwrap_var = tk.BooleanVar(value=True)

        # Displacement
        self.disp_mode_var = tk.StringVar(value="dinsar")
        self.disp_max_temporal_var = tk.IntVar(value=180)
        self.disp_max_perp_var = tk.DoubleVar(value=200.0)
        self.disp_atm_filter_var = tk.BooleanVar(value=False)
        self.disp_temp_coh_var = tk.DoubleVar(value=0.7)
        self.disp_ref_lon_var = tk.DoubleVar(value=0.0)
        self.disp_ref_lat_var = tk.DoubleVar(value=0.0)
        self.disp_use_ref_point_var = tk.BooleanVar(value=False)
        self.disp_output_vertical_var = tk.BooleanVar(value=True)
        self.disp_output_los_var = tk.BooleanVar(value=True)

        # Status
        self.status_var = tk.StringVar(value="Ready")
        self.progress_var = tk.DoubleVar(value=0.0)

    def _setup_gui(self):
        # Menu bar
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Save Config...", command=lambda: save_config(self))
        file_menu.add_command(label="Load Config...", command=lambda: load_config(self))
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=lambda: on_closing(self))
        menubar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menubar)

        # Notebook
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        create_search_tab(self, self.notebook)
        create_download_tab(self, self.notebook)
        create_processing_tab(self, self.notebook)
        create_insar_tab(self, self.notebook)
        create_displacement_tab(self, self.notebook)
        create_results_tab(self, self.notebook)

        # Status bar
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        ttk.Label(status_frame, textvariable=self.status_var,
                  style='Status.TLabel').pack(side=tk.LEFT, padx=5)
        ttk.Progressbar(status_frame, variable=self.progress_var,
                       maximum=100, length=200).pack(side=tk.RIGHT, padx=5)

        # Research-grade disclaimer
        disclaimer = ttk.Label(
            self.root,
            text="Research-grade software \u2014 not validated for production displacement monitoring",
            style='Status.TLabel'
        )
        disclaimer.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=2)

        self.root.protocol("WM_DELETE_WINDOW", lambda: on_closing(self))

    def run(self):
        self.root.mainloop()
