"""
Unified GUI for Sentinel-2 TSS Pipeline.

Professional tkinter interface for complete S2 processing and TSS estimation.

5-tab layout:
    1. Processing - Mode, I/O, options
    2. Spatial - Resampling + subset + map preview
    3. C2RCC - Atmospheric correction parameters
    4. Outputs - 6 category toggles, Jiang config, water mask
    5. Monitor - Progress, system info, statistics

Reference:
    Jiang, D., Matsushita, B., Pahlevan, N., et al. (2021).
    "Remotely Estimating Total Suspended Solids Concentration in Clear to
    Extremely Turbid Waters Using a Novel Semi-Analytical Method."
    Remote Sensing of Environment, 258, 112386.
"""

import sys
import os
import logging
import tkinter as tk
from tkinter import ttk, messagebox

logger = logging.getLogger('sentinel2_tss_pipeline')

# Import configuration classes
from ..config import (
    ResamplingConfig,
    SubsetConfig,
    C2RCCConfig,
    TSSConfig,
    OutputCategoryConfig,
    WaterQualityConfig,
)

# Import utilities
from ..utils.product_detector import SystemMonitor

# Import tab creators
from .tabs import (
    create_processing_tab,
    create_c2rcc_tab,
    create_monitoring_tab,
)
from .tabs.spatial_tab import create_spatial_tab
from .tabs.outputs_tab import create_outputs_tab

# Import handlers
from .handlers import (
    on_mode_change,
    update_tab_visibility,
    validate_input_directory,
    on_closing,
)

# Import config I/O
from .config_io import save_config, load_config

# Import processing controller
from .processing_controller import (
    start_processing,
    stop_processing,
    start_gui_updates,
)

# Import theme
from .theme import ThemeManager


def bring_window_to_front(window):
    """Bring a tkinter window to the front and give it focus."""
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
            except Exception:
                pass

    except Exception as e:
        logger.warning(f"Could not bring window to front: {e}")


class UnifiedS2TSSGUI:
    """
    Unified GUI for Complete S2 Processing and TSS Estimation Pipeline.

    Features:
        - 5-tab interface with modern theme
        - 6 output category toggles (replaces 13 sub-toggles)
        - Interactive map preview for subset geometry
        - Auto water mask (NDWI + NIR) enabled by default
        - NN preset auto-adjustment for C2RCC thresholds
    """

    def __init__(self):
        """Initialize the GUI with all components."""
        self.root = tk.Tk()
        self.root.title("Sentinel-2 TSS Pipeline v2.0")

        # Apply modern theme
        self.theme = ThemeManager(self.root)

        # Window sizing
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = 1000
        window_height = max(900, int(screen_height * 0.8))
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2

        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.minsize(1000, 900)
        self.root.resizable(True, True)

        bring_window_to_front(self.root)

        # Initialize
        self._init_configurations()
        self._init_state_variables()
        self._init_tk_variables()
        self._setup_gui()

        # Start background updates
        start_gui_updates(self)

        logger.info("GUI initialization complete")

    def _init_configurations(self):
        """Initialize configuration objects."""
        self.resampling_config = ResamplingConfig()
        self.subset_config = SubsetConfig()
        self.c2rcc_config = C2RCCConfig()
        self.tss_config = TSSConfig()

        logger.info("Configuration objects initialized")

    def _init_state_variables(self):
        """Initialize GUI state variables."""
        self.input_validation_result = {"valid": False, "message": "", "products": []}
        self.system_monitor = SystemMonitor()
        self.system_monitor.start_monitoring()

        # Processing state
        self.processor = None
        self.processing_thread = None
        self.processing_active = False

        # Tab management
        self.tab_indices = {}

    def _init_tk_variables(self):
        """Initialize all tkinter variables."""
        # --- Core ---
        self.processing_mode = tk.StringVar(value="complete_pipeline")
        self.progress_var = tk.DoubleVar()
        self.status_var = tk.StringVar(value="Ready")
        self.eta_var = tk.StringVar(value="")

        # --- I/O ---
        self.input_dir_var = tk.StringVar()
        self.output_dir_var = tk.StringVar()

        # --- Processing options ---
        self.skip_existing_var = tk.BooleanVar(value=True)
        self.test_mode_var = tk.BooleanVar(value=False)
        self.delete_intermediate_var = tk.BooleanVar(value=False)
        self.memory_limit_var = tk.StringVar(value="8")
        self.thread_count_var = tk.StringVar(value="4")

        # --- Spatial (Resampling) ---
        self.resolution_var = tk.StringVar(value="10")
        self.upsampling_var = tk.StringVar(value="Bilinear")
        self.downsampling_var = tk.StringVar(value="Mean")
        self.flag_downsampling_var = tk.StringVar(value="First")
        self.pyramid_var = tk.BooleanVar(value=True)

        # --- Spatial (Subset) ---
        self.subset_method_var = tk.StringVar(value="none")
        self.pixel_start_x_var = tk.StringVar()
        self.pixel_start_y_var = tk.StringVar()
        self.pixel_width_var = tk.StringVar()
        self.pixel_height_var = tk.StringVar()

        # --- C2RCC ---
        self.net_set_var = tk.StringVar(value="C2RCC-Nets")
        self.dem_name_var = tk.StringVar(value="Copernicus 30m Global DEM")
        self.elevation_var = tk.DoubleVar(value=0.0)
        self.salinity_var = tk.DoubleVar(value=35.0)
        self.temperature_var = tk.DoubleVar(value=15.0)
        self.ozone_var = tk.DoubleVar(value=330.0)
        self.pressure_var = tk.DoubleVar(value=1000.0)
        self.use_ecmwf_var = tk.BooleanVar(value=True)

        # --- C2RCC output products ---
        self.output_rrs_var = tk.BooleanVar(value=True)
        self.output_rhow_var = tk.BooleanVar(value=True)
        self.output_kd_var = tk.BooleanVar(value=True)
        self.output_uncertainties_var = tk.BooleanVar(value=True)
        self.output_ac_reflectance_var = tk.BooleanVar(value=True)
        self.output_rtoa_var = tk.BooleanVar(value=True)

        # --- Outputs (Jiang TSS) ---
        self.enable_jiang_var = tk.BooleanVar(value=True)
        self.jiang_intermediates_var = tk.BooleanVar(value=True)
        self.jiang_comparison_var = tk.BooleanVar(value=True)

        # --- Outputs (Water mask) ---
        self.auto_water_mask_var = tk.BooleanVar(value=True)
        self.water_mask_shapefile_var = tk.StringVar(value="")

        # --- Outputs (6 output categories) ---
        self.enable_tss_var = tk.BooleanVar(value=True)
        self.enable_rgb_var = tk.BooleanVar(value=True)
        self.enable_indices_var = tk.BooleanVar(value=True)
        self.enable_water_clarity_var = tk.BooleanVar(value=False)
        self.enable_hab_var = tk.BooleanVar(value=False)
        self.enable_trophic_state_var = tk.BooleanVar(value=False)

    def _setup_gui(self):
        """Setup the GUI interface."""
        try:
            main_container = ttk.Frame(self.root, padding="10")
            main_container.pack(fill=tk.BOTH, expand=True)

            # Title
            self._setup_title_section(main_container)

            # Tabbed notebook
            self.notebook = ttk.Notebook(main_container)
            self.notebook.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

            # 5 tabs
            self.tab_indices['processing'] = create_processing_tab(self, self.notebook)
            self.tab_indices['spatial'] = create_spatial_tab(self, self.notebook)
            self.tab_indices['c2rcc'] = create_c2rcc_tab(self, self.notebook)
            self.tab_indices['outputs'] = create_outputs_tab(self, self.notebook)
            self.tab_indices['monitoring'] = create_monitoring_tab(self, self.notebook)

            # Status bar and controls
            self._setup_status_bar(main_container)
            self._setup_control_buttons(main_container)

            # Update tab visibility based on initial mode
            update_tab_visibility(self)

            # Handle window close
            self.root.protocol("WM_DELETE_WINDOW", lambda: on_closing(self))

        except Exception as e:
            logger.error(f"GUI setup error: {e}")
            messagebox.showerror("GUI Error", f"Failed to setup GUI: {str(e)}")

    def _setup_title_section(self, parent):
        """Setup title section."""
        title_frame = ttk.Frame(parent)
        title_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(
            title_frame,
            text="Sentinel-2 TSS Pipeline",
            font=("Segoe UI", 16, "bold")
        ).pack()

        ttk.Label(
            title_frame,
            text="L1C \u2192 C2RCC (SNAP TSM/CHL) \u2192 Jiang TSS \u2192 Visualization",
            font=("Segoe UI", 10),
            foreground="gray"
        ).pack()

    def _setup_status_bar(self, parent):
        """Setup the status bar with progress information."""
        status_frame = ttk.LabelFrame(parent, text="Processing Status", padding="5")
        status_frame.pack(fill=tk.X, pady=(10, 5))

        self.progress_bar = ttk.Progressbar(
            status_frame, variable=self.progress_var,
            maximum=100, mode='determinate'
        )
        self.progress_bar.pack(fill=tk.X, pady=(0, 5))

        info_frame = ttk.Frame(status_frame)
        info_frame.pack(fill=tk.X)

        ttk.Label(info_frame, text="Status:").pack(side=tk.LEFT)
        self.status_label = ttk.Label(
            info_frame, textvariable=self.status_var, foreground="blue"
        )
        self.status_label.pack(side=tk.LEFT, padx=(5, 20))

        ttk.Label(info_frame, text="ETA:").pack(side=tk.LEFT)
        self.eta_label = ttk.Label(
            info_frame, textvariable=self.eta_var, foreground="gray"
        )
        self.eta_label.pack(side=tk.LEFT, padx=5)

    def _setup_control_buttons(self, parent):
        """Setup control buttons."""
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=5)

        # Left: Config buttons
        left_frame = ttk.Frame(button_frame)
        left_frame.pack(side=tk.LEFT)

        ttk.Button(
            left_frame, text="Save Config",
            command=lambda: save_config(self)
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            left_frame, text="Load Config",
            command=lambda: load_config(self)
        ).pack(side=tk.LEFT, padx=2)

        # Right: Processing buttons
        right_frame = ttk.Frame(button_frame)
        right_frame.pack(side=tk.RIGHT)

        self.start_button = ttk.Button(
            right_frame, text="Start Processing",
            command=lambda: start_processing(self)
        )
        self.start_button.pack(side=tk.LEFT, padx=2)

        self.stop_button = ttk.Button(
            right_frame, text="Stop",
            command=lambda: stop_processing(self),
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=2)

        ttk.Button(
            right_frame, text="Exit",
            command=lambda: on_closing(self)
        ).pack(side=tk.LEFT, padx=2)

    # Handler wrappers (delegate to module functions)
    def on_mode_change(self):
        on_mode_change(self)

    def update_tab_visibility(self):
        update_tab_visibility(self)

    def validate_input_directory(self, *args):
        validate_input_directory(self)

    def run(self):
        """Run the GUI main loop."""
        self.root.mainloop()


__all__ = ['UnifiedS2TSSGUI', 'bring_window_to_front']
