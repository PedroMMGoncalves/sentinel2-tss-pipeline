"""
Unified GUI for Sentinel-2 TSS Pipeline.

Professional interface that combines S2 pre-processing with TSS estimation,
featuring automatic SNAP TSM/CHL generation and optional Jiang methodology.

Reference:
    Jiang, D., Matsushita, B., Pahlevan, N., et al. (2021).
    "Remotely Estimating Total Suspended Solids Concentration in Clear to
    Extremely Turbid Waters Using a Novel Semi-Analytical Method."
    Remote Sensing of Environment, 258, 112386.
    DOI: https://doi.org/10.1016/j.rse.2021.112386
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
    JiangTSSConfig,
    MarineVisualizationConfig,
    WaterQualityConfig,
)

# Import utilities
from ..utils.product_detector import SystemMonitor
from ..utils.geometry_utils import (
    load_geometry,
    validate_wkt,
    generate_area_name,
    # Backwards compatibility aliases
    load_geometry_from_file,
    validate_wkt_geometry,
    get_area_name,
)

# Import tab creators
from .tabs import (
    create_processing_tab,
    create_resampling_tab,
    create_subset_tab,
    create_c2rcc_tab,
    create_tss_tab,
    create_monitoring_tab,
)

# Import handlers
from .handlers import (
    on_mode_change,
    update_tab_visibility,
    update_subset_visibility,
    update_jiang_visibility,
    on_ecmwf_toggle,
    on_rhow_toggle,
    validate_input_directory,
    validate_geometry,
    browse_input_dir,
    browse_output_dir,
    apply_water_preset,
    apply_snap_defaults,
    apply_essential_outputs,
    apply_scientific_outputs,
    reset_all_outputs,
    on_closing,
)

# Import config I/O
from .config_io import (
    update_configurations,
    save_config,
    load_config,
)

# Import processing controller
from .processing_controller import (
    start_processing,
    stop_processing,
    start_gui_updates,
    update_system_info,
    update_processing_stats,
)


def bring_window_to_front(window):
    """
    Enhanced window focus management.

    Brings a tkinter window to the front of other windows
    and ensures it receives focus.
    """
    try:
        window.lift()
        window.attributes('-topmost', True)
        window.focus_force()
        window.grab_set()
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

    Professional interface that combines S2 pre-processing with TSS estimation,
    featuring automatic SNAP TSM/CHL generation and optional Jiang methodology.

    Features:
        - Complete S2 processing pipeline (L1C -> C2RCC)
        - Automatic SNAP TSM/CHL calculation
        - Optional Jiang TSS methodology
        - Marine visualization products
        - Batch processing capabilities
        - Progress monitoring and logging
    """

    def __init__(self):
        """Initialize the GUI with all components."""
        self.root = tk.Tk()
        self.root.title("Unified S2 Processing & TSS Estimation Pipeline v2.0")

        # Get screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # Calculate window size (80% of screen height, min 900px)
        window_width = 1000
        window_height = max(900, int(screen_height * 0.8))

        # Center window on screen
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2

        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.configure(bg='#f0f0f0')

        # Make resizable with minimum size
        self.root.minsize(1000, 900)
        self.root.resizable(True, True)

        # Bring window to front
        bring_window_to_front(self.root)

        # Initialize configuration objects
        self._init_configurations()

        # Initialize GUI state variables
        self._init_state_variables()

        # Initialize tk variables
        self._init_tk_variables()

        # Setup GUI components
        self._setup_gui()

        # Start background updates
        start_gui_updates(self)

        logger.info("GUI initialization complete")

    def _init_configurations(self):
        """Initialize configuration objects."""
        self.resampling_config = ResamplingConfig()
        self.subset_config = SubsetConfig()
        self.c2rcc_config = C2RCCConfig()
        self.jiang_config = JiangTSSConfig()

        # Initialize marine visualization
        if not hasattr(self.jiang_config, 'enable_marine_visualization'):
            self.jiang_config.enable_marine_visualization = True

        if not hasattr(self.jiang_config, 'marine_viz_config'):
            self.jiang_config.marine_viz_config = MarineVisualizationConfig()

        # Initialize advanced algorithms
        if not hasattr(self.jiang_config, 'enable_advanced_algorithms'):
            self.jiang_config.enable_advanced_algorithms = True

        if not hasattr(self.jiang_config, 'water_quality_config'):
            self.jiang_config.water_quality_config = WaterQualityConfig()

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
        # Processing mode
        self.processing_mode = tk.StringVar(value="complete_pipeline")

        # Progress tracking
        self.progress_var = tk.DoubleVar()
        self.status_var = tk.StringVar(value="Ready")
        self.eta_var = tk.StringVar(value="")

        # Input/Output
        self.input_dir_var = tk.StringVar()
        self.output_dir_var = tk.StringVar()

        # Processing options
        self.skip_existing_var = tk.BooleanVar(value=True)
        self.test_mode_var = tk.BooleanVar(value=False)
        self.delete_intermediate_var = tk.BooleanVar(value=False)
        self.memory_limit_var = tk.StringVar(value="8")
        self.thread_count_var = tk.StringVar(value="4")

        # Resampling configuration
        self.resolution_var = tk.StringVar(value="10")
        self.upsampling_var = tk.StringVar(value="Bilinear")
        self.downsampling_var = tk.StringVar(value="Mean")
        self.flag_downsampling_var = tk.StringVar(value="First")
        self.pyramid_var = tk.BooleanVar(value=True)

        # Subset configuration
        self.subset_method_var = tk.StringVar(value="none")
        self.pixel_start_x_var = tk.StringVar()
        self.pixel_start_y_var = tk.StringVar()
        self.pixel_width_var = tk.StringVar()
        self.pixel_height_var = tk.StringVar()

        # Geometry subset variables
        self.wkt_geometry_var = tk.StringVar()
        self.geometry_file_var = tk.StringVar()
        self.geometry_crs_var = tk.StringVar(value="EPSG:4326")
        self.copy_metadata_var = tk.BooleanVar(value=True)

        # C2RCC configuration
        self.net_set_var = tk.StringVar(value="C2RCC-Nets")
        self.dem_name_var = tk.StringVar(value="Copernicus 30m Global DEM")
        self.elevation_var = tk.DoubleVar(value=0.0)

        # Water and atmospheric parameters
        self.salinity_var = tk.DoubleVar(value=35.0)
        self.temperature_var = tk.DoubleVar(value=15.0)
        self.ozone_var = tk.DoubleVar(value=330.0)
        self.pressure_var = tk.DoubleVar(value=1000.0)
        self.use_ecmwf_var = tk.BooleanVar(value=True)

        # Essential output products
        self.output_rrs_var = tk.BooleanVar(value=True)
        self.output_rhow_var = tk.BooleanVar(value=True)
        self.output_kd_var = tk.BooleanVar(value=True)
        self.output_uncertainties_var = tk.BooleanVar(value=True)
        self.output_ac_reflectance_var = tk.BooleanVar(value=True)
        self.output_rtoa_var = tk.BooleanVar(value=True)

        # Advanced atmospheric products
        self.output_rtosa_gc_var = tk.BooleanVar(value=False)
        self.output_rtosa_gc_aann_var = tk.BooleanVar(value=False)
        self.output_rpath_var = tk.BooleanVar(value=False)
        self.output_tdown_var = tk.BooleanVar(value=False)
        self.output_tup_var = tk.BooleanVar(value=False)
        self.output_oos_var = tk.BooleanVar(value=False)

        # Advanced C2RCC parameters
        self.valid_pixel_var = tk.StringVar(value="B8 > 0 && B8 < 0.1")
        self.threshold_rtosa_oos_var = tk.DoubleVar(value=0.05)
        self.threshold_ac_reflec_oos_var = tk.DoubleVar(value=0.1)
        self.threshold_cloud_tdown865_var = tk.DoubleVar(value=0.955)

        # TSM and CHL parameters
        self.tsm_fac_var = tk.DoubleVar(value=1.06)
        self.tsm_exp_var = tk.DoubleVar(value=0.942)
        self.chl_fac_var = tk.DoubleVar(value=21.0)
        self.chl_exp_var = tk.DoubleVar(value=1.04)

        # Jiang TSS configuration
        self.enable_jiang_var = tk.BooleanVar(value=True)
        self.jiang_intermediates_var = tk.BooleanVar(value=True)
        self.jiang_comparison_var = tk.BooleanVar(value=True)

        # Water mask options (independent controls)
        self.apply_nir_water_mask_var = tk.BooleanVar(value=False)
        self.water_mask_threshold_var = tk.DoubleVar(value=0.01)
        self.water_mask_shapefile_var = tk.StringVar(value="")

        self.enable_advanced_var = tk.BooleanVar(value=True)
        self.trophic_state_var = tk.BooleanVar(value=True)
        self.water_clarity_var = tk.BooleanVar(value=True)
        self.hab_detection_var = tk.BooleanVar(value=True)
        self.upwelling_detection_var = tk.BooleanVar(value=True)
        self.river_plumes_var = tk.BooleanVar(value=True)
        self.particle_size_var = tk.BooleanVar(value=True)
        self.primary_productivity_var = tk.BooleanVar(value=True)

        # Marine visualization variables
        self.enable_marine_viz_var = tk.BooleanVar(value=True)
        self.natural_color_var = tk.BooleanVar(value=True)
        self.false_color_var = tk.BooleanVar(value=True)
        self.water_specific_var = tk.BooleanVar(value=True)
        self.research_rgb_var = tk.BooleanVar(value=False)
        self.water_quality_indices_var = tk.BooleanVar(value=True)
        self.chlorophyll_indices_var = tk.BooleanVar(value=True)
        self.turbidity_indices_var = tk.BooleanVar(value=True)
        self.advanced_indices_var = tk.BooleanVar(value=False)

    def _setup_gui(self):
        """Setup the GUI interface."""
        try:
            # Create main container with padding
            main_container = ttk.Frame(self.root, padding="10")
            main_container.pack(fill=tk.BOTH, expand=True)

            # Title section
            self._setup_title_section(main_container)

            # Create notebook for tabbed interface
            self.notebook = ttk.Notebook(main_container)
            self.notebook.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

            # Create tabs using modular creators
            self.tab_indices['processing'] = create_processing_tab(self, self.notebook)
            self.tab_indices['resampling'] = create_resampling_tab(self, self.notebook)
            self.tab_indices['subset'] = create_subset_tab(self, self.notebook)
            self.tab_indices['c2rcc'] = create_c2rcc_tab(self, self.notebook)
            self.tab_indices['tss'] = create_tss_tab(self, self.notebook)
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
        """Setup title and system info section."""
        title_frame = ttk.Frame(parent)
        title_frame.pack(fill=tk.X, pady=(0, 10))

        # Main title
        title_label = ttk.Label(
            title_frame,
            text="Unified S2 Processing & TSS Estimation Pipeline",
            font=("Arial", 16, "bold")
        )
        title_label.pack()

        # Subtitle
        subtitle_label = ttk.Label(
            title_frame,
            text="Complete pipeline: L1C -> C2RCC (automatic SNAP TSM/CHL) -> Optional Jiang TSS",
            font=("Arial", 10),
            foreground="gray"
        )
        subtitle_label.pack()

    def _setup_status_bar(self, parent):
        """Setup the status bar with progress information."""
        status_frame = ttk.LabelFrame(parent, text="Processing Status", padding="5")
        status_frame.pack(fill=tk.X, pady=(10, 5))

        # Progress bar
        self.progress_bar = ttk.Progressbar(
            status_frame,
            variable=self.progress_var,
            maximum=100,
            mode='determinate'
        )
        self.progress_bar.pack(fill=tk.X, pady=(0, 5))

        # Status info row
        info_frame = ttk.Frame(status_frame)
        info_frame.pack(fill=tk.X)

        ttk.Label(info_frame, text="Status:").pack(side=tk.LEFT)
        self.status_label = ttk.Label(
            info_frame,
            textvariable=self.status_var,
            foreground="blue"
        )
        self.status_label.pack(side=tk.LEFT, padx=(5, 20))

        ttk.Label(info_frame, text="ETA:").pack(side=tk.LEFT)
        self.eta_label = ttk.Label(
            info_frame,
            textvariable=self.eta_var,
            foreground="gray"
        )
        self.eta_label.pack(side=tk.LEFT, padx=5)

    def _setup_control_buttons(self, parent):
        """Setup control buttons."""
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=5)

        # Left side: Config buttons
        left_frame = ttk.Frame(button_frame)
        left_frame.pack(side=tk.LEFT)

        ttk.Button(
            left_frame,
            text="Save Config",
            command=lambda: save_config(self)
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            left_frame,
            text="Load Config",
            command=lambda: load_config(self)
        ).pack(side=tk.LEFT, padx=2)

        # Right side: Processing buttons
        right_frame = ttk.Frame(button_frame)
        right_frame.pack(side=tk.RIGHT)

        self.start_button = ttk.Button(
            right_frame,
            text="Start Processing",
            command=lambda: start_processing(self)
        )
        self.start_button.pack(side=tk.LEFT, padx=2)

        self.stop_button = ttk.Button(
            right_frame,
            text="Stop",
            command=lambda: stop_processing(self),
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=2)

        ttk.Button(
            right_frame,
            text="Exit",
            command=lambda: on_closing(self)
        ).pack(side=tk.LEFT, padx=2)

    # Handler wrapper methods (delegate to module functions)
    def on_mode_change(self):
        """Handle processing mode change."""
        on_mode_change(self)

    def update_tab_visibility(self):
        """Update tab visibility based on processing mode."""
        update_tab_visibility(self)

    def update_subset_visibility(self):
        """Update subset options visibility."""
        update_subset_visibility(self)

    def update_jiang_visibility(self):
        """Update Jiang options visibility."""
        update_jiang_visibility(self)

    def on_ecmwf_toggle(self):
        """Handle ECMWF toggle."""
        on_ecmwf_toggle(self)

    def on_rhow_toggle(self):
        """Handle rhow toggle."""
        on_rhow_toggle(self)

    def validate_input_directory(self, *args):
        """Validate input directory."""
        validate_input_directory(self)

    def validate_geometry(self):
        """Validate geometry input."""
        validate_geometry(self)

    def browse_input_dir(self):
        """Browse for input directory."""
        browse_input_dir(self)

    def browse_output_dir(self):
        """Browse for output directory."""
        browse_output_dir(self)

    def apply_water_preset(self, preset_name):
        """Apply water parameter preset."""
        apply_water_preset(self, preset_name)

    def apply_snap_defaults(self):
        """Apply SNAP default values."""
        apply_snap_defaults(self)

    def apply_essential_outputs(self):
        """Enable essential output products."""
        apply_essential_outputs(self)

    def apply_scientific_outputs(self):
        """Enable all scientific output products."""
        apply_scientific_outputs(self)

    def reset_all_outputs(self):
        """Reset all output options."""
        reset_all_outputs(self)

    def run(self):
        """Run the GUI main loop."""
        self.root.mainloop()


__all__ = [
    'UnifiedS2TSSGUI',
    'bring_window_to_front',
    # Re-exported from geometry_utils for backwards compatibility
    'load_geometry_from_file',
    'validate_wkt_geometry',
    'get_area_name',
    # New names
    'load_geometry',
    'validate_wkt',
    'generate_area_name',
]
