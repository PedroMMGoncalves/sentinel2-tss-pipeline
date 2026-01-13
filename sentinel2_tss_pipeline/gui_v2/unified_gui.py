"""
Unified GUI v2 for Sentinel-2 TSS Pipeline.

Improved version with modern styling, collapsible sections,
tooltips, and better layout organization.

All processing options are exposed with clear, descriptive labels.
"""

import sys
import os
import logging
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

logger = logging.getLogger('sentinel2_tss_pipeline')

# Import theme
from .theme import ThemeManager

# Import widgets
from .widgets import CollapsibleFrame, CheckboxGroup, Tooltip, create_tooltip

# Import modular tabs (updated imports)
from .tabs import (
    create_processing_tab,
    create_spatial_tab,
    create_c2rcc_tab,
    create_tss_outputs_tab,
    create_monitoring_tab,
)

# Import configuration classes from main package
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


class UnifiedS2TSSGUI:
    """
    Unified GUI v2 for Sentinel-2 TSS Pipeline.

    Features:
    - Complete exposure of all processing options
    - Clear, descriptive labels for every parameter
    - Logical organization by processing workflow stage
    - Modern styling with collapsible sections
    """

    def __init__(self):
        """Initialize the GUI."""
        self.root = tk.Tk()
        self.root.title("Sentinel-2 TSS Pipeline v2.0")

        # Apply theme
        self.theme = ThemeManager(self.root)

        # Window setup
        self._setup_window()

        # Initialize configurations
        self._init_configurations()

        # Initialize state variables
        self._init_state_variables()

        # Initialize ALL tk variables
        self._init_tk_variables()

        # Build GUI
        self._setup_gui()

        logger.info("GUI v2 initialization complete")

    def _setup_window(self):
        """Configure main window."""
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        window_width = 1100
        window_height = max(900, int(screen_height * 0.85))

        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2

        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.configure(bg=self.theme.COLORS['bg_main'])
        self.root.minsize(1000, 800)
        self.root.resizable(True, True)

        self.root.lift()
        self.root.focus_force()

    def _init_configurations(self):
        """Initialize configuration objects."""
        self.resampling_config = ResamplingConfig()
        self.subset_config = SubsetConfig()
        self.c2rcc_config = C2RCCConfig()
        self.jiang_config = JiangTSSConfig()

        if not hasattr(self.jiang_config, 'marine_viz_config'):
            self.jiang_config.marine_viz_config = MarineVisualizationConfig()
        if not hasattr(self.jiang_config, 'water_quality_config'):
            self.jiang_config.water_quality_config = WaterQualityConfig()

    def _init_state_variables(self):
        """Initialize state variables."""
        self.input_validation_result = {"valid": False, "message": "", "products": []}
        self.system_monitor = SystemMonitor()
        self.system_monitor.start_monitoring()
        self.processor = None
        self.processing_thread = None
        self.processing_active = False
        self.tab_indices = {}

    def _init_tk_variables(self):
        """Initialize ALL tkinter variables for complete GUI control."""

        # =====================================================
        # PROCESSING TAB
        # =====================================================
        self.processing_mode = tk.StringVar(value="complete_pipeline")
        self.input_dir_var = tk.StringVar()
        self.output_dir_var = tk.StringVar()
        self.skip_existing_var = tk.BooleanVar(value=True)
        self.test_mode_var = tk.BooleanVar(value=False)
        self.memory_limit_var = tk.StringVar(value="8")
        self.thread_count_var = tk.StringVar(value="4")

        # Progress tracking
        self.progress_var = tk.DoubleVar()
        self.status_var = tk.StringVar(value="Ready")
        self.eta_var = tk.StringVar(value="")

        # =====================================================
        # SPATIAL TAB (Resampling + Subset + Mask)
        # =====================================================

        # --- Resampling ---
        self.resolution_var = tk.StringVar(value="10")
        self.upsampling_var = tk.StringVar(value="Bilinear")
        self.downsampling_var = tk.StringVar(value="Mean")
        self.flag_downsampling_var = tk.StringVar(value="First")
        self.pyramid_var = tk.BooleanVar(value=True)

        # --- Spatial Subset ---
        self.enable_subset_var = tk.BooleanVar(value=False)
        self.subset_method_var = tk.StringVar(value="none")  # none, geometry, bbox, pixels
        self.geometry_file_var = tk.StringVar()
        self.geometry_wkt_var = tk.StringVar()
        self.bbox_north_var = tk.StringVar()
        self.bbox_south_var = tk.StringVar()
        self.bbox_east_var = tk.StringVar()
        self.bbox_west_var = tk.StringVar()
        self.pixel_start_x_var = tk.StringVar(value="0")
        self.pixel_start_y_var = tk.StringVar(value="0")
        self.pixel_size_x_var = tk.StringVar(value="1000")
        self.pixel_size_y_var = tk.StringVar(value="1000")

        # --- Processing Mask ---
        self.valid_pixel_expression_var = tk.StringVar(value="B8 > 0 && B8 < 0.1")
        self.water_mask_threshold_var = tk.DoubleVar(value=0.01)

        # =====================================================
        # C2RCC TAB (Atmospheric Correction)
        # =====================================================

        # --- Neural Network & DEM (NEW) ---
        self.neural_network_var = tk.StringVar(value="C2RCC-Nets")
        self.dem_var = tk.StringVar(value="Copernicus 30m Global DEM")
        self.elevation_var = tk.DoubleVar(value=0.0)
        self.alternative_nn_path_var = tk.StringVar()

        # --- Atmospheric Parameters ---
        self.use_ecmwf_var = tk.BooleanVar(value=True)
        self.salinity_var = tk.DoubleVar(value=35.0)
        self.temperature_var = tk.DoubleVar(value=15.0)
        self.ozone_var = tk.DoubleVar(value=330.0)
        self.pressure_var = tk.DoubleVar(value=1000.0)

        # --- TSM/CHL Coefficients ---
        self.tsm_fac_var = tk.DoubleVar(value=1.06)
        self.tsm_exp_var = tk.DoubleVar(value=0.942)
        self.chl_fac_var = tk.DoubleVar(value=21.0)
        self.chl_exp_var = tk.DoubleVar(value=1.04)

        # --- C2RCC Output Products (Essential - always on) ---
        self.output_rrs_var = tk.BooleanVar(value=True)
        self.output_rhow_var = tk.BooleanVar(value=True)
        self.output_kd_var = tk.BooleanVar(value=True)

        # --- C2RCC Output Products (Optional) ---
        self.output_uncertainties_var = tk.BooleanVar(value=True)
        self.output_rtoa_var = tk.BooleanVar(value=False)
        self.output_ac_reflectance_var = tk.BooleanVar(value=False)

        # --- C2RCC Output Products (IOPs) ---
        self.output_iop_var = tk.BooleanVar(value=False)  # iop_adet
        self.output_agelb_var = tk.BooleanVar(value=False)  # iop_agelb (CDOM)

        # --- C2RCC Output Products (Advanced Atmospheric) ---
        self.output_rpath_var = tk.BooleanVar(value=False)
        self.output_rtosa_gc_var = tk.BooleanVar(value=False)
        self.output_rtosa_gc_aann_var = tk.BooleanVar(value=False)
        self.output_tdown_var = tk.BooleanVar(value=False)
        self.output_tup_var = tk.BooleanVar(value=False)
        self.output_oos_var = tk.BooleanVar(value=False)
        self.output_flags_var = tk.BooleanVar(value=False)  # c2rcc_flags

        # =====================================================
        # TSS & OUTPUTS TAB
        # =====================================================

        # --- Jiang TSS ---
        self.enable_jiang_var = tk.BooleanVar(value=True)
        self.enable_jiang_tss_var = self.enable_jiang_var  # Alias
        self.jiang_intermediates_var = tk.BooleanVar(value=True)
        self.jiang_comparison_var = tk.BooleanVar(value=False)
        self.compare_snap_tsm_var = self.jiang_comparison_var  # Alias

        # --- Jiang TSS Output Products ---
        self.output_tss_var = tk.BooleanVar(value=True)
        self.output_water_types_var = tk.BooleanVar(value=True)
        self.output_absorption_var = tk.BooleanVar(value=True)
        self.output_backscattering_var = tk.BooleanVar(value=True)

        # --- Marine Visualization Master Toggle ---
        self.enable_marine_viz_var = tk.BooleanVar(value=True)

        # --- RGB Composites (Individual Controls) ---
        # Standard Visualization
        self.rgb_true_color_var = tk.BooleanVar(value=True)
        self.rgb_false_color_infrared_var = tk.BooleanVar(value=True)
        self.rgb_enhanced_contrast_var = tk.BooleanVar(value=False)
        self.rgb_false_color_nir_var = tk.BooleanVar(value=False)
        self.rgb_natural_color_var = tk.BooleanVar(value=False)

        # Water Quality
        self.rgb_turbidity_enhanced_var = tk.BooleanVar(value=True)
        self.rgb_chlorophyll_enhanced_var = tk.BooleanVar(value=True)
        self.rgb_coastal_aerosol_var = tk.BooleanVar(value=False)
        self.rgb_cyanobacteria_var = tk.BooleanVar(value=False)

        # Specialized
        self.rgb_sediment_transport_var = tk.BooleanVar(value=False)
        self.rgb_bathymetric_var = tk.BooleanVar(value=False)
        self.rgb_ocean_color_standard_var = tk.BooleanVar(value=False)
        self.rgb_ocean_color_var = self.rgb_ocean_color_standard_var  # Alias
        self.rgb_water_quality_var = tk.BooleanVar(value=False)
        self.rgb_atmospheric_penetration_var = tk.BooleanVar(value=False)
        self.rgb_swir_nir_var = tk.BooleanVar(value=False)
        self.rgb_geology_var = tk.BooleanVar(value=False)

        # Research
        self.rgb_algal_bloom_var = tk.BooleanVar(value=False)
        self.rgb_coastal_turbidity_var = tk.BooleanVar(value=False)
        self.rgb_deep_water_clarity_var = tk.BooleanVar(value=False)
        self.rgb_riverine_waters_var = tk.BooleanVar(value=False)
        self.rgb_cdom_enhanced_var = tk.BooleanVar(value=False)
        self.rgb_water_change_detection_var = tk.BooleanVar(value=False)
        self.rgb_research_marine_var = tk.BooleanVar(value=False)

        # --- Spectral Indices (Individual Controls) ---
        # Water Detection
        self.idx_ndwi_var = tk.BooleanVar(value=True)
        self.idx_mndwi_var = tk.BooleanVar(value=True)
        self.idx_awei_var = tk.BooleanVar(value=False)
        self.idx_wri_var = tk.BooleanVar(value=False)
        self.idx_wi_var = tk.BooleanVar(value=False)
        self.idx_ndmi_var = tk.BooleanVar(value=False)

        # Chlorophyll & Algae
        self.idx_ndci_var = tk.BooleanVar(value=True)
        self.idx_gndvi_var = tk.BooleanVar(value=False)
        self.idx_fai_var = tk.BooleanVar(value=False)
        self.idx_flh_var = tk.BooleanVar(value=False)  # Fluorescence Line Height
        self.idx_mci_var = tk.BooleanVar(value=False)  # Maximum Chlorophyll Index
        self.idx_pc_var = tk.BooleanVar(value=False)
        self.idx_chl_red_edge_var = tk.BooleanVar(value=False)

        # Turbidity & Sediment
        self.idx_ndti_var = tk.BooleanVar(value=True)
        self.idx_tsi_var = tk.BooleanVar(value=False)
        self.idx_ngrdi_var = tk.BooleanVar(value=False)

        # Advanced Properties
        self.idx_fui_var = tk.BooleanVar(value=False)
        self.idx_sdd_var = tk.BooleanVar(value=False)
        self.idx_cdom_var = tk.BooleanVar(value=False)

        # --- Advanced Aquatic Algorithms ---
        # Water Clarity
        self.enable_water_clarity_var = tk.BooleanVar(value=False)
        self.enable_secchi_depth_var = tk.BooleanVar(value=False)
        self.enable_euphotic_depth_var = tk.BooleanVar(value=False)
        self.enable_kd_mapping_var = tk.BooleanVar(value=False)
        self.solar_zenith_angle_var = tk.DoubleVar(value=30.0)

        # HAB Detection
        self.enable_hab_detection_var = tk.BooleanVar(value=False)
        self.enable_cyanobacteria_map_var = tk.BooleanVar(value=False)
        self.hab_cyano_map_var = self.enable_cyanobacteria_map_var  # Alias
        self.enable_bloom_risk_var = tk.BooleanVar(value=False)
        self.hab_risk_class_var = self.enable_bloom_risk_var  # Alias
        self.enable_biomass_alerts_var = tk.BooleanVar(value=False)
        self.hab_alert_flags_var = self.enable_biomass_alerts_var  # Alias
        self.hab_biomass_threshold_var = tk.DoubleVar(value=20.0)
        self.hab_extreme_threshold_var = tk.DoubleVar(value=100.0)

        # Trophic State Index
        self.enable_tsi_var = tk.BooleanVar(value=False)
        self.tsi_include_secchi_var = tk.BooleanVar(value=False)
        self.tsi_include_phosphorus_var = tk.BooleanVar(value=False)

        # Oceanographic Features
        self.enable_upwelling_var = tk.BooleanVar(value=False)
        self.upwelling_chl_threshold_var = tk.DoubleVar(value=10.0)
        self.enable_river_plume_var = tk.BooleanVar(value=False)
        self.enable_river_plumes_var = self.enable_river_plume_var  # Alias
        self.plume_tss_threshold_var = tk.DoubleVar(value=15.0)
        self.river_plume_tss_threshold_var = self.plume_tss_threshold_var  # Alias
        self.enable_particle_size_var = tk.BooleanVar(value=False)
        self.enable_primary_productivity_var = tk.BooleanVar(value=False)

        # Legacy compatibility (keep for backward compat)
        self.natural_color_var = self.rgb_true_color_var
        self.false_color_var = self.rgb_false_color_infrared_var
        self.water_specific_var = self.rgb_turbidity_enhanced_var
        self.research_rgb_var = self.rgb_research_marine_var
        self.water_quality_indices_var = self.idx_ndwi_var
        self.chlorophyll_indices_var = self.idx_ndci_var
        self.turbidity_indices_var = self.idx_ndti_var
        self.advanced_indices_var = self.idx_fui_var
        self.enable_advanced_var = self.enable_water_clarity_var

    def _setup_gui(self):
        """Build the GUI with 5 tabs."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        self._create_title(main_frame)

        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        # Create 5 tabs
        self.tab_indices['processing'] = create_processing_tab(self, self.notebook)
        self.tab_indices['spatial'] = create_spatial_tab(self, self.notebook)
        self.tab_indices['c2rcc'] = create_c2rcc_tab(self, self.notebook)
        self.tab_indices['tss_outputs'] = create_tss_outputs_tab(self, self.notebook)
        self.tab_indices['monitoring'] = create_monitoring_tab(self, self.notebook)

        self._create_status_bar(main_frame)
        self._create_control_buttons(main_frame)

    def _create_title(self, parent):
        """Create title section."""
        title_frame = ttk.Frame(parent)
        title_frame.pack(fill=tk.X)

        ttk.Label(
            title_frame,
            text="Sentinel-2 TSS Pipeline",
            style='Title.TLabel'
        ).pack()

        ttk.Label(
            title_frame,
            text="L1C → C2RCC → TSS Estimation",
            style='Muted.TLabel'
        ).pack()

    def _create_status_bar(self, parent):
        """Create status bar."""
        status_frame = ttk.LabelFrame(parent, text="Status", padding="5")
        status_frame.pack(fill=tk.X, pady=(10, 5))

        self.progress_bar = ttk.Progressbar(
            status_frame,
            variable=self.progress_var,
            maximum=100,
            mode='determinate',
            style='TProgressbar'
        )
        self.progress_bar.pack(fill=tk.X, pady=(0, 5), ipady=3)

        info_frame = ttk.Frame(status_frame)
        info_frame.pack(fill=tk.X)

        ttk.Label(info_frame, text="Status:").pack(side=tk.LEFT)
        ttk.Label(
            info_frame,
            textvariable=self.status_var,
            style='Info.TLabel'
        ).pack(side=tk.LEFT, padx=5)

        ttk.Label(info_frame, text="ETA:").pack(side=tk.LEFT, padx=(20, 0))
        ttk.Label(
            info_frame,
            textvariable=self.eta_var,
            style='Muted.TLabel'
        ).pack(side=tk.LEFT, padx=5)

    def _create_control_buttons(self, parent):
        """Create control buttons."""
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=5)

        left_frame = ttk.Frame(button_frame)
        left_frame.pack(side=tk.LEFT)

        ttk.Button(
            left_frame,
            text="Save Config",
            command=self._save_config
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            left_frame,
            text="Load Config",
            command=self._load_config
        ).pack(side=tk.LEFT, padx=2)

        right_frame = ttk.Frame(button_frame)
        right_frame.pack(side=tk.RIGHT)

        self.start_button = ttk.Button(
            right_frame,
            text="Start Processing",
            style='Primary.TButton',
            command=self._start_processing
        )
        self.start_button.pack(side=tk.LEFT, padx=2)

        self.stop_button = ttk.Button(
            right_frame,
            text="Stop",
            state=tk.DISABLED,
            command=self._stop_processing
        )
        self.stop_button.pack(side=tk.LEFT, padx=2)

        ttk.Button(
            right_frame,
            text="Exit",
            command=self.root.quit
        ).pack(side=tk.LEFT, padx=2)

    # === Handler Methods ===

    def validate_input(self):
        """Validate input directory and update label."""
        input_dir = self.input_dir_var.get()
        if not input_dir:
            self.input_validation_label.configure(
                text="No input directory selected",
                style='Warning.TLabel'
            )
            return

        if not os.path.isdir(input_dir):
            self.input_validation_label.configure(
                text="Invalid directory",
                style='Warning.TLabel'
            )
            return

        try:
            from ..utils.product_detector import ProductDetector
            detector = ProductDetector()
            products = detector.find_products(input_dir, self.processing_mode.get())
            count = len(products)

            if count > 0:
                self.input_validation_label.configure(
                    text=f"Found {count} product(s)",
                    style='Success.TLabel'
                )
                self.input_validation_result = {
                    "valid": True,
                    "message": f"Found {count} products",
                    "products": products
                }
            else:
                self.input_validation_label.configure(
                    text="No valid products found",
                    style='Warning.TLabel'
                )
                self.input_validation_result = {
                    "valid": False,
                    "message": "No products",
                    "products": []
                }
        except Exception as e:
            self.input_validation_label.configure(
                text=f"Error: {str(e)[:30]}",
                style='Warning.TLabel'
            )
            logger.warning(f"Input validation error: {e}")

    def browse_input_dir(self):
        """Browse for input directory."""
        directory = filedialog.askdirectory(
            title="Select Input Directory",
            initialdir=self.input_dir_var.get() or os.path.expanduser("~")
        )
        if directory:
            self.input_dir_var.set(directory)
            self.validate_input()

    def browse_output_dir(self):
        """Browse for output directory."""
        directory = filedialog.askdirectory(
            title="Select Output Directory",
            initialdir=self.output_dir_var.get() or os.path.expanduser("~")
        )
        if directory:
            self.output_dir_var.set(directory)

    def browse_geometry_file(self):
        """Browse for geometry file."""
        filepath = filedialog.askopenfilename(
            title="Select Geometry File",
            filetypes=[
                ("All supported", "*.shp;*.kml;*.geojson;*.json"),
                ("Shapefiles", "*.shp"),
                ("KML files", "*.kml"),
                ("GeoJSON files", "*.geojson;*.json"),
                ("All files", "*.*")
            ]
        )
        if filepath:
            self.geometry_file_var.set(filepath)

    def _save_config(self):
        """Save configuration to file."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save Configuration"
        )
        if filepath:
            try:
                import json
                config = self._collect_config()
                with open(filepath, 'w') as f:
                    json.dump(config, f, indent=2)
                logger.info(f"Configuration saved to: {filepath}")
                messagebox.showinfo("Success", f"Configuration saved to:\n{filepath}")
            except Exception as e:
                logger.error(f"Failed to save config: {e}")
                messagebox.showerror("Error", f"Failed to save configuration:\n{e}")

    def _load_config(self):
        """Load configuration from file."""
        filepath = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Load Configuration"
        )
        if filepath:
            try:
                import json
                with open(filepath, 'r') as f:
                    config = json.load(f)
                self._apply_config(config)
                logger.info(f"Configuration loaded from: {filepath}")
                messagebox.showinfo("Success", f"Configuration loaded from:\n{filepath}")
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
                messagebox.showerror("Error", f"Failed to load configuration:\n{e}")

    def _collect_config(self):
        """Collect all configuration values."""
        # Get all tk variables that have a .get() method
        config = {}
        for attr_name in dir(self):
            if attr_name.endswith('_var'):
                attr = getattr(self, attr_name)
                if hasattr(attr, 'get'):
                    config[attr_name] = attr.get()
        return config

    def _apply_config(self, config):
        """Apply loaded configuration values."""
        for key, value in config.items():
            if hasattr(self, key):
                attr = getattr(self, key)
                if hasattr(attr, 'set'):
                    try:
                        attr.set(value)
                    except Exception:
                        pass  # Skip incompatible values

    def _start_processing(self):
        """Start processing."""
        if not self.input_dir_var.get():
            messagebox.showwarning("Warning", "Please select an input directory.")
            return

        if not self.output_dir_var.get():
            messagebox.showwarning("Warning", "Please select an output directory.")
            return

        self.processing_active = True
        self.start_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.status_var.set("Starting...")

        # TODO: Connect to actual processing
        logger.info("Processing started")
        messagebox.showinfo("Info", "Processing will be connected in next phase.")

        self.processing_active = False
        self.start_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)
        self.status_var.set("Ready")

    def _stop_processing(self):
        """Stop processing."""
        self.processing_active = False
        self.start_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)
        self.status_var.set("Stopped")
        logger.info("Processing stopped by user")

    def run(self):
        """Run the GUI main loop."""
        self.root.mainloop()
