"""
Configuration I/O for Sentinel-2 TSS Pipeline GUI.

Provides save/load functionality for processing configurations.
"""

import json
import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime
from dataclasses import asdict
import logging

from ..config.s2_config import ResamplingConfig, SubsetConfig, C2RCCConfig
from ..config.jiang_config import JiangTSSConfig
from ..config.water_quality_config import WaterQualityConfig
from ..config.marine_config import MarineVisualizationConfig

logger = logging.getLogger('sentinel2_tss_pipeline')


def update_configurations(gui):
    """
    Update configuration objects from GUI variables.

    Args:
        gui: GUI instance with tk variables and config objects.

    Returns:
        bool: True if successful, False if validation failed.
    """
    try:
        # Update resampling config
        gui.resampling_config.target_resolution = gui.resolution_var.get()
        gui.resampling_config.upsampling_method = gui.upsampling_var.get()
        gui.resampling_config.downsampling_method = gui.downsampling_var.get()
        gui.resampling_config.flag_downsampling = gui.flag_downsampling_var.get()
        gui.resampling_config.resample_on_pyramid_levels = gui.pyramid_var.get()

        # Update subset config
        subset_method = gui.subset_method_var.get()

        if subset_method == "geometry":
            geometry_text = gui.geometry_text.get(1.0, tk.END).strip()
            if geometry_text:
                gui.subset_config.geometry_wkt = geometry_text
                gui.subset_config.pixel_start_x = None
                gui.subset_config.pixel_start_y = None
                gui.subset_config.pixel_size_x = None
                gui.subset_config.pixel_size_y = None
            else:
                messagebox.showerror(
                    "Error",
                    "Geometry method selected but no WKT provided!",
                    parent=gui.root
                )
                return False
        elif subset_method == "pixel":
            try:
                start_x = int(gui.pixel_start_x_var.get()) if gui.pixel_start_x_var.get() else None
                start_y = int(gui.pixel_start_y_var.get()) if gui.pixel_start_y_var.get() else None
                width = int(gui.pixel_width_var.get()) if gui.pixel_width_var.get() else None
                height = int(gui.pixel_height_var.get()) if gui.pixel_height_var.get() else None

                if all(v is not None for v in [start_x, start_y, width, height]):
                    gui.subset_config.pixel_start_x = start_x
                    gui.subset_config.pixel_start_y = start_y
                    gui.subset_config.pixel_size_x = width
                    gui.subset_config.pixel_size_y = height
                    gui.subset_config.geometry_wkt = None
                else:
                    messagebox.showerror(
                        "Error",
                        "Pixel method selected but incomplete coordinates!",
                        parent=gui.root
                    )
                    return False
            except ValueError:
                messagebox.showerror(
                    "Error",
                    "Invalid pixel coordinates (must be integers)!",
                    parent=gui.root
                )
                return False
        else:
            # No subset
            gui.subset_config.geometry_wkt = None
            gui.subset_config.pixel_start_x = None
            gui.subset_config.pixel_start_y = None
            gui.subset_config.pixel_size_x = None
            gui.subset_config.pixel_size_y = None

        # Update C2RCC config
        gui.c2rcc_config.salinity = gui.salinity_var.get()
        gui.c2rcc_config.temperature = gui.temperature_var.get()
        gui.c2rcc_config.ozone = gui.ozone_var.get()
        gui.c2rcc_config.pressure = gui.pressure_var.get()
        gui.c2rcc_config.use_ecmwf_aux_data = gui.use_ecmwf_var.get()

        # Neural network, DEM and elevation
        if hasattr(gui, 'net_set_var'):
            gui.c2rcc_config.net_set = gui.net_set_var.get()
        if hasattr(gui, 'dem_name_var'):
            gui.c2rcc_config.dem_name = gui.dem_name_var.get()
        if hasattr(gui, 'elevation_var'):
            gui.c2rcc_config.elevation = gui.elevation_var.get()

        # Output products
        gui.c2rcc_config.output_rhown = gui.output_rhow_var.get()
        gui.c2rcc_config.output_kd = gui.output_kd_var.get()
        gui.c2rcc_config.output_uncertainties = gui.output_uncertainties_var.get()

        # Update Jiang config
        gui.jiang_config.enable_jiang_tss = gui.enable_jiang_var.get()
        gui.jiang_config.output_intermediates = gui.jiang_intermediates_var.get()
        gui.jiang_config.output_comparison_stats = gui.jiang_comparison_var.get()

        # Water mask options
        if hasattr(gui, 'apply_nir_water_mask_var'):
            gui.jiang_config.apply_nir_water_mask = gui.apply_nir_water_mask_var.get()
        if hasattr(gui, 'water_mask_threshold_var'):
            gui.jiang_config.water_mask_threshold = gui.water_mask_threshold_var.get()
        if hasattr(gui, 'water_mask_shapefile_var'):
            shapefile_path = gui.water_mask_shapefile_var.get()
            gui.jiang_config.water_mask_shapefile = shapefile_path if shapefile_path else None

        # Marine visualization
        if hasattr(gui, 'enable_marine_viz_var'):
            gui.jiang_config.enable_marine_visualization = gui.enable_marine_viz_var.get()
            if gui.jiang_config.enable_marine_visualization:
                if gui.jiang_config.marine_viz_config is None:
                    gui.jiang_config.marine_viz_config = MarineVisualizationConfig()
            else:
                gui.jiang_config.marine_viz_config = None

        # Advanced algorithms
        if hasattr(gui, 'enable_advanced_var'):
            gui.jiang_config.enable_advanced_algorithms = gui.enable_advanced_var.get()
            if gui.jiang_config.enable_advanced_algorithms:
                if gui.jiang_config.water_quality_config is None:
                    gui.jiang_config.water_quality_config = WaterQualityConfig()
            else:
                gui.jiang_config.water_quality_config = None

        logger.debug("Configuration updated from GUI")
        return True

    except Exception as e:
        logger.error(f"Configuration update error: {e}")
        messagebox.showerror(
            "Configuration Error",
            f"Failed to update configurations: {str(e)}",
            parent=gui.root
        )
        return False


def save_config(gui):
    """
    Save configuration to JSON file.

    Args:
        gui: GUI instance with configuration objects.
    """
    try:
        if not update_configurations(gui):
            return

        config_file = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            parent=gui.root
        )

        if not config_file:
            return

        config = {
            'processing_mode': gui.processing_mode.get(),
            'resampling': asdict(gui.resampling_config),
            'subset': asdict(gui.subset_config),
            'c2rcc': asdict(gui.c2rcc_config),
            'jiang': _serialize_jiang_config(gui.jiang_config),
            'skip_existing': gui.skip_existing_var.get(),
            'test_mode': gui.test_mode_var.get(),
            'delete_intermediate_files': gui.delete_intermediate_var.get(),
            'memory_limit': int(gui.memory_limit_var.get()),
            'thread_count': int(gui.thread_count_var.get()),
            'saved_at': datetime.now().isoformat()
        }

        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        messagebox.showinfo(
            "Success",
            f"Configuration saved to:\n{config_file}",
            parent=gui.root
        )
        gui.status_var.set("Configuration saved")
        logger.info(f"Configuration saved to: {config_file}")

    except Exception as e:
        messagebox.showerror(
            "Error",
            f"Failed to save configuration: {str(e)}",
            parent=gui.root
        )
        logger.error(f"Failed to save configuration: {e}")


def load_config(gui):
    """
    Load configuration from JSON file.

    Args:
        gui: GUI instance with configuration objects and GUI variables.
    """
    try:
        config_file = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            parent=gui.root
        )

        if not config_file:
            return

        with open(config_file, 'r') as f:
            config = json.load(f)

        logger.info(f"Loading configuration from: {config_file}")

        # Processing mode
        if 'processing_mode' in config:
            gui.processing_mode.set(config['processing_mode'])
            logger.info(f"Processing mode: {config['processing_mode']}")

        # Resampling configuration
        if 'resampling' in config:
            _load_resampling_config(gui, config['resampling'])

        # Subset configuration
        if 'subset' in config:
            _load_subset_config(gui, config['subset'])

        # C2RCC configuration
        if 'c2rcc' in config:
            _load_c2rcc_config(gui, config['c2rcc'])

        # Jiang TSS configuration
        if 'jiang' in config:
            _load_jiang_config(gui, config['jiang'])

        # Processing options
        if 'skip_existing' in config:
            gui.skip_existing_var.set(config['skip_existing'])
        if 'test_mode' in config:
            gui.test_mode_var.set(config['test_mode'])
        if 'delete_intermediate_files' in config:
            gui.delete_intermediate_var.set(config['delete_intermediate_files'])
        if 'memory_limit' in config:
            gui.memory_limit_var.set(config['memory_limit'])
        if 'thread_count' in config:
            gui.thread_count_var.set(config['thread_count'])

        messagebox.showinfo(
            "Success",
            f"Configuration loaded from:\n{config_file}",
            parent=gui.root
        )
        gui.status_var.set("Configuration loaded")
        logger.info("Configuration loaded successfully")

    except Exception as e:
        messagebox.showerror(
            "Error",
            f"Failed to load configuration: {str(e)}",
            parent=gui.root
        )
        logger.error(f"Failed to load configuration: {e}")


def _serialize_jiang_config(jiang_config):
    """Serialize JiangTSSConfig to dictionary."""
    try:
        data = asdict(jiang_config)
        # Handle tuple serialization
        if 'tss_valid_range' in data and isinstance(data['tss_valid_range'], tuple):
            data['tss_valid_range'] = list(data['tss_valid_range'])
        return data
    except Exception:
        # Fallback for configs with non-standard attributes
        return {
            'enable_jiang_tss': getattr(jiang_config, 'enable_jiang_tss', True),
            'output_intermediates': getattr(jiang_config, 'output_intermediates', True),
            'tss_valid_range': list(getattr(jiang_config, 'tss_valid_range', (0.01, 10000))),
            'output_comparison_stats': getattr(jiang_config, 'output_comparison_stats', True),
            # Water mask options
            'apply_nir_water_mask': getattr(jiang_config, 'apply_nir_water_mask', False),
            'water_mask_threshold': getattr(jiang_config, 'water_mask_threshold', 0.01),
            'water_mask_shapefile': getattr(jiang_config, 'water_mask_shapefile', None),
            'enable_advanced_algorithms': getattr(jiang_config, 'enable_advanced_algorithms', True),
            'enable_marine_visualization': getattr(jiang_config, 'enable_marine_visualization', True),
        }


def _load_resampling_config(gui, data):
    """Load resampling configuration from dictionary."""
    gui.resampling_config = ResamplingConfig(
        target_resolution=data.get('target_resolution', '10'),
        upsampling_method=data.get('upsampling_method', 'Bilinear'),
        downsampling_method=data.get('downsampling_method', 'Mean'),
        flag_downsampling=data.get('flag_downsampling', 'First'),
        resample_on_pyramid_levels=data.get('resample_on_pyramid_levels', True)
    )

    # Update GUI variables
    if hasattr(gui, 'resolution_var'):
        gui.resolution_var.set(gui.resampling_config.target_resolution)
    if hasattr(gui, 'upsampling_var'):
        gui.upsampling_var.set(gui.resampling_config.upsampling_method)
    if hasattr(gui, 'downsampling_var'):
        gui.downsampling_var.set(gui.resampling_config.downsampling_method)
    if hasattr(gui, 'flag_downsampling_var'):
        gui.flag_downsampling_var.set(gui.resampling_config.flag_downsampling)
    if hasattr(gui, 'pyramid_var'):
        gui.pyramid_var.set(gui.resampling_config.resample_on_pyramid_levels)

    logger.info(f"Resampling config loaded: {gui.resampling_config.target_resolution}m")


def _load_subset_config(gui, data):
    """Load subset configuration from dictionary."""
    gui.subset_config = SubsetConfig(
        geometry_wkt=data.get('geometry_wkt'),
        sub_sampling_x=data.get('sub_sampling_x', 1),
        sub_sampling_y=data.get('sub_sampling_y', 1),
        full_swath=data.get('full_swath', False),
        copy_metadata=data.get('copy_metadata', True),
        pixel_start_x=data.get('pixel_start_x'),
        pixel_start_y=data.get('pixel_start_y'),
        pixel_size_x=data.get('pixel_size_x'),
        pixel_size_y=data.get('pixel_size_y')
    )

    # Update GUI variables
    if hasattr(gui, 'subset_method_var'):
        if gui.subset_config.geometry_wkt:
            gui.subset_method_var.set("geometry")
            if hasattr(gui, 'geometry_text'):
                gui.geometry_text.delete(1.0, tk.END)
                gui.geometry_text.insert(1.0, gui.subset_config.geometry_wkt)
        elif gui.subset_config.pixel_start_x is not None:
            gui.subset_method_var.set("pixel")
            if hasattr(gui, 'pixel_start_x_var'):
                gui.pixel_start_x_var.set(str(gui.subset_config.pixel_start_x or ''))
            if hasattr(gui, 'pixel_start_y_var'):
                gui.pixel_start_y_var.set(str(gui.subset_config.pixel_start_y or ''))
            if hasattr(gui, 'pixel_width_var'):
                gui.pixel_width_var.set(str(gui.subset_config.pixel_size_x or ''))
            if hasattr(gui, 'pixel_height_var'):
                gui.pixel_height_var.set(str(gui.subset_config.pixel_size_y or ''))
        else:
            gui.subset_method_var.set("none")

    logger.info("Subset config loaded")


def _load_c2rcc_config(gui, data):
    """Load C2RCC configuration from dictionary."""
    gui.c2rcc_config = C2RCCConfig(
        salinity=data.get('salinity', 35.0),
        temperature=data.get('temperature', 15.0),
        ozone=data.get('ozone', 330.0),
        pressure=data.get('pressure', 1000.0),
        elevation=data.get('elevation', 0.0),
        net_set=data.get('net_set', 'C2RCC-Nets'),
        dem_name=data.get('dem_name', 'Copernicus 30m Global DEM'),
        use_ecmwf_aux_data=data.get('use_ecmwf_aux_data', True),
        atmospheric_aux_data_path=data.get('atmospheric_aux_data_path', ''),
        alternative_nn_path=data.get('alternative_nn_path', ''),
        output_as_rrs=data.get('output_as_rrs', True),
        output_rhown=data.get('output_rhown', True),
        output_kd=data.get('output_kd', True),
        output_uncertainties=data.get('output_uncertainties', True),
        output_ac_reflectance=data.get('output_ac_reflectance', True),
        output_rtoa=data.get('output_rtoa', True),
        output_rtosa_gc=data.get('output_rtosa_gc', False),
        output_rtosa_gc_aann=data.get('output_rtosa_gc_aann', False),
        output_rpath=data.get('output_rpath', False),
        output_tdown=data.get('output_tdown', False),
        output_tup=data.get('output_tup', False),
        output_oos=data.get('output_oos', False),
        derive_rw_from_path_and_transmittance=data.get('derive_rw_from_path_and_transmittance', False),
        valid_pixel_expression=data.get('valid_pixel_expression', 'B8 > 0 && B8 < 0.1'),
        threshold_rtosa_oos=data.get('threshold_rtosa_oos', 0.05),
        threshold_ac_reflec_oos=data.get('threshold_ac_reflec_oos', 0.1),
        threshold_cloud_tdown865=data.get('threshold_cloud_tdown865', 0.955),
        tsm_fac=data.get('tsm_fac', 1.06),
        tsm_exp=data.get('tsm_exp', 0.942),
        chl_fac=data.get('chl_fac', 21.0),
        chl_exp=data.get('chl_exp', 1.04)
    )

    # Update GUI variables
    _update_c2rcc_gui_variables(gui)
    logger.info(f"C2RCC config loaded: {gui.c2rcc_config.salinity} PSU, {gui.c2rcc_config.temperature}C")


def _update_c2rcc_gui_variables(gui):
    """Update C2RCC GUI variables from config object."""
    c = gui.c2rcc_config

    # Basic parameters
    if hasattr(gui, 'salinity_var'):
        gui.salinity_var.set(c.salinity)
    if hasattr(gui, 'temperature_var'):
        gui.temperature_var.set(c.temperature)
    if hasattr(gui, 'ozone_var'):
        gui.ozone_var.set(c.ozone)
    if hasattr(gui, 'pressure_var'):
        gui.pressure_var.set(c.pressure)
    if hasattr(gui, 'elevation_var'):
        gui.elevation_var.set(c.elevation)

    # Neural network and DEM
    if hasattr(gui, 'net_set_var'):
        gui.net_set_var.set(c.net_set)
    if hasattr(gui, 'dem_name_var'):
        gui.dem_name_var.set(c.dem_name)
    if hasattr(gui, 'use_ecmwf_var'):
        gui.use_ecmwf_var.set(c.use_ecmwf_aux_data)

    # Output products
    if hasattr(gui, 'output_rrs_var'):
        gui.output_rrs_var.set(c.output_as_rrs)
    if hasattr(gui, 'output_rhow_var'):
        gui.output_rhow_var.set(c.output_rhown)
    if hasattr(gui, 'output_kd_var'):
        gui.output_kd_var.set(c.output_kd)
    if hasattr(gui, 'output_uncertainties_var'):
        gui.output_uncertainties_var.set(c.output_uncertainties)
    if hasattr(gui, 'output_ac_reflectance_var'):
        gui.output_ac_reflectance_var.set(c.output_ac_reflectance)
    if hasattr(gui, 'output_rtoa_var'):
        gui.output_rtoa_var.set(c.output_rtoa)


def _load_jiang_config(gui, data):
    """Load Jiang TSS configuration from dictionary."""
    # Handle tuple conversion
    tss_range = data.get('tss_valid_range', [0.01, 10000])
    if isinstance(tss_range, list):
        tss_range = tuple(tss_range)

    gui.jiang_config = JiangTSSConfig(
        enable_jiang_tss=data.get('enable_jiang_tss', True),
        output_intermediates=data.get('output_intermediates', True),
        tss_valid_range=tss_range,
        output_comparison_stats=data.get('output_comparison_stats', True),
        # Water mask options
        apply_nir_water_mask=data.get('apply_nir_water_mask', False),
        water_mask_threshold=data.get('water_mask_threshold', 0.01),
        water_mask_shapefile=data.get('water_mask_shapefile', None),
        enable_advanced_algorithms=data.get('enable_advanced_algorithms', True)
    )

    # Add marine visualization attributes
    gui.jiang_config.enable_marine_visualization = data.get('enable_marine_visualization', True)

    # Initialize marine_viz_config if enabled
    if gui.jiang_config.enable_marine_visualization:
        gui.jiang_config.marine_viz_config = MarineVisualizationConfig()
    else:
        gui.jiang_config.marine_viz_config = None

    # Initialize water_quality_config if enabled
    if gui.jiang_config.enable_advanced_algorithms:
        if 'water_quality_config' in data and data['water_quality_config']:
            adv_data = data['water_quality_config']
            gui.jiang_config.water_quality_config = WaterQualityConfig(
                enable_water_clarity=adv_data.get('enable_water_clarity', True),
                solar_zenith_angle=adv_data.get('solar_zenith_angle', 30.0),
                enable_hab_detection=adv_data.get('enable_hab_detection', True),
                hab_biomass_threshold=adv_data.get('hab_biomass_threshold', 20.0),
                hab_extreme_threshold=adv_data.get('hab_extreme_threshold', 100.0),
            )
        else:
            gui.jiang_config.water_quality_config = WaterQualityConfig()
    else:
        gui.jiang_config.water_quality_config = None

    # Update GUI variables
    _update_jiang_gui_variables(gui)
    logger.info(f"Jiang TSS config loaded: enabled={gui.jiang_config.enable_jiang_tss}")


def _update_jiang_gui_variables(gui):
    """Update Jiang TSS GUI variables from config object."""
    j = gui.jiang_config

    if hasattr(gui, 'enable_jiang_var'):
        gui.enable_jiang_var.set(j.enable_jiang_tss)
    if hasattr(gui, 'jiang_intermediates_var'):
        gui.jiang_intermediates_var.set(j.output_intermediates)
    if hasattr(gui, 'jiang_comparison_var'):
        gui.jiang_comparison_var.set(j.output_comparison_stats)

    # Water mask options
    if hasattr(gui, 'apply_nir_water_mask_var'):
        gui.apply_nir_water_mask_var.set(getattr(j, 'apply_nir_water_mask', False))
    if hasattr(gui, 'water_mask_threshold_var'):
        gui.water_mask_threshold_var.set(getattr(j, 'water_mask_threshold', 0.01))
    if hasattr(gui, 'water_mask_shapefile_var'):
        shapefile = getattr(j, 'water_mask_shapefile', None)
        gui.water_mask_shapefile_var.set(shapefile if shapefile else "")

    if hasattr(gui, 'enable_marine_viz_var'):
        gui.enable_marine_viz_var.set(getattr(j, 'enable_marine_visualization', True))
    if hasattr(gui, 'enable_advanced_var'):
        gui.enable_advanced_var.set(j.enable_advanced_algorithms)
