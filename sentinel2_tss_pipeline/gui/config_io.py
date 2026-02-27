"""
Configuration I/O for Sentinel-2 TSS Pipeline GUI.

Provides save/load functionality for processing configurations.
Uses TSSConfig and OutputCategoryConfig (replaces JiangTSSConfig).
"""

import json
import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime
from dataclasses import asdict
import logging

from ..config.s2_config import ResamplingConfig, SubsetConfig, C2RCCConfig
from ..config.tss_config import TSSConfig
from ..config.output_categories import OutputCategoryConfig

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
        gui.c2rcc_config.net_set = gui.net_set_var.get()
        gui.c2rcc_config.dem_name = gui.dem_name_var.get()
        gui.c2rcc_config.elevation = gui.elevation_var.get()
        gui.c2rcc_config.output_rhown = gui.output_rhow_var.get()
        gui.c2rcc_config.output_kd = gui.output_kd_var.get()
        gui.c2rcc_config.output_uncertainties = gui.output_uncertainties_var.get()

        # Apply NN presets for auto-threshold adjustment
        gui.c2rcc_config.apply_nn_presets()

        # Update TSS config
        gui.tss_config.enable_tss_processing = gui.enable_jiang_var.get()
        gui.tss_config.output_intermediates = gui.jiang_intermediates_var.get()
        gui.tss_config.output_comparison_stats = gui.jiang_comparison_var.get()
        gui.tss_config.auto_water_mask = gui.auto_water_mask_var.get()
        shapefile_path = gui.water_mask_shapefile_var.get()
        gui.tss_config.water_mask_shapefile = shapefile_path if shapefile_path else None

        # Update output categories
        gui.tss_config.output_categories = OutputCategoryConfig(
            enable_tss=gui.enable_tss_var.get(),
            enable_rgb=gui.enable_rgb_var.get(),
            enable_indices=gui.enable_indices_var.get(),
            enable_water_clarity=gui.enable_water_clarity_var.get(),
            enable_hab=gui.enable_hab_var.get(),
            enable_trophic_state=gui.enable_trophic_state_var.get(),
        )

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
    """Save configuration to JSON file."""
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
            'version': '2.0',
            'processing_mode': gui.processing_mode.get(),
            'resampling': asdict(gui.resampling_config),
            'subset': asdict(gui.subset_config),
            'c2rcc': asdict(gui.c2rcc_config),
            'tss': _serialize_tss_config(gui.tss_config),
            'skip_existing': gui.skip_existing_var.get(),
            'test_mode': gui.test_mode_var.get(),
            'delete_intermediate_files': gui.delete_intermediate_var.get(),
            'memory_limit': int(gui.memory_limit_var.get()),
            'thread_count': int(gui.thread_count_var.get()),
            'saved_at': datetime.now().isoformat(),
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
    """Load configuration from JSON file."""
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

        # Check config version
        version = config.get('version', '1.0')
        if version < '2.0':
            messagebox.showerror(
                "Config Error",
                "This config file uses an outdated format (v1.x).\n"
                "Please reconfigure using the GUI and save a new config.",
                parent=gui.root
            )
            return

        # Processing mode
        if 'processing_mode' in config:
            gui.processing_mode.set(config['processing_mode'])

        # Resampling
        if 'resampling' in config:
            _load_resampling_config(gui, config['resampling'])

        # Subset
        if 'subset' in config:
            _load_subset_config(gui, config['subset'])

        # C2RCC
        if 'c2rcc' in config:
            _load_c2rcc_config(gui, config['c2rcc'])

        # TSS + Output Categories
        if 'tss' in config:
            _load_tss_config(gui, config['tss'])

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


# ===== SERIALIZATION HELPERS =====

def _serialize_tss_config(tss_config):
    """Serialize TSSConfig to dictionary."""
    data = {
        'enable_tss_processing': tss_config.enable_tss_processing,
        'output_intermediates': tss_config.output_intermediates,
        'tss_valid_range': list(tss_config.tss_valid_range),
        'output_comparison_stats': tss_config.output_comparison_stats,
        'auto_water_mask': tss_config.auto_water_mask,
        'water_mask_shapefile': tss_config.water_mask_shapefile,
        'output_categories': asdict(tss_config.output_categories),
    }
    return data


def _load_resampling_config(gui, data):
    """Load resampling configuration from dictionary."""
    gui.resampling_config = ResamplingConfig(
        target_resolution=data.get('target_resolution', '10'),
        upsampling_method=data.get('upsampling_method', 'Bilinear'),
        downsampling_method=data.get('downsampling_method', 'Mean'),
        flag_downsampling=data.get('flag_downsampling', 'First'),
        resample_on_pyramid_levels=data.get('resample_on_pyramid_levels', True),
    )

    gui.resolution_var.set(gui.resampling_config.target_resolution)
    gui.upsampling_var.set(gui.resampling_config.upsampling_method)
    gui.downsampling_var.set(gui.resampling_config.downsampling_method)
    gui.flag_downsampling_var.set(gui.resampling_config.flag_downsampling)
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
        pixel_size_y=data.get('pixel_size_y'),
    )

    if gui.subset_config.geometry_wkt:
        gui.subset_method_var.set("geometry")
        if hasattr(gui, 'geometry_text'):
            gui.geometry_text.delete(1.0, tk.END)
            gui.geometry_text.insert(1.0, gui.subset_config.geometry_wkt)
    elif gui.subset_config.pixel_start_x is not None:
        gui.subset_method_var.set("pixel")
        gui.pixel_start_x_var.set(str(gui.subset_config.pixel_start_x or ''))
        gui.pixel_start_y_var.set(str(gui.subset_config.pixel_start_y or ''))
        gui.pixel_width_var.set(str(gui.subset_config.pixel_size_x or ''))
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
        chl_exp=data.get('chl_exp', 1.04),
    )

    # Update GUI variables
    gui.salinity_var.set(gui.c2rcc_config.salinity)
    gui.temperature_var.set(gui.c2rcc_config.temperature)
    gui.ozone_var.set(gui.c2rcc_config.ozone)
    gui.pressure_var.set(gui.c2rcc_config.pressure)
    gui.elevation_var.set(gui.c2rcc_config.elevation)
    gui.net_set_var.set(gui.c2rcc_config.net_set)
    gui.dem_name_var.set(gui.c2rcc_config.dem_name)
    gui.use_ecmwf_var.set(gui.c2rcc_config.use_ecmwf_aux_data)
    gui.output_rrs_var.set(gui.c2rcc_config.output_as_rrs)
    gui.output_rhow_var.set(gui.c2rcc_config.output_rhown)
    gui.output_kd_var.set(gui.c2rcc_config.output_kd)
    gui.output_uncertainties_var.set(gui.c2rcc_config.output_uncertainties)
    gui.output_ac_reflectance_var.set(gui.c2rcc_config.output_ac_reflectance)
    gui.output_rtoa_var.set(gui.c2rcc_config.output_rtoa)

    logger.info(f"C2RCC config loaded: {gui.c2rcc_config.salinity} PSU, {gui.c2rcc_config.temperature}C")


def _load_tss_config(gui, data):
    """Load TSS configuration and output categories from dictionary."""
    # Handle tuple conversion
    tss_range = data.get('tss_valid_range', [0.01, 10000])
    if isinstance(tss_range, list):
        tss_range = tuple(tss_range)

    # Load output categories
    cat_data = data.get('output_categories', {})
    output_categories = OutputCategoryConfig(
        enable_tss=cat_data.get('enable_tss', True),
        enable_rgb=cat_data.get('enable_rgb', True),
        enable_indices=cat_data.get('enable_indices', True),
        enable_water_clarity=cat_data.get('enable_water_clarity', False),
        enable_hab=cat_data.get('enable_hab', False),
        enable_trophic_state=cat_data.get('enable_trophic_state', False),
    )

    gui.tss_config = TSSConfig(
        enable_tss_processing=data.get('enable_tss_processing', True),
        output_intermediates=data.get('output_intermediates', True),
        tss_valid_range=tss_range,
        output_comparison_stats=data.get('output_comparison_stats', True),
        auto_water_mask=data.get('auto_water_mask', True),
        water_mask_shapefile=data.get('water_mask_shapefile', None),
        output_categories=output_categories,
    )

    # Update GUI variables
    gui.enable_jiang_var.set(gui.tss_config.enable_tss_processing)
    gui.jiang_intermediates_var.set(gui.tss_config.output_intermediates)
    gui.jiang_comparison_var.set(gui.tss_config.output_comparison_stats)
    gui.auto_water_mask_var.set(gui.tss_config.auto_water_mask)
    gui.water_mask_shapefile_var.set(gui.tss_config.water_mask_shapefile or "")

    # Output categories
    gui.enable_tss_var.set(output_categories.enable_tss)
    gui.enable_rgb_var.set(output_categories.enable_rgb)
    gui.enable_indices_var.set(output_categories.enable_indices)
    gui.enable_water_clarity_var.set(output_categories.enable_water_clarity)
    gui.enable_hab_var.set(output_categories.enable_hab)
    gui.enable_trophic_state_var.set(output_categories.enable_trophic_state)

    logger.info(f"TSS config loaded: enabled={gui.tss_config.enable_tss_processing}")
