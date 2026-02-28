"""
Output Structure Manager.

Manages scene-based output folder structure for Sentinel-2 TSS Pipeline.

Output Structure:
    Output/
    ├── S2A_20190105_T29TNF/           # Scene folder
    │   ├── TSS/                        # TSS products (Jiang et al. 2021)
    │   ├── RGB/                        # RGB composites (15 unique)
    │   ├── Indices/                    # Spectral indices (13 indices)
    │   ├── WaterClarity/              # Water clarity products
    │   ├── HAB/                        # Harmful algal bloom detection
    │   └── TrophicState/              # Trophic state index (Carlson 1977)
    ├── Intermediate/                   # Deletable after processing
    │   ├── C2RCC/
    │   └── Geometric/
    └── Logs/
"""

import os
import re
import logging
from typing import Optional, Dict

logger = logging.getLogger('sentinel2_tss_pipeline')


class OutputStructure:
    """Manages scene-based output folder structure."""

    # Scene-level output category folders
    TSS_FOLDER = "TSS"
    RGB_FOLDER = "RGB"
    INDICES_FOLDER = "Indices"
    WATER_CLARITY_FOLDER = "WaterClarity"
    HAB_FOLDER = "HAB"
    TROPHIC_STATE_FOLDER = "TrophicState"

    # Infrastructure folders
    INTERMEDIATE_FOLDER = "Intermediate"
    C2RCC_FOLDER = "C2RCC"
    GEOMETRIC_FOLDER = "Geometric"
    IOP_FOLDER = "IOP"
    LOGS_FOLDER = "Logs"

    # RGB band mappings (wavelength -> band number)
    WAVELENGTH_TO_BAND = {
        443: '1',    # B1 - Coastal aerosol
        490: '2',    # B2 - Blue
        560: '3',    # B3 - Green
        665: '4',    # B4 - Red
        705: '5',    # B5 - Red Edge 1
        740: '6',    # B6 - Red Edge 2
        783: '7',    # B7 - Red Edge 3
        842: '8',    # B8 - NIR (broad)
        865: '8A',   # B8A - NIR (narrow)
        945: '9',    # B9 - Water vapour
        1375: '10',  # B10 - SWIR Cirrus
        1610: '11',  # B11 - SWIR 1
        2190: '12'   # B12 - SWIR 2
    }

    @staticmethod
    def extract_clean_scene_name(product_path: str) -> str:
        """
        Extract clean scene name from input path.

        Converts full S2 product name to short form:
        S2A_MSIL1C_20190105T103401_N0207_R108_T29TNF_20190105T123456
        -> S2A_20190105_T29TNF

        Args:
            product_path: Full path to input product

        Returns:
            Clean scene name (e.g., S2A_20190105_T29TNF)
        """
        basename = os.path.basename(product_path)

        # Remove extensions
        for ext in ['.zip', '.SAFE', '.dim', '.data']:
            basename = basename.replace(ext, '')

        # Remove common prefixes from processing
        for prefix in ['Resampled_', 'Subset_', 'C2RCC_']:
            if basename.startswith(prefix):
                basename = basename[len(prefix):]

        # Remove suffixes
        for suffix in ['_Subset_C2RCC', '_C2RCC', '_Subset']:
            if suffix in basename:
                basename = basename.replace(suffix, '')

        # Try to extract components from full S2 name
        # Pattern: S2A_MSIL1C_20190105T103401_N0207_R108_T29TNF_20190105T123456
        match = re.match(
            r'(S2[AB])_MSI[^_]+_(\d{8})T\d+_[^_]+_[^_]+_(T\d{2}[A-Z]{3}).*',
            basename
        )

        if match:
            satellite, date, tile = match.groups()
            return f"{satellite}_{date}_{tile}"

        # Fallback: return as-is if already clean or unrecognized format
        return basename

    @staticmethod
    def get_scene_folder(output_root: str, scene_name: str) -> str:
        """
        Get or create scene folder.

        Args:
            output_root: Root output directory
            scene_name: Clean scene name (e.g., S2A_20190105_T29TNF)

        Returns:
            Full path to scene folder
        """
        scene_folder = os.path.join(output_root, scene_name)
        os.makedirs(scene_folder, exist_ok=True)
        return scene_folder

    @staticmethod
    def get_category_folder(scene_folder: str, category: str) -> str:
        """
        Get or create category subfolder within scene folder.

        Args:
            scene_folder: Path to scene folder
            category: Category name (TSS, SNAP, RGB, Indices, Advanced)

        Returns:
            Full path to category folder
        """
        category_folder = os.path.join(scene_folder, category)
        os.makedirs(category_folder, exist_ok=True)
        return category_folder

    @staticmethod
    def get_intermediate_folder(output_root: str, category: str) -> str:
        """
        Get or create intermediate folder for C2RCC/Geometric products.

        Args:
            output_root: Root output directory
            category: Category (C2RCC or Geometric)

        Returns:
            Full path to intermediate folder
        """
        intermediate_folder = os.path.join(
            output_root,
            OutputStructure.INTERMEDIATE_FOLDER,
            category
        )
        os.makedirs(intermediate_folder, exist_ok=True)
        return intermediate_folder

    @staticmethod
    def get_logs_folder(output_root: str) -> str:
        """
        Get or create logs folder.

        Args:
            output_root: Root output directory

        Returns:
            Full path to logs folder
        """
        logs_folder = os.path.join(output_root, OutputStructure.LOGS_FOLDER)
        os.makedirs(logs_folder, exist_ok=True)
        return logs_folder

    @staticmethod
    def get_product_filename(scene_name: str, product_type: str,
                             suffix: Optional[str] = None) -> str:
        """
        Generate product filename with scene name.

        Args:
            scene_name: Clean scene name (e.g., S2A_20190105_T29TNF)
            product_type: Product type (TSS, WaterTypes, NDWI, etc.)
            suffix: Optional suffix (e.g., band combo for RGB)

        Returns:
            Filename (e.g., S2A_20190105_T29TNF_TSS.tif)
        """
        if suffix:
            return f"{scene_name}_{product_type}_{suffix}.tif"
        return f"{scene_name}_{product_type}.tif"

    @staticmethod
    def get_rgb_filename(scene_name: str, red_wl: int, green_wl: int,
                         blue_wl: int) -> str:
        """
        Generate RGB filename with band combination.

        Args:
            scene_name: Clean scene name
            red_wl: Red channel wavelength (nm)
            green_wl: Green channel wavelength (nm)
            blue_wl: Blue channel wavelength (nm)

        Returns:
            Filename (e.g., S2A_20190105_T29TNF_RGB_432.tif)
        """
        band_map = OutputStructure.WAVELENGTH_TO_BAND

        red_band = band_map.get(red_wl, str(red_wl))
        green_band = band_map.get(green_wl, str(green_wl))
        blue_band = band_map.get(blue_wl, str(blue_wl))

        band_combo = f"{red_band}{green_band}{blue_band}"
        return f"{scene_name}_RGB_{band_combo}.tif"

    @staticmethod
    def get_full_product_path(output_root: str, scene_name: str,
                              category: str, product_type: str,
                              suffix: Optional[str] = None,
                              subcategory: Optional[str] = None) -> str:
        """
        Get full path for a product file.

        Args:
            output_root: Root output directory
            scene_name: Clean scene name
            category: Category (TSS, RGB, Indices, WaterClarity, HAB, TrophicState)
            product_type: Product type (TSS, NDWI, etc.)
            suffix: Optional suffix for filename
            subcategory: Optional subcategory folder (e.g., HAB, IOP)

        Returns:
            Full path to product file
        """
        scene_folder = OutputStructure.get_scene_folder(output_root, scene_name)
        category_folder = OutputStructure.get_category_folder(scene_folder, category)

        if subcategory:
            category_folder = os.path.join(category_folder, subcategory)
            os.makedirs(category_folder, exist_ok=True)

        filename = OutputStructure.get_product_filename(scene_name, product_type, suffix)
        return os.path.join(category_folder, filename)

    @staticmethod
    def get_full_rgb_path(output_root: str, scene_name: str,
                          red_wl: int, green_wl: int, blue_wl: int) -> str:
        """
        Get full path for an RGB composite.

        Args:
            output_root: Root output directory
            scene_name: Clean scene name
            red_wl: Red channel wavelength (nm)
            green_wl: Green channel wavelength (nm)
            blue_wl: Blue channel wavelength (nm)

        Returns:
            Full path to RGB file
        """
        scene_folder = OutputStructure.get_scene_folder(output_root, scene_name)
        rgb_folder = OutputStructure.get_category_folder(
            scene_folder, OutputStructure.RGB_FOLDER
        )
        filename = OutputStructure.get_rgb_filename(scene_name, red_wl, green_wl, blue_wl)
        return os.path.join(rgb_folder, filename)

    @staticmethod
    def create_scene_structure(output_root: str, scene_name: str,
                               enable_water_clarity: bool = False,
                               enable_hab: bool = False,
                               enable_trophic_state: bool = False) -> Dict[str, str]:
        """
        Create complete scene folder structure.

        Args:
            output_root: Root output directory
            scene_name: Clean scene name
            enable_water_clarity: Create WaterClarity folder
            enable_hab: Create HAB folder
            enable_trophic_state: Create TrophicState folder

        Returns:
            Dictionary of folder paths
        """
        folders = {}

        scene_folder = OutputStructure.get_scene_folder(output_root, scene_name)
        folders['scene'] = scene_folder

        # Always create core output folders
        for category in [OutputStructure.TSS_FOLDER,
                         OutputStructure.RGB_FOLDER,
                         OutputStructure.INDICES_FOLDER]:
            folders[category.lower()] = OutputStructure.get_category_folder(
                scene_folder, category
            )

        # Create optional category folders based on OutputCategoryConfig
        if enable_water_clarity:
            folders['waterclarity'] = OutputStructure.get_category_folder(
                scene_folder, OutputStructure.WATER_CLARITY_FOLDER
            )

        if enable_hab:
            folders['hab'] = OutputStructure.get_category_folder(
                scene_folder, OutputStructure.HAB_FOLDER
            )

        if enable_trophic_state:
            folders['trophicstate'] = OutputStructure.get_category_folder(
                scene_folder, OutputStructure.TROPHIC_STATE_FOLDER
            )

        logger.debug(f"Created scene structure for {scene_name}")
        return folders
