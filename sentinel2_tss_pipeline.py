import os
import warnings

# Fix PROJ database conflict
def fix_proj_database_comprehensive():
    """
    Comprehensive fix for PROJ database issues
    This resolves the "no database context specified" error
    """
    print("🔧 Fixing PROJ database configuration...")
    
    # Method 1: Set PROJ_DATA environment variable
    proj_data_paths = []
    
    # Check conda environment first
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        conda_proj = os.path.join(conda_prefix, 'share', 'proj')
        if os.path.exists(conda_proj):
            proj_data_paths.append(conda_proj)
    
    # Check common PROJ installation paths
    common_paths = [
        # Conda-forge typical locations
        os.path.join(sys.prefix, 'share', 'proj'),
        os.path.join(sys.prefix, 'Library', 'share', 'proj'),  # Windows conda
        # System installations
        '/usr/share/proj',
        '/usr/local/share/proj',
        # OSGeo4W (Windows)
        'C:/OSGeo4W64/share/proj',
        'C:/OSGeo4W/share/proj',
        # QGIS installations
        'C:\Program Files\QGIS 3.42.3\share\proj',
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            proj_data_paths.append(path)
    
    # Set PROJ_DATA to the first valid path found
    if proj_data_paths:
        proj_data = proj_data_paths[0]
        os.environ['PROJ_DATA'] = proj_data
        print(f"✓ PROJ_DATA set to: {proj_data}")
    else:
        print("❌ No PROJ data directory found")
        return False
    
    # Method 2: Set additional PROJ environment variables
    proj_lib_paths = []
    
    if conda_prefix:
        conda_lib = os.path.join(conda_prefix, 'lib')
        if os.path.exists(conda_lib):
            proj_lib_paths.append(conda_lib)
    
    # Common library paths
    lib_paths = [
        os.path.join(sys.prefix, 'lib'),
        os.path.join(sys.prefix, 'Library', 'lib'),  # Windows conda
        '/usr/lib',
        '/usr/local/lib',
        'C:/OSGeo4W64/lib',
        'C:/OSGeo4W/lib',
    ]
    
    for path in lib_paths:
        if os.path.exists(path):
            proj_lib_paths.append(path)
    
    if proj_lib_paths:
        os.environ['PROJ_LIB'] = proj_lib_paths[0]
        print(f"✓ PROJ_LIB set to: {proj_lib_paths[0]}")
    
    # Method 3: Configure GDAL to use PROJ properly
    try:
        from osgeo import gdal, osr
        
        # Enable GDAL exceptions
        gdal.UseExceptions()
        
        # Test PROJ functionality
        try:
            source_srs = osr.SpatialReference()
            source_srs.ImportFromEPSG(4326)  # WGS84
            print(f"✓ PROJ test successful")
            return True
            
        except Exception as e:
            print(f"❌ PROJ test failed: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ Could not import GDAL/OSR: {e}")
        return False

# Call this before any other imports
import sys
gdal_available = fix_proj_database_comprehensive()

# Rest of imports with error handling
import os
import glob
import subprocess
import time
import logging
import json
import gc
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, NamedTuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import logging

# Add these imports for geometry handling with better error handling
try:
    import geopandas as gpd
    from shapely.geometry import Polygon, MultiPolygon, shape
    from shapely.wkt import loads as wkt_loads
    import fiona
    HAS_GEOPANDAS = True
    print("✓ GeoPandas available")
except ImportError as e:
    HAS_GEOPANDAS = False
    print(f"⚠ GeoPandas not available: {e}")
    print("Install with: conda install -c conda-forge geopandas")

# Required dependencies with error handling
try:
    import numpy as np
    print("✓ NumPy available")
except ImportError:
    print("❌ NumPy not found - install with: pip install numpy")
    sys.exit(1)

try:
    import psutil
    print("✓ psutil available")
except ImportError:
    print("❌ psutil not found - install with: pip install psutil")
    sys.exit(1)

# GDAL import with proper error handling
if gdal_available:
    from osgeo import gdal, gdalconst
    print("✓ GDAL available")
else:
    print("❌ GDAL not available - install with: conda install gdal")
    sys.exit(1)

# Optional imports with fallbacks
try:
    from tqdm import tqdm
    HAS_TQDM = True
    print("✓ tqdm available")
except ImportError:
    HAS_TQDM = False
    print("⚠ tqdm not available - install with: pip install tqdm")

print("="*60)
print("Dependency check completed")
print("="*60)


# Configure enhanced logging
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors and enhanced formatting"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green  
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Add color to level name
        record.levelname = f"{color}{record.levelname}{reset}"
        
        # Enhanced format with more info
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%H:%M:%S'
        )
        return formatter.format(record)

# Setup enhanced logging
def setup_enhanced_logging(log_level=logging.INFO, output_folder: str = None):
    """Setup enhanced logging with proper file placement"""
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Determine log file location
    if output_folder:
        log_dir = os.path.join(output_folder, "Logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'unified_s2_tss_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    else:
        log_file = f'unified_s2_tss_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    # File handler - detailed logging
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler - colored output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(ColoredFormatter())
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging configured - File: {log_file}")
    return logger, log_file

def get_default_logger():
    """Create logger with smart default location - prevents logs in code directory"""
    try:
        # Create a default results folder to avoid cluttering code directory
        default_output = os.path.join(os.getcwd(), "S2_TSS_Results")
        os.makedirs(default_output, exist_ok=True)
        
        # Setup logging in the default location
        logger, log_file = setup_enhanced_logging(log_level=logging.INFO, output_folder=default_output)
        
        # Print to console so user knows where logs are going
        print(f"📋 Default logging location: {log_file}")
        print(f"📁 Logs will be saved to: {default_output}/Logs/")
        
        return logger
        
    except Exception as e:
        # Fallback to current directory if anything fails
        print(f"⚠️ Could not create default log folder: {e}")
        print("📋 Using current directory for logs")
        return setup_enhanced_logging()[0]

# Initialize logger without output folder initially (will be updated later)
logger = get_default_logger()

# ===== ENUMS AND DATA CLASSES =====

class ProcessingMode(Enum):
    """Processing mode enumeration"""
    COMPLETE_PIPELINE = "complete_pipeline"
    S2_PROCESSING_ONLY = "s2_processing_only"
    TSS_PROCESSING_ONLY = "tss_processing_only"

class ProductType(Enum):
    """Product type enumeration"""
    L1C_ZIP = "l1c_zip"
    L1C_SAFE = "l1c_safe"
    GEOMETRIC_DIM = "geometric_dim"
    C2RCC_DIM = "c2rcc_dim"
    UNKNOWN = "unknown"

@dataclass
class ResamplingConfig:
    """S2 Resampling configuration"""
    target_resolution: str = "10"
    upsampling_method: str = "Bilinear"
    downsampling_method: str = "Mean"
    flag_downsampling: str = "First"
    resample_on_pyramid_levels: bool = True

@dataclass
class SubsetConfig:
    """Spatial subset configuration"""
    geometry_wkt: Optional[str] = None
    sub_sampling_x: int = 1
    sub_sampling_y: int = 1
    full_swath: bool = False
    copy_metadata: bool = True
    pixel_start_x: Optional[int] = None
    pixel_start_y: Optional[int] = None
    pixel_size_x: Optional[int] = None
    pixel_size_y: Optional[int] = None

@dataclass
class C2RCCConfig:
    """Enhanced C2RCC atmospheric correction configuration with SNAP defaults"""
    
    # Basic water parameters
    salinity: float = 35.0
    temperature: float = 15.0
    ozone: float = 330.0
    pressure: float = 1000.0  # SNAP default
    elevation: float = 0.0
    
    # Neural network configuration
    net_set: str = "C2RCC-Nets"
    
    # DEM configuration
    dem_name: str = "Copernicus 30m Global DEM"
    
    # Auxiliary data - ECMWF enabled by default as requested
    use_ecmwf_aux_data: bool = True  # Set to True by default
    atmospheric_aux_data_path: str = ""
    alternative_nn_path: str = ""
    
    # Essential output products (SNAP defaults + uncertainties)
    output_as_rrs: bool = True
    output_rhown: bool = True          # Required for TSS
    output_kd: bool = True
    output_uncertainties: bool = True # Ensures unc_tsm.img and unc_chl.img
    output_ac_reflectance: bool = True
    output_rtoa: bool = True
       
    # Advanced atmospheric products (SNAP defaults)
    output_rtosa_gc: bool = False
    output_rtosa_gc_aann: bool = False
    output_rpath: bool = False
    output_tdown: bool = False
    output_tup: bool = False
    output_oos: bool = False
    
    # Advanced parameters
    derive_rw_from_path_and_transmittance: bool = False
    valid_pixel_expression: str = "B8 > 0 && B8 < 0.1"
    
    # Thresholds
    threshold_rtosa_oos: float = 0.05
    threshold_ac_reflec_oos: float = 0.1
    threshold_cloud_tdown865: float = 0.955
    
    # TSM and CHL parameters (SNAP defaults)
    tsm_fac: float = 1.06
    tsm_exp: float = 0.942
    chl_fac: float = 21.0
    chl_exp: float = 1.04

@dataclass
class JiangTSSConfig:
    """Jiang TSS methodology configuration - CLEAN VERSION"""
    enable_jiang_tss: bool = True  # Enable by default
    output_intermediates: bool = True
    water_mask_threshold: float = 0.01
    tss_valid_range: tuple = (0.01, 10000)  # g/m³
    output_comparison_stats: bool = True
    
    # Advanced algorithms configuration
    enable_advanced_algorithms: bool = True
    advanced_config: Optional['AdvancedAquaticConfig'] = None
    
    # NEW: Marine visualization configuration
    enable_marine_visualization: bool = True  # ENABLED BY DEFAULT
    marine_viz_config: Optional['MarineVisualizationConfig'] = None
    
    def __post_init__(self):
        """Initialize advanced and marine visualization configs"""
        if self.enable_advanced_algorithms and self.advanced_config is None:
            self.advanced_config = AdvancedAquaticConfig()
            
        if self.enable_marine_visualization and self.marine_viz_config is None:
            self.marine_viz_config = MarineVisualizationConfig()

@dataclass
class AdvancedAquaticConfig:
    """Configuration for advanced aquatic algorithms"""
    
    # WORKING ALGORITHMS ONLY
    enable_water_clarity: bool = True
    solar_zenith_angle: float = 30.0
    
    enable_hab_detection: bool = True
    hab_biomass_threshold: float = 20.0
    hab_extreme_threshold: float = 100.0
    
    # Output options
    save_intermediate_products: bool = True
    create_classification_maps: bool = True
    generate_statistics: bool = True
@dataclass
class ProcessingConfig:
    """Complete processing configuration"""
    processing_mode: ProcessingMode
    input_folder: str
    output_folder: str
    resampling_config: ResamplingConfig
    subset_config: SubsetConfig
    c2rcc_config: C2RCCConfig
    jiang_config: JiangTSSConfig
    skip_existing: bool = True
    test_mode: bool = False
    memory_limit_gb: int = 8
    thread_count: int = 4

@dataclass
class ProcessingResult:
    """Result container for processing outputs with intermediate paths support"""
    success: bool
    output_path: str
    statistics: Optional[Dict]
    error_message: Optional[str]
    intermediate_paths: Dict[str, str] = field(default_factory=dict)

class ProcessingStatus(NamedTuple):
    """Processing status information"""
    total_products: int
    processed: int
    failed: int
    skipped: int
    current_product: str
    current_stage: str
    progress_percent: float
    eta_minutes: float
    processing_speed: float

# ===== UTILITY CLASSES =====

class MemoryManager:
    """Memory management utilities"""
    
    @staticmethod
    def cleanup_variables(*variables):
        """Clean up variables and force garbage collection"""
        for var in variables:
            if var is not None:
                try:
                    del var
                except:
                    pass
        gc.collect()
    
    @staticmethod
    def monitor_memory(threshold_mb=8000):
        """Monitor memory usage"""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb > threshold_mb:
                logger.warning(f"High memory usage: {memory_mb:.1f} MB")
                return True
            return False
        except:
            return False

class SafeMathNumPy:
    """Safe mathematical operations for NumPy arrays"""
    
    @staticmethod
    def safe_divide(numerator, denominator, default_value=0.0, min_denominator=1e-10):
        """Safely divide arrays with protection against division by zero"""
        if not isinstance(numerator, np.ndarray):
            numerator = np.array(numerator, dtype=np.float32)
        if not isinstance(denominator, np.ndarray):
            denominator = np.array(denominator, dtype=np.float32)
            
        result = np.full_like(numerator, default_value, dtype=np.float32)
        
        if isinstance(denominator, (int, float)):
            if abs(denominator) >= min_denominator:
                result = numerator / denominator
        else:
            valid_mask = np.abs(denominator) >= min_denominator
            result[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
        
        return result
    @staticmethod
    def safe_sqrt(value, min_value=0.0, default_value=0.0):
        """Safely calculate square root"""
        if not isinstance(value, np.ndarray):
            value = np.array(value, dtype=np.float32)
        
        result = np.full_like(value, default_value, dtype=np.float32)
        valid_mask = value >= min_value
        result[valid_mask] = np.sqrt(value[valid_mask])
        
        return result
    
    @staticmethod
    def safe_log(value, base=10, min_value=1e-10, default_value=-999):
        """Safely calculate logarithm"""
        if not isinstance(value, np.ndarray):
            value = np.array(value, dtype=np.float32)
            
        result = np.full_like(value, default_value, dtype=np.float32)
        
        if isinstance(value, (int, float)):
            if value > min_value:
                result = np.log(value) / np.log(base)
        else:
            valid_mask = value > min_value
            result[valid_mask] = np.log(value[valid_mask]) / np.log(base)
        
        return result

    @staticmethod
    def safe_power(base, exponent, min_base=1e-10, max_exponent=100, default_value=0.0):
        """Safely calculate power operations"""
        if not isinstance(base, np.ndarray):
            base = np.array(base, dtype=np.float32)
        if not isinstance(exponent, np.ndarray):
            exponent = np.array(exponent, dtype=np.float32)
            
        result = np.full_like(base, default_value, dtype=np.float32)
        
        # Avoid extreme values that could cause overflow
        valid_mask = (base >= min_base) & (np.abs(exponent) <= max_exponent)
        
        if np.any(valid_mask):
            try:
                result[valid_mask] = np.power(base[valid_mask], exponent[valid_mask])
            except (OverflowError, RuntimeWarning):
                logger.warning("Power calculation overflow detected, using default values")
        
        return result


class RasterIO:
    """Utilities for raster input/output operations using GDAL"""
    
    @staticmethod
    def read_raster(file_path: str) -> Tuple[np.ndarray, dict]:
        """Read raster file and return data array with metadata"""
        try:
            dataset = gdal.Open(file_path, gdalconst.GA_ReadOnly)
            if dataset is None:
                raise ValueError(f"Could not open raster file: {file_path}")
            
            band = dataset.GetRasterBand(1)
            data = band.ReadAsArray().astype(np.float32)
            nodata = band.GetNoDataValue()
            
            # Apply nodata mask
            if nodata is not None:
                data[data == nodata] = np.nan
            
            metadata = {
                'geotransform': dataset.GetGeoTransform(),
                'projection': dataset.GetProjection(),
                'width': dataset.RasterXSize,
                'height': dataset.RasterYSize,
                'nodata': nodata if nodata is not None else -9999
            }
            
            dataset = None  # Close dataset
            return data, metadata
            
        except Exception as e:
            logger.error(f"Error reading raster {file_path}: {e}")
            raise
    
    @staticmethod
    def write_raster(data: np.ndarray, output_path: str, metadata: dict, 
                    description: str = "", nodata: float = -9999) -> bool:
        """FIXED: Write numpy array to raster file with enhanced validation"""
        dataset = None
        try:
            # CRITICAL FIX: Validate inputs first
            if not isinstance(data, np.ndarray):
                logger.error(f"Error writing raster {output_path}: not a sequence")
                logger.error(f"Data type: {type(data)}, shape: {getattr(data, 'shape', 'no shape')}")
                return False
            
            if not isinstance(metadata, dict):
                logger.error(f"Invalid metadata type: {type(metadata)}")
                return False
            
            if data.ndim != 2:
                logger.error(f"Data must be 2D array, got {data.ndim}D with shape {data.shape}")
                return False
            
            # CRITICAL FIX: Ensure contiguous array for GDAL
            try:
                output_data = np.ascontiguousarray(data, dtype=np.float32)
                
                # Validate the array is suitable for GDAL
                if not output_data.flags.c_contiguous:
                    raise ValueError("Failed to create contiguous array")
                    
            except Exception as array_error:
                logger.error(f"Array preparation failed: {array_error}")
                return False
            
            # Replace NaN with nodata
            nan_mask = np.isnan(output_data)
            output_data[nan_mask] = nodata
            
            # Get dimensions
            height, width = output_data.shape
            
            # Handle metadata safely
            geotransform = metadata.get('geotransform')
            projection = metadata.get('projection')
            
            if geotransform is None:
                logger.warning("No geotransform in metadata, using default")
                geotransform = (0.0, 1.0, 0.0, 0.0, 0.0, -1.0)
            
            if projection is None:
                logger.warning("No projection in metadata, using default WGS84")
                projection = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'
            
            # Create GDAL dataset
            driver = gdal.GetDriverByName('GTiff')
            if driver is None:
                logger.error("GTiff driver not available")
                return False
            
            dataset = driver.Create(
                output_path, 
                width, 
                height, 
                1,  # Single band
                gdal.GDT_Float32,
                ['COMPRESS=LZW', 'PREDICTOR=2', 'TILED=YES']
            )
            
            if dataset is None:
                logger.error(f"Failed to create GDAL dataset: {output_path}")
                return False
            
            # Set georeference information
            try:
                dataset.SetGeoTransform(geotransform)
                dataset.SetProjection(projection)
            except Exception as georef_error:
                logger.warning(f"Error setting georeference: {georef_error}")
            
            # Write data
            band = dataset.GetRasterBand(1)
            write_result = band.WriteArray(output_data)
            
            if write_result != 0:  # GDAL returns 0 for success
                logger.error(f"GDAL WriteArray failed with code: {write_result}")
                return False
            
            band.SetNoDataValue(nodata)
            if description:
                band.SetDescription(description)
            
            # Calculate statistics
            try:
                band.ComputeStatistics(False)
                band.FlushCache()
                dataset.FlushCache()
            except Exception as stats_error:
                logger.warning(f"Could not compute statistics: {stats_error}")
            
            dataset = None  # Close dataset
            
            logger.info(f"Successfully wrote raster: {os.path.basename(output_path)}")
            return True
            
        except Exception as e:
            logger.error(f"Error writing raster {output_path}: {e}")
            logger.error(f"Data type: {type(data)}, shape: {getattr(data, 'shape', 'no shape')}")
            return False
        finally:
            if dataset is not None:
                dataset = None

    
    @staticmethod
    def calculate_statistics(data: np.ndarray, nodata: float = -9999) -> Dict:
        """Calculate statistics for data array"""
        valid_data = data[~np.isnan(data) & (data != nodata)]
        
        if len(valid_data) == 0:
            return {
                'count': 0, 'min': nodata, 'max': nodata, 
                'mean': nodata, 'std': nodata, 'coverage_percent': 0.0
            }
        
        return {
            'count': len(valid_data),
            'min': float(np.min(valid_data)),
            'max': float(np.max(valid_data)),
            'mean': float(np.mean(valid_data)),
            'std': float(np.std(valid_data)),
            'coverage_percent': (len(valid_data) / data.size) * 100
        }

class ProductDetector:
    """Smart product type detection and validation"""
    
    @staticmethod
    def detect_product_type(file_path: str) -> ProductType:
        """Detect product type from file/folder structure"""
        basename = os.path.basename(file_path)
        
        if basename.endswith('.zip') and 'MSIL1C' in basename:
            return ProductType.L1C_ZIP
        elif basename.endswith('.SAFE') and 'MSIL1C' in basename:
            return ProductType.L1C_SAFE
        elif basename.endswith('.dim'):
            if 'C2RCC' in basename:
                return ProductType.C2RCC_DIM
            elif 'Resampled' in basename and 'Subset' in basename:
                return ProductType.GEOMETRIC_DIM
            else:
                return ProductType.UNKNOWN
        else:
            return ProductType.UNKNOWN
    
    @staticmethod
    def scan_input_folder(folder_path: str) -> Dict[ProductType, List[str]]:
        """Scan folder and categorize all products"""
        products = {ptype: [] for ptype in ProductType}
        
        if not os.path.exists(folder_path):
            return products
        
        # Scan for files and directories
        for root, dirs, files in os.walk(folder_path):
            # Check .dim files
            for file in files:
                if file.endswith('.dim'):
                    file_path = os.path.join(root, file)
                    ptype = ProductDetector.detect_product_type(file_path)
                    products[ptype].append(file_path)
                elif file.endswith('.zip'):
                    file_path = os.path.join(root, file)
                    ptype = ProductDetector.detect_product_type(file_path)
                    products[ptype].append(file_path)
            
            # Check .SAFE directories
            for dir_name in dirs:
                if dir_name.endswith('.SAFE'):
                    dir_path = os.path.join(root, dir_name)
                    ptype = ProductDetector.detect_product_type(dir_path)
                    products[ptype].append(dir_path)
        
        # Sort all lists
        for ptype in products:
            products[ptype].sort()
        
        return products
    
    @staticmethod
    def validate_processing_mode(products: Dict[ProductType, List[str]], mode: ProcessingMode) -> Tuple[bool, str, List[str]]:
        """Validate that products match the selected processing mode"""
        if mode == ProcessingMode.COMPLETE_PIPELINE:
            l1c_products = products[ProductType.L1C_ZIP] + products[ProductType.L1C_SAFE]
            if l1c_products:
                return True, f"Found {len(l1c_products)} L1C products for complete pipeline", l1c_products
            else:
                return False, "No L1C products found for complete pipeline", []
        
        elif mode == ProcessingMode.S2_PROCESSING_ONLY:
            l1c_products = products[ProductType.L1C_ZIP] + products[ProductType.L1C_SAFE]
            if l1c_products:
                return True, f"Found {len(l1c_products)} L1C products for S2 processing", l1c_products
            else:
                return False, "No L1C products found for S2 processing", []
        
        elif mode == ProcessingMode.TSS_PROCESSING_ONLY:
            c2rcc_products = products[ProductType.C2RCC_DIM]
            if c2rcc_products:
                return True, f"Found {len(c2rcc_products)} C2RCC products for TSS processing", c2rcc_products
            else:
                return False, "No C2RCC products found for TSS processing", []
        
        return False, "Unknown processing mode", []

class SystemMonitor:
    """Real-time system monitoring"""
    
    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
        self.current_info = {
            'cpu_percent': 0,
            'memory_used_gb': 0,
            'memory_total_gb': 0,
            'disk_free_gb': 0
        }
        
    def start_monitoring(self):
        """Start system monitoring in background thread"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                # Get system info
                memory = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent(interval=1)
                
                # Get disk info for current directory
                disk_usage = psutil.disk_usage('.')
                
                self.current_info = {
                    'cpu_percent': cpu_percent,
                    'memory_used_gb': memory.used / (1024**3),
                    'memory_total_gb': memory.total / (1024**3),
                    'disk_free_gb': disk_usage.free / (1024**3)
                }
                
            except Exception as e:
                logger.warning(f"System monitoring error: {e}")
            
            time.sleep(2)  # Update every 2 seconds
    
    def get_current_info(self) -> dict:
        """Get current system information"""
        return self.current_info.copy()
    
    def check_system_health(self) -> Tuple[bool, List[str]]:
        """Check system health and return warnings"""
        warnings = []
        healthy = True
        
        info = self.get_current_info()
        
        # Memory checks
        memory_percent = (info['memory_used_gb'] / info['memory_total_gb']) * 100
        if memory_percent > 90:
            warnings.append(f"Very high memory usage: {memory_percent:.1f}%")
            healthy = False
        elif memory_percent > 75:
            warnings.append(f"High memory usage: {memory_percent:.1f}%")
        
        # CPU checks
        if info['cpu_percent'] > 95:
            warnings.append(f"Very high CPU usage: {info['cpu_percent']:.1f}%")
        
        # Disk space checks
        if info['disk_free_gb'] < 1:
            warnings.append(f"Critically low disk space: {info['disk_free_gb']:.1f} GB")
            healthy = False
        elif info['disk_free_gb'] < 10:
            warnings.append(f"Low disk space: {info['disk_free_gb']:.1f} GB")
        
        return healthy, warnings

# ===== SNAP TSM/CHL CALCULATOR =====

class SNAPTSMCHLCalculator:
    """Calculate TSM and CHL from SNAP C2RCC IOP outputs using official SNAP formulas"""
    
    def __init__(self, tsm_fac: float = 1.06, tsm_exp: float = 0.942, 
                 chl_fac: float = 21.0, chl_exp: float = 1.04):
        self.tsm_fac = tsm_fac
        self.tsm_exp = tsm_exp
        self.chl_fac = chl_fac
        self.chl_exp = chl_exp
        
        logger.info(f"SNAP TSM/CHL Calculator initialized:")
        logger.info(f"  TSM formula: TSM = {tsm_fac} * (bpart + bwit)^{tsm_exp}")
        logger.info(f"  CHL formula: CHL = apig^{chl_exp} * {chl_fac}")
    
    def calculate_snap_tsm_chl(self, c2rcc_path: str) -> Dict[str, ProcessingResult]:
        """Calculate TSM and CHL from SNAP IOPs using official formulas - FIXED VERSION"""
        try:
            logger.info("Calculating SNAP TSM/CHL from IOP products...")
            
            # Determine data folder
            if c2rcc_path.endswith('.dim'):
                data_folder = c2rcc_path.replace('.dim', '.data')
            else:
                data_folder = f"{c2rcc_path}.data"
            
            if not os.path.exists(data_folder):
                return {'error': ProcessingResult(False, "", None, f"Data folder not found: {data_folder}")}
            
            # Load required IOPs with robust error handling
            iop_files = {
                'apig': os.path.join(data_folder, 'iop_apig.img'),     # For CHL
                'bpart': os.path.join(data_folder, 'iop_bpart.img'),   # For TSM
                'bwit': os.path.join(data_folder, 'iop_bwit.img')      # For TSM
            }
            
            # Check and load available IOPs
            available_iops = {}
            
            for iop_name, iop_path in iop_files.items():
                if os.path.exists(iop_path) and os.path.getsize(iop_path) > 1024:
                    try:
                        data, metadata = RasterIO.read_raster(iop_path)
                        available_iops[iop_name] = {'data': data, 'metadata': metadata}
                        logger.info(f"✓ Loaded {iop_name}: {data.shape}, mean={np.nanmean(data):.4f}")
                    except Exception as e:
                        logger.error(f"Error loading {iop_name}: {e}")
                else:
                    logger.warning(f"Missing or empty: {iop_name}")
            
            results = {}
            
            # Calculate CHL from apig using SNAP formula - FIXED
            if 'apig' in available_iops:
                logger.info("Calculating CHL concentration from iop_apig...")
                
                apig_data = available_iops['apig']['data']
                metadata = available_iops['apig']['metadata']
                
                # FIXED: Direct numpy calculation instead of SafeMathNumPy
                # Handle edge cases properly
                valid_mask = (apig_data > 0) & (~np.isnan(apig_data)) & (~np.isinf(apig_data))
                
                # Initialize result array
                chl_concentration = np.full_like(apig_data, np.nan, dtype=np.float32)
                
                if np.any(valid_mask):
                    # Apply SNAP CHL formula: CHL = apig^CHLexp * CHLfac
                    # Use numpy power function directly with proper handling
                    try:
                        valid_apig = apig_data[valid_mask]
                        chl_values = np.power(valid_apig, self.chl_exp) * self.chl_fac
                        
                        # Check for invalid results
                        valid_chl_mask = (~np.isnan(chl_values)) & (~np.isinf(chl_values)) & (chl_values >= 0)
                        
                        # Only assign valid CHL values
                        if np.any(valid_chl_mask):
                            # Create a temporary mask for the original array
                            temp_mask = valid_mask.copy()
                            temp_mask[valid_mask] = valid_chl_mask
                            chl_concentration[temp_mask] = chl_values[valid_chl_mask]
                            
                            logger.info(f"CHL calculation: {np.sum(temp_mask)} valid pixels out of {np.sum(valid_mask)} processed")
                        else:
                            logger.warning("No valid CHL values after calculation")
                            
                    except Exception as calc_error:
                        logger.error(f"Error in CHL calculation: {calc_error}")
                        # Fill with NaN if calculation fails
                        chl_concentration = np.full_like(apig_data, np.nan, dtype=np.float32)
                
                # Save CHL concentration
                output_path = os.path.join(data_folder, 'conc_chl.img')
                success = RasterIO.write_raster(
                    chl_concentration, output_path, metadata,
                    f"SNAP Chlorophyll concentration (mg/m³) - CHL = apig^{self.chl_exp} * {self.chl_fac}",
                    nodata=-9999
                )
                
                if success:
                    stats = RasterIO.calculate_statistics(chl_concentration)
                    logger.info(f"✓ CHL concentration saved: {stats['coverage_percent']:.1f}% coverage, mean={stats['mean']:.3f} mg/m³")
                    results['snap_chl'] = ProcessingResult(True, output_path, stats, None)
                else:
                    results['snap_chl'] = ProcessingResult(False, output_path, None, "Failed to save CHL")
            else:
                logger.error("Cannot calculate CHL: iop_apig.img not available")
                results['snap_chl'] = ProcessingResult(False, "", None, "Missing iop_apig for CHL calculation")
            
            # Calculate TSM from bpart + bwit (btot approximation) - FIXED
            if 'bpart' in available_iops and 'bwit' in available_iops:
                logger.info("Calculating TSM concentration from bpart + bwit (btot approximation)...")
                
                bpart_data = available_iops['bpart']['data']
                bwit_data = available_iops['bwit']['data']
                metadata = available_iops['bpart']['metadata']
                
                # FIXED: Direct numpy calculation
                # Approximate btot as bpart + bwit (since iop_btot.img is missing)
                btot_approx = bpart_data + bwit_data
                
                # Handle edge cases properly
                valid_mask = (btot_approx > 0) & (~np.isnan(btot_approx)) & (~np.isinf(btot_approx))
                
                # Initialize result array
                tsm_concentration = np.full_like(btot_approx, np.nan, dtype=np.float32)
                
                if np.any(valid_mask):
                    # Apply SNAP TSM formula: TSM = TSMfac * btot^TSMexp
                    try:
                        valid_btot = btot_approx[valid_mask]
                        tsm_values = self.tsm_fac * np.power(valid_btot, self.tsm_exp)
                        
                        # Check for invalid results
                        valid_tsm_mask = (~np.isnan(tsm_values)) & (~np.isinf(tsm_values)) & (tsm_values >= 0)
                        
                        # Only assign valid TSM values
                        if np.any(valid_tsm_mask):
                            # Create a temporary mask for the original array
                            temp_mask = valid_mask.copy()
                            temp_mask[valid_mask] = valid_tsm_mask
                            tsm_concentration[temp_mask] = tsm_values[valid_tsm_mask]
                            
                            logger.info(f"TSM calculation: {np.sum(temp_mask)} valid pixels out of {np.sum(valid_mask)} processed")
                        else:
                            logger.warning("No valid TSM values after calculation")
                            
                    except Exception as calc_error:
                        logger.error(f"Error in TSM calculation: {calc_error}")
                        # Fill with NaN if calculation fails
                        tsm_concentration = np.full_like(btot_approx, np.nan, dtype=np.float32)
                
                # Save TSM concentration
                output_path = os.path.join(data_folder, 'conc_tsm.img')
                success = RasterIO.write_raster(
                    tsm_concentration, output_path, metadata,
                    f"SNAP TSM concentration (g/m³) - TSM = {self.tsm_fac} * (bpart + bwit)^{self.tsm_exp}",
                    nodata=-9999
                )
                
                if success:
                    stats = RasterIO.calculate_statistics(tsm_concentration)
                    logger.info(f"✓ TSM concentration saved: {stats['coverage_percent']:.1f}% coverage, mean={stats['mean']:.3f} g/m³")
                    results['snap_tsm'] = ProcessingResult(True, output_path, stats, None)
                else:
                    results['snap_tsm'] = ProcessingResult(False, output_path, None, "Failed to save TSM")
            else:
                logger.error("Cannot calculate TSM: missing bpart or bwit")
                results['snap_tsm'] = ProcessingResult(False, "", None, "Missing bpart/bwit for TSM calculation")
            
            return results
            
        except Exception as e:
            error_msg = f"Error calculating SNAP TSM/CHL: {str(e)}"
            logger.error(error_msg)
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return {'error': ProcessingResult(False, "", None, error_msg)}

    def calculate_uncertainties(self, c2rcc_path: str) -> Dict[str, ProcessingResult]:
        """Calculate uncertainties for TSM and CHL products - ADD THIS METHOD HERE"""
        try:
            data_folder = c2rcc_path.replace('.dim', '.data')
            
            # Check if uncertainty files exist
            unc_tsm_path = os.path.join(data_folder, 'unc_tsm.img')
            unc_chl_path = os.path.join(data_folder, 'unc_chl.img')
            
            tsm_path = os.path.join(data_folder, 'conc_tsm.img')
            chl_path = os.path.join(data_folder, 'conc_chl.img')
            
            results = {}
            
            # Calculate TSM uncertainty (typically 15-30% of TSM value)
            if os.path.exists(tsm_path) and not (os.path.exists(unc_tsm_path) and os.path.getsize(unc_tsm_path) > 1024):
                logger.info("🔄 Calculating TSM uncertainties...")
                
                tsm_data, tsm_meta = RasterIO.read_raster(tsm_path)
                
                # Simple uncertainty model: 20% of TSM value + 0.1 g/m³ base uncertainty
                unc_tsm_data = np.where(
                    tsm_data > 0,
                    tsm_data * 0.20 + 0.1,  # 20% relative + 0.1 g/m³ absolute
                    0
                )
                
                success = RasterIO.write_raster(unc_tsm_path, unc_tsm_data, tsm_meta)
                if success:
                    logger.info("✅ TSM uncertainties calculated and saved")
                    
                    # Calculate statistics
                    valid_pixels = np.sum(unc_tsm_data > 0)
                    mean_uncertainty = np.mean(unc_tsm_data[unc_tsm_data > 0]) if valid_pixels > 0 else 0
                    
                    stats = {
                        'coverage_percent': (valid_pixels / unc_tsm_data.size) * 100,
                        'mean': mean_uncertainty,
                        'valid_pixels': int(valid_pixels),
                        'file_size_mb': os.path.getsize(unc_tsm_path) / (1024 * 1024)
                    }
                    
                    results['unc_tsm'] = ProcessingResult(True, unc_tsm_path, stats, None)
                else:
                    logger.error("❌ Failed to save TSM uncertainties")
                    results['unc_tsm'] = ProcessingResult(False, "", None, "Failed to save TSM uncertainties")
            
            # Calculate CHL uncertainty (typically 25-40% of CHL value)
            if os.path.exists(chl_path) and not (os.path.exists(unc_chl_path) and os.path.getsize(unc_chl_path) > 1024):
                logger.info("🔄 Calculating CHL uncertainties...")
                
                chl_data, chl_meta = RasterIO.read_raster(chl_path)
                
                # Simple uncertainty model: 30% of CHL value + 0.05 mg/m³ base uncertainty
                unc_chl_data = np.where(
                    chl_data > 0,
                    chl_data * 0.30 + 0.05,  # 30% relative + 0.05 mg/m³ absolute
                    0
                )
                
                success = RasterIO.write_raster(unc_chl_path, unc_chl_data, chl_meta)
                if success:
                    logger.info("✅ CHL uncertainties calculated and saved")
                    
                    # Calculate statistics
                    valid_pixels = np.sum(unc_chl_data > 0)
                    mean_uncertainty = np.mean(unc_chl_data[unc_chl_data > 0]) if valid_pixels > 0 else 0
                    
                    stats = {
                        'coverage_percent': (valid_pixels / unc_chl_data.size) * 100,
                        'mean': mean_uncertainty,
                        'valid_pixels': int(valid_pixels),
                        'file_size_mb': os.path.getsize(unc_chl_path) / (1024 * 1024)
                    }
                    
                    results['unc_chl'] = ProcessingResult(True, unc_chl_path, stats, None)
                else:
                    logger.error("❌ Failed to save CHL uncertainties")
                    results['unc_chl'] = ProcessingResult(False, "", None, "Failed to save CHL uncertainties")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error calculating uncertainties: {e}")
            return {
                'unc_error': ProcessingResult(False, "", None, f"Error calculating uncertainties: {e}")
            }
            
# ===== JIANG TSS PROCESSOR =====

@dataclass  
class JiangTSSConstants:
    """Complete TSS configuration constants from Jiang et al. (2023) - FULL IMPLEMENTATION"""
    
    # Pure water absorption coefficients (aw) in m^-1 - FROM ORIGINAL R CODE
    PURE_WATER_ABSORPTION = {
        443: 0.00515124,    # Band B1
        490: 0.01919594,    # Band B2  
        560: 0.06299986,    # Band B3 (Type I)
        665: 0.41395333,    # Band B4 (Type II)
        705: 0.70385758,    # Band B5
        740: 2.71167020,    # Band B6 (Type III)
        783: 2.62000141,    # Band B7
        865: 4.61714226     # Band B8A (Type IV)
    }
    
    # Pure water backscattering coefficients (bbw) in m^-1 - FROM ORIGINAL R CODE
    PURE_WATER_BACKSCATTERING = {
        443: 0.00215037,    # Band B1
        490: 0.00138116,    # Band B2
        560: 0.00078491,    # Band B3 (Type I)
        665: 0.00037474,    # Band B4 (Type II)
        705: 0.00029185,    # Band B5
        740: 0.00023499,    # Band B6 (Type III)
        783: 0.00018516,    # Band B7
        865: 0.00012066     # Band B8A (Type IV)
    }
    
    # TSS conversion factors from Jiang et al. (2023) - FROM ORIGINAL R CODE
    TSS_CONVERSION_FACTORS = {
        560: 94.48785,      # Type I: Clear water
        665: 113.87498,     # Type II: Moderately turbid
        740: 134.91845,     # Type III: Highly turbid
        865: 166.07382      # Type IV: Extremely turbid
    }
    
    # Rrs620 estimation coefficients - EXACT from R code
    RRS620_COEFFICIENTS = {
        'a': 1.693846e+02,  # a <- 1.693846e+02
        'b': -1.557556e+01, # b <- -1.557556e+01
        'c': 1.316727e+00,  # c <- 1.316727e+00
        'd': 1.484814e-04   # d <- 1.484814e-04
    }

@dataclass
class JiangTSSConfig:
    """Jiang TSS methodology configuration - CLEAN VERSION"""
    enable_jiang_tss: bool = True  # ENABLED BY DEFAULT
    output_intermediates: bool = True
    water_mask_threshold: float = 0.01
    tss_valid_range: tuple = (0.01, 10000)  # g/m³
    output_comparison_stats: bool = True
    
    # Advanced algorithms configuration - SIMPLIFIED
    enable_advanced_algorithms: bool = True
    advanced_config: Optional['AdvancedAquaticConfig'] = None
    
    def __post_init__(self):
        """Initialize advanced config with only working algorithms"""
        if self.enable_advanced_algorithms and self.advanced_config is None:
            self.advanced_config = AdvancedAquaticConfig()
            
class JiangTSSProcessor:
    """Complete implementation of Jiang et al. 2023 TSS methodology - FULL VERSION"""
    
    def __init__(self, config: JiangTSSConfig):
        """Initialize Jiang TSS Processor with clean configuration and marine visualization"""
        # Direct assignment - no patching needed
        self.config = config
        self.constants = JiangTSSConstants()
        
        # NEW: Initialize marine visualization processor
        if hasattr(config, 'enable_marine_visualization') and config.enable_marine_visualization:
            self.marine_viz_processor = S2MarineVisualizationProcessor(config.marine_viz_config)
            logger.info("Marine visualization processor initialized")
        else:
            self.marine_viz_processor = None
            logger.info("Marine visualization disabled")
        
        # Initialize advanced processor if enabled
        if self.config.enable_advanced_algorithms:
            self.advanced_processor = AdvancedAquaticProcessor()
            if self.config.advanced_config is None:
                self.config.advanced_config = AdvancedAquaticConfig()
        else:
            self.advanced_processor = None
            
        logger.info("Initialized Jiang TSS Processor with enhanced methodology")
        
        # Enhanced logging with marine visualization status
        logger.info(f"Jiang TSS enabled: {self.config.enable_jiang_tss}")
        logger.info(f"Advanced algorithms enabled: {self.config.enable_advanced_algorithms}")
        logger.info(f"Marine visualization enabled: {getattr(self.config, 'enable_marine_visualization', False)}")
        
        if self.config.enable_advanced_algorithms and self.config.advanced_config:
            logger.info("Working algorithms available:")
            logger.info(f"  ✓ Water Clarity: {self.config.advanced_config.enable_water_clarity}")
            logger.info(f"  ✓ HAB Detection: {self.config.advanced_config.enable_hab_detection}")
            logger.info("Note: Only Sentinel-2 compatible algorithms included")
        
        # Marine visualization configuration details
        if hasattr(self.config, 'enable_marine_visualization') and self.config.enable_marine_visualization:
            viz_config = self.config.marine_viz_config
            if viz_config:
                logger.info("Marine visualization configuration:")
                logger.info(f"  ✓ Natural color RGB: {viz_config.generate_natural_color}")
                logger.info(f"  ✓ False color RGB: {viz_config.generate_false_color}")
                logger.info(f"  ✓ Water-specific RGB: {viz_config.generate_water_specific}")
                logger.info(f"  ✓ Research RGB: {viz_config.generate_research_combinations}")
                logger.info(f"  ✓ Water quality indices: {viz_config.generate_water_quality_indices}")
                logger.info(f"  ✓ Chlorophyll indices: {viz_config.generate_chlorophyll_indices}")
                logger.info(f"  ✓ Turbidity indices: {viz_config.generate_turbidity_indices}")
                logger.info(f"  ✓ Advanced indices: {viz_config.generate_advanced_indices}")
                logger.info(f"  Output format: {viz_config.rgb_format}")
                logger.info(f"  Contrast enhancement: {viz_config.apply_contrast_enhancement}")
            else:
                logger.info("Marine visualization enabled with default configuration")
        
        logger.info("Jiang TSS Processor initialization completed successfully")
    
    
    def _load_bands_data(self, band_paths: Dict[int, str]) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Load band data arrays from file paths
        
        Args:
            band_paths: Dictionary mapping wavelength -> file path
            
        Returns:
            Tuple of (bands_data, reference_metadata) or (None, None) on failure
        """
        bands_data = {}
        reference_metadata = None
        
        logger.info(f"Loading {len(band_paths)} spectral bands into memory")
        
        for wavelength, file_path in band_paths.items():
            try:
                data, metadata = RasterIO.read_raster(file_path)
                bands_data[wavelength] = data
                
                if reference_metadata is None:
                    reference_metadata = metadata
                
                # Log loading progress
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                logger.debug(f"✓ Loaded {wavelength}nm: {file_size_mb:.1f}MB")
                
            except Exception as e:
                logger.error(f"❌ Failed to load band {wavelength}nm from {file_path}: {e}")
                return None, None
        
        logger.info(f"✅ Successfully loaded {len(bands_data)} bands into memory")
        return bands_data, reference_metadata
    
    def _apply_simple_unit_conversion(self, bands_data: Dict[int, np.ndarray], 
                                    band_paths: Dict[int, str]) -> Dict[int, np.ndarray]:
        """
        SIMPLE: Apply unit conversions based on detected band types
        Works with your _update_band_mapping_for_mixed_types method
        """
        converted_data = {}
        conversion_log = []
        
        # Analyze what band types we have
        band_types = set()
        for file_path in band_paths.values():
            filename = os.path.basename(file_path)
            if filename.startswith('rhow_'):
                band_types.add('rhow')
            elif filename.startswith('rrs_'):
                band_types.add('rrs')
        
        # Log detected types
        logger.info(f"🔧 Detected band types: {list(band_types)}")
        
        # Apply conversions
        for wavelength, data in bands_data.items():
            file_path = band_paths[wavelength]
            filename = os.path.basename(file_path)
            
            if filename.startswith('rhow_'):
                # Convert rhow to Rrs: Rrs = rhow / π
                converted_data[wavelength] = data / np.pi
                conversion_log.append(f"{wavelength}nm: rhow → Rrs (÷π)")
                
            elif filename.startswith('rrs_'):
                # Already in Rrs units, no conversion needed
                converted_data[wavelength] = data.copy()
                conversion_log.append(f"{wavelength}nm: rrs (no conversion)")
            
            else:
                # This shouldn't happen with your current band mapping
                logger.warning(f"Unexpected band type for {wavelength}nm: {filename}")
                converted_data[wavelength] = data.copy()
                conversion_log.append(f"{wavelength}nm: unknown (no conversion)")
        
        # Log all conversions applied
        if len(band_types) > 1:
            logger.info("🔄 Mixed band types - individual conversions:")
        else:
            band_type = list(band_types)[0]
            if band_type == 'rhow':
                logger.info("🔄 All rhow bands - converting to Rrs:")
            else:
                logger.info("✅ All rrs bands - no conversion needed:")
        
        for log_entry in conversion_log:
            logger.info(f"  {log_entry}")
        
        # Validate converted data ranges
        self._validate_rrs_ranges(converted_data)
        
        return converted_data
    
    def _validate_rrs_ranges(self, rrs_data: Dict[int, np.ndarray]):
        """
        Validate that Rrs values are in reasonable ranges after conversion
        """
        for wavelength, data in rrs_data.items():
            # Remove NaN values for statistics
            valid_data = data[~np.isnan(data)]
            
            if len(valid_data) == 0:
                logger.warning(f"Band {wavelength}nm: All values are NaN")
                continue
                
            min_val = np.min(valid_data)
            max_val = np.max(valid_data)
            mean_val = np.mean(valid_data)
            
            # Typical Rrs ranges for water (sr^-1)
            # Blue/Green: 0.001-0.05, Red: 0.0001-0.02, NIR: 0.00001-0.01
            if wavelength < 600:  # Blue/Green
                expected_max = 0.08
            elif wavelength < 700:  # Red
                expected_max = 0.04
            else:  # NIR
                expected_max = 0.02
            
            # Check for unreasonable values
            if max_val > expected_max:
                logger.warning(f"Band {wavelength}nm: High Rrs values detected (max={max_val:.6f})")
                logger.warning("  This may indicate incorrect unit conversion or data issues")
            
            if min_val < -0.01:  # Allow small negative values (sensor noise)
                negative_count = np.sum(valid_data < -0.01)
                logger.warning(f"Band {wavelength}nm: {negative_count} significantly negative values detected")
            
            logger.debug(f"Band {wavelength}nm Rrs stats: min={min_val:.6f}, max={max_val:.6f}, mean={mean_val:.6f}")

    
    def _update_band_mapping_for_mixed_types(self, c2rcc_path: str) -> Dict[int, str]:
        """
        JIANG-FOCUSED: Load bands that can actually be used for Jiang TSS algorithm
        Priority: rhow > rrs (ONLY - these are the only complete 8-band datasets)
        
        Removed rhown: Only has 6 bands, missing B7 and B8A needed for Jiang algorithm
        Removed TOA: Not atmospherically corrected, unsuitable for water analysis
        """
        if c2rcc_path.endswith('.dim'):
            data_folder = c2rcc_path.replace('.dim', '.data')
        else:
            data_folder = f"{c2rcc_path}.data"
        
        if not os.path.exists(data_folder):
            logger.error(f"Data folder not found: {data_folder}")
            return {}
        
        # CLEAN: Only complete 8-band water reflectance products
        band_mapping = {
            443: ['rhow_B1.img', 'rrs_B1.img'],      # Blue
            490: ['rhow_B2.img', 'rrs_B2.img'],      # Blue
            560: ['rhow_B3.img', 'rrs_B3.img'],      # Green (Type I reference)
            665: ['rhow_B4.img', 'rrs_B4.img'],      # Red (Type II reference)
            705: ['rhow_B5.img', 'rrs_B5.img'],      # Red Edge
            740: ['rhow_B6.img', 'rrs_B6.img'],      # Red Edge (Type III reference)
            783: ['rhow_B7.img', 'rrs_B7.img'],      # NIR (needed for calculations)
            865: ['rhow_B8A.img', 'rrs_B8A.img']     # NIR (Type IV reference)
        }
        
        found_bands = {}
        band_type_summary = {}
        
        logger.info(f"🎯 Searching for COMPLETE 8-band datasets in: {data_folder}")
        
        for wavelength, possible_names in band_mapping.items():
            for name in possible_names:
                file_path = os.path.join(data_folder, name)
                if os.path.exists(file_path) and os.path.getsize(file_path) > 1024:  # >1KB
                    found_bands[wavelength] = file_path
                    
                    # Track band type for summary
                    if name.startswith('rhow_'):
                        band_type = 'rhow'
                    elif name.startswith('rrs_'):
                        band_type = 'rrs'
                    
                    band_type_summary[band_type] = band_type_summary.get(band_type, 0) + 1
                    logger.info(f"✓ Found {wavelength}nm: {name} ({band_type})")
                    break
            else:
                logger.warning(f"✗ Missing {wavelength}nm - CRITICAL for Jiang algorithm")
        
        # Detailed analysis for Jiang requirements
        logger.info(f"📊 Band type summary: {band_type_summary}")
        
        total_found = len(found_bands)
        required_bands = [443, 490, 560, 665, 705, 740, 783, 865]
        missing_critical = [wl for wl in required_bands if wl not in found_bands]
        
        # Jiang algorithm readiness assessment
        if total_found == 8 and not missing_critical:
            logger.info("🎉 PERFECT: Complete 8-band dataset found - Jiang algorithm ready")
        elif total_found >= 6 and 783 in found_bands and 865 in found_bands:
            logger.info(f"✅ USABLE: {total_found}/8 bands found including critical NIR bands")
            if missing_critical:
                logger.warning(f"   Missing non-critical: {missing_critical}nm")
        else:
            logger.error(f"❌ INSUFFICIENT: {total_found}/8 bands found")
            if missing_critical:
                logger.error(f"   Missing CRITICAL bands: {missing_critical}nm")
            
            # Specific warnings for missing critical bands
            if 783 not in found_bands:
                logger.error("   ⚠️  Missing 783nm (B7) - needed for QAA calculations")
            if 865 not in found_bands:
                logger.error("   ⚠️  Missing 865nm (B8A) - needed for extremely turbid waters")
            
            logger.error("   🚫 Jiang TSS algorithm cannot proceed reliably")
        
        # Unit conversion info
        if len(band_type_summary) > 1:
            logger.info("🔧 Mixed band types detected - automatic unit conversions will be applied")
        elif 'rrs' in band_type_summary:
            logger.info("🔧 RRS bands detected - no unit conversion needed (already in sr⁻¹)")
        elif 'rhow' in band_type_summary:
            logger.info("🔧 RHOW bands detected - will convert to Rrs (÷π)")
        
        return found_bands
 
    
    def process_jiang_tss(self, c2rcc_path: str, output_folder: str, product_name: str, intermediate_paths: Optional[Dict[str, str]] = None) -> Dict[str, ProcessingResult]:
        """
        Process Jiang TSS methodology from C2RCC outputs
        
        Advanced algorithms results are now properly passed to save method
        Advanced results are integrated into the main results dictionary
        """
        try:
            logger.info(f"Starting Jiang TSS processing for: {product_name}")
            
            # CRITICAL FIX: Extract proper georeference from C2RCC output
            try:
                # Get reference metadata from C2RCC data
                data_folder = c2rcc_path.replace('.dim', '.data')
                sample_band_path = os.path.join(data_folder, 'rrs_B4.img')  # Use red band as reference
                
                if os.path.exists(sample_band_path):
                    _, reference_metadata = RasterIO.read_raster(sample_band_path)
                    logger.info("✅ Using C2RCC georeference for proper geographic positioning")
                else:
                    logger.error("❌ Cannot find reference band for georeference - products will have incorrect coordinates")
                    reference_metadata = {
                        'geotransform': None,
                        'projection': None,
                        'width': None,
                        'height': None,
                        'nodata': -9999
                    }
            except Exception as e:
                logger.error(f"Error extracting georeference: {e}")
                    
            # Create intermediate tracking
            intermediate_paths = {}
            
            # Step 1: Load spectral bands directly
            logger.info("📡 Step 1: Loading spectral bands from C2RCC output")
            data_folder = c2rcc_path.replace('.dim', '.data')

            # Simple band mapping - RRS FIRST (Jiang algorithm native input)
            band_files = {
                443: ['rrs_B1.img', 'rhow_B1.img'],    # Try rrs first, fallback to rhow
                490: ['rrs_B2.img', 'rhow_B2.img'],
                560: ['rrs_B3.img', 'rhow_B3.img'],
                665: ['rrs_B4.img', 'rhow_B4.img'],
                705: ['rrs_B5.img', 'rhow_B5.img'],
                740: ['rrs_B6.img', 'rhow_B6.img'],
                783: ['rrs_B7.img', 'rhow_B7.img'],
                865: ['rrs_B8A.img', 'rhow_B8A.img']
            }

            bands_data = {}
            band_paths = {}
            for wavelength, possible_files in band_files.items():
                for filename in possible_files:
                    file_path = os.path.join(data_folder, filename)
                    if os.path.exists(file_path):
                        try:
                            data, _ = RasterIO.read_raster(file_path)
                            bands_data[wavelength] = data
                            band_paths[wavelength] = file_path
                            logger.info(f"✓ Loaded {wavelength}nm: {filename}")
                            break  # Found the band, stop looking
                        except Exception as e:
                            logger.error(f"Failed to load {wavelength}nm from {filename}: {e}")

            if not bands_data:
                error_msg = "No spectral bands found in C2RCC output"
                logger.error(error_msg)
                return {'error': ProcessingResult(False, "", None, error_msg)}
            
            # Step 3: Apply unit conversion 
            logger.info("🔄 Step 3: Converting rhow to Rrs")
            converted_bands_data = self._apply_simple_unit_conversion(bands_data, band_paths)
            
            # Step 4: Apply Jiang methodology with converted data
            logger.info("🧮 Step 4: Applying Jiang TSS methodology")
            jiang_results = self._apply_full_jiang_methodology(converted_bands_data)
            
            # Step 5: Process advanced algorithms (ONLY ONCE)
            advanced_results = {}
            if (hasattr(self.config, 'enable_advanced_algorithms') and 
                self.config.enable_advanced_algorithms and 
                hasattr(self, 'advanced_processor') and 
                self.advanced_processor is not None):
                
                logger.info("🔬 Step 5: Processing advanced algorithms")
                
                # SINGLE CALL to _process_advanced_algorithms
                advanced_results = self._process_advanced_algorithms(
                    c2rcc_path, jiang_results, converted_bands_data, product_name
                )
                
                # CRITICAL FIX: Check if advanced_results is None before using len()
                if advanced_results is None:
                    logger.warning("Advanced algorithms returned None, using empty dictionary")
                    advanced_results = {}
                
                logger.info(f"Advanced algorithms completed: {len(advanced_results)} additional products")

            # Combine all results including advanced algorithms
            all_algorithm_results = jiang_results.copy()

            # Add advanced algorithms data to the combined results for saving
            for key, result in advanced_results.items():
                if isinstance(result, ProcessingResult) and result.statistics and 'numpy_data' in result.statistics:
                    # FIXED: Extract the raw numpy array from statistics for saving
                    all_algorithm_results[key] = result.statistics['numpy_data']
                
            # Step 6: Save complete results (now includes advanced algorithms)
            logger.info("💾 Step 6: Saving complete results including advanced algorithms")
            saved_results = self._save_complete_results(all_algorithm_results, output_folder, product_name, reference_metadata)
            
            # Update ProcessingResult objects with actual file paths
            final_results = saved_results.copy()
            
            # Update advanced algorithm results with actual output paths from saved_results
            for key, advanced_result in advanced_results.items():
                if key in saved_results:
                    # Update the ProcessingResult with the actual saved path
                    advanced_result.output_path = saved_results[key].output_path
                    final_results[key] = advanced_result
            
            # Step 7: Marine visualization processing (if enabled)
            if (hasattr(self.config, 'enable_marine_visualization') and 
                self.config.enable_marine_visualization and 
                hasattr(self, 'marine_viz_processor') and 
                self.marine_viz_processor is not None):
                
                logger.info("🌊 Step 7: Processing marine visualizations")
                
                # CRITICAL FIX: Calculate correct paths based on actual folder structure
                # Current output_folder points to: TSS_Products/SCENE_NAME/
                # We need to go up 2 levels to get to Results folder where Geometric_Products is located
                
                try:
                    results_folder = os.path.dirname(output_folder)
                    geometric_folder = os.path.join(results_folder, "Geometric_Products")
                    
                    # Set up intermediate_paths with correct geometric location
                    if intermediate_paths is None:
                        intermediate_paths = {}
                    
                    # Find the geometric product file
                    clean_product_name = product_name.replace('.zip', '').replace('.SAFE', '')
                    if 'MSIL1C' in clean_product_name:
                        parts = clean_product_name.split('_')
                        if len(parts) >= 6:
                            clean_name = f"{parts[0]}_{parts[2]}_{parts[5]}"
                        else:
                            clean_name = clean_product_name.replace('MSIL1C_', '')
                    else:
                        clean_name = clean_product_name
                    
                    geometric_filename = f"Resampled_{clean_name}_Subset.dim"
                    geometric_path = os.path.join(geometric_folder, geometric_filename)
                    
                    # FIXED: Set the correct geometric path
                    intermediate_paths['geometric_path'] = geometric_path
                    
                    # Debug logging
                    logger.info(f"Current output_folder: {output_folder}")
                    logger.info(f"Results folder: {results_folder}")
                    logger.info(f"Looking for geometric products: {geometric_path}")
                    
                    # Verify geometric products exist
                    if os.path.exists(geometric_path):
                        logger.info("✅ Geometric products found for marine visualization")
                        
                        # Call marine visualization with correct scene output folder
                        # Marine viz will create RGB_Composites/ and Spectral_Indices/ in the scene folder
                        viz_results = self.marine_viz_processor.process_marine_visualizations(
                            c2rcc_path, output_folder, product_name, intermediate_paths
                        )
                        
                        # Add visualization results to final results
                        final_results.update(viz_results)
                        
                        # Count successful visualizations
                        rgb_count = len([k for k in viz_results.keys() if k.startswith('rgb_')])
                        index_count = len([k for k in viz_results.keys() if k.startswith('index_')])
                        logger.info(f"Marine visualization completed: {rgb_count} RGB + {index_count} indices")
                        
                    else:
                        logger.error(f"❌ Geometric products not found at: {geometric_path}")
                        logger.error("Cannot proceed with marine visualization")
                        
                        # Add error to results
                        final_results['visualization_error'] = ProcessingResult(
                            success=False,
                            output_path="",
                            statistics=None,
                            error_message="Geometric products not found for visualization"
                        )
                        
                except Exception as viz_error:
                    logger.error(f"Marine visualization processing failed: {viz_error}")
                    import traceback
                    logger.error(f"Visualization error traceback: {traceback.format_exc()}")
                    
                    # Add error to results
                    final_results['visualization_error'] = ProcessingResult(
                        success=False,
                        output_path="",
                        statistics=None,
                        error_message=str(viz_error)
                    )
            
            else:
                if not getattr(self.config, 'enable_marine_visualization', False):
                    logger.info("🌊 Marine visualization disabled in configuration")
                else:
                    logger.warning("🌊 Marine visualization processor not available")
        
            # Final validation and cleanup
            success_count = 0
            for key, result in final_results.items():
                if isinstance(result, ProcessingResult) and hasattr(result, 'success'):
                    if result.success:
                        success_count += 1
                elif result is not None:
                    success_count += 1
            total_count = len(final_results)

            logger.info("=" * 80)
            logger.info(f"✅ COMPLETE TSS PROCESSING FINISHED: {product_name}")
            logger.info(f"   Total products generated: {success_count}/{total_count}")
            logger.info(f"   Success rate: {(success_count/total_count)*100:.1f}%")
            
            # Log product breakdown
            if success_count > 0:
                jiang_count = len([k for k in final_results.keys() if k.startswith('jiang_')])
                advanced_count = len([k for k in final_results.keys() if k.startswith('advanced_')])
                rgb_count = len([k for k in final_results.keys() if k.startswith('rgb_')])
                index_count = len([k for k in final_results.keys() if k.startswith('index_')])
                
                logger.info(f"   Product breakdown:")
                logger.info(f"     • Jiang TSS products: {jiang_count}")
                if advanced_count > 0:
                    logger.info(f"     • Advanced algorithms: {advanced_count}")
                if rgb_count > 0:
                    logger.info(f"     • RGB composites: {rgb_count}")
                if index_count > 0:
                    logger.info(f"     • Spectral indices: {index_count}")
            
            logger.info("=" * 80)
            
            return final_results
            
        except Exception as e:
            error_msg = f"Jiang TSS processing failed: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return {'error': ProcessingResult(False, "", None, error_msg)}

        
    def _process_advanced_algorithms(self, c2rcc_path: str, jiang_results: Dict, 
                                    rrs_bands_data: Dict, product_name: str) -> Dict[str, ProcessingResult]:
        """
        Process advanced algorithms with properly converted Rrs data
        
        This method now returns ProcessingResults that can be saved properly
        Returns ProcessingResult objects instead of raw numpy arrays
        """
        try:
            logger.info("Processing advanced algorithms with unit-converted data")
            
            # CRITICAL FIX: Initialize advanced_results as empty dictionary
            advanced_results = {}
            
            config = self.config.advanced_config
            
            if config is None:
                logger.warning("No advanced config available, using defaults")
                config = AdvancedAquaticConfig()
            
            # Get output folder for this product
            output_folder = self.config.output_folder if hasattr(self.config, 'output_folder') else ""
            clean_product_name = product_name.replace('.zip', '').replace('.SAFE', '')
            
            # =======================================================================
            # 1. WATER CLARITY CALCULATION
            # =======================================================================
            if config.enable_water_clarity and 'absorption' in jiang_results and 'backscattering' in jiang_results:
                logger.info("Calculating water clarity indices")
                
                absorption = jiang_results['absorption']
                backscattering = jiang_results['backscattering']
                
                try:
                    clarity_results = self.advanced_processor.calculate_water_clarity(
                        absorption, backscattering, config.solar_zenith_angle
                    )
                    
                    # FIXED: Store everything in statistics dictionary
                    for key, value in clarity_results.items():
                        if value is not None and isinstance(value, np.ndarray):
                            # Calculate basic statistics
                            stats = RasterIO.calculate_statistics(value) if hasattr(self, 'RasterIO') else {
                                'coverage_percent': 100.0, 
                                'valid_pixels': value.size, 
                                'data_range': [value.min(), value.max()]
                            }
                            
                            # FIXED: Store all metadata and data in statistics dictionary
                            stats['numpy_data'] = value  # Store the actual numpy array
                            stats['product_type'] = 'water_clarity'
                            stats['algorithm'] = 'advanced_aquatic'
                            stats['description'] = f"Water clarity {key} - Advanced Aquatic Processing"
                            stats['processing_date'] = datetime.now().isoformat()
                            stats['source_absorption'] = 'jiang_methodology'
                            stats['source_backscattering'] = 'jiang_methodology'
                            
                            # FIXED: Only use parameters that ProcessingResult accepts
                            advanced_results[f'advanced_clarity_{key}'] = ProcessingResult(
                                success=True,
                                output_path="",  # Will be set during save
                                statistics=stats,
                                error_message=None
                            )
                    
                    logger.info(f"Water clarity calculation completed: {len(clarity_results)} products")
                    
                except Exception as e:
                    logger.error(f"Water clarity calculation failed: {e}")
            
            # =======================================================================
            # 2. HAB DETECTION CALCULATION - FIXED STRUCTURE
            # =======================================================================
            if config.enable_hab_detection and rrs_bands_data:
                logger.info("Detecting harmful algal blooms using converted Rrs data")
                
                try:
                    hab_results = self.advanced_processor.detect_harmful_algal_blooms(
                        chlorophyll=None,  # Not needed - calculated from spectral data
                        phycocyanin=None,  # Not available from S2
                        rrs_bands=rrs_bands_data  # Already in correct Rrs units
                    )
                    
                    # FIXED: Store everything in statistics dictionary
                    for key, value in hab_results.items():
                        if value is not None and isinstance(value, np.ndarray):
                            # Calculate basic statistics
                            stats = RasterIO.calculate_statistics(value) if hasattr(self, 'RasterIO') else {
                                'coverage_percent': 100.0, 
                                'valid_pixels': value.size, 
                                'data_range': [value.min(), value.max()]
                            }
                            
                            # FIXED: Store all metadata and data in statistics dictionary
                            stats['numpy_data'] = value  # Store the actual numpy array
                            stats['product_type'] = 'hab_detection'
                            stats['algorithm'] = 'advanced_aquatic'
                            stats['description'] = f"HAB {key} - Advanced Aquatic Processing"
                            stats['processing_date'] = datetime.now().isoformat()
                            stats['source_rrs'] = 'c2rcc_atmospheric_correction'
                            stats['biomass_threshold'] = config.hab_biomass_threshold
                            stats['extreme_threshold'] = config.hab_extreme_threshold
                            
                            # FIXED: Only use parameters that ProcessingResult accepts
                            advanced_results[f'advanced_hab_{key}'] = ProcessingResult(
                                success=True,
                                output_path="",  # Will be set during save
                                statistics=stats,
                                error_message=None
                            )
                    
                    logger.info(f"HAB detection completed: {len(hab_results)} products")
                    
                except Exception as e:
                    logger.error(f"HAB detection failed: {e}")
            
            else:
                # FIXED: This else is properly attached to the if statement
                logger.warning("No suitable converted spectral bands available for HAB detection")
            
            logger.info(f"✅ Advanced algorithms completed: {len(advanced_results)} products generated")
            
            # CRITICAL FIX: Always return a dictionary, never None
            return advanced_results
            
        except Exception as e:
            logger.error(f"Error in advanced algorithms processing: {e}")
            import traceback
            traceback.print_exc()
            # CRITICAL FIX: Return empty dictionary instead of None
            return {}
            
    # This method enables to extract SNAP chlorophyll
    def _extract_snap_chlorophyll(self, c2rcc_path: str) -> Optional[np.ndarray]:
        """Extract chlorophyll from SNAP C2RCC output"""
        try:
            data_folder = c2rcc_path.replace('.dim', '.data')
            chl_path = os.path.join(data_folder, 'conc_chl.img')
            
            if os.path.exists(chl_path):
                chl_data, _ = RasterIO.read_raster(chl_path)
                logger.info("Successfully extracted SNAP chlorophyll data")
                return chl_data
            else:
                logger.warning("SNAP chlorophyll data not found")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting SNAP chlorophyll: {e}")
            return None
        
    def _apply_full_jiang_methodology(self, rrs_data: Dict[int, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Complete Jiang methodology implementation - UPDATED to work with pre-converted Rrs data
        """
        logger.info("Applying Jiang methodology to pre-converted Rrs data")
        
        # Get array shape
        shape = rrs_data[443].shape
        
        # Initialize output arrays
        absorption = np.full(shape, np.nan, dtype=np.float32)
        backscattering = np.full(shape, np.nan, dtype=np.float32)
        reference_band = np.full(shape, np.nan, dtype=np.float32)
        tss_concentration = np.full(shape, np.nan, dtype=np.float32)
        
        # Add water type classification array
        water_type_classification = np.full(shape, 0, dtype=np.uint8)  # 0 = Invalid/Land
        
        # NOTE: Data is already in Rrs units from unit conversion step
        logger.info("ℹ️  Using pre-converted Rrs data (no additional π division)")
        
        # Create valid pixel mask
        valid_mask = self._create_valid_pixel_mask(rrs_data)
        
        if np.any(valid_mask):
            # Apply methodology to valid pixels
            pixel_results = self._process_valid_pixels(rrs_data, valid_mask)
            
            # Fill output arrays
            absorption[valid_mask] = pixel_results['absorption']
            backscattering[valid_mask] = pixel_results['backscattering']
            reference_band[valid_mask] = pixel_results['reference_band']
            tss_concentration[valid_mask] = pixel_results['tss']
            
            # Create water type classification map
            ref_bands = pixel_results['reference_band']
            
            # Map wavelengths to water type codes
            water_type_classification[valid_mask] = np.select(
                [
                    ref_bands == 560,  # Clear water (Type I)
                    ref_bands == 665,  # Moderately turbid (Type II)
                    ref_bands == 740,  # Highly turbid (Type III)
                    ref_bands == 865   # Extremely turbid (Type IV)
                ],
                [1, 2, 3, 4],  # Water type codes
                default=0  # Invalid/Land
            )
        
        # Calculate statistics
        valid_pixels = np.sum(valid_mask)
        total_pixels = shape[0] * shape[1]
        coverage_percent = (valid_pixels / total_pixels) * 100
        
        logger.info(f"✅ Jiang processing completed:")
        logger.info(f"  Valid pixels: {valid_pixels}/{total_pixels} ({coverage_percent:.1f}%)")
        
        # Log water type distribution
        if valid_pixels > 0:
            ref_bands_valid = reference_band[valid_mask]
            ref_bands_valid = ref_bands_valid[~np.isnan(ref_bands_valid)]
            
            if len(ref_bands_valid) > 0:
                logger.info("Water type distribution:")
                for band in [560, 665, 740, 865]:
                    count = np.sum(ref_bands_valid == band)
                    if count > 0:
                        percentage = (count / len(ref_bands_valid)) * 100
                        water_type = {
                            560: "Type I (Clear water)",
                            665: "Type II (Moderately turbid)", 
                            740: "Type III (Highly turbid)",
                            865: "Type IV (Extremely turbid)"
                        }[band]
                        logger.info(f"  {band}nm ({water_type}): {count} pixels ({percentage:.1f}%)")
        
        return {
            'absorption': absorption,
            'backscattering': backscattering, 
            'reference_band': reference_band,
            'tss': tss_concentration,
            'valid_mask': valid_mask,
            'water_type_classification': water_type_classification
        }
    
    def _create_visualization_summary(self, viz_results: Dict[str, ProcessingResult], 
                                     output_folder: str, product_name: str):
        """Create visualization summary report"""
        try:
            summary_file = os.path.join(output_folder, f"{product_name}_Visualization_Summary.txt")
            
            with open(summary_file, 'w') as f:
                f.write(f"MARINE VISUALIZATION SUMMARY\n")
                f.write(f"=" * 50 + "\n")
                f.write(f"Product: {product_name}\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # RGB Composites
                rgb_products = {k: v for k, v in viz_results.items() if k.startswith('rgb_')}
                if rgb_products:
                    f.write(f"RGB COMPOSITES ({len(rgb_products)} products):\n")
                    f.write(f"{'-'*30}\n")
                    for product_key, result in rgb_products.items():
                        if result.success and result.statistics:
                            stats = result.statistics
                            f.write(f"✓ {os.path.basename(result.output_path)}\n")
                            f.write(f"  Description: {stats.get('description', 'N/A')}\n")
                            f.write(f"  Bands: {stats.get('bands_used', 'N/A')}\n")
                            f.write(f"  Coverage: {stats.get('coverage_percent', 0):.1f}%\n")
                            f.write(f"  File Size: {stats.get('file_size_mb', 0):.1f} MB\n\n")
                        else:
                            f.write(f"✗ {product_key} (failed)\n\n")
                
                # Spectral Indices
                index_products = {k: v for k, v in viz_results.items() if k.startswith('index_')}
                if index_products:
                    f.write(f"SPECTRAL INDICES ({len(index_products)} products):\n")
                    f.write(f"{'-'*30}\n")
                    for product_key, result in index_products.items():
                        if result.success and result.statistics:
                            stats = result.statistics
                            f.write(f"✓ {os.path.basename(result.output_path)}\n")
                            f.write(f"  Description: {stats.get('description', 'N/A')}\n")
                            f.write(f"  Formula: {stats.get('formula', 'N/A')}\n")
                            f.write(f"  Application: {stats.get('application', 'N/A')}\n")
                            f.write(f"  Range: {stats.get('range', 'N/A')}\n")
                            f.write(f"  Coverage: {stats.get('coverage_percent', 0):.1f}%\n")
                            f.write(f"  Mean Value: {stats.get('mean', 0):.4f}\n\n")
                        else:
                            f.write(f"✗ {product_key} (failed)\n\n")
                
                f.write(f"Total visualization products: {len([r for r in viz_results.values() if r.success])}\n")
            
            logger.info(f"Visualization summary created: {os.path.basename(summary_file)}")
            
        except Exception as e:
            logger.warning(f"Could not create visualization summary: {e}")

    def _create_processing_summary(self, output_results: Dict[str, ProcessingResult], 
                                image_folder: str, product_name: str):
        """Create a comprehensive processing summary file"""
        try:
            summary_file = os.path.join(image_folder, f"{product_name}_ProcessingSummary.txt")
            
            # Count products by category
            categories = {
                'TSS Products': [k for k in output_results.keys() if not k.startswith(('advanced_', 'rgb_', 'index_'))],
                'Advanced Products': [k for k in output_results.keys() if k.startswith('advanced_')],
                'RGB Composites': [k for k in output_results.keys() if k.startswith('rgb_')],
                'Spectral Indices': [k for k in output_results.keys() if k.startswith('index_')]
            }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"SENTINEL-2 TSS PROCESSING SUMMARY\n")
                f.write(f"{'='*50}\n")
                f.write(f"Product: {product_name}\n")
                f.write(f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Pipeline: Unified S2-TSS Processing v1.0\n")
                f.write(f"Structure: Scene-based organization\n\n")
                
                # Processing statistics
                total_products = len(output_results)
                successful_products = len([r for r in output_results.values() if r.success])
                
                f.write(f"PROCESSING STATISTICS:\n")
                f.write(f"{'-'*25}\n")
                f.write(f"Total products generated: {total_products}\n")
                f.write(f"Successfully saved: {successful_products}\n")
                f.write(f"Success rate: {(successful_products/total_products)*100:.1f}%\n\n")
                
                # Product breakdown by category
                for category, products in categories.items():
                    if products:
                        f.write(f"{category.upper()}:\n")
                        f.write(f"{'-'*len(category)}\n")
                        for product_key in products:
                            if product_key in output_results:
                                result = output_results[product_key]
                                status = "✓" if result.success else "✗"
                                f.write(f"{status} {os.path.basename(result.output_path)}\n")
                        f.write(f"\n")
                
                # Folder structure
                f.write(f"FOLDER STRUCTURE:\n")
                f.write(f"{'-'*17}\n")
                f.write(f"{product_name}/\n")
                f.write(f"├── TSS_Products/ ({len(categories['TSS Products'])} files)\n")
                f.write(f"├── Advanced_Products/ ({len(categories['Advanced Products'])} files)\n")
                f.write(f"├── RGB_Composites/ ({len(categories['RGB Composites'])} files)\n")
                f.write(f"├── Spectral_Indices/ ({len(categories['Spectral Indices'])} files)\n")
                f.write(f"└── Metadata files (*.txt)\n\n")
                
                # Algorithm information
                f.write(f"ALGORITHMS APPLIED:\n")
                f.write(f"{'-'*19}\n")
                f.write(f"• Jiang et al. (2023) TSS methodology\n")
                if categories['Advanced Products']:
                    f.write(f"• Advanced aquatic algorithms\n")
                if categories['RGB Composites']:
                    f.write(f"• Marine visualization processing\n")
                if categories['Spectral Indices']:
                    f.write(f"• Spectral index calculations\n")
            
            logger.info(f"✓ Processing summary created: {os.path.basename(summary_file)}")
            
        except Exception as e:
            logger.warning(f"Could not create processing summary: {e}")
            
    def _create_valid_pixel_mask(self, rrs_data: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Validation matching R algorithm requirements
        """
        # Required bands for processing (from R validation)
        required_bands = [490, 560, 665, 740]
        
        # Initialize valid mask
        valid_mask = np.ones(rrs_data[443].shape, dtype=bool)
        
        # Check for valid data in required bands
        for band in required_bands:
            if band in rrs_data:
                # Not zero, not NaN, and positive values
                band_valid = (~np.isnan(rrs_data[band])) & (rrs_data[band] > 0)
                valid_mask &= band_valid
            else:
                # If required band is missing, mark all as invalid
                valid_mask[:] = False
                break
        
        return valid_mask
    
    def _process_valid_pixels(self, rrs_data: Dict[int, np.ndarray], 
                                      valid_mask: np.ndarray) -> Dict[str, np.ndarray]:
        """
        CORRECTED pixel processing using exact R algorithm
        """
        # Extract valid pixel data
        valid_pixels = {}
        for wavelength, data in rrs_data.items():
            valid_pixels[wavelength] = data[valid_mask]
        
        n_pixels = len(valid_pixels[443])
        logger.info(f"Processing {n_pixels} valid pixels with corrected Jiang algorithm")
        
        # Initialize output arrays for valid pixels
        absorption_out = np.full(n_pixels, np.nan, dtype=np.float32)
        backscattering_out = np.full(n_pixels, np.nan, dtype=np.float32)
        reference_band_out = np.full(n_pixels, np.nan, dtype=np.float32)
        tss_out = np.full(n_pixels, np.nan, dtype=np.float32)
        
        # Process each valid pixel using corrected algorithm
        for i in range(n_pixels):
            # Extract Rrs values for this pixel
            pixel_rrs = {wl: valid_pixels[wl][i] for wl in valid_pixels.keys()}
            
            # Apply corrected Jiang methodology to this pixel
            result = self._estimate_tss_single_pixel(pixel_rrs)
            
            if result is not None:
                absorption_out[i] = result['a']
                backscattering_out[i] = result['bbp']
                reference_band_out[i] = result['band']
                tss_out[i] = result['tss']
        
        return {
            'absorption': absorption_out,
            'backscattering': backscattering_out,
            'reference_band': reference_band_out,
            'tss': tss_out
        }
    
    def _estimate_tss_single_pixel(self, pixel_rrs: Dict[int, float]) -> Optional[Dict]:
        """
        EXACT R implementation: Estimate_TSS_Jiang_MSI <- function(site_Rrs)
        
        This is the main function that implements the complete algorithm exactly as in R.
        """
        try:
            # R code validation logic - EXACT translation
            # if (all(site_Rrs == 0) | all(is.na(site_Rrs) == TRUE) | 
            #     any(is.na(c(site_Rrs["Rrs490"],site_Rrs["Rrs560"],site_Rrs["Rrs665"],site_Rrs["Rrs740"])))){
            #     tmp_tss <- rep(NA,4)
            # }
            
            required_bands = [490, 560, 665, 740]
            
            # Check if all values are zero
            if all(v == 0 for v in pixel_rrs.values()):
                return None
            
            # Check if all values are NaN
            if all(np.isnan(v) for v in pixel_rrs.values()):
                return None
            
            # Check if any required bands are NaN
            if any(np.isnan(pixel_rrs.get(band, np.nan)) for band in required_bands):
                return None
            
            # R code: Rrs620 <- estimate_Rrs620(site_Rrs["Rrs665"])
            rrs620 = self._estimate_rrs620_from_rrs665(pixel_rrs[665])
            
            # R code band selection logic - EXACT translation
            # if (site_Rrs["Rrs490"] > site_Rrs["Rrs560"]){
            #     tmp_tss <- QAA_560(site_Rrs)
            # }else if (site_Rrs["Rrs490"] > Rrs620){
            #     tmp_tss <- QAA_665(site_Rrs)    
            # }else if (site_Rrs["Rrs740"] > site_Rrs["Rrs490"] & site_Rrs["Rrs740"] > 0.010){
            #     tmp_tss <- QAA_865(site_Rrs)      
            # }else{
            #     tmp_tss <- QAA_740(site_Rrs)     
            # }
            
            if pixel_rrs[490] > pixel_rrs[560]:
                # Type I: Clear water
                result = self._qaa_560(pixel_rrs)
            elif pixel_rrs[490] > rrs620:
                # Type II: Moderately turbid
                result = self._qaa_665(pixel_rrs)
            elif pixel_rrs[740] > pixel_rrs[490] and pixel_rrs[740] > 0.010:
                # Type IV: Extremely turbid (note: uses 865nm algorithm)
                result = self._qaa_865(pixel_rrs)
            else:
                # Type III: Highly turbid
                result = self._qaa_740(pixel_rrs)
            
            return result
            
        except Exception as e:
            logger.debug(f"Error processing pixel with Jiang algorithm: {e}")
            return None
        
    def _estimate_rrs620_from_rrs665(self, rrs665: float) -> float:
        """
        EXACT R implementation: estimate_Rrs620 <- function(in665)
        """
        coeffs = self.constants.RRS620_COEFFICIENTS
        a, b, c, d = coeffs['a'], coeffs['b'], coeffs['c'], coeffs['d']
        
        # R code: est620 <- a*in665^3+b*in665^2+c*in665+d
        rrs620 = a * (rrs665**3) + b * (rrs665**2) + c * rrs665 + d
        
        return rrs620
    
    def _qaa_560(self, site_rrs: Dict[int, float]) -> Dict:
        """
        EXACT R implementation: QAA_560 <- function(site_Rrs)
        """
        aw = self.constants.PURE_WATER_ABSORPTION
        bbw = self.constants.PURE_WATER_BACKSCATTERING
        
        # R code: rrs <- site_Rrs/(0.52+1.7*site_Rrs)
        rrs = {}
        for wl, rrs_val in site_rrs.items():
            rrs[wl] = rrs_val / (0.52 + 1.7 * rrs_val)
        
        # R code: u <- (-0.089+sqrt((0.089^2)+4*0.125*rrs))/(2*0.125)
        u = {}
        for wl, rrs_val in rrs.items():
            # Safe u calculation to avoid sqrt warnings
            if rrs_val > 0:
                discriminant = (0.089**2) + 4 * 0.125 * rrs_val
                if discriminant >= 0:
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', 'invalid value encountered in sqrt')
                        u[wl] = (-0.089 + np.sqrt(discriminant)) / (2 * 0.125)
                else:
                    u[wl] = np.nan
            else:
                u[wl] = np.nan
        
        # R code: x <- log((rrs["Rrs443"]+rrs["Rrs490"])/(rrs["Rrs560"]+5*rrs["Rrs665"]*rrs["Rrs665"]/rrs["Rrs490"]),10)
        numerator = rrs[443] + rrs[490]
        denominator = rrs[560] + 5 * rrs[665] * rrs[665] / rrs[490]
        x = np.log10(numerator / denominator)
        
        # R code: a560 <- aw["aw560"]+10^(-1.146-1.366*x-0.469*(x^2))
        a560 = aw[560] + 10**(-1.146 - 1.366*x - 0.469*(x**2))
        
        # R code: bbp560 <- ((u["Rrs560"]*a560)/(1-u["Rrs560"]))-bbw["bbw560"]
        bbp560 = ((u[560] * a560) / (1 - u[560])) - bbw[560]
        
        # R code: one_tss <- 94.48785*bbp560
        tss = self.constants.TSS_CONVERSION_FACTORS[560] * bbp560
        
        # R code: return(c(a560,bbp560,bbp_wave,one_tss))
        return {
            'a': a560,
            'bbp': bbp560,
            'band': 560,
            'tss': tss
        }
    
    def _qaa_665(self, site_rrs: Dict[int, float]) -> Dict:
        """
        EXACT R implementation: QAA_665 <- function(site_Rrs)
        """
        aw = self.constants.PURE_WATER_ABSORPTION
        bbw = self.constants.PURE_WATER_BACKSCATTERING
        
        # R code: rrs <- site_Rrs/(0.52+1.7*site_Rrs)
        rrs = {}
        for wl, rrs_val in site_rrs.items():
            rrs[wl] = rrs_val / (0.52 + 1.7 * rrs_val)
        
        # R code: u <- (-0.089+sqrt((0.089^2)+4*0.125*rrs))/(2*0.125)
        u = {}
        for wl, rrs_val in rrs.items():
            # Safe u calculation to avoid sqrt warnings
            if rrs_val > 0:
                discriminant = (0.089**2) + 4 * 0.125 * rrs_val
                if discriminant >= 0:
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', 'invalid value encountered in sqrt')
                        u[wl] = (-0.089 + np.sqrt(discriminant)) / (2 * 0.125)
                else:
                    u[wl] = np.nan
            else:
                u[wl] = np.nan
        
        # R code: a665 <- aw["aw665"]+0.39*((site_Rrs["Rrs665"]/(site_Rrs["Rrs443"]+site_Rrs["Rrs490"]))^1.14)
        # CRITICAL: Use original site_Rrs, NOT converted rrs
        ratio = site_rrs[665] / (site_rrs[443] + site_rrs[490])
        a665 = aw[665] + 0.39 * (ratio**1.14)
        
        # R code: bbp665 <- ((u["Rrs665"]*a665)/(1-u["Rrs665"]))-bbw["bbw665"]
        bbp665 = ((u[665] * a665) / (1 - u[665])) - bbw[665]
        
        # R code: one_tss <- 113.87498*bbp665
        tss = self.constants.TSS_CONVERSION_FACTORS[665] * bbp665
        
        # R code: return(c(a665,bbp665,bbp_wave,one_tss))
        return {
            'a': a665,
            'bbp': bbp665,
            'band': 665,
            'tss': tss
        }
    
    def _qaa_740(self, site_rrs: Dict[int, float]) -> Dict:
        """
        EXACT R implementation: QAA_740 <- function(site_Rrs)
        """
        aw = self.constants.PURE_WATER_ABSORPTION
        bbw = self.constants.PURE_WATER_BACKSCATTERING
        
        # R code: rrs <- site_Rrs/(0.52+1.7*site_Rrs)
        rrs = {}
        for wl, rrs_val in site_rrs.items():
            rrs[wl] = rrs_val / (0.52 + 1.7 * rrs_val)
        
        # R code: u <- (-0.089+sqrt((0.089^2)+4*0.125*rrs))/(2*0.125)
        u = {}
        for wl, rrs_val in rrs.items():
            # Safe u calculation to avoid sqrt warnings
            if rrs_val > 0:
                discriminant = (0.089**2) + 4 * 0.125 * rrs_val
                if discriminant >= 0:
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', 'invalid value encountered in sqrt')
                        u[wl] = (-0.089 + np.sqrt(discriminant)) / (2 * 0.125)
                else:
                    u[wl] = np.nan
            else:
                u[wl] = np.nan
        
        # R code: bbp740 <- ((u["Rrs740"]*aw["aw740"])/(1-u["Rrs740"]))-bbw["bbw740"]
        bbp740 = ((u[740] * aw[740]) / (1 - u[740])) - bbw[740]
        
        # R code: one_tss <- 134.91845*bbp740
        tss = self.constants.TSS_CONVERSION_FACTORS[740] * bbp740
        
        # R code: return(c(aw["aw740"],bbp740,bbp_wave,one_tss))
        # IMPORTANT: Returns aw[740] as absorption, not calculated value
        return {
            'a': aw[740],  # Uses pure water absorption directly
            'bbp': bbp740,
            'band': 740,
            'tss': tss
        }
    
    def _qaa_865(self, site_rrs: Dict[int, float]) -> Dict:
        """
        EXACT R implementation: QAA_865 <- function(site_Rrs)
        """
        aw = self.constants.PURE_WATER_ABSORPTION
        bbw = self.constants.PURE_WATER_BACKSCATTERING
        
        # R code: rrs <- site_Rrs/(0.52+1.7*site_Rrs)
        rrs = {}
        for wl, rrs_val in site_rrs.items():
            rrs[wl] = rrs_val / (0.52 + 1.7 * rrs_val)
        
        # R code: u <- (-0.089+sqrt((0.089^2)+4*0.125*rrs))/(2*0.125)
        u = {}
        for wl, rrs_val in rrs.items():
            # Safe u calculation to avoid sqrt warnings
            if rrs_val > 0:
                discriminant = (0.089**2) + 4 * 0.125 * rrs_val
                if discriminant >= 0:
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', 'invalid value encountered in sqrt')
                        u[wl] = (-0.089 + np.sqrt(discriminant)) / (2 * 0.125)
                else:
                    u[wl] = np.nan
            else:
                u[wl] = np.nan
        
        # R code: bbp865 <- ((u["Rrs865"]*aw["aw865"])/(1-u["Rrs865"]))-bbw["bbw865"]
        bbp865 = ((u[865] * aw[865]) / (1 - u[865])) - bbw[865]
        
        # R code: one_tss <- 166.07382*bbp865
        tss = self.constants.TSS_CONVERSION_FACTORS[865] * bbp865
        
        # R code: return(c(aw["aw865"],bbp865,bbp_wave,one_tss))
        # IMPORTANT: Returns aw[865] as absorption, not calculated value
        return {
            'a': aw[865],  # Uses pure water absorption directly
            'bbp': bbp865,
            'band': 865,
            'tss': tss
        }
    
    def _save_complete_results(self, results: Dict[str, np.ndarray], output_folder: str, 
                        product_name: str, reference_metadata: Dict) -> Dict[str, ProcessingResult]:
        """
        Save complete results - Core Jiang + Water Types + Advanced algorithms
        
        ENHANCED VERSION: Now properly handles advanced algorithm products with surgical fixes
        """
        try:
            output_results = {}
            
            # Clean product name (remove .zip extension)
            clean_product_name = product_name.replace('.zip', '').replace('.SAFE', '')
            
            # Create main output structure with scene-based folders
            scene_folder = os.path.join(output_folder, clean_product_name)
            tss_folder = os.path.join(scene_folder, "TSS_Products") 
            advanced_folder = os.path.join(scene_folder, "Advanced_Products")
            # SURGICAL FIX: Create additional folders for RGB and spectral indices
            rgb_folder = os.path.join(scene_folder, "RGB_Composites")
            spectral_folder = os.path.join(scene_folder, "Spectral_Indices")
            
            os.makedirs(tss_folder, exist_ok=True)
            os.makedirs(advanced_folder, exist_ok=True)
            os.makedirs(rgb_folder, exist_ok=True)
            os.makedirs(spectral_folder, exist_ok=True)
            
            # ========================================================================
            # CORE JIANG PRODUCTS (TSS_Products folder) - INCLUDING WATER TYPES
            # ========================================================================
            jiang_products = {
                'absorption': {
                    'data': results.get('absorption'),
                    'filename': f"{clean_product_name}_Jiang_Absorption.tif",
                    'description': "Absorption coefficient (m⁻¹) - Jiang et al. 2023",
                    'folder': tss_folder
                },
                'backscattering': {
                    'data': results.get('backscattering'),
                    'filename': f"{clean_product_name}_Jiang_Backscattering.tif", 
                    'description': "Particulate backscattering coefficient (m⁻¹) - Jiang et al. 2023",
                    'folder': tss_folder
                },
                'reference_band': {
                    'data': results.get('reference_band'),
                    'filename': f"{clean_product_name}_Jiang_ReferenceBand.tif",
                    'description': "Reference wavelength used (nm) - Jiang et al. 2023",
                    'folder': tss_folder
                },
                'tss': {
                    'data': results.get('tss'),
                    'filename': f"{clean_product_name}_Jiang_TSS.tif",
                    'description': "Total Suspended Solids (g/m³) - Jiang et al. 2023",
                    'folder': tss_folder
                },
                # Water Type Classification
                'water_type_classification': {
                    'data': results.get('water_type_classification'),
                    'filename': f"{clean_product_name}_Jiang_WaterTypes.tif",
                    'description': "Water Type Classification (0=Invalid, 1=Clear, 2=Moderate, 3=Highly turbid, 4=Extremely turbid) - Jiang et al. 2023",
                    'folder': tss_folder
                },
                'valid_mask': {
                    'data': results.get('valid_mask'),
                    'filename': f"{clean_product_name}_Jiang_ValidMask.tif",
                    'description': "Valid pixel mask - Jiang processing",
                    'folder': tss_folder
                }
            }
            
            # ========================================================================
            # ENHANCED ADVANCED ALGORITHM PRODUCTS HANDLING
            # ========================================================================
            advanced_products = {}
            
            # Handle all advanced products dynamically
            for key, data in results.items():
                if key.startswith('advanced_'):
                    # Determine product category and create appropriate filename
                    if 'clarity' in key:
                        category = 'WaterClarity'
                        description = "Water clarity analysis - Advanced Aquatic Processing"
                    elif 'hab' in key:
                        category = 'HAB'
                        description = "Harmful Algal Bloom detection - Advanced Aquatic Processing"
                    elif 'tsi' in key:
                        category = 'TrophicState'
                        description = "Trophic state assessment - Advanced Aquatic Processing"
                    else:
                        category = 'General'
                        description = "Advanced algorithm product"
                    
                    # Create clean filename
                    clean_key = key.replace('advanced_', '').replace('_', '').title()
                    filename = f"{clean_product_name}_Advanced_{category}_{clean_key}.tif"
                    
                    advanced_products[key] = {
                        'data': data,
                        'filename': filename,
                        'description': description,
                        'folder': advanced_folder
                    }
            
            # Keep existing reliable water clarity products if they exist
            clarity_products = {
                'secchi_depth': ('clarity_secchi_depth', "Secchi Depth (m) - Tyler 1968"),
                'clarity_index': ('clarity_clarity_index', "Water Clarity Index (0-1)"),
                'euphotic_depth': ('clarity_euphotic_depth', "Euphotic Depth (m) - 1% light level"),
                'diffuse_attenuation': ('clarity_diffuse_attenuation', "Diffuse Attenuation Coefficient (m⁻¹)"),
                'beam_attenuation': ('clarity_beam_attenuation', "Beam Attenuation Coefficient (m⁻¹)"),
                'turbidity_proxy': ('clarity_turbidity_proxy', "Turbidity Proxy (NTU equivalent)")
            }
            
            for product_key, (result_key, description) in clarity_products.items():
                if result_key in results:
                    advanced_products[product_key] = {
                        'data': results[result_key],
                        'filename': f"{clean_product_name}_Clarity_{product_key.replace('_', '').title()}.tif",
                        'description': description,
                        'folder': advanced_folder
                    }
            
            # Keep existing reliable HAB products if they exist
            hab_products = {
                'hab_probability': ('hab_hab_probability', "Harmful Algal Bloom Probability (0-1)"),
                'hab_risk_level': ('hab_hab_risk_level', "HAB Risk Level (0=None, 1=Low, 2=Medium, 3=High)"),
                'high_biomass_alert': ('hab_high_biomass_alert', "High Biomass Alert (threshold > 0.6)"),
                'extreme_biomass_alert': ('hab_extreme_biomass_alert', "Extreme Biomass Alert (threshold > 0.8)"),
                'ndci_bloom': ('hab_ndci_bloom', "NDCI Bloom Detection (Mishra & Mishra 2012)"),
                'flh_bloom': ('hab_flh_bloom', "Fluorescence Line Height Bloom (Gower et al. 1999)"),
                'mci_bloom': ('hab_mci_bloom', "Maximum Chlorophyll Index Bloom (Gitelson et al. 2008)"),
                'cyanobacteria_bloom': ('hab_cyanobacteria_bloom', "Cyanobacteria Bloom Detection"),
                'ndci_values': ('hab_ndci_values', "NDCI Values"),
                'flh_values': ('hab_flh_values', "Fluorescence Line Height Values"),
                'mci_values': ('hab_mci_values', "Maximum Chlorophyll Index Values")
            }
            
            for product_key, (result_key, description) in hab_products.items():
                if result_key in results:
                    advanced_products[product_key] = {
                        'data': results[result_key],
                        'filename': f"{clean_product_name}_HAB_{product_key.replace('hab_', '').title()}.tif",
                        'description': description,
                        'folder': advanced_folder
                    }
            
            # ========================================================================
            # RGB COMPOSITES HANDLING
            # ========================================================================
            rgb_products = {}
            for key, data in results.items():
                if key.startswith('rgb_'):
                    rgb_name = key.replace('rgb_', '').replace('_', '').title()
                    rgb_products[key] = {
                        'data': data,
                        'filename': f"{clean_product_name}_RGB_{rgb_name}.tif",
                        'description': f"RGB composite - {rgb_name}",
                        'folder': rgb_folder
                    }
            
            # ========================================================================
            # SPECTRAL INDICES HANDLING
            # ========================================================================
            spectral_products = {}
            for key, data in results.items():
                if key.startswith('index_'):
                    index_name = key.replace('index_', '').replace('_', '').upper()
                    spectral_products[key] = {
                        'data': data,
                        'filename': f"{clean_product_name}_Index_{index_name}.tif",
                        'description': f"Spectral index - {index_name}",
                        'folder': spectral_folder
                    }
            
            # ========================================================================
            # SAVE ALL PRODUCTS WITH FIXED DATA TYPE HANDLING
            # ========================================================================
            all_products = {**jiang_products, **advanced_products, **rgb_products, **spectral_products}
            
            logger.info(f"Saving {len(all_products)} products:")
            logger.info(f"  Core Jiang products: {len(jiang_products)}")
            logger.info(f"  Advanced products: {len(advanced_products)}")
            if rgb_products:
                logger.info(f"  RGB products: {len(rgb_products)}")
            if spectral_products:
                logger.info(f"  Spectral products: {len(spectral_products)}")
            
            # Define classification products that need uint8 and nodata=255
            classification_product_keys = [
                'hab_risk_level', 'reference_band', 'valid_mask', 'high_biomass_alert', 
                'extreme_biomass_alert', 'ndci_bloom', 'flh_bloom', 'mci_bloom', 
                'cyanobacteria_bloom', 'water_type_classification'
            ]
            
            # Save each product with FIXED data type handling
            saved_count = 0
            skipped_count = 0
            
            for product_key, product_info in all_products.items():
                if product_info['data'] is not None:
                    output_path = os.path.join(product_info['folder'], product_info['filename'])
                    
                    # Determine appropriate nodata value and handle data types properly
                    if any(class_key in product_key for class_key in classification_product_keys):
                        # Classification products: use uint8 and nodata=255
                        nodata_value = 255
                        
                        # Handle NaN values before casting to uint8
                        data_to_save = product_info['data'].copy().astype(np.float64)
                        
                        # Replace NaN with nodata value
                        data_to_save[np.isnan(data_to_save)] = nodata_value
                        
                        # Ensure values are in valid uint8 range (0-254, reserve 255 for nodata)
                        data_to_save = np.clip(data_to_save, 0, 254)
                        
                        # Set nodata pixels back to 255
                        original_data = product_info['data']
                        data_to_save[np.isnan(original_data)] = 255
                        
                        # Now safe to cast to uint8
                        data_to_save = data_to_save.astype(np.uint8)
                    else:
                        # Continuous products: use float32 and nodata=-9999
                        nodata_value = -9999
                        data_to_save = product_info['data'].astype(np.float32)
                    
                    # Use write_raster instead of save_raster for compatibility
                    success = RasterIO.write_raster(
                        data_to_save, 
                        output_path, 
                        reference_metadata, 
                        product_info['description'],
                        nodata=nodata_value
                    )
                    
                    if success:
                        # Use safe statistics calculation
                        try:
                            stats = RasterIO.calculate_statistics(product_info['data'])
                        except:
                            # Fallback statistics if calculation fails
                            valid_pixels = np.sum(~np.isnan(product_info['data'])) if isinstance(product_info['data'], np.ndarray) else 0
                            total_pixels = product_info['data'].size if isinstance(product_info['data'], np.ndarray) else 1
                            stats = {
                                'coverage_percent': (valid_pixels / total_pixels) * 100,
                                'valid_pixels': valid_pixels,
                                'data_range': [0, 1]  # Default range
                            }
                        
                        logger.debug(f"Saved {product_key}: {stats.get('coverage_percent', 0):.1f}% coverage")
                        
                        output_results[product_key] = ProcessingResult(
                            True, output_path, stats, None
                        )
                        saved_count += 1
                    else:
                        logger.error(f"Failed to save {product_key}")
                        output_results[product_key] = ProcessingResult(
                            False, output_path, None, f"Failed to write {product_key}"
                        )
                else:
                    logger.debug(f"Skipping {product_key}: no data available")
                    skipped_count += 1
            
            # ========================================================================
            # CREATE WATER TYPE LEGEND FILE (if method exists)
            # ========================================================================
            if 'water_type_classification' in results and results['water_type_classification'] is not None:
                try:
                    if hasattr(self, '_create_water_type_legend'):
                        self._create_water_type_legend(scene_folder, clean_product_name)
                except Exception as e:
                    logger.warning(f"Could not create water type legend: {e}")
            
            # ========================================================================
            # FINAL SUMMARY WITH ENHANCED LOGGING
            # ========================================================================
            logger.info(f"Product saving completed:")
            logger.info(f"  📁 Scene folder: {clean_product_name}/")
            logger.info(f"     ├── TSS_Products/ ({len(jiang_products)} files)")
            logger.info(f"     ├── Advanced_Products/ ({len(advanced_products)} files)")
            if rgb_products:
                logger.info(f"     ├── RGB_Composites/ ({len(rgb_products)} files)")
            if spectral_products:
                logger.info(f"     ├── Spectral_Indices/ ({len(spectral_products)} files)")
            logger.info(f"  ✅ Successfully saved: {saved_count} products")
            logger.info(f"  ⏭️  Skipped (no data): {skipped_count} products")
            logger.info(f"  📊 Total attempted: {len(all_products)} products")
            logger.info(f"  📂 Output folder: {scene_folder}")
            
            # Create processing summary (if method exists)
            try:
                if 'tss' in output_results and output_results['tss'].success:
                    if hasattr(self, '_log_processing_summary'):
                        self._log_processing_summary(results, clean_product_name)
            except Exception as e:
                logger.debug(f"Could not create processing summary: {e}")
            
            # Create product index file (if method exists)
            try:
                if hasattr(self, '_create_product_index'):
                    self._create_product_index(output_results, scene_folder, clean_product_name)
            except Exception as e:
                logger.debug(f"Could not create product index: {e}")
            
            return output_results
            
        except Exception as e:
            error_msg = f"Error saving complete results: {str(e)}"
            logger.error(error_msg)
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {'error': ProcessingResult(False, "", None, error_msg)}
        
    def _create_water_type_legend(self, output_folder: str, product_name: str):
        """Create a legend file for water type classification"""
        
        legend_file = os.path.join(output_folder, f"{product_name}_WaterTypes_Legend.txt")
        
        legend_content = """JIANG WATER TYPE CLASSIFICATION LEGEND
    ======================================

    Value | Water Type          | Algorithm Used | Characteristics
    ------|--------------------|--------------  |----------------
    0   | Invalid/Land       | N/A            | No valid data or land pixels
    1   | Clear Water        | QAA-560        | Low turbidity, high transparency
    2   | Moderately Turbid  | QAA-665        | Moderate suspended matter
    3   | Highly Turbid      | QAA-740        | High suspended matter concentration
    4   | Extremely Turbid   | QAA-865        | Very high turbidity, possible algal blooms

    ALGORITHM SELECTION CRITERIA (Jiang et al. 2023):
    ================================================

    Type I (Clear Water - 560nm):
    - Condition: Rrs(490) > Rrs(560)
    - Typical TSS: < 2 g/m³
    - Water clarity: High (Secchi depth > 10m)

    Type II (Moderately Turbid - 665nm):
    - Condition: Rrs(490) > Rrs(620) AND Rrs(490) ≤ Rrs(560)  
    - Typical TSS: 2-10 g/m³
    - Water clarity: Moderate (Secchi depth 3-10m)

    Type III (Highly Turbid - 740nm):
    - Condition: Rrs(740) ≤ Rrs(490) OR Rrs(740) ≤ 0.010
    - Typical TSS: 10-50 g/m³
    - Water clarity: Low (Secchi depth 1-3m)

    Type IV (Extremely Turbid - 865nm):
    - Condition: Rrs(740) > Rrs(490) AND Rrs(740) > 0.010
    - Typical TSS: > 50 g/m³
    - Water clarity: Very low (Secchi depth < 1m)

    COLOR SCHEME SUGGESTION:
    ========================
    Value 0 (Invalid): Transparent or Black
    Value 1 (Clear): Deep Blue (#0066CC)
    Value 2 (Moderate): Light Blue (#66B2FF) 
    Value 3 (Highly Turbid): Yellow (#FFFF00)
    Value 4 (Extremely Turbid): Red (#FF0000)

    Reference: Jiang, D., et al. (2023). A practical atmospheric correction algorithm 
    for Sentinel-2 images and extension to total suspended matter.
    """
        
        try:
            with open(legend_file, 'w', encoding='utf-8') as f:
                f.write(legend_content)
            
            logger.info(f"Water type legend created: {os.path.basename(legend_file)}")
        except Exception as e:
            logger.warning(f"Could not create water type legend: {e}")

    def _create_product_index(self, output_results: Dict[str, ProcessingResult], 
                            output_folder: str, product_name: str):
        """Create an index file listing all generated products"""
        try:
            index_file = os.path.join(output_folder, f"{product_name}_ProductIndex.txt")
            
            with open(index_file, 'w', encoding='utf-8') as f:
                f.write(f"SENTINEL-2 TSS PROCESSING RESULTS\n")
                f.write(f"{'='*50}\n")
                f.write(f"Product: {product_name}\n")
                f.write(f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Pipeline: Unified S2-TSS Processing v1.0\n\n")
                
                # Core Jiang Products
                f.write(f"CORE TSS PRODUCTS:\n")
                f.write(f"{'-'*20}\n")
                jiang_products = ['absorption', 'backscattering', 'reference_band', 'tss', 'valid_mask']
                for product in jiang_products:
                    if product in output_results and output_results[product].success:
                        f.write(f"✓ {os.path.basename(output_results[product].output_path)}\n")
                    else:
                        f.write(f"✗ {product} (failed or not generated)\n")
                
                # Advanced Products
                f.write(f"\nADVANCED ALGORITHM PRODUCTS:\n")
                f.write(f"{'-'*30}\n")
                
                categories = {
                    'Trophic State': ['tsi_chlorophyll', 'tsi_classification'],
                    'Water Clarity': ['secchi_depth', 'clarity_index', 'euphotic_depth'],
                    'Harmful Algal Blooms': ['hab_probability', 'hab_risk_level'],
                    'Upwelling Detection': ['upwelling_signature', 'upwelling_strength'],
                    'River Plumes': ['plume_intensity', 'plume_classification'],
                    'Particle Size': ['spectral_slope', 'size_classification'],
                    'Primary Productivity': ['primary_productivity', 'productivity_classification']
                }
                
                for category, products in categories.items():
                    f.write(f"\n{category}:\n")
                    for product in products:
                        if product in output_results and output_results[product].success:
                            f.write(f"  ✓ {os.path.basename(output_results[product].output_path)}\n")
                
                f.write(f"\nTotal products generated: {len([r for r in output_results.values() if r.success])}\n")
            
            logger.info(f"Product index created: {os.path.basename(index_file)}")
            
        except Exception as e:
            logger.warning(f"Could not create product index: {e}")

    def _log_processing_summary(self, results: Dict[str, np.ndarray], product_name: str):
        """Enhanced processing summary with advanced algorithm statistics"""
        
        tss_data = results.get('tss')
        if tss_data is None:
            return
        
        reference_bands = results.get('reference_band')
        valid_mask = results.get('valid_mask')
        
        # Overall statistics
        tss_stats = RasterIO.calculate_statistics(tss_data)
        
        logger.info(f"=== COMPLETE PROCESSING SUMMARY: {product_name} ===")
        logger.info(f"Total coverage: {tss_stats['coverage_percent']:.1f}%")
        logger.info(f"TSS range: {tss_stats['min']:.2f} - {tss_stats['max']:.2f} g/m³")
        logger.info(f"TSS mean: {tss_stats['mean']:.2f} g/m³")
        
        # Jiang algorithm usage statistics
        if reference_bands is not None and valid_mask is not None:
            ref_bands_valid = reference_bands[valid_mask]
            ref_bands_valid = ref_bands_valid[~np.isnan(ref_bands_valid)]
            
            if len(ref_bands_valid) > 0:
                logger.info("Jiang water type classification results:")
                for band in [560, 665, 740, 865]:
                    count = np.sum(ref_bands_valid == band)
                    percentage = (count / len(ref_bands_valid)) * 100
                    if count > 0:
                        water_type = {
                            560: "Type I (Clear)",
                            665: "Type II (Moderately turbid)", 
                            740: "Type III (Highly turbid)",
                            865: "Type IV (Extremely turbid)"
                        }[band]
                        logger.info(f"  {band}nm ({water_type}): {count} pixels ({percentage:.1f}%)")
        
        # Advanced algorithm summaries
        if 'tsi_trophic_classification' in results:
            tsi_class = results['tsi_trophic_classification']
            valid_tsi = tsi_class[~np.isnan(tsi_class)]
            if len(valid_tsi) > 0:
                logger.info("Trophic state distribution:")
                tsi_names = ['Invalid', 'Oligotrophic', 'Mesotrophic', 'Eutrophic', 'Hypereutrophic']
                for i, name in enumerate(tsi_names):
                    count = np.sum(valid_tsi == i)
                    if count > 0:
                        percentage = (count / len(valid_tsi)) * 100
                        logger.info(f"  {name}: {count} pixels ({percentage:.1f}%)")
        
        if 'hab_hab_risk_level' in results:
            hab_risk = results['hab_hab_risk_level']
            valid_hab = hab_risk[~np.isnan(hab_risk)]
            if len(valid_hab) > 0:
                high_risk_count = np.sum(valid_hab == 3)
                total_risk_count = np.sum(valid_hab > 0)
                if total_risk_count > 0:
                    logger.info(f"HAB detection: {total_risk_count} pixels with bloom risk ({high_risk_count} high risk)")
        
        logger.info("=" * 60)
    
    def _log_processing_summary(self, results: Dict[str, np.ndarray], product_name: str):
            """Log comprehensive processing summary"""
            
            tss_data = results['tss']
            reference_bands = results['reference_band']
            valid_mask = results['valid_mask']
            
            # Overall statistics
            tss_stats = RasterIO.calculate_statistics(tss_data)
            
            logger.info(f"=== FULL JIANG TSS PROCESSING SUMMARY: {product_name} ===")
            logger.info(f"Total coverage: {tss_stats['coverage_percent']:.1f}%")
            logger.info(f"TSS range: {tss_stats['min']:.2f} - {tss_stats['max']:.2f} g/m³")
            logger.info(f"TSS mean: {tss_stats['mean']:.2f} g/m³")
            
            # Band usage statistics
            if np.any(valid_mask):
                ref_bands_valid = reference_bands[valid_mask]
                ref_bands_valid = ref_bands_valid[~np.isnan(ref_bands_valid)]
                
                if len(ref_bands_valid) > 0:
                    logger.info("Water type classification results:")
                    for band in [560, 665, 740, 865]:
                        count = np.sum(ref_bands_valid == band)
                        percentage = (count / len(ref_bands_valid)) * 100
                        if count > 0:
                            water_type = {
                                560: "Type I (Clear)",
                                665: "Type II (Moderately turbid)", 
                                740: "Type III (Highly turbid)",
                                865: "Type IV (Extremely turbid)"
                            }[band]
                            logger.info(f"  {band}nm ({water_type}): {count} pixels ({percentage:.1f}%)")
            
            logger.info("=" * 60)    

# =============================================================================
# ADVANCED AQUATIC ALGORITHMS WITH BIBLIOGRAPHIC REFERENCES
# =============================================================================
# Implementation of cutting-edge algorithms for comprehensive water quality analysis
# All algorithms based on peer-reviewed scientific literature

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import os
import logging

logger = logging.getLogger(__name__)

@dataclass
class AdvancedAlgorithmConstants:
    """Constants for advanced aquatic algorithms based on scientific literature"""
    
        
    # OC3 algorithm coefficients (O'Reilly et al., 1998)
    OC3_COEFFS = [0.3272, -2.9940, 2.7218, -1.2259, -0.5683]
    
    # VGPM coefficients (Behrenfeld & Falkowski, 1997)
    VGPM_PBOPT_MAX = 4.0  # mg C (mg Chl)⁻¹ h⁻¹
    VGPM_SST_OPT = 20.0   # Optimal temperature (°C)
    
    # Particle size coefficients (Boss et al., 2001)
    PARTICLE_SIZE_THRESHOLDS = {
        'small': 1.5,    # η > 1.5: Small particles (<2 µm)
        'medium': 0.5,   # 0.5 < η < 1.5: Medium particles (2-20 µm)
        'large': 0.0     # η < 0.5: Large particles (>20 µm)
    }

class AdvancedAquaticProcessor:
    """
    Advanced aquatic algorithms processor with complete scientific implementations
    """
    
    def __init__(self):
        self.constants = AdvancedAlgorithmConstants()
        logger.info("Initialized Advanced Aquatic Processor with scientific algorithms")
    
        
    def calculate_water_clarity(self, absorption: np.ndarray, 
                               backscattering: np.ndarray,
                               solar_zenith: float = 30.0) -> Dict[str, np.ndarray]:
        """
        Calculate water clarity indices from bio-optical properties
        
        References:
        - Kirk, J.T.O. (2011). Light and photosynthesis in aquatic ecosystems. Cambridge University Press.
        - Lee, Z. et al. (2002). Deriving inherent optical properties from water color. Applied Optics, 41(27), 5755-5772.
        - Tyler, J.E. (1968). The Secchi disc. Limnology and Oceanography, 13(1), 1-6.
        - Preisendorfer, R.W. (1986). Secchi disk science: Visual optics of natural waters. 
          Limnology and Oceanography, 31(5), 909-926.
        
        Args:
            absorption: Absorption coefficient at 443nm (m⁻¹)
            backscattering: Backscattering coefficient at 443nm (m⁻¹)
            solar_zenith: Solar zenith angle (degrees)
            
        Returns:
            Dictionary with clarity indices
        """
        try:
            logger.info("Calculating water clarity indices")
            
            # Convert solar zenith to cosine
            mu0 = np.cos(np.radians(solar_zenith))
            
            # Gordon equation for diffuse attenuation coefficient (Gordon, 1989)
            kd = absorption + backscattering * (1 + 0.425 * mu0) / mu0
            
            # Tyler (1968) Secchi depth approximation
            secchi_depth = 1.7 / kd
            
            # Water clarity index (0-1 scale)
            clarity_index = 1 / (1 + kd)
            
            # Euphotic depth (1% light level)
            euphotic_depth = 4.605 / kd  # ln(100) / kd
            
            # Beam attenuation coefficient (approximate)
            beam_attenuation = absorption + backscattering
            
            # Turbidity proxy (NTU approximation)
            turbidity_proxy = backscattering * 1000  # Rough conversion
            
            results = {
                'diffuse_attenuation': kd,
                'secchi_depth': secchi_depth,
                'clarity_index': clarity_index,
                'euphotic_depth': euphotic_depth,
                'beam_attenuation': beam_attenuation,
                'turbidity_proxy': turbidity_proxy
            }
            
            # Calculate statistics
            valid_pixels = np.sum(~np.isnan(kd))
            if valid_pixels > 0:
                logger.info(f"Water clarity calculated for {valid_pixels} pixels")
                logger.info(f"Mean Secchi depth: {np.nanmean(secchi_depth):.2f} m")
                logger.info(f"Mean clarity index: {np.nanmean(clarity_index):.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating water clarity: {e}")
            return {}
    
    def detect_harmful_algal_blooms(self, chlorophyll: Optional[np.ndarray],
                                phycocyanin: Optional[np.ndarray],
                                rrs_bands: Dict[int, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        FIXED HAB detection using only Sentinel-2 spectral bands
        
        References:
        - Wynne, T.T. et al. (2008). Relating spectral shape to cyanobacterial blooms. 
        - Mishra, S. & Mishra, D.R. (2012). Normalized difference chlorophyll index.
        - Gower, J. et al. (1999). Detection of intense plankton blooms using 709 nm band.
        """
        try:
            logger.info("Detecting harmful algal blooms using Sentinel-2 spectral analysis")
            
            results = {}
            
            if not rrs_bands:
                logger.warning("No Rrs bands available for HAB detection")
                return {}
            
            # Get reference shape from any available band
            ref_band = list(rrs_bands.values())[0]
            shape = ref_band.shape
            
            # Initialize result arrays
            hab_probability = np.zeros(shape, dtype=np.float32)
            ndci_values = np.full(shape, np.nan, dtype=np.float32)
            flh_values = np.full(shape, np.nan, dtype=np.float32)
            mci_values = np.full(shape, np.nan, dtype=np.float32)
            
            algorithms_applied = []
            
            # Method 1: Normalized Difference Chlorophyll Index (NDCI) - FIXED
            if 705 in rrs_bands and 665 in rrs_bands:
                logger.info("Calculating NDCI (Normalized Difference Chlorophyll Index)")
                
                band_705 = rrs_bands[705]
                band_665 = rrs_bands[665]
                
                # FIXED: Proper array validation using numpy operations
                valid_mask = (
                    (~np.isnan(band_705)) & 
                    (~np.isnan(band_665)) & 
                    (band_705 > 0) & 
                    (band_665 > 0)
                )
                
                denominator = band_705 + band_665
                valid_mask = valid_mask & (denominator > 1e-8)
                
                if np.any(valid_mask):
                    ndci_values[valid_mask] = ((band_705[valid_mask] - band_665[valid_mask]) / 
                                            denominator[valid_mask])
                    
                    # NDCI bloom threshold (Mishra & Mishra, 2012)
                    ndci_bloom = (ndci_values > 0.05).astype(np.float32)
                    results['ndci_bloom'] = ndci_bloom
                    results['ndci_values'] = ndci_values
                    
                    # Add to probability calculation
                    hab_probability += ndci_bloom * 0.3
                    algorithms_applied.append("NDCI")
                    
                    valid_count = np.sum(valid_mask)
                    logger.info(f"NDCI calculated for {valid_count} pixels")
            
            # Method 2: Fluorescence Line Height (FLH) approximation - FIXED
            if all(band in rrs_bands for band in [665, 705, 740]):
                logger.info("Calculating Fluorescence Line Height (FLH)")
                
                band_665 = rrs_bands[665]
                band_705 = rrs_bands[705]
                band_740 = rrs_bands[740]
                
                # FIXED: Proper array validation
                valid_mask = (
                    (~np.isnan(band_665)) & 
                    (~np.isnan(band_705)) & 
                    (~np.isnan(band_740))
                )
                
                if np.any(valid_mask):
                    # FLH calculation (baseline correction)
                    slope_factor = (705 - 665) / (740 - 665)  # 0.533
                    baseline = band_665 + (band_740 - band_665) * slope_factor
                    flh_values[valid_mask] = band_705[valid_mask] - baseline[valid_mask]
                    
                    # FLH bloom threshold
                    flh_bloom = (flh_values > 0.004).astype(np.float32)
                    results['flh_bloom'] = flh_bloom
                    results['flh_values'] = flh_values
                    
                    # Add to probability calculation
                    hab_probability += flh_bloom * 0.3
                    algorithms_applied.append("FLH")
                    
                    valid_count = np.sum(valid_mask)
                    logger.info(f"FLH calculated for {valid_count} pixels")
            
            # Method 3: Maximum Chlorophyll Index (MCI) approximation - FIXED
            if all(band in rrs_bands for band in [665, 705, 740, 865]):
                logger.info("Calculating Maximum Chlorophyll Index (MCI)")
                
                band_665 = rrs_bands[665]
                band_705 = rrs_bands[705]
                band_740 = rrs_bands[740]
                band_865 = rrs_bands[865]
                
                # FIXED: Proper validation for all bands using numpy operations
                valid_mask = (
                    (~np.isnan(band_665)) & 
                    (~np.isnan(band_705)) & 
                    (~np.isnan(band_740)) & 
                    (~np.isnan(band_865))
                )
                
                if np.any(valid_mask):
                    # MCI calculation (simplified for S2 bands)
                    slope = (740 - 665) / (865 - 665)
                    mci_values[valid_mask] = (
                        band_705[valid_mask] - band_665[valid_mask] - 
                        slope * (band_865[valid_mask] - band_665[valid_mask])
                    )
                    
                    # MCI bloom threshold
                    mci_bloom = (mci_values > 0.004).astype(np.float32)
                    results['mci_bloom'] = mci_bloom
                    results['mci_values'] = mci_values
                    
                    # Add to probability calculation
                    hab_probability += mci_bloom * 0.3
                    algorithms_applied.append("MCI")
                    
                    valid_count = np.sum(valid_mask)
                    logger.info(f"MCI calculated for {valid_count} pixels")
            
            # Calculate combined HAB probability and risk levels
            if algorithms_applied:
                # Normalize probability to 0-1 range
                hab_probability = np.clip(hab_probability, 0, 1)
                results['hab_probability'] = hab_probability
                
                # Create risk level classification
                hab_risk = np.zeros(shape, dtype=np.uint8)
                hab_risk[hab_probability > 0.7] = 3  # High risk
                hab_risk[(hab_probability > 0.4) & (hab_probability <= 0.7)] = 2  # Medium risk
                hab_risk[(hab_probability > 0.2) & (hab_probability <= 0.4)] = 1  # Low risk
                # hab_risk = 0 for probability <= 0.2 (no risk)
                
                results['hab_risk_level'] = hab_risk
                
                # Additional detection flags
                if 'ndci_bloom' in results or 'flh_bloom' in results:
                    # Cyanobacteria-like bloom detection
                    cyano_bloom = np.zeros(shape, dtype=np.float32)
                    
                    if 'ndci_bloom' in results:
                        cyano_bloom = np.maximum(cyano_bloom, results['ndci_bloom'])
                    if 'flh_bloom' in results:
                        cyano_bloom = np.maximum(cyano_bloom, results['flh_bloom'])
                    
                    results['cyanobacteria_bloom'] = cyano_bloom
                
                # Biomass alert levels
                high_biomass = (hab_probability > 0.6).astype(np.float32)
                extreme_biomass = (hab_probability > 0.8).astype(np.float32)
                
                results['high_biomass_alert'] = high_biomass
                results['extreme_biomass_alert'] = extreme_biomass
                
                # Calculate statistics
                total_pixels = np.sum(~np.isnan(hab_probability))
                if total_pixels > 0:
                    high_risk_pixels = np.sum(hab_risk == 3)
                    medium_risk_pixels = np.sum(hab_risk == 2)
                    low_risk_pixels = np.sum(hab_risk == 1)
                    
                    logger.info(f"HAB detection completed using algorithms: {', '.join(algorithms_applied)}")
                    logger.info(f"Processed {total_pixels} pixels")
                    logger.info(f"High risk: {high_risk_pixels} pixels ({100*high_risk_pixels/total_pixels:.1f}%)")
                    logger.info(f"Medium risk: {medium_risk_pixels} pixels ({100*medium_risk_pixels/total_pixels:.1f}%)")
                    logger.info(f"Low risk: {low_risk_pixels} pixels ({100*low_risk_pixels/total_pixels:.1f}%)")
            else:
                logger.warning("No suitable spectral bands found for HAB detection")
                # Return empty arrays to avoid missing data issues
                results['hab_probability'] = np.zeros(shape, dtype=np.float32)
                results['hab_risk_level'] = np.zeros(shape, dtype=np.uint8)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in HAB detection: {e}")
            import traceback
            traceback.print_exc()
            return {}
              
    
@dataclass
class AdvancedAquaticConfig:
    """Configuration for advanced aquatic algorithms"""
    
    # Trophic state calculation
    enable_trophic_state: bool = False
    tsi_include_secchi: bool = False
    tsi_include_phosphorus: bool = False
    
    # Water clarity calculation
    enable_water_clarity: bool = True
    solar_zenith_angle: float = 30.0
    
    # HAB detection
    enable_hab_detection: bool = True
    hab_biomass_threshold: float = 20.0
    hab_extreme_threshold: float = 100.0
    
    # Upwelling detection
    enable_upwelling_detection: bool = True
    upwelling_chl_threshold: float = 10.0
    
    # River plume tracking
    enable_river_plume_tracking: bool = True
    plume_tss_threshold: float = 15.0
    plume_distance_threshold: float = 10000
    
    # Particle size estimation
    enable_particle_size: bool = True
    particle_size_wavelengths: List[int] = None
    
    # Primary productivity
    enable_primary_productivity: bool = True
    productivity_model: str = 'vgpm'
    day_length: float = 12.0
    
    # Output options
    save_intermediate_products: bool = True
    create_classification_maps: bool = True
    generate_statistics: bool = True
    
    def __post_init__(self):
        if self.particle_size_wavelengths is None:
            self.particle_size_wavelengths = [443, 490, 560, 665, 705]

def create_advanced_processor(config: AdvancedAquaticConfig = None) -> AdvancedAquaticProcessor:
    """
    Factory function to create advanced aquatic processor
    
    Args:
        config: Advanced aquatic configuration
        
    Returns:
        Configured AdvancedAquaticProcessor instance
    """
    if config is None:
        config = AdvancedAquaticConfig()
    
    processor = AdvancedAquaticProcessor()
    processor.config = config
    
    return processor

# Integration helper functions
def integrate_with_existing_pipeline(c2rcc_results: Dict, 
                                   jiang_results: Dict,
                                   advanced_config: AdvancedAquaticConfig) -> Dict:
    """
    Integrate advanced algorithms with existing TSS pipeline
    
    Args:
        c2rcc_results: Results from C2RCC processing
        jiang_results: Results from Jiang TSS processing
        advanced_config: Configuration for advanced algorithms
        
    Returns:
        Dictionary with all advanced algorithm results
    """
    try:
        logger.info("Integrating advanced algorithms with existing pipeline")
        
        processor = create_advanced_processor(advanced_config)
        advanced_results = {}
        
        # Extract required data from existing results
        if 'chlorophyll' in c2rcc_results:
            chlorophyll = c2rcc_results['chlorophyll']
            
            # Trophic state index
            if advanced_config.enable_trophic_state:
                tsi_results = processor.calculate_trophic_state(chlorophyll)
                advanced_results.update({f'tsi_{k}': v for k, v in tsi_results.items()})
            
            # HAB detection
            if advanced_config.enable_hab_detection:
                rrs_bands = c2rcc_results.get('rrs_bands', {})
                phycocyanin = c2rcc_results.get('phycocyanin', None)
                hab_results = processor.detect_harmful_algal_blooms(chlorophyll, phycocyanin, rrs_bands)
                advanced_results.update({f'hab_{k}': v for k, v in hab_results.items()})
        
        # Bio-optical parameters from Jiang results
        if 'absorption' in jiang_results and 'backscattering' in jiang_results:
            absorption = jiang_results['absorption']
            backscattering = jiang_results['backscattering']
            
            # Water clarity
            if advanced_config.enable_water_clarity:
                clarity_results = processor.calculate_water_clarity(absorption, backscattering)
                advanced_results.update({f'clarity_{k}': v for k, v in clarity_results.items()})
            
            # Particle size estimation
            if advanced_config.enable_particle_size:
                # Create backscattering spectrum from available bands
                backscattering_spectrum = {}
                for wl in advanced_config.particle_size_wavelengths:
                    if f'backscattering_{wl}' in jiang_results:
                        backscattering_spectrum[wl] = jiang_results[f'backscattering_{wl}']
                
                if backscattering_spectrum:
                    particle_results = processor.estimate_particle_size(backscattering_spectrum)
                    advanced_results.update({f'particle_{k}': v for k, v in particle_results.items()})
        
        logger.info(f"Advanced algorithm integration completed: {len(advanced_results)} products generated")
        return advanced_results
        
    except Exception as e:
        logger.error(f"Error integrating advanced algorithms: {e}")
        return {}

@dataclass
class MarineVisualizationConfig:
    """Configuration for marine visualization products"""
    
    # RGB options
    generate_natural_color: bool = True
    generate_false_color: bool = True
    generate_water_specific: bool = True
    generate_research_combinations: bool = False
    
    # Spectral indices options
    generate_water_quality_indices: bool = True
    generate_chlorophyll_indices: bool = True
    generate_turbidity_indices: bool = True
    generate_advanced_indices: bool = False
    
    # Output format options
    rgb_format: str = 'GeoTIFF'
    export_metadata: bool = True
    create_overview_images: bool = True
    
    # Enhancement options
    apply_contrast_enhancement: bool = True
    contrast_method: str = 'percentile_stretch'
    percentile_range: tuple = (2, 98)

class S2MarineRGBGenerator:
    """Generate marine-optimized RGB composites and spectral indices"""
    
    def __init__(self):
        # Marine-optimized RGB combinations using Sentinel-2 bands
        self.rgb_combinations = {
            # Natural color combinations
            'natural_color': {
                'red': 665, 'green': 560, 'blue': 490,  # B4, B3, B2
                'description': 'Natural color (True color)',
                'application': 'General visualization, publications',
                'priority': 'essential'
            },
            'natural_with_contrast': {
                'red': 665, 'green': 560, 'blue': 443,  # B4, B3, B1
                'description': 'Natural color with enhanced contrast',
                'application': 'Better water-land contrast',
                'priority': 'important'
            },
            
            # False color for vegetation/water
            'false_color_infrared': {
                'red': 842, 'green': 665, 'blue': 560,  # B8, B4, B3
                'description': 'False color infrared',
                'application': 'Vegetation (red), clear water (dark)',
                'priority': 'important'
            },
            'false_color_nir': {
                'red': 865, 'green': 665, 'blue': 560,  # B8A, B4, B3
                'description': 'False color NIR',
                'application': 'Enhanced vegetation/water contrast',
                'priority': 'important'
            },
            
            # Water-specific combinations
            'turbidity_enhanced': {
                'red': 865, 'green': 705, 'blue': 560,  # B8A, B5, B3
                'description': 'Turbidity and suspended sediment',
                'application': 'Sediment plumes, river discharge',
                'priority': 'marine'
            },
            'chlorophyll_enhanced': {
                'red': 705, 'green': 665, 'blue': 560,  # B5, B4, B3
                'description': 'Chlorophyll and algae detection',
                'application': 'Algal blooms, phytoplankton',
                'priority': 'marine'
            },
            'coastal_aerosol': {
                'red': 865, 'green': 665, 'blue': 443,  # B8A, B4, B1
                'description': 'Coastal aerosol enhanced',
                'application': 'Atmospheric correction, haze penetration',
                'priority': 'marine'
            },
            
            # Specialized marine combinations
            'sediment_transport': {
                'red': 2190, 'green': 865, 'blue': 665,  # B12, B8A, B4
                'description': 'Sediment transport visualization',
                'application': 'River plumes, coastal erosion',
                'priority': 'marine'
            },
            'water_quality': {
                'red': 842, 'green': 705, 'blue': 490,  # B8, B5, B2
                'description': 'Water quality assessment',
                'application': 'Turbidity, organic matter',
                'priority': 'marine'
            },
            'atmospheric_penetration': {
                'red': 2190, 'green': 1610, 'blue': 865,  # B12, B11, B8A
                'description': 'Atmospheric penetration',
                'application': 'Hazy conditions, atmospheric interference',
                'priority': 'optional'
            },
            
            # Research combinations
            'research_marine': {
                'red': 740, 'green': 705, 'blue': 665,  # B6, B5, B4
                'description': 'Marine research combination',
                'application': 'Red edge analysis, chlorophyll research',
                'priority': 'research'
            },
            'bathymetric': {
                'red': 560, 'green': 490, 'blue': 443,  # B3, B2, B1
                'description': 'Bathymetric analysis',
                'application': 'Shallow water depth estimation',
                'priority': 'research'
            }
        }
        
        # Spectral indices for marine applications
        self.spectral_indices = {
            # Water quality indices
            'NDTI': {
                'formula': '(B4 - B3) / (B4 + B3)',
                'description': 'Normalized Difference Turbidity Index',
                'application': 'Turbidity, suspended sediment',
                'range': '(-1, 1)',
                'interpretation': 'Higher values = more turbid',
                'priority': 'essential'
            },
            'NDWI': {
                'formula': '(B3 - B8) / (B3 + B8)',
                'description': 'Normalized Difference Water Index',
                'application': 'Water body delineation',
                'range': '(-1, 1)',
                'interpretation': 'Positive values = water',
                'priority': 'essential'
            },
            'MNDWI': {
                'formula': '(B3 - B11) / (B3 + B11)',
                'description': 'Modified Normalized Difference Water Index',
                'application': 'Water extraction, better for turbid water',
                'range': '(-1, 1)',
                'interpretation': 'Positive values = water',
                'priority': 'important'
            },
            
            # Chlorophyll and algae indices
            'NDVI': {
                'formula': '(B8 - B4) / (B8 + B4)',
                'description': 'Normalized Difference Vegetation Index',
                'application': 'Vegetation, floating algae',
                'range': '(-1, 1)',
                'interpretation': 'Higher values = more vegetation/algae',
                'priority': 'important'
            },
            'GNDVI': {
                'formula': '(B8 - B3) / (B8 + B3)',
                'description': 'Green Normalized Difference Vegetation Index',
                'application': 'Chlorophyll, green algae',
                'range': '(-1, 1)',
                'interpretation': 'Sensitive to chlorophyll concentration',
                'priority': 'marine'
            },
            'NDCI': {
                'formula': '(B5 - B4) / (B5 + B4)',
                'description': 'Normalized Difference Chlorophyll Index',
                'application': 'Chlorophyll in turbid waters',
                'range': '(-1, 1)',
                'interpretation': 'Red edge based chlorophyll indicator',
                'priority': 'marine'
            },
            'CHL_RED_EDGE': {
                'formula': '(B6 / B5) - 1',
                'description': 'Chlorophyll Red Edge',
                'application': 'Chlorophyll concentration',
                'range': '(0, inf)',
                'interpretation': 'Red edge position indicator',
                'priority': 'marine'
            },
            
            # Turbidity and sediment indices
            'TSI': {
                'formula': 'B4 / B2',
                'description': 'Turbidity Sediment Index',
                'application': 'Suspended sediment concentration',
                'range': '(0, inf)',
                'interpretation': 'Higher values = more suspended sediment',
                'priority': 'essential'
            },
            'NGRDI': {
                'formula': '(B3 - B4) / (B3 + B4)',
                'description': 'Normalized Green Red Difference Index',
                'application': 'Water-vegetation separation',
                'range': '(-1, 1)',
                'interpretation': 'Water quality assessment',
                'priority': 'marine'
            },
            'BSI': {
                'formula': '((B11 + B4) - (B8 + B2)) / ((B11 + B4) + (B8 + B2))',
                'description': 'Bare Soil Index',
                'application': 'Sediment, bare areas',
                'range': '(-1, 1)',
                'interpretation': 'Higher values = more bare soil/sediment',
                'priority': 'marine'
            },
            
            # Advanced marine indices
            'FUI': {
                'formula': 'arctan2(B4 - B3, B2 - B3) * 180 / π',
                'description': 'Forel-Ule Index',
                'application': 'Water color classification',
                'range': '(0, 360)',
                'interpretation': 'Water color angle in degrees',
                'priority': 'advanced'
            },
            'SDD': {
                'formula': 'ln(0.14 / B4) / 1.7',
                'description': 'Secchi Disk Depth proxy',
                'application': 'Water transparency',
                'range': '(0, inf)',
                'interpretation': 'Higher values = clearer water',
                'priority': 'advanced'
            },
            'CDOM': {
                'formula': 'B1 / B3',
                'description': 'Colored Dissolved Organic Matter proxy',
                'application': 'CDOM concentration',
                'range': '(0, inf)',
                'interpretation': 'Higher values = more CDOM',
                'priority': 'advanced'
            }
        }

class S2MarineVisualizationProcessor:
    """Complete marine visualization processor for RGB and indices"""
    
    def __init__(self, config: Optional[MarineVisualizationConfig] = None):
        """Initialize marine visualization processor with configuration and logging"""
        # Configuration setup
        self.config = config or MarineVisualizationConfig()
        
        # Initialize RGB generator
        self.rgb_generator = S2MarineRGBGenerator()
        
        # Use global logger (updated for centralized logging)
        self.logger = logger
        
        # Log initialization details
        self.logger.info("Initialized S2 Marine Visualization Processor")
        self.logger.info(f"Configuration settings:")
        self.logger.info(f"  Natural color RGB: {self.config.generate_natural_color}")
        self.logger.info(f"  False color RGB: {self.config.generate_false_color}")
        self.logger.info(f"  Water-specific RGB: {self.config.generate_water_specific}")
        self.logger.info(f"  Research RGB: {self.config.generate_research_combinations}")
        self.logger.info(f"  Water quality indices: {self.config.generate_water_quality_indices}")
        self.logger.info(f"  Chlorophyll indices: {self.config.generate_chlorophyll_indices}")
        self.logger.info(f"  Turbidity indices: {self.config.generate_turbidity_indices}")
        self.logger.info(f"  Advanced indices: {self.config.generate_advanced_indices}")
        self.logger.info(f"  Output format: {self.config.rgb_format}")
        self.logger.info(f"  Contrast enhancement: {self.config.apply_contrast_enhancement}")
        self.logger.info(f"  Contrast method: {self.config.contrast_method}")
        self.logger.info(f"  Export metadata: {self.config.export_metadata}")
        self.logger.info(f"  Create overview images: {self.config.create_overview_images}")
        
        # Estimate expected products
        expected_rgb = 0
        if self.config.generate_natural_color: expected_rgb += 2  # natural_color, natural_with_contrast
        if self.config.generate_false_color: expected_rgb += 2   # false_color_infrared, false_color_nir
        if self.config.generate_water_specific: expected_rgb += 4  # turbidity, chlorophyll, coastal_aerosol, water_quality
        if self.config.generate_research_combinations: expected_rgb += 3  # sediment_transport, atmospheric_penetration, research_marine, bathymetric
        
        expected_indices = 0
        if self.config.generate_water_quality_indices: expected_indices += 3  # NDTI, NDWI, MNDWI
        if self.config.generate_chlorophyll_indices: expected_indices += 4   # NDVI, GNDVI, NDCI, CHL_RED_EDGE
        if self.config.generate_turbidity_indices: expected_indices += 3     # TSI, NGRDI, BSI
        if self.config.generate_advanced_indices: expected_indices += 3      # FUI, SDD, CDOM
        
        self.logger.info(f"Expected products: ~{expected_rgb} RGB composites + ~{expected_indices} spectral indices")
        self.logger.info("Marine visualization processor ready for processing")
        

    def process_marine_visualizations(self, c2rcc_path: str, output_folder: str, 
                                    product_name: str, intermediate_paths: Optional[Dict[str, str]] = None) -> Dict[str, ProcessingResult]:
        """
        Generate RGB composites and spectral indices from geometric products - FIXED VERSION
        
        Args:
            c2rcc_path: Path to C2RCC output directory (fallback only)
            output_folder: Output directory for visualizations
            product_name: Name of the product being processed
            intermediate_paths: Optional dictionary containing geometric_path
            
        Returns:
            Dictionary of visualization results (RGB composites + spectral indices)
        """
        try:
            self.logger.info(f"🌊 Starting marine visualization processing for {product_name}")
            
            # CRITICAL FIX: PRIORITIZE INTERMEDIATE_PATHS FIRST
            input_source = None
            input_type = "unknown"
            
            # =======================================================================
            # PRIORITY 1: Use intermediate_paths geometric_path (HIGHEST PRIORITY)
            # =======================================================================
            if intermediate_paths and 'geometric_path' in intermediate_paths:
                geometric_path = intermediate_paths['geometric_path']
                self.logger.info(f"🔍 Checking intermediate_paths geometric_path: {geometric_path}")
                
                if os.path.exists(geometric_path):
                    input_source = geometric_path
                    input_type = "geometric"
                    self.logger.info(f"✅ Using GEOMETRIC products from intermediate_paths: {os.path.basename(geometric_path)}")
                    self.logger.info(f"   Source: Provided geometric path (highest priority)")
                    self.logger.info(f"   Full path: {geometric_path}")
                else:
                    self.logger.warning(f"⚠️ Geometric path in intermediate_paths not found: {geometric_path}")
            
            # =======================================================================
            # PRIORITY 2: Look in standard output_folder location (FALLBACK)
            # =======================================================================
            if input_source is None:
                self.logger.info("🔍 Trying standard Geometric_Products folder location")
                geometric_folder = os.path.join(output_folder, "Geometric_Products")
                
                if os.path.exists(geometric_folder):
                    # Extract clean product name to find matching geometric product
                    clean_product_name = product_name.replace('.zip', '').replace('.SAFE', '')
                    if 'MSIL1C' in clean_product_name:
                        parts = clean_product_name.split('_')
                        if len(parts) >= 6:
                            clean_name = f"{parts[0]}_{parts[2]}_{parts[5]}"
                        else:
                            clean_name = clean_product_name.replace('MSIL1C_', '')
                    else:
                        clean_name = clean_product_name
                    
                    # Look for geometric product file
                    geometric_filename = f"Resampled_{clean_name}_Subset.dim"
                    geometric_path = os.path.join(geometric_folder, geometric_filename)
                    
                    if os.path.exists(geometric_path):
                        input_source = geometric_path
                        input_type = "geometric"
                        self.logger.info(f"✅ Using GEOMETRIC products from standard location: {geometric_filename}")
                    else:
                        # Fallback: search for any geometric product in the folder
                        for file in os.listdir(geometric_folder):
                            if file.endswith('.dim') and 'Subset' in file:
                                input_source = os.path.join(geometric_folder, file)
                                input_type = "geometric"
                                self.logger.info(f"✅ Found geometric product: {file}")
                                break
                      
            # =======================================================================
            # Create visualization output directories
            # =======================================================================
            # Create RGB_Composites and Spectral_Indices folders as expected
            #rgb_output_dir = os.path.join(output_folder, "RGB_Composites")
            #indices_output_dir = os.path.join(output_folder, "Spectral_Indices")
            # Clean product name (remove .zip extension)
            clean_product_name = product_name.replace('.zip', '').replace('.SAFE', '')

            # Create scene-specific folder structure (matching Jiang TSS processor)
            scene_folder = os.path.join(output_folder, clean_product_name)
            rgb_output_dir = os.path.join(scene_folder, "RGB_Composites")
            indices_output_dir = os.path.join(scene_folder, "Spectral_Indices")
            
            os.makedirs(rgb_output_dir, exist_ok=True)
            os.makedirs(indices_output_dir, exist_ok=True)
            
            self.logger.info(f"📁 RGB output directory: {rgb_output_dir}")
            self.logger.info(f"📁 Indices output directory: {indices_output_dir}")
            
            # Initialize results dictionary
            viz_results = {}
            
            # =======================================================================
            # STEP 2: LOAD SPECTRAL BANDS FROM DETERMINED SOURCE
            # =======================================================================
            self.logger.info(f"📊 Step 2: Loading spectral bands from {input_type.upper()} products")
            
            try:
                if input_type == "geometric":
                    # Load bands from geometric products
                    band_paths = self._load_bands_from_geometric_products(input_source)
                else:
                    # Load bands from C2RCC products (fallback)
                    band_paths = self._load_available_bands(input_source)
                
                if not band_paths:
                    error_msg = f"No suitable spectral bands found in {input_type} products"
                    self.logger.error(error_msg)
                    return {'error': ProcessingResult(False, "", None, error_msg)}
                
                self.logger.info(f"Found {len(band_paths)} spectral bands for visualization:")
                for wavelength, path in band_paths.items():
                    self.logger.info(f"  {wavelength}nm: {os.path.basename(path)}")
                
                # Load band data arrays
                bands_data, reference_metadata = self._load_bands_data_from_paths(band_paths)
                
                if bands_data is None or reference_metadata is None:
                    error_msg = "Failed to load band data arrays"
                    self.logger.error(error_msg)
                    return {'error': ProcessingResult(False, "", None, error_msg)}
                
                self.logger.info(f"✅ Successfully loaded {len(bands_data)} bands for visualization")
                
                # Log band statistics
                sample_band = list(bands_data.values())[0]
                data_min = np.nanmin(sample_band)
                data_max = np.nanmax(sample_band)
                self.logger.info(f"Data characteristics:")
                self.logger.info(f"  Input type: {input_type.upper()}")
                self.logger.info(f"  Value range: {data_min:.4f} to {data_max:.4f}")
                self.logger.info(f"  Spatial dimensions: {sample_band.shape}")
                
            except Exception as band_error:
                error_msg = f"Error loading spectral bands: {str(band_error)}"
                self.logger.error(error_msg)
                import traceback
                self.logger.error(f"Band loading traceback: {traceback.format_exc()}")
                return {'error': ProcessingResult(False, "", None, error_msg)}
            
            # =======================================================================
            # STEP 3: GENERATE RGB COMPOSITES
            # =======================================================================
            if bands_data and len(bands_data) >= 3:
                self.logger.info("🎨 Step 3: Generating RGB composites")
                
                try:
                    rgb_results = self._generate_rgb_composites(bands_data, reference_metadata, rgb_output_dir, product_name)
                    viz_results.update(rgb_results)
                    rgb_count = len([k for k in rgb_results.keys() if k.startswith('rgb_')])
                    self.logger.info(f"Generated {rgb_count} RGB composites")
                    
                except Exception as rgb_error:
                    self.logger.error(f"Error generating RGB composites: {rgb_error}")
                    viz_results['rgb_error'] = ProcessingResult(False, "", None, str(rgb_error))
            
            # =======================================================================
            # STEP 4: GENERATE SPECTRAL INDICES
            # =======================================================================
            if bands_data and len(bands_data) >= 2:
                self.logger.info("📊 Step 4: Generating spectral indices")
                
                try:
                    index_results = self._generate_spectral_indices(bands_data, reference_metadata, indices_output_dir, product_name)
                    viz_results.update(index_results)
                    index_count = len([k for k in index_results.keys() if k.startswith('index_')])
                    self.logger.info(f"Generated {index_count} spectral indices")
                    
                except Exception as index_error:
                    self.logger.error(f"Error generating spectral indices: {index_error}")
                    viz_results['index_error'] = ProcessingResult(False, "", None, str(index_error))
            
            # =======================================================================
            # STEP 5: CREATE PROCESSING SUMMARY
            # =======================================================================
            self.logger.info("📄 Step 5: Creating visualization summary")
            try:
                self._create_visualization_summary(viz_results, output_folder, product_name)
            except Exception as summary_error:
                self.logger.warning(f"Could not create visualization summary: {summary_error}")
            
            # =======================================================================
            # FINAL RESULTS
            # =======================================================================
            successful_viz = len([r for r in viz_results.values() if isinstance(r, ProcessingResult) and r.success])
            total_viz = len([r for r in viz_results.values() if isinstance(r, ProcessingResult)])
            success_rate = (successful_viz / total_viz) * 100 if total_viz > 0 else 0
            
            self.logger.info("🌊 Marine visualization processing completed!")
            self.logger.info(f"   Input source: {input_type.upper()} products")
            self.logger.info(f"   Results: {successful_viz}/{total_viz} products successful ({success_rate:.1f}%)")
            self.logger.info(f"   RGB output: {rgb_output_dir}")
            self.logger.info(f"   Indices output: {indices_output_dir}")
            
            return viz_results
            
        except Exception as e:
            error_msg = f"Marine visualization processing failed: {str(e)}"
            self.logger.error(error_msg)
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return {'error': ProcessingResult(False, "", None, error_msg)}

    def _load_bands_from_geometric_products(self, geometric_path: str) -> Dict[int, str]:
        """Load bands from geometric products - basic validation only"""
        
        if geometric_path.endswith('.dim'):
            data_folder = geometric_path.replace('.dim', '.data')
        else:
            data_folder = f"{geometric_path}.data"
        
        if not os.path.exists(data_folder):
            self.logger.error(f"Geometric data folder not found: {data_folder}")
            return {}
        
        # Standard S2 band mapping
        band_mapping = {
            443: ['B1.img'],    # Coastal aerosol
            490: ['B2.img'],    # Blue  
            560: ['B3.img'],    # Green
            665: ['B4.img'],    # Red
            705: ['B5.img'],    # Red Edge 1
            740: ['B6.img'],    # Red Edge 2
            783: ['B7.img'],    # Red Edge 3
            842: ['B8.img'],    # NIR (broad)
            865: ['B8A.img'],   # NIR (narrow)
            945: ['B9.img'],    # Water vapour
            1375: ['B10.img'],  # SWIR Cirrus
            1610: ['B11.img'],  # SWIR 1
            2190: ['B12.img']   # SWIR 2
        }
        
        available_bands = {}
        
        self.logger.info(f"Loading bands from: {os.path.basename(data_folder)}")
        
        for wavelength, possible_files in band_mapping.items():
            for filename in possible_files:
                band_path = os.path.join(data_folder, filename)
                if os.path.exists(band_path):
                    file_size = os.path.getsize(band_path)
                    if file_size > 1024:  # Basic size check
                        available_bands[wavelength] = band_path
                        self.logger.debug(f"✓ {wavelength}nm: {filename}")
                        break
        
        self.logger.info(f"Found {len(available_bands)} bands: {sorted(available_bands.keys())}")
        return available_bands
    
    def _load_available_bands(self, c2rcc_path: str) -> Dict[int, str]:
        """Load bands from actual SNAP C2RCC output structure"""
        # Determine data folder
        if c2rcc_path.endswith('.dim'):
            data_folder = c2rcc_path.replace('.dim', '.data')
        else:
            data_folder = f"{c2rcc_path}.data"
        
        if not os.path.exists(data_folder):
            self.logger.error(f"Data folder not found: {data_folder}")
            return {}
        
        # Band mapping using actual file structure from your C2RCC output
        band_mapping = {
            443: ['rho_toa_B1.img', 'rtoa_B1.img'],      # Band 1
            490: ['rho_toa_B2.img', 'rtoa_B2.img'],      # Band 2  
            560: ['rho_toa_B3.img', 'rtoa_B3.img'],      # Band 3
            665: ['rho_toa_B4.img', 'rtoa_B4.img'],      # Band 4
            705: ['rho_toa_B5.img', 'rtoa_B5.img'],      # Band 5
            740: ['rho_toa_B6.img', 'rtoa_B6.img'],      # Band 6
            783: ['rho_toa_B7.img', 'rtoa_B7.img'],      # Band 7
            842: ['rho_toa_B8.img', 'rtoa_B8.img'],      # Band 8
            865: ['rho_toa_B8A.img', 'rtoa_B8A.img'],    # Band 8A
            945: ['rho_toa_B9.img', 'rtoa_B9.img'],      # Band 9
            1610: ['rho_toa_B11.img', 'rtoa_B11.img'],   # Band 11
            2190: ['rho_toa_B12.img', 'rtoa_B12.img']    # Band 12
        }
        
        available_bands = {}
        found_files = []
        missing_files = []
        
        for wavelength, possible_files in band_mapping.items():
            band_found = False
            for filename in possible_files:
                band_path = os.path.join(data_folder, filename)
                if os.path.exists(band_path):
                    file_size = os.path.getsize(band_path)
                    if file_size > 1024:  # At least 1KB
                        available_bands[wavelength] = band_path
                        found_files.append(filename)
                        band_found = True
                        break
            
            if not band_found:
                missing_files.append(f"{wavelength}nm ({possible_files[0]})")
        
        self.logger.info(f"Band loading results:")
        self.logger.info(f"  Found {len(available_bands)} bands: {sorted(list(available_bands.keys()))}")
        if missing_files:
            self.logger.warning(f"  Missing: {missing_files}")
        
        return available_bands
    
    def _load_bands_data_from_paths(self, band_paths: Dict[int, str]) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Load band data from geometric products - straightforward approach"""
        bands_data = {}
        reference_metadata = None
        
        self.logger.info(f"Loading {len(band_paths)} bands into memory")
        
        for wavelength, file_path in band_paths.items():
            try:
                # Simple file validation
                if not os.path.exists(file_path):
                    self.logger.error(f"Band file missing: {file_path}")
                    return None, None
                
                # Load using existing RasterIO
                data, metadata = RasterIO.read_raster(file_path)
                
                # Basic validation
                if data is None or data.size == 0:
                    self.logger.error(f"Invalid data from {file_path}")
                    return None, None
                
                # Convert to float32
                data = data.astype(np.float32)
                
                # Handle nodata
                if metadata and 'nodata' in metadata and metadata['nodata'] is not None:
                    data[data == metadata['nodata']] = np.nan
                
                bands_data[wavelength] = data
                
                if reference_metadata is None:
                    reference_metadata = metadata
                
                self.logger.debug(f"✓ Loaded {wavelength}nm: shape={data.shape}")
                
            except Exception as e:
                self.logger.error(f"Failed to load {wavelength}nm: {e}")
                return None, None
        
        self.logger.info(f"Successfully loaded {len(bands_data)} bands")
        return bands_data, reference_metadata
    
    def _generate_rgb_composites(self, bands_data: Dict[int, np.ndarray], reference_metadata: Dict, 
                           output_folder: str, product_name: str) -> Dict[str, ProcessingResult]:
        """Generate RGB composites based on available bands and configuration"""
        results = {}
        
        try:
            available_wavelengths = set(bands_data.keys())
            self.logger.info(f"Available wavelengths for RGB: {sorted(available_wavelengths)}")
            
            # Define RGB combinations with priorities
            rgb_combinations = {
                # Natural color combinations
                'natural_color': {
                    'red': 665, 'green': 560, 'blue': 490,  # B4, B3, B2
                    'description': 'Natural color (True color)',
                    'application': 'General visualization, publications',
                    'priority': 'essential',
                    'enabled': self.config.generate_natural_color
                },
                'natural_enhanced': {
                    'red': 665, 'green': 560, 'blue': 443,  # B4, B3, B1
                    'description': 'Natural color with enhanced contrast',
                    'application': 'Better water-land contrast',
                    'priority': 'important',
                    'enabled': self.config.generate_natural_color
                },
                
                # False color combinations
                'false_color_infrared': {
                    'red': 842, 'green': 665, 'blue': 560,  # B8, B4, B3
                    'description': 'False color infrared',
                    'application': 'Vegetation (red), clear water (dark)',
                    'priority': 'important',
                    'enabled': self.config.generate_false_color
                },
                'false_color_nir': {
                    'red': 865, 'green': 665, 'blue': 560,  # B8A, B4, B3
                    'description': 'False color NIR',
                    'application': 'Enhanced vegetation/water contrast',
                    'priority': 'important',
                    'enabled': self.config.generate_false_color
                },
                
                # Water-specific combinations
                'turbidity_enhanced': {
                    'red': 705, 'green': 665, 'blue': 560,  # B5, B4, B3
                    'description': 'Turbidity-enhanced RGB',
                    'application': 'Enhanced turbidity visualization',
                    'priority': 'marine',
                    'enabled': self.config.generate_water_specific
                },
                'chlorophyll_enhanced': {
                    'red': 705, 'green': 665, 'blue': 490,  # B5, B4, B2
                    'description': 'Chlorophyll-enhanced RGB',
                    'application': 'Enhanced chlorophyll visualization',
                    'priority': 'marine',
                    'enabled': self.config.generate_water_specific
                },
                'coastal_aerosol': {
                    'red': 665, 'green': 490, 'blue': 443,  # B4, B2, B1
                    'description': 'Coastal aerosol RGB',
                    'application': 'Coastal water analysis',
                    'priority': 'marine',
                    'enabled': self.config.generate_water_specific
                },
                'water_quality': {
                    'red': 740, 'green': 665, 'blue': 560,  # B6, B4, B3
                    'description': 'Water quality RGB',
                    'application': 'General water quality assessment',
                    'priority': 'marine',
                    'enabled': self.config.generate_water_specific
                },
                
                # Research combinations
                'sediment_transport': {
                    'red': 783, 'green': 705, 'blue': 665,  # B7, B5, B4
                    'description': 'Sediment transport RGB',
                    'application': 'Sediment plume visualization',
                    'priority': 'research',
                    'enabled': self.config.generate_research_combinations
                },
                'atmospheric_penetration': {
                    'red': 865, 'green': 783, 'blue': 740,  # B8A, B7, B6
                    'description': 'Atmospheric penetration RGB',
                    'application': 'Deep water analysis',
                    'priority': 'research',
                    'enabled': self.config.generate_research_combinations
                }
            }
            
            # Filter combinations based on configuration
            active_combinations = {name: config for name, config in rgb_combinations.items() 
                                if config['enabled'] and config['priority'] in ['essential', 'important', 'marine', 'research']}
            
            self.logger.info(f"Processing {len(active_combinations)} RGB combinations")
            
            for rgb_name, config in active_combinations.items():
                try:
                    required_bands = [config['red'], config['green'], config['blue']]
                    missing_bands = [wl for wl in required_bands if wl not in available_wavelengths]
                    
                    if missing_bands:
                        self.logger.info(f"Skipping {rgb_name}: missing wavelengths {missing_bands}")
                        continue
                    
                    self.logger.info(f"Creating {rgb_name}: R={config['red']}nm, G={config['green']}nm, B={config['blue']}nm")
                    
                    # Get band data
                    red_data = bands_data[config['red']]
                    green_data = bands_data[config['green']]
                    blue_data = bands_data[config['blue']]
                    
                    # Create RGB composite
                    rgb_array = self._create_robust_rgb_composite(red_data, green_data, blue_data)
                    
                    # Save RGB composite
                    output_path = os.path.join(output_folder, f"{product_name}_{rgb_name}.tif")
                    success = self._save_rgb_geotiff(rgb_array, output_path, reference_metadata, config['description'])
                    
                    if success:
                        # Calculate statistics
                        valid_pixels = np.sum(np.any(rgb_array > 0, axis=2))
                        total_pixels = rgb_array.shape[0] * rgb_array.shape[1]
                        coverage_percent = (valid_pixels / total_pixels) * 100
                        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                        
                        stats = {
                            'description': config['description'],
                            'application': config['application'],
                            'bands_used': f"R:{config['red']}nm, G:{config['green']}nm, B:{config['blue']}nm",
                            'priority': config['priority'],
                            'file_size_mb': file_size_mb,
                            'coverage_percent': coverage_percent,
                            'output_path': output_path
                        }
                        
                        results[f'rgb_{rgb_name}'] = ProcessingResult(True, output_path, stats, None)
                        self.logger.info(f"✓ {rgb_name}: {coverage_percent:.1f}% coverage, {file_size_mb:.1f}MB")
                    else:
                        results[f'rgb_{rgb_name}'] = ProcessingResult(False, output_path, None, "Failed to save RGB composite")
                        
                except Exception as e:
                    self.logger.error(f"Error generating {rgb_name}: {e}")
                    results[f'rgb_{rgb_name}'] = ProcessingResult(False, "", None, str(e))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in RGB composite generation: {e}")
            return {'rgb_error': ProcessingResult(False, "", None, str(e))}
    
    def _generate_spectral_indices(self, bands_data: Dict[int, np.ndarray], reference_metadata: Dict,
                                output_folder: str, product_name: str) -> Dict[str, ProcessingResult]:
        """Generate spectral indices based on configuration and available bands"""
        results = {}
        
        try:
            available_wavelengths = set(bands_data.keys())
            self.logger.info(f"Available wavelengths for indices: {sorted(available_wavelengths)}")
            
            # Define spectral indices with requirements
            spectral_indices = {
                # Water quality indices
                'NDWI': {
                    'formula': '(B3 - B8A) / (B3 + B8A)',
                    'required_bands': [560, 865],
                    'description': 'Normalized Difference Water Index',
                    'application': 'Water body delineation',
                    'enabled': self.config.generate_water_quality_indices,
                    'category': 'water_quality'
                },
                'MNDWI': {
                    'formula': '(B3 - B11) / (B3 + B11)',
                    'required_bands': [560, 1610],  # B11 might not be available in C2RCC
                    'fallback_bands': [560, 865],   # Use B8A as fallback
                    'description': 'Modified Normalized Difference Water Index',
                    'application': 'Enhanced water detection',
                    'enabled': self.config.generate_water_quality_indices,
                    'category': 'water_quality'
                },
                'NDTI': {
                    'formula': '(B4 - B3) / (B4 + B3)',
                    'required_bands': [665, 560],
                    'description': 'Normalized Difference Turbidity Index',
                    'application': 'Turbidity assessment',
                    'enabled': self.config.generate_water_quality_indices,
                    'category': 'water_quality'
                },
                
                # Chlorophyll indices
                'NDCI': {
                    'formula': '(B5 - B4) / (B5 + B4)',
                    'required_bands': [705, 665],
                    'description': 'Normalized Difference Chlorophyll Index',
                    'application': 'Chlorophyll concentration',
                    'enabled': self.config.generate_chlorophyll_indices,
                    'category': 'chlorophyll'
                },
                'CHL_RED_EDGE': {
                    'formula': '(B5 / B4) - 1',
                    'required_bands': [705, 665],
                    'description': 'Chlorophyll Red Edge',
                    'application': 'Chlorophyll using red edge',
                    'enabled': self.config.generate_chlorophyll_indices,
                    'category': 'chlorophyll'
                },
                'GNDVI': {
                    'formula': '(B8 - B3) / (B8 + B3)',
                    'required_bands': [842, 560],
                    'description': 'Green Normalized Difference Vegetation Index',
                    'application': 'Aquatic vegetation',
                    'enabled': self.config.generate_chlorophyll_indices,
                    'category': 'chlorophyll'
                },
                
                # Turbidity indices
                'TSI': {
                    'formula': '(B4 + B3) / 2',
                    'required_bands': [665, 560],
                    'description': 'Turbidity Spectral Index',
                    'application': 'Turbidity estimation',
                    'enabled': self.config.generate_turbidity_indices,
                    'category': 'turbidity'
                },
                'NGRDI': {
                    'formula': '(B3 - B4) / (B3 + B4)',
                    'required_bands': [560, 665],
                    'description': 'Normalized Green Red Difference Index',
                    'application': 'Water-vegetation separation',
                    'enabled': self.config.generate_turbidity_indices,
                    'category': 'turbidity'
                },
                
                # Advanced indices
                'FUI': {
                    'formula': 'arctan2(B4 - B3, B2 - B3) * 180 / π',
                    'required_bands': [665, 560, 490],
                    'description': 'Forel-Ule Index',
                    'application': 'Water color classification',
                    'enabled': self.config.generate_advanced_indices,
                    'category': 'advanced'
                },
                'SDD': {
                    'formula': 'ln(0.14 / B4) / 1.7',
                    'required_bands': [665],
                    'description': 'Secchi Disk Depth proxy',
                    'application': 'Water transparency',
                    'enabled': self.config.generate_advanced_indices,
                    'category': 'advanced'
                },
                'CDOM': {
                    'formula': 'B1 / B3',
                    'required_bands': [443, 560],
                    'description': 'Colored Dissolved Organic Matter proxy',
                    'application': 'CDOM concentration',
                    'enabled': self.config.generate_advanced_indices,
                    'category': 'advanced'
                }
            }
            
            # Filter indices based on configuration
            active_indices = {name: config for name, config in spectral_indices.items() if config['enabled']}
            
            self.logger.info(f"Processing {len(active_indices)} spectral indices")
            
            for index_name, config in active_indices.items():
                try:
                    # Check band availability
                    required_bands = config['required_bands']
                    fallback_bands = config.get('fallback_bands', None)
                    
                    if all(band in available_wavelengths for band in required_bands):
                        bands_to_use = required_bands
                    elif fallback_bands and all(band in available_wavelengths for band in fallback_bands):
                        bands_to_use = fallback_bands
                        self.logger.info(f"Using fallback bands for {index_name}")
                    else:
                        missing = [b for b in required_bands if b not in available_wavelengths]
                        self.logger.info(f"Skipping {index_name}: missing wavelengths {missing}")
                        continue
                    
                    self.logger.info(f"Calculating {index_name} using bands: {bands_to_use}")
                    
                    # Calculate index based on formula
                    index_data = self._calculate_spectral_index(index_name, config, bands_data, bands_to_use)
                    
                    if index_data is not None:
                        # Save index
                        output_path = os.path.join(output_folder, f"{product_name}_{index_name.lower()}.tif")
                        success = self._save_single_band_geotiff(index_data, output_path, reference_metadata, config['description'])
                        
                        if success:
                            # Calculate statistics
                            valid_data = index_data[~np.isnan(index_data)]
                            if len(valid_data) > 0:
                                stats = {
                                    'description': config['description'],
                                    'application': config['application'],
                                    'formula': config['formula'],
                                    'bands_used': bands_to_use,
                                    'category': config['category'],
                                    'min_value': float(np.min(valid_data)),
                                    'max_value': float(np.max(valid_data)),
                                    'mean_value': float(np.mean(valid_data)),
                                    'std_value': float(np.std(valid_data)),
                                    'valid_pixels': len(valid_data),
                                    'coverage_percent': (len(valid_data) / index_data.size) * 100,
                                    'file_size_mb': os.path.getsize(output_path) / (1024 * 1024)
                                }
                            else:
                                stats = {'description': config['description'], 'error': 'No valid data'}
                            
                            results[f'index_{index_name.lower()}'] = ProcessingResult(True, output_path, stats, None)
                            self.logger.info(f"✓ {index_name}: {stats.get('coverage_percent', 0):.1f}% coverage")
                        else:
                            results[f'index_{index_name.lower()}'] = ProcessingResult(False, output_path, None, "Failed to save index")
                    else:
                        results[f'index_{index_name.lower()}'] = ProcessingResult(False, "", None, "Failed to calculate index")
                        
                except Exception as e:
                    self.logger.error(f"Error calculating {index_name}: {e}")
                    results[f'index_{index_name.lower()}'] = ProcessingResult(False, "", None, str(e))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in spectral index generation: {e}")
            return {'index_error': ProcessingResult(False, "", None, str(e))}

    def _calculate_spectral_index(self, index_name: str, config: Dict, bands_data: Dict[int, np.ndarray], 
                                bands_to_use: List[int]) -> Optional[np.ndarray]:
        """Calculate specific spectral index using provided bands"""
        try:
            if index_name == 'NDWI':
                # (B3 - B8A) / (B3 + B8A)
                b3 = bands_data[bands_to_use[0]]  # 560nm
                b8a = bands_data[bands_to_use[1]]  # 865nm
                return (b3 - b8a) / (b3 + b8a + 1e-8)
                
            elif index_name == 'MNDWI':
                # Modified NDWI using available bands
                b3 = bands_data[bands_to_use[0]]  # 560nm
                b_swir = bands_data[bands_to_use[1]]  # 865nm or 1610nm
                return (b3 - b_swir) / (b3 + b_swir + 1e-8)
                
            elif index_name == 'NDTI':
                # (B4 - B3) / (B4 + B3)
                b4 = bands_data[bands_to_use[0]]  # 665nm
                b3 = bands_data[bands_to_use[1]]  # 560nm
                return (b4 - b3) / (b4 + b3 + 1e-8)
                
            elif index_name == 'NDCI':
                # (B5 - B4) / (B5 + B4)
                b5 = bands_data[bands_to_use[0]]  # 705nm
                b4 = bands_data[bands_to_use[1]]  # 665nm
                return (b5 - b4) / (b5 + b4 + 1e-8)
                
            elif index_name == 'CHL_RED_EDGE':
                # (B5 / B4) - 1
                b5 = bands_data[bands_to_use[0]]  # 705nm
                b4 = bands_data[bands_to_use[1]]  # 665nm
                return (b5 / (b4 + 1e-8)) - 1
                
            elif index_name == 'GNDVI':
                # (B8 - B3) / (B8 + B3)
                b8 = bands_data[bands_to_use[0]]  # 842nm
                b3 = bands_data[bands_to_use[1]]  # 560nm
                return (b8 - b3) / (b8 + b3 + 1e-8)
                
            elif index_name == 'TSI':
                # (B4 + B3) / 2
                b4 = bands_data[bands_to_use[0]]  # 665nm
                b3 = bands_data[bands_to_use[1]]  # 560nm
                return (b4 + b3) / 2
                
            elif index_name == 'NGRDI':
                # (B3 - B4) / (B3 + B4)
                b3 = bands_data[bands_to_use[0]]  # 560nm
                b4 = bands_data[bands_to_use[1]]  # 665nm
                return (b3 - b4) / (b3 + b4 + 1e-8)
                
            elif index_name == 'FUI':
                # arctan2(B4 - B3, B2 - B3) * 180 / π
                b4 = bands_data[bands_to_use[0]]  # 665nm
                b3 = bands_data[bands_to_use[1]]  # 560nm
                b2 = bands_data[bands_to_use[2]]  # 490nm
                return np.arctan2(b4 - b3, b2 - b3) * 180 / np.pi
                
            elif index_name == 'SDD':
                # ln(0.14 / B4) / 1.7
                b4 = bands_data[bands_to_use[0]]  # 665nm
                return np.log(0.14 / (b4 + 1e-8)) / 1.7
                
            elif index_name == 'CDOM':
                # B1 / B3
                b1 = bands_data[bands_to_use[0]]  # 443nm
                b3 = bands_data[bands_to_use[1]]  # 560nm
                return b1 / (b3 + 1e-8)
                
            else:
                self.logger.warning(f"Unknown index calculation for {index_name}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error calculating {index_name}: {e}")
            return None
    
    def _create_robust_rgb_composite(self, red: np.ndarray, green: np.ndarray, blue: np.ndarray) -> np.ndarray:
        """Create RGB composite from geometric products - SIMPLIFIED VERSION"""
        try:
            # Input validation
            if red.shape != green.shape or red.shape != blue.shape:
                raise ValueError(f"Band shape mismatch: R={red.shape}, G={green.shape}, B={blue.shape}")
            
            height, width = red.shape
            rgb_array = np.zeros((height, width, 3), dtype=np.float32)
            
            bands = [red, green, blue]
            band_names = ['Red', 'Green', 'Blue']
            
            for i, (band, name) in enumerate(zip(bands, band_names)):
                # Simple valid data mask
                valid_mask = (~np.isnan(band)) & (~np.isinf(band)) & (band >= 0)
                
                if not np.any(valid_mask):
                    self.logger.warning(f"{name} band: No valid data found")
                    rgb_array[:, :, i] = 0
                    continue
                
                # Simple percentile normalization for TOA reflectance
                valid_data = band[valid_mask]
                p2, p98 = np.percentile(valid_data, [2, 98])
                
                if p98 > p2:
                    # Standard 2-98% stretch
                    normalized_band = np.clip((band - p2) / (p98 - p2), 0, 1)
                else:
                    # Fallback: direct clipping
                    normalized_band = np.clip(band, 0, 1)
                
                rgb_array[:, :, i] = normalized_band
                
                # Log simple statistics
                mean_val = np.mean(valid_data)
                self.logger.debug(f"{name}: mean={mean_val:.4f}, range=[{p2:.4f}, {p98:.4f}]")
            
            return rgb_array
            
        except Exception as e:
            self.logger.error(f"Error in RGB composite creation: {e}")
            return np.zeros((red.shape[0], red.shape[1], 3), dtype=np.float32)

    def _apply_additional_contrast_enhancement(self, rgb_array: np.ndarray) -> np.ndarray:
        """Apply additional contrast enhancement based on configuration"""
        try:
            if self.config.contrast_method == 'percentile_stretch':
                # Apply percentile stretch to the combined RGB
                for i in range(3):  # R, G, B channels
                    channel = rgb_array[:, :, i]
                    valid_mask = channel > 0
                    valid_data = channel[valid_mask]
                    
                    if len(valid_data) > 100:
                        p_low, p_high = self.config.percentile_range
                        p_low_val, p_high_val = np.percentile(valid_data, [p_low, p_high])
                        
                        if p_high_val > p_low_val:
                            enhanced_channel = np.clip(
                                (channel - p_low_val) / (p_high_val - p_low_val), 
                                0, 1
                            )
                            rgb_array[:, :, i] = enhanced_channel
            
            elif self.config.contrast_method == 'histogram_equalization':
                # Simple histogram equalization
                for i in range(3):
                    channel = rgb_array[:, :, i]
                    valid_mask = channel > 0
                    
                    if np.any(valid_mask):
                        # Apply histogram equalization to valid pixels only
                        valid_data = channel[valid_mask]
                        if len(np.unique(valid_data)) > 10:  # Enough unique values
                            hist, bins = np.histogram(valid_data, bins=256, range=(0, 1))
                            cdf = hist.cumsum()
                            cdf = cdf / cdf[-1]  # Normalize
                            
                            # Interpolate to get equalized values
                            enhanced_data = np.interp(valid_data, bins[:-1], cdf)
                            channel[valid_mask] = enhanced_data
                            rgb_array[:, :, i] = channel
            
            return rgb_array
            
        except Exception as e:
            self.logger.warning(f"Error in additional contrast enhancement: {e}")
            return rgb_array

    def _save_rgb_geotiff(self, rgb_array: np.ndarray, output_path: str, 
                        metadata: Dict, description: str) -> bool:
        """Save RGB array as GeoTIFF with proper metadata"""
        try:
            # Convert to uint8 for better visualization and smaller file size
            rgb_uint8 = (rgb_array * 255).astype(np.uint8)
            
            # Get dimensions
            height, width, bands = rgb_uint8.shape
            
            # Create the dataset
            driver = gdal.GetDriverByName('GTiff')
            if driver is None:
                self.logger.error("GTiff driver not available")
                return False
            
            dataset = driver.Create(
                output_path, 
                width, 
                height, 
                bands, 
                gdal.GDT_Byte,
                options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=IF_SAFER']
            )
            
            if dataset is None:
                self.logger.error(f"Could not create dataset: {output_path}")
                return False
            
            # Set geotransform and projection
            if 'geotransform' in metadata:
                dataset.SetGeoTransform(metadata['geotransform'])
            if 'projection' in metadata:
                dataset.SetProjection(metadata['projection'])
            
            # Write the RGB bands
            for i in range(bands):
                band = dataset.GetRasterBand(i + 1)
                band.WriteArray(rgb_uint8[:, :, i])
                
                # Set band metadata
                if i == 0:
                    band.SetDescription('Red')
                    band.SetColorInterpretation(gdal.GCI_RedBand)
                elif i == 1:
                    band.SetDescription('Green')
                    band.SetColorInterpretation(gdal.GCI_GreenBand)
                elif i == 2:
                    band.SetDescription('Blue')
                    band.SetColorInterpretation(gdal.GCI_BlueBand)
                
                # Set statistics
                band.SetStatistics(0, 255, float(np.mean(rgb_uint8[:, :, i])), float(np.std(rgb_uint8[:, :, i])))
            
            # Set dataset metadata
            dataset.SetDescription(description)
            dataset.SetMetadataItem('DESCRIPTION', description)
            dataset.SetMetadataItem('CREATION_DATE', datetime.now().isoformat())
            dataset.SetMetadataItem('SOURCE', 'Unified S2 TSS Pipeline - Marine Visualization')
            
            if self.config.export_metadata:
                dataset.SetMetadataItem('PROCESSING_METHOD', 'Marine RGB Composite')
                dataset.SetMetadataItem('CONTRAST_ENHANCEMENT', str(self.config.apply_contrast_enhancement))
                dataset.SetMetadataItem('CONTRAST_METHOD', self.config.contrast_method)
            
            # Build overviews if configured
            if self.config.create_overview_images:
                dataset.BuildOverviews('AVERAGE', [2, 4, 8, 16])
            
            # Flush and close
            dataset.FlushCache()
            dataset = None
            
            self.logger.debug(f"✓ Saved RGB composite: {os.path.basename(output_path)}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving RGB GeoTIFF {output_path}: {e}")
            return False

    def _save_single_band_geotiff(self, data: np.ndarray, output_path: str, 
                                metadata: Dict, description: str) -> bool:
        """Save single band data as GeoTIFF with proper metadata"""
        try:
            # Get dimensions
            height, width = data.shape
            
            # Create the dataset
            driver = gdal.GetDriverByName('GTiff')
            if driver is None:
                self.logger.error("GTiff driver not available")
                return False
            
            dataset = driver.Create(
                output_path, 
                width, 
                height, 
                1, 
                gdal.GDT_Float32,
                options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=IF_SAFER']
            )
            
            if dataset is None:
                self.logger.error(f"Could not create dataset: {output_path}")
                return False
            
            # Set geotransform and projection
            if 'geotransform' in metadata:
                dataset.SetGeoTransform(metadata['geotransform'])
            if 'projection' in metadata:
                dataset.SetProjection(metadata['projection'])
            
            # Write the data
            band = dataset.GetRasterBand(1)
            band.WriteArray(data.astype(np.float32))
            
            # Set nodata value
            nodata_value = metadata.get('nodata', -9999)
            band.SetNoDataValue(float(nodata_value))
            
            # Calculate and set statistics for valid data
            valid_data = data[~np.isnan(data)]
            if len(valid_data) > 0:
                band.SetStatistics(
                    float(np.min(valid_data)), 
                    float(np.max(valid_data)), 
                    float(np.mean(valid_data)), 
                    float(np.std(valid_data))
                )
            
            # Set band metadata
            band.SetDescription(description)
            
            # Set dataset metadata
            dataset.SetDescription(description)
            dataset.SetMetadataItem('DESCRIPTION', description)
            dataset.SetMetadataItem('CREATION_DATE', datetime.now().isoformat())
            dataset.SetMetadataItem('SOURCE', 'Unified S2 TSS Pipeline - Marine Visualization')
            
            if self.config.export_metadata:
                dataset.SetMetadataItem('PROCESSING_METHOD', 'Spectral Index Calculation')
                dataset.SetMetadataItem('UNITS', 'Dimensionless')
            
            # Build overviews if configured
            if self.config.create_overview_images:
                dataset.BuildOverviews('AVERAGE', [2, 4, 8, 16])
            
            # Flush and close
            dataset.FlushCache()
            dataset = None
            
            self.logger.debug(f"✓ Saved spectral index: {os.path.basename(output_path)}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving single band GeoTIFF {output_path}: {e}")
            return False

    def _create_visualization_summary(self, viz_results: Dict[str, ProcessingResult], 
                                    output_folder: str, product_name: str):
        """Create comprehensive visualization summary report"""
        try:
            summary_file = os.path.join(output_folder, f"{product_name}_Marine_Visualization_Summary.txt")
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                # Header
                f.write("MARINE VISUALIZATION PROCESSING SUMMARY\n")
                f.write("=" * 60 + "\n")
                f.write(f"Product: {product_name}\n")
                f.write(f"Processing completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Pipeline: Unified S2-TSS Processing with Marine Visualization\n")
                f.write(f"Output directory: {output_folder}\n\n")
                
                # Overall statistics
                total_products = len(viz_results)
                successful_products = len([r for r in viz_results.values() if r.success])
                failed_products = total_products - successful_products
                success_rate = (successful_products / total_products) * 100 if total_products > 0 else 0
                
                f.write("PROCESSING STATISTICS:\n")
                f.write(f"  Total products attempted: {total_products}\n")
                f.write(f"  Successfully generated: {successful_products}\n")
                f.write(f"  Failed: {failed_products}\n")
                f.write(f"  Success rate: {success_rate:.1f}%\n\n")
                
                # Product categories
                rgb_products = {k: v for k, v in viz_results.items() if k.startswith('rgb_')}
                index_products = {k: v for k, v in viz_results.items() if k.startswith('index_')}
                
                f.write("PRODUCT CATEGORIES:\n")
                f.write(f"  RGB Composites: {len(rgb_products)} products\n")
                f.write(f"  Spectral Indices: {len(index_products)} products\n\n")
                
                # RGB Composites section
                if rgb_products:
                    f.write("RGB COMPOSITES:\n")
                    f.write("-" * 40 + "\n")
                    
                    for product_key, result in rgb_products.items():
                        product_name_clean = product_key.replace('rgb_', '').replace('_', ' ').title()
                        
                        if result.success and result.statistics:
                            stats = result.statistics
                            f.write(f"✓ {product_name_clean}\n")
                            f.write(f"  Description: {stats.get('description', 'N/A')}\n")
                            f.write(f"  Application: {stats.get('application', 'N/A')}\n")
                            f.write(f"  Bands used: {stats.get('bands_used', 'N/A')}\n")
                            f.write(f"  Coverage: {stats.get('coverage_percent', 0):.1f}%\n")
                            f.write(f"  File size: {stats.get('file_size_mb', 0):.1f} MB\n")
                            f.write(f"  Output: {os.path.basename(result.output_path)}\n\n")
                        else:
                            f.write(f"✗ {product_name_clean} (failed)\n")
                            if result.error_message:
                                f.write(f"  Error: {result.error_message}\n\n")
                            else:
                                f.write("  Error: Unknown failure\n\n")
                
                # Spectral Indices section
                if index_products:
                    f.write("SPECTRAL INDICES:\n")
                    f.write("-" * 40 + "\n")
                    
                    for product_key, result in index_products.items():
                        index_name = product_key.replace('index_', '').upper()
                        
                        if result.success and result.statistics:
                            stats = result.statistics
                            f.write(f"✓ {index_name}\n")
                            f.write(f"  Description: {stats.get('description', 'N/A')}\n")
                            f.write(f"  Application: {stats.get('application', 'N/A')}\n")
                            f.write(f"  Formula: {stats.get('formula', 'N/A')}\n")
                            f.write(f"  Bands used: {stats.get('bands_used', 'N/A')}\n")
                            f.write(f"  Value range: [{stats.get('min_value', 0):.4f}, {stats.get('max_value', 0):.4f}]\n")
                            f.write(f"  Mean ± Std: {stats.get('mean_value', 0):.4f} ± {stats.get('std_value', 0):.4f}\n")
                            f.write(f"  Coverage: {stats.get('coverage_percent', 0):.1f}%\n")
                            f.write(f"  File size: {stats.get('file_size_mb', 0):.1f} MB\n")
                            f.write(f"  Output: {os.path.basename(result.output_path)}\n\n")
                        else:
                            f.write(f"✗ {index_name} (failed)\n")
                            if result.error_message:
                                f.write(f"  Error: {result.error_message}\n\n")
                            else:
                                f.write("  Error: Unknown failure\n\n")
                
                # Configuration used
                f.write("CONFIGURATION SETTINGS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"  RGB Format: {self.config.rgb_format}\n")
                f.write(f"  Contrast Enhancement: {self.config.apply_contrast_enhancement}\n")
                f.write(f"  Contrast Method: {self.config.contrast_method}\n")
                f.write(f"  Percentile Range: {self.config.percentile_range}\n")
                f.write(f"  Export Metadata: {self.config.export_metadata}\n")
                f.write(f"  Create Overviews: {self.config.create_overview_images}\n\n")
                
                # Enabled features
                f.write("ENABLED FEATURES:\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Natural Color RGB: {self.config.generate_natural_color}\n")
                f.write(f"  False Color RGB: {self.config.generate_false_color}\n")
                f.write(f"  Water-specific RGB: {self.config.generate_water_specific}\n")
                f.write(f"  Research RGB: {self.config.generate_research_combinations}\n")
                f.write(f"  Water Quality Indices: {self.config.generate_water_quality_indices}\n")
                f.write(f"  Chlorophyll Indices: {self.config.generate_chlorophyll_indices}\n")
                f.write(f"  Turbidity Indices: {self.config.generate_turbidity_indices}\n")
                f.write(f"  Advanced Indices: {self.config.generate_advanced_indices}\n\n")
                
                # File listing
                f.write("OUTPUT FILES:\n")
                f.write("-" * 40 + "\n")
                successful_files = [result.output_path for result in viz_results.values() 
                                if result.success and result.output_path]
                
                for file_path in sorted(successful_files):
                    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    f.write(f"  {os.path.basename(file_path)} ({file_size_mb:.1f} MB)\n")
                
                total_size_mb = sum(os.path.getsize(fp) / (1024 * 1024) for fp in successful_files)
                f.write(f"\nTotal output size: {total_size_mb:.1f} MB\n")
            
            self.logger.info(f"✓ Visualization summary created: {os.path.basename(summary_file)}")
            
        except Exception as e:
            self.logger.warning(f"Could not create visualization summary: {e}")

    def _cleanup_geometric_products(self, output_folder: str, product_name: str) -> bool:
        """Clean up geometric products after processing"""
        try:
            geometric_folder = os.path.join(output_folder, "Geometric_Products")
            
            if not os.path.exists(geometric_folder):
                return True
            
            # Find files to delete based on product name
            clean_product_name = product_name.replace('.zip', '').replace('.SAFE', '')
            if 'MSIL1C' in clean_product_name:
                parts = clean_product_name.split('_')
                if len(parts) >= 6:
                    clean_name = f"{parts[0]}_{parts[2]}_{parts[5]}"
                else:
                    clean_name = clean_product_name.replace('MSIL1C_', '')
            else:
                clean_name = clean_product_name
            
            files_to_delete = []
            total_size = 0
            
            # Find matching files
            for filename in os.listdir(geometric_folder):
                if clean_name in filename and filename.startswith('Resampled_'):
                    file_path = os.path.join(geometric_folder, filename)
                    files_to_delete.append(file_path)
                    
                    # Calculate size
                    if os.path.isfile(file_path):
                        total_size += os.path.getsize(file_path)
                    
                    # Check for .data folder
                    if filename.endswith('.dim'):
                        data_folder = file_path.replace('.dim', '.data')
                        if os.path.exists(data_folder):
                            files_to_delete.append(data_folder)
                            # Calculate folder size
                            for root, dirs, files in os.walk(data_folder):
                                for file in files:
                                    total_size += os.path.getsize(os.path.join(root, file))
            
            # Delete files
            import shutil
            deleted_count = 0
            for file_path in files_to_delete:
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                    deleted_count += 1
                    self.logger.debug(f"Deleted: {os.path.basename(file_path)}")
                except Exception as e:
                    self.logger.warning(f"Could not delete {file_path}: {e}")
            
            total_size_mb = total_size / (1024 * 1024)
            self.logger.info(f"🧹 Cleaned up {deleted_count} files ({total_size_mb:.1f} MB freed)")
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Cleanup error: {e}")
            return False


    def _calculate_rgb_statistics(self, rgb_array: np.ndarray) -> Dict:
        """Calculate statistics for RGB composite"""
        try:
            stats = {}
            
            # Overall coverage
            valid_pixels = np.sum(np.any(rgb_array > 0, axis=2))
            total_pixels = rgb_array.shape[0] * rgb_array.shape[1]
            stats['coverage_percent'] = (valid_pixels / total_pixels) * 100
            
            # Per-channel statistics
            for i, channel_name in enumerate(['red', 'green', 'blue']):
                channel = rgb_array[:, :, i]
                valid_data = channel[channel > 0]
                
                if len(valid_data) > 0:
                    stats[f'{channel_name}_min'] = float(np.min(valid_data))
                    stats[f'{channel_name}_max'] = float(np.max(valid_data))
                    stats[f'{channel_name}_mean'] = float(np.mean(valid_data))
                    stats[f'{channel_name}_std'] = float(np.std(valid_data))
                else:
                    stats[f'{channel_name}_min'] = 0.0
                    stats[f'{channel_name}_max'] = 0.0
                    stats[f'{channel_name}_mean'] = 0.0
                    stats[f'{channel_name}_std'] = 0.0
            
            return stats
            
        except Exception as e:
            self.logger.warning(f"Error calculating RGB statistics: {e}")
            return {'coverage_percent': 0.0}

    def _calculate_band_statistics(self, data: np.ndarray, band_name: str) -> Dict:
        """Calculate statistics for single band data"""
        try:
            valid_data = data[~np.isnan(data)]
            
            if len(valid_data) > 0:
                stats = {
                    'band_name': band_name,
                    'min_value': float(np.min(valid_data)),
                    'max_value': float(np.max(valid_data)),
                    'mean_value': float(np.mean(valid_data)),
                    'std_value': float(np.std(valid_data)),
                    'valid_pixels': len(valid_data),
                    'coverage_percent': (len(valid_data) / data.size) * 100
                }
            else:
                stats = {
                    'band_name': band_name,
                    'min_value': 0.0,
                    'max_value': 0.0,
                    'mean_value': 0.0,
                    'std_value': 0.0,
                    'valid_pixels': 0,
                    'coverage_percent': 0.0
                }
            
            return stats
            
        except Exception as e:
            self.logger.warning(f"Error calculating band statistics for {band_name}: {e}")
            return {'band_name': band_name, 'coverage_percent': 0.0}

# ===== S2 PROCESSOR =====

class S2Processor:
    """Enhanced S2 processor with complete pipeline"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.processed_count = 0
        self.failed_count = 0
        self.skipped_count = 0
        self.start_time = time.time()
        self.current_product = ""
        self.current_stage = ""
        
        # Track intermediate products
        self.intermediate_products = {}
        
        # Validate SNAP installation
        self.validate_snap_installation()
        
        # Create processing graphs
        self.setup_processing_graphs()
    
    def validate_snap_installation(self):
        """Enhanced SNAP validation"""
        snap_home = os.environ.get('SNAP_HOME')
        if not snap_home:
            logger.error("SNAP_HOME environment variable not set!")
            raise RuntimeError("SNAP installation not found")
        
        logger.info(f"SNAP_HOME: {snap_home}")
        
        gpt_cmd = self.get_gpt_command()
        if not os.path.exists(gpt_cmd):
            logger.error(f"GPT executable not found: {gpt_cmd}")
            raise RuntimeError("GPT executable not found")
        
        try:
            result = subprocess.run([gpt_cmd, '-h'], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                logger.info(f"✓ GPT validated: {gpt_cmd}")
            else:
                logger.error(f"GPT validation failed: {result.stderr}")
                raise RuntimeError("GPT validation failed")
        except subprocess.TimeoutExpired:
            logger.error("GPT validation timeout")
            raise RuntimeError("GPT validation timeout")
        except Exception as e:
            logger.error(f"GPT validation error: {e}")
            raise RuntimeError(f"GPT validation error: {e}")
    
    def get_gpt_command(self) -> str:
        """Get GPT command for the operating system"""
        snap_home = os.environ.get('SNAP_HOME')
        if sys.platform.startswith('win'):
            return os.path.join(snap_home, 'bin', 'gpt.exe')
        else:
            return os.path.join(snap_home, 'bin', 'gpt')
    
    def setup_processing_graphs(self):
        """Create processing graphs based on configuration"""
        mode = self.config.processing_mode
        
        if mode in [ProcessingMode.COMPLETE_PIPELINE, ProcessingMode.S2_PROCESSING_ONLY]:
            if self.config.subset_config.geometry_wkt or self.config.subset_config.pixel_start_x is not None:
                self.main_graph_file = self.create_s2_graph_with_subset()
            else:
                self.main_graph_file = self.create_s2_graph_no_subset()
        
        logger.info(f"✓ Processing graph created for mode: {mode.value}")
    
    def create_s2_graph_with_subset(self) -> str:
        """Create S2 processing graph with spatial subset"""
        subset_params = self._get_subset_parameters()
        
        graph_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<graph id="S2_Complete_Processing_WithSubset">
  <version>1.0</version>
  
  <!-- Step 1: Read Input Product -->
  <node id="Read">
    <operator>Read</operator>
    <sources/>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <file>${{sourceProduct}}</file>
    </parameters>
  </node>
  
  <!-- Step 2: S2 Resampling -->
  <node id="S2Resampling">
    <operator>S2Resampling</operator>
    <sources>
      <sourceProduct refid="Read"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <resolution>{self.config.resampling_config.target_resolution}</resolution>
      <upsampling>{self.config.resampling_config.upsampling_method}</upsampling>
      <downsampling>{self.config.resampling_config.downsampling_method}</downsampling>
      <flagDownsampling>{self.config.resampling_config.flag_downsampling}</flagDownsampling>
      <resampleOnPyramidLevels>{str(self.config.resampling_config.resample_on_pyramid_levels).lower()}</resampleOnPyramidLevels>
    </parameters>
  </node>
  
  <!-- Step 3: Spatial Subset -->
  <node id="Subset">
    <operator>Subset</operator>
    <sources>
      <sourceProduct refid="S2Resampling"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      {subset_params}
      <subSamplingX>{self.config.subset_config.sub_sampling_x}</subSamplingX>
      <subSamplingY>{self.config.subset_config.sub_sampling_y}</subSamplingY>
      <fullSwath>{str(self.config.subset_config.full_swath).lower()}</fullSwath>
      <copyMetadata>{str(self.config.subset_config.copy_metadata).lower()}</copyMetadata>
    </parameters>
  </node>
  
  <!-- Step 3.5: SAVE GEOMETRIC PRODUCTS -->
<node id="WriteGeometric">
  <operator>Write</operator>
  <sources>
    <source refid="Subset"/>
  </sources>
  <parameters class="com.bc.ceres.binding.dom.XppDomElement">
    <file>${{geometricProduct}}</file>
    <formatName>BEAM-DIMAP</formatName>
  </parameters>
</node>

  <!-- Step 4: C2RCC Atmospheric Correction -->
  <node id="c2rcc_msi">
    <operator>c2rcc.msi</operator>
    <sources>
      <sourceProduct refid="Subset"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      {self._get_c2rcc_parameters()}
    </parameters>
  </node>
  
  <!-- Step 5: Write Output -->
  <node id="Write">
    <operator>Write</operator>
    <sources>
      <sourceProduct refid="c2rcc_msi"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <file>${{targetProduct}}</file>
      <formatName>BEAM-DIMAP</formatName>
    </parameters>
  </node>
  
</graph>'''
        
        graph_file = 's2_complete_processing_with_subset.xml'
        with open(graph_file, 'w', encoding='utf-8') as f:
            f.write(graph_content)
        
        logger.info(f"Complete processing graph saved: {graph_file}")
        return graph_file
    
    def create_s2_graph_no_subset(self) -> str:
        """Create S2 processing graph without spatial subset"""
        graph_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<graph id="S2_Complete_Processing_NoSubset">
  <version>1.0</version>
  
  <!-- Step 1: Read Input Product -->
  <node id="Read">
    <operator>Read</operator>
    <sources/>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <file>${{sourceProduct}}</file>
    </parameters>
  </node>
  
  <!-- Step 2: S2 Resampling -->
  <node id="S2Resampling">
    <operator>S2Resampling</operator>
    <sources>
      <sourceProduct refid="Read"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <resolution>{self.config.resampling_config.target_resolution}</resolution>
      <upsampling>{self.config.resampling_config.upsampling_method}</upsampling>
      <downsampling>{self.config.resampling_config.downsampling_method}</downsampling>
      <flagDownsampling>{self.config.resampling_config.flag_downsampling}</flagDownsampling>
      <resampleOnPyramidLevels>{str(self.config.resampling_config.resample_on_pyramid_levels).lower()}</resampleOnPyramidLevels>
    </parameters>
  </node>
  
  <!-- Step 3: C2RCC Atmospheric Correction -->
  <node id="c2rcc_msi">
    <operator>c2rcc.msi</operator>
    <sources>
      <sourceProduct refid="S2Resampling"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      {self._get_c2rcc_parameters()}
    </parameters>
  </node>
  
  <!-- Step 4: Write Output -->
  <node id="Write">
    <operator>Write</operator>
    <sources>
      <sourceProduct refid="c2rcc_msi"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <file>${{targetProduct}}</file>
      <formatName>BEAM-DIMAP</formatName>
    </parameters>
  </node>
  
</graph>'''
        
        graph_file = 's2_complete_processing_no_subset.xml'
        with open(graph_file, 'w', encoding='utf-8') as f:
            f.write(graph_content)
        
        logger.info(f"Complete processing graph saved: {graph_file}")
        return graph_file
    
    def _get_subset_parameters(self) -> str:
        """Generate subset parameters for XML with proper escaping"""
        subset_config = self.config.subset_config
        
        if subset_config.geometry_wkt:
            escaped_wkt = subset_config.geometry_wkt.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            return f"<geoRegion>{escaped_wkt}</geoRegion>"
        elif subset_config.pixel_start_x is not None:
            return f"<region>{subset_config.pixel_start_x},{subset_config.pixel_start_y},{subset_config.pixel_size_x},{subset_config.pixel_size_y}</region>"
        else:
            return ""
    
    def _get_c2rcc_parameters(self) -> str:
        """Generate complete C2RCC parameters with correct SNAP parameter names"""
        c2rcc = self.config.c2rcc_config
        
        # Escape XML special characters
        valid_pixel_expr = c2rcc.valid_pixel_expression.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        
        # Build complete parameter set using exact SNAP parameter names (CORRECT ORDER)
        params = f'''<validPixelExpression>{valid_pixel_expr}</validPixelExpression>
        <salinity>{c2rcc.salinity}</salinity>
        <temperature>{c2rcc.temperature}</temperature>
        <ozone>{c2rcc.ozone}</ozone>
        <press>{c2rcc.pressure}</press>
        <elevation>{c2rcc.elevation}</elevation>
        <TSMfac>{c2rcc.tsm_fac}</TSMfac>
        <TSMexp>{c2rcc.tsm_exp}</TSMexp>
        <CHLexp>{c2rcc.chl_exp}</CHLexp>
        <CHLfac>{c2rcc.chl_fac}</CHLfac>
        <thresholdRtosaOOS>{c2rcc.threshold_rtosa_oos}</thresholdRtosaOOS>
        <thresholdAcReflecOos>{c2rcc.threshold_ac_reflec_oos}</thresholdAcReflecOos>
        <thresholdCloudTDown865>{c2rcc.threshold_cloud_tdown865}</thresholdCloudTDown865>
        <netSet>{c2rcc.net_set}</netSet>
        <useEcmwfAuxData>{str(c2rcc.use_ecmwf_aux_data).lower()}</useEcmwfAuxData>
        <demName>{c2rcc.dem_name}</demName>
        <outputAsRrs>{str(c2rcc.output_as_rrs).lower()}</outputAsRrs>
        <deriveRwFromPathAndTransmittance>{str(c2rcc.derive_rw_from_path_and_transmittance).lower()}</deriveRwFromPathAndTransmittance>
        <outputRtoa>{str(c2rcc.output_rtoa).lower()}</outputRtoa>
        <outputRtosaGc>{str(c2rcc.output_rtosa_gc).lower()}</outputRtosaGc>
        <outputRtosaGcAann>{str(c2rcc.output_rtosa_gc_aann).lower()}</outputRtosaGcAann>
        <outputRpath>{str(c2rcc.output_rpath).lower()}</outputRpath>
        <outputTdown>{str(c2rcc.output_tdown).lower()}</outputTdown>
        <outputTup>{str(c2rcc.output_tup).lower()}</outputTup>
        <outputAcReflectance>{str(c2rcc.output_ac_reflectance).lower()}</outputAcReflectance>
        <outputRhown>{str(c2rcc.output_rhown).lower()}</outputRhown>
        <outputOos>{str(c2rcc.output_oos).lower()}</outputOos>
        <outputKd>{str(c2rcc.output_kd).lower()}</outputKd>
        <outputUncertainties>{str(c2rcc.output_uncertainties).lower()}</outputUncertainties>'''
        
        # Add optional paths if specified
        if c2rcc.atmospheric_aux_data_path:
            params += f'\n      <atmosphericAuxDataPath>{c2rcc.atmospheric_aux_data_path}</atmosphericAuxDataPath>'
        
        if c2rcc.alternative_nn_path:
            params += f'\n      <alternativeNNPath>{c2rcc.alternative_nn_path}</alternativeNNPath>'
        
        return params
    
    def _get_geometric_output_path(self, input_path: str, output_folder: str) -> str:
        """Generate geometric product output path"""
        basename = os.path.basename(input_path)
        if basename.endswith('.zip'):
            product_name = basename.replace('.zip', '')
        elif basename.endswith('.SAFE'):
            product_name = basename.replace('.SAFE', '')
        else:
            product_name = basename
        
        geometric_folder = os.path.join(output_folder, "Geometric_Products")
        os.makedirs(geometric_folder, exist_ok=True)
        
        # Clean product name for geometric output
        if 'MSIL1C' in product_name:
            parts = product_name.split('_')
            if len(parts) >= 6:
                clean_name = f"Resampled_{parts[0]}_{parts[2]}_{parts[5]}_Subset"
            else:
                clean_name = f"Resampled_{product_name}_Subset"
        else:
            clean_name = f"Resampled_{product_name}_Subset"
        
        return os.path.join(geometric_folder, f"{clean_name}.dim")
    
    def get_output_filename(self, input_path: str, output_dir: str, stage: str) -> str:
        """Generate output filename based on processing stage"""
        basename = os.path.basename(input_path)
        
        # Extract base product name
        if basename.endswith('.zip'):
            product_name = basename.replace('.zip', '')
        elif basename.endswith('.SAFE'):
            product_name = basename.replace('.SAFE', '')
        elif basename.endswith('.dim'):
            product_name = basename.replace('.dim', '')
        else:
            product_name = basename
            
        # Remove MSIL1C prefix for cleaner naming
        if 'MSIL1C' in product_name:
            # Extract key parts: S2A_MSIL1C_20230615T113321_N0509_R080_T29TNE_20230615T134426
            parts = product_name.split('_')
            if len(parts) >= 6:
                # Create cleaner name: S2A_20230615T113321_T29TNE
                clean_name = f"{parts[0]}_{parts[2]}_{parts[5]}"
            else:
                clean_name = product_name.replace('MSIL1C_', '')
        else:
            clean_name = product_name
            
        # Stage-specific naming
        if stage == "geometric":
            output_name = f"Resampled_{clean_name}_Subset.dim"
            return os.path.join(output_dir, "Geometric_Products", output_name)
        elif stage == "c2rcc":
            output_name = f"Resampled_{clean_name}_Subset_C2RCC.dim"
            return os.path.join(output_dir, "C2RCC_Products", output_name)
        else:
            return os.path.join(output_dir, f"{clean_name}_{stage}.dim")
    
    def process_single_product(self, input_path: str, output_folder: str) -> Dict[str, ProcessingResult]:
        """
        Process single product through complete S2 pipeline - CORRECTED FULL VERSION
        
        Responsibilities:
        - L1C → Resampling → Subset → C2RCC
        - Calculate SNAP TSM/CHL from IOPs
        - Verify C2RCC output
        """
        processing_start = time.time()
        results = {}
        
        try:
            product_name = os.path.basename(input_path)
            self.current_product = product_name
            
            logger.info(f"Processing: {product_name}")
            logger.info(f"  Mode: {self.config.processing_mode.value}")
            logger.info(f"  Resolution: {self.config.resampling_config.target_resolution}m")
            logger.info(f"  ECMWF: {self.config.c2rcc_config.use_ecmwf_aux_data}")
            
            # Check system health before processing
            if hasattr(self, 'system_monitor'):
                healthy, warnings = self.system_monitor.check_system_health()
                if not healthy:
                    logger.warning("System health issues detected:")
                    for warning in warnings:
                        logger.warning(f"  - {warning}")
            
            # Ensure output directories exist
            os.makedirs(os.path.join(output_folder, "Geometric_Products"), exist_ok=True)
            os.makedirs(os.path.join(output_folder, "C2RCC_Products"), exist_ok=True)
            os.makedirs(os.path.join(output_folder, "TSS_Products"), exist_ok=True)
            os.makedirs(os.path.join(output_folder, "Logs"), exist_ok=True)
            
            # Step 1: S2 Processing (Resampling + Subset + C2RCC)
            self.current_stage = "S2 Processing (Complete)"
            c2rcc_output_path = self.get_output_filename(input_path, output_folder, "c2rcc")
            
            # Check if output already exists and is valid
            if self.config.skip_existing and os.path.exists(c2rcc_output_path):
                file_size = os.path.getsize(c2rcc_output_path)
                if file_size > 1024 * 1024:  # > 1MB
                    logger.info(f"C2RCC output exists ({file_size/1024/1024:.1f}MB), skipping S2 processing")
                    self.skipped_count += 1
                    
                    # Verify required bands exist for potential TSS processing
                    data_folder = c2rcc_output_path.replace('.dim', '.data')
                    required_bands = ['conc_tsm.img', 'conc_chl.img', 'unc_tsm.img', 'unc_chl.img']
                    if self.config.jiang_config.enable_jiang_tss:
                        required_bands.extend(['rhow_B1.img', 'rhow_B2.img', 'rhow_B3.img', 'rhow_B4.img',
                                            'rhow_B5.img', 'rhow_B6.img', 'rhow_B7.img', 'rhow_B8A.img'])
                    
                    missing_bands = []
                    for band in required_bands:
                        if not os.path.exists(os.path.join(data_folder, band)):
                            missing_bands.append(band)
                    
                    if missing_bands:
                        logger.warning(f"Missing bands for TSS processing: {missing_bands}")
                        logger.warning("Will reprocess to ensure all required bands are available")
                    else:
                        # Verify and return existing output
                        c2rcc_stats = self._verify_c2rcc_output(c2rcc_output_path)
                        if c2rcc_stats:
                            results['s2_processing'] = ProcessingResult(True, c2rcc_output_path, 
                                                                    c2rcc_stats, None)
                            
                            # ===== S2PROCESSOR STOPS HERE - NO TSS PROCESSING =====
                            # TSS processing is handled by UnifiedS2TSSProcessor/JiangTSSProcessor
                            logger.info("✅ S2 processing completed - ready for TSS processing")
                            
                            return results
                else:
                    logger.warning(f"Removing incomplete C2RCC output file ({file_size} bytes)")
                    os.remove(c2rcc_output_path)
            
            # Run S2 processing
            logger.info("Starting S2 processing with C2RCC...")
            s2_success = self._run_s2_processing(input_path, c2rcc_output_path)
            
            processing_time = time.time() - processing_start
            
            if s2_success:
                # Verify C2RCC output and extract SNAP TSM/CHL info
                c2rcc_stats = self._verify_c2rcc_output(c2rcc_output_path)
                
                if c2rcc_stats:
                    # Initial S2 processing success
                    logger.info(f"✅ S2 processing SUCCESS: {product_name}")
                    logger.info(f"   Output: {os.path.basename(c2rcc_output_path)} ({c2rcc_stats['file_size_mb']:.1f}MB)")
                    logger.info(f"   Processing time: {processing_time/60:.1f} minutes")
                    
                    self.processed_count += 1
                    results['s2_processing'] = ProcessingResult(True, c2rcc_output_path, c2rcc_stats, None)
                    
                    # Calculate SNAP TSM/CHL from IOPs if missing
                    if not c2rcc_stats['has_tsm'] or not c2rcc_stats['has_chl']:
                        logger.info("Calculating missing SNAP TSM/CHL concentrations from IOP products...")
                        snap_calculator = SNAPTSMCHLCalculator(
                            tsm_fac=self.config.c2rcc_config.tsm_fac,
                            tsm_exp=self.config.c2rcc_config.tsm_exp,
                            chl_fac=self.config.c2rcc_config.chl_fac,
                            chl_exp=self.config.c2rcc_config.chl_exp
                        )
                        
                        snap_tsm_chl_results = snap_calculator.calculate_snap_tsm_chl(c2rcc_output_path)
                        results.update(snap_tsm_chl_results)
                        
                        # Log SNAP TSM/CHL results
                        if 'snap_tsm' in snap_tsm_chl_results and snap_tsm_chl_results['snap_tsm'].success:
                            logger.info("✅ SNAP TSM calculation successful!")
                        
                        if 'snap_chl' in snap_tsm_chl_results and snap_tsm_chl_results['snap_chl'].success:
                            logger.info("✅ SNAP CHL calculation successful!")
                        
                        # Re-verify after calculating TSM/CHL
                        final_stats = self._verify_c2rcc_output(c2rcc_output_path)
                        if final_stats:
                            # Store intermediate paths for marine visualization
                            self.intermediate_products['geometric_path'] = self._get_geometric_output_path(input_path, output_folder)
                            self.intermediate_products['c2rcc_path'] = c2rcc_output_path
                            
                            # Create enhanced ProcessingResult with intermediate paths
                            result = ProcessingResult(True, c2rcc_output_path, final_stats, None)
                            result.intermediate_paths = self.intermediate_products.copy()
                            results['s2_processing'] = result
                            logger.info("📊 UPDATED SNAP PRODUCTS STATUS:")
                            logger.info(f"   TSM={final_stats['has_tsm']}, CHL={final_stats['has_chl']}, Uncertainties={final_stats['has_uncertainties']}")
                    
                    # ===== CRITICAL FIX: REMOVE THE PROBLEMATIC TSS CALL =====
                    # REMOVED: The problematic _process_tss_stage call that was causing the error
                    # OLD CODE (CAUSING ERROR):
                    # if self.config.processing_mode == ProcessingMode.COMPLETE_PIPELINE and self.config.jiang_config.enable_jiang_tss:
                    #     tss_results = self._process_tss_stage(c2rcc_output_path, output_folder, product_name)  # ❌ ERROR
                    #     results.update(tss_results)
                    
                    # NEW CODE: Clean separation - S2Processor only does S2 processing
                    logger.info("✅ S2 processing completed - ready for TSS processing")
                    logger.info("ℹ️  TSS processing (Jiang + Marine Viz) handled by separate processors")
                    
                    # Clean up memory after S2 processing
                    try:
                        import gc
                        gc.collect()
                        
                        # Clear any large temporary variables
                        temp_vars = ['temp_data', 'large_array', 'processing_data']
                        for var in temp_vars:
                            if var in locals():
                                del locals()[var]
                        
                        gc.collect()
                        logger.debug("Memory cleanup completed after S2 processing")
                    except Exception as cleanup_error:
                        logger.debug(f"Memory cleanup warning: {cleanup_error}")
                    
                else:
                    logger.error("C2RCC output verification failed")
                    self.failed_count += 1
                    results['s2_error'] = ProcessingResult(False, "", None, "C2RCC verification failed")
            else:
                logger.error(f"S2 processing failed: {product_name}")
                self.failed_count += 1
                results['s2_error'] = ProcessingResult(False, "", None, "S2 processing failed")
            
            return results
            
        except Exception as e:
            error_msg = f"Unexpected error processing {product_name}: {str(e)}"
            logger.error(error_msg)
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            self.failed_count += 1
            return {'s2_error': ProcessingResult(False, "", None, error_msg)}
    
    def _run_s2_processing(self, input_path: str, output_path: str) -> bool:
        """Run S2 processing using GPT - CORRECTED VERSION"""
        try:
            # Prepare GPT command
            gpt_cmd = self.get_gpt_command()
            
            # FIXED: Get the main output folder (parent of C2RCC_Products)
            # output_path is like: /path/to/output/C2RCC_Products/product.dim
            # We need: /path/to/output/ (the main output folder)
            output_folder = os.path.dirname(os.path.dirname(output_path))
            geometric_output_path = self._get_geometric_output_path(input_path, output_folder)
            
            # Ensure geometric products directory exists
            os.makedirs(os.path.dirname(geometric_output_path), exist_ok=True)
            
            cmd = [
                gpt_cmd,
                self.main_graph_file,
                f'-PsourceProduct={input_path}',
                f'-PtargetProduct={output_path}',
                f'-PgeometricProduct={geometric_output_path}',  # NOW PROPERLY FIXED
                f'-c', f'{self.config.memory_limit_gb}G',
                f'-q', str(self.config.thread_count)
            ]
            
            logger.info(f"GPT processing paths:")
            logger.info(f"  Input: {os.path.basename(input_path)}")
            logger.info(f"  C2RCC Output: {os.path.basename(output_path)}")
            logger.info(f"  Geometric Output: {os.path.basename(geometric_output_path)}")
            logger.debug(f"GPT command: {' '.join(cmd)}")
            
            # Run GPT processing with timeout
            logger.info("Executing GPT processing...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            # Check processing results
            if result.returncode == 0:
                # Check main C2RCC output
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path)
                    if file_size > 1024 * 1024:  # > 1MB
                        logger.info(f"✅ C2RCC output created: {file_size / (1024*1024):.1f} MB")
                        
                        # Also check if geometric product was created
                        if os.path.exists(geometric_output_path):
                            geom_size = os.path.getsize(geometric_output_path)
                            logger.info(f"✅ Geometric output created: {geom_size / (1024*1024):.1f} MB")
                        else:
                            logger.warning("⚠️  Geometric product not created (but C2RCC succeeded)")
                        
                        return True
                    else:
                        logger.error(f"❌ C2RCC output file too small ({file_size} bytes)")
                        return False
                else:
                    logger.error(f"❌ C2RCC output file not created")
                    return False
            else:
                logger.error(f"❌ GPT processing failed")
                logger.error(f"Return code: {result.returncode}")
                if result.stderr:
                    logger.error(f"GPT stderr: {result.stderr[:1000]}...")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"❌ GPT processing timeout")
            return False
        except Exception as e:
            logger.error(f"❌ GPT processing error: {str(e)}")
            return False
    
    def _verify_c2rcc_output(self, c2rcc_path: str) -> Optional[Dict]:
        """Enhanced C2RCC output verification with automatic fixes"""
        try:
            if not os.path.exists(c2rcc_path):
                logger.error(f"C2RCC file not found: {c2rcc_path}")
                return None
            
            # Basic file info
            file_size = os.path.getsize(c2rcc_path)
            file_size_mb = file_size / (1024 * 1024)
            data_folder = c2rcc_path.replace('.dim', '.data')
            
            logger.info("=" * 60)
            logger.info("C2RCC OUTPUT VERIFICATION REPORT")
            logger.info("=" * 60)
            logger.info(f"C2RCC file: {os.path.basename(c2rcc_path)} ({file_size_mb:.1f} MB)")
            logger.info(f"Data folder: {os.path.basename(data_folder)}")
            
            # Check critical products
            critical_products = {
                'conc_tsm.img': 'TSM',
                'conc_chl.img': 'CHL',
                'unc_tsm.img': 'TSM uncertainties',
                'unc_chl.img': 'CHL uncertainties'
            }
            
            logger.info("\n🧪 SNAP TSM/CHL PRODUCTS (Critical for Analysis):")
            
            has_tsm = False
            has_chl = False
            has_uncertainties = True  # Start optimistic
            
            for filename, description in critical_products.items():
                file_path = os.path.join(data_folder, filename)
                exists, size_kb, is_valid = self._verify_file_integrity(file_path, min_size_kb=10)
                
                if exists and is_valid:
                    logger.info(f"   ✓ {description}: {filename} ({size_kb:.1f} KB)")
                    if 'tsm' in filename and 'unc' not in filename:
                        has_tsm = True
                    elif 'chl' in filename and 'unc' not in filename:
                        has_chl = True
                elif exists and not is_valid:
                    logger.warning(f"   ⚠ {description}: {filename} ({size_kb:.1f} KB) - File too small!")
                    if 'unc' in filename:
                        has_uncertainties = False
                else:
                    logger.warning(f"   ✗ {description}: {filename} (missing/empty)")
                    if 'unc' in filename:
                        has_uncertainties = False
                    elif 'tsm' in filename:
                        has_tsm = False
                    elif 'chl' in filename:
                        has_chl = False
            
            # Success message or issues
            if has_tsm and has_chl:
                if has_uncertainties:
                    logger.info("✅ SUCCESS: Both TSM and CHL products available with uncertainties!")
                else:
                    logger.info("✅ SUCCESS: Both TSM and CHL products available!")
                    logger.info("   ⚠ Uncertainties: Not available (calculated products)")
            else:
                logger.info("ℹ️  TSM/CHL products will be calculated from IOP products")
                logger.info("   Using SNAP formulas:")
                logger.info("   • TSM = TSMfac * (bpart + bwit)^TSMexp")
                logger.info("   • CHL = apig^CHLexp * CHLfac")
            
            # Check IOP products
            logger.info("\n🔬 SNAP IOP PRODUCTS (5/6 available):")
            iop_products = ['iop_apig.img', 'iop_adet.img', 'iop_agelb.img', 
                        'iop_bpart.img', 'iop_bwit.img', 'iop_btot.img']
            
            iop_count = 0
            for iop_file in iop_products:
                file_path = os.path.join(data_folder, iop_file)
                exists, size_kb, is_valid = self._verify_file_integrity(file_path, min_size_kb=100)
                
                if exists and is_valid:
                    logger.info(f"   ✓ {iop_file} ({size_kb:.1f} KB)")
                    iop_count += 1
                else:
                    if iop_file == 'iop_btot.img':
                        # Try to calculate missing btot
                        if self._calculate_missing_btot(c2rcc_path):
                            logger.info(f"   ✓ {iop_file} (calculated from bpart + bwit)")
                            iop_count += 1
                        else:
                            logger.info(f"   ✗ {iop_file} (missing/empty)")
                    else:
                        logger.info(f"   ✗ {iop_file} (missing/empty)")
            
            # Check rhow bands for Jiang TSS
            logger.info("\n🌊 JIANG TSS READINESS:")
            rhow_bands = self._check_rhow_bands_availability(c2rcc_path)
            rhow_count = len(rhow_bands)
            
            if rhow_count == 8:
                logger.info(f"✅ EXCELLENT: All {rhow_count}/8 rhow bands available")
                ready_for_jiang = True
            elif rhow_count >= 6:
                logger.info(f"✅ GOOD: {rhow_count}/8 rhow bands available (sufficient for processing)")
                ready_for_jiang = True
            elif rhow_count > 0:
                logger.warning(f"⚠ PARTIAL: Only {rhow_count}/8 rhow bands available")
                missing_bands = []
                for wl in [443, 490, 560, 665, 705, 740, 783, 865]:
                    if wl not in rhow_bands:
                        band_names = ['rhow_B1.img', 'rhow_B2.img', 'rhow_B3.img', 'rhow_B4.img', 
                                    'rhow_B5.img', 'rhow_B6.img', 'rhow_B7.img', 'rhow_B8A.img']
                        wavelengths = [443, 490, 560, 665, 705, 740, 783, 865]
                        if wl in wavelengths:
                            missing_bands.append(band_names[wavelengths.index(wl)])
                logger.warning(f"   Missing bands: {missing_bands}")
                ready_for_jiang = False
            else:
                logger.warning("⚠ PARTIAL: Only 0/8 rhow bands available")
                logger.warning("   Missing bands: ['rhow_B1.img', 'rhow_B2.img', 'rhow_B3.img', 'rhow_B4.img', 'rhow_B5.img', 'rhow_B6.img', 'rhow_B7.img', 'rhow_B8A.img']")
                ready_for_jiang = False
            
            # Overall assessment
            logger.info("\n📊 OVERALL ASSESSMENT:")
            
            if has_tsm and has_chl and has_uncertainties and ready_for_jiang and iop_count >= 5:
                logger.info("🎉 EXCELLENT: Complete processing capabilities")
                logger.info("   - SNAP TSM/CHL: Available")
                logger.info("   - Jiang TSS: Ready")
                logger.info("   - Advanced algorithms: Can proceed")
                overall_status = "excellent"
            elif has_tsm and has_chl and ready_for_jiang:
                logger.info("👍 GOOD: Jiang TSS ready, SNAP products will be calculated")
                overall_status = "good"
            elif ready_for_jiang:
                logger.info("✅ ADEQUATE: Jiang TSS ready, some products missing")
                overall_status = "adequate"
            else:
                logger.warning("⚠ LIMITED: Missing critical components")
                overall_status = "limited"
            
            logger.info("=" * 60)
            
            # Return comprehensive stats
            stats = {
                'file_size_mb': file_size_mb,
                'has_tsm': has_tsm,
                'has_chl': has_chl,
                'has_uncertainties': has_uncertainties,
                'iop_count': iop_count,
                'rhow_bands_count': rhow_count,
                'ready_for_jiang_tss': ready_for_jiang,
                'overall_status': overall_status,
                'rhow_bands': rhow_bands
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Unexpected error verifying C2RCC output {c2rcc_path}: {e}")
            import traceback
            traceback.print_exc()
            return None
        
    def _get_geometric_output_path(self, input_path: str, output_folder: str) -> str:
        """Get the geometric processing output path"""
        product_name = os.path.basename(input_path)
        clean_name = self._extract_clean_product_name(product_name)
        
        # Check both possible geometric output names
        subset_path = os.path.join(output_folder, "Geometric_Products", f"Resampled_{clean_name}_Subset.dim")
        resample_path = os.path.join(output_folder, "Geometric_Products", f"Resampled_{clean_name}.dim")
        
        # Return the one that exists, or default to subset version
        if os.path.exists(subset_path):
            return subset_path
        elif os.path.exists(resample_path):
            return resample_path
        else:
            # Default assumption: subset is applied
            return subset_path

    def _extract_clean_product_name(self, product_name: str) -> str:
        """Extract clean product name from input path"""
        clean_name = os.path.basename(product_name)
        clean_name = clean_name.replace('.zip', '').replace('.SAFE', '')
        
        # Create cleaner name from Sentinel-2 product name
        if 'MSIL1C' in clean_name:
            parts = clean_name.split('_')
            if len(parts) >= 6:
                # S2A_MSIL1C_20190105T112441_N0500_R037_T29TNF_20221215T164753
                # becomes: S2A_20190105T112441_T29TNF
                clean_name = f"{parts[0]}_{parts[2]}_{parts[5]}"
            else:
                clean_name = clean_name.replace('MSIL1C_', '')
        
        return clean_name
    
    def _verify_file_integrity(self, file_path: str, min_size_kb: int = 1) -> tuple:
        """Verify file exists and has reasonable size"""
        try:
            if not os.path.exists(file_path):
                return False, 0.0, False
            
            size_bytes = os.path.getsize(file_path)
            size_kb = size_bytes / 1024
            is_valid = size_kb >= min_size_kb
            
            return True, size_kb, is_valid
            
        except Exception as e:
            logger.error(f"Error checking file {file_path}: {e}")
            return False, 0.0, False

    def _calculate_missing_btot(self, c2rcc_path: str) -> bool:
        """Calculate missing iop_btot from bpart + bwit - FIXED VERSION"""
        try:
            data_folder = c2rcc_path.replace('.dim', '.data')
            bpart_path = os.path.join(data_folder, 'iop_bpart.img')
            bwit_path = os.path.join(data_folder, 'iop_bwit.img')
            btot_path = os.path.join(data_folder, 'iop_btot.img')
            
            # Check if btot already exists and is valid
            if os.path.exists(btot_path) and os.path.getsize(btot_path) > 1024:
                logger.info("✓ iop_btot.img already exists")
                return True
            
            # Check if source files exist
            if not os.path.exists(bpart_path):
                logger.error(f"Missing bpart file: {bpart_path}")
                return False
            
            if not os.path.exists(bwit_path):
                logger.error(f"Missing bwit file: {bwit_path}")
                return False
            
            logger.info("🔄 Calculating missing iop_btot from bpart + bwit...")
            
            # Load bpart and bwit
            bpart_data, bpart_meta = RasterIO.read_raster(bpart_path)
            bwit_data, _ = RasterIO.read_raster(bwit_path)
            
            # Validate data loaded correctly
            if not isinstance(bpart_data, np.ndarray):
                logger.error(f"Invalid bpart_data type: {type(bpart_data)}")
                return False
            
            if not isinstance(bwit_data, np.ndarray):
                logger.error(f"Invalid bwit_data type: {type(bwit_data)}")
                return False
            
            # Check shapes match
            if bpart_data.shape != bwit_data.shape:
                logger.error(f"Shape mismatch: bpart {bpart_data.shape} vs bwit {bwit_data.shape}")
                return False
            
            # Calculate btot = bpart + bwit
            btot_data = bpart_data + bwit_data
            
            # Validate metadata
            if not isinstance(bpart_meta, dict):
                logger.error(f"Invalid metadata type: {type(bpart_meta)}")
                return False
            
            logger.info(f"Calculated btot: shape={btot_data.shape}, valid_pixels={np.sum(~np.isnan(btot_data))}")
            
            # FIXED: Call write_raster with proper parameters
            success = RasterIO.write_raster(
                data=btot_data,
                output_path=btot_path,
                metadata=bpart_meta,
                description="Total backscattering coefficient (bpart + bwit) in m^-1",
                nodata=-9999
            )
            
            if success and os.path.exists(btot_path):
                file_size_kb = os.path.getsize(btot_path) / 1024
                logger.info(f"✅ Successfully calculated and saved iop_btot.img ({file_size_kb:.1f} KB)")
                return True
            else:
                logger.error("❌ Failed to save iop_btot.img")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error calculating btot: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
        
    def _get_file_size_kb(self, file_path: str) -> float:
        """Get file size in KB"""
        try:
            if os.path.exists(file_path):
                return os.path.getsize(file_path) / 1024
            return 0.0
        except:
            return 0.0

    def _check_rhow_bands_availability(self, c2rcc_path: str) -> Dict[int, str]:
        """Load water reflectance bands from SNAP C2RCC output - UPDATED to include RRS"""
        if c2rcc_path.endswith('.dim'):
            data_folder = c2rcc_path.replace('.dim', '.data')
        else:
            data_folder = f"{c2rcc_path}.data"
        
        if not os.path.exists(data_folder):
            logger.error(f"Data folder not found: {data_folder}")
            return {}
        
        # UPDATED: Include rrs_ bands as first priority since both rhow and rrs are equally valid
        # for the Jiang algorithm (both are water-leaving reflectance, just different units)
        band_mapping = {
            443: ['rhow_B1.img', 'rrs_B1.img', 'rhown_B1.img', 'rho_toa_B1.img', 'rtoa_B1.img'],
            490: ['rhow_B2.img', 'rrs_B2.img', 'rhown_B2.img', 'rho_toa_B2.img', 'rtoa_B2.img'],
            560: ['rhow_B3.img', 'rrs_B3.img', 'rhown_B3.img', 'rho_toa_B3.img', 'rtoa_B3.img'],
            665: ['rhow_B4.img', 'rrs_B4.img', 'rhown_B4.img', 'rho_toa_B4.img', 'rtoa_B4.img'],
            705: ['rhow_B5.img', 'rrs_B5.img', 'rhown_B5.img', 'rho_toa_B5.img', 'rtoa_B5.img'],
            740: ['rhow_B6.img', 'rrs_B6.img', 'rhown_B6.img', 'rho_toa_B6.img', 'rtoa_B6.img'],
            783: ['rhow_B7.img', 'rrs_B7.img', 'rho_toa_B7.img', 'rtoa_B7.img'],  # No rhown_B7
            865: ['rhow_B8A.img', 'rrs_B8A.img', 'rho_toa_B8A.img', 'rtoa_B8A.img']  # No rhown_B8A
        }
        
        rhow_bands = {}
        logger.info(f"🔍 Checking for water reflectance bands in: {data_folder}")
        
        found_band_types = {}  # Track which type of bands we're finding
        
        for wavelength, possible_names in band_mapping.items():
            for name in possible_names:
                file_path = os.path.join(data_folder, name)
                if os.path.exists(file_path) and os.path.getsize(file_path) > 1024:  # >1KB
                    rhow_bands[wavelength] = file_path
                    
                    # Track band type for logging
                    if name.startswith('rhow_'):
                        band_type = 'rhow'
                    elif name.startswith('rrs_'):
                        band_type = 'rrs'
                    elif name.startswith('rhown_'):
                        band_type = 'rhown'
                    else:
                        band_type = 'toa'
                    
                    found_band_types[wavelength] = band_type
                    logger.info(f"✓ Found band {wavelength}nm: {name} ({band_type})")
                    break
            else:
                logger.warning(f"✗ Missing band {wavelength}nm")
        
        # Summary of band types found
        type_counts = {}
        for band_type in found_band_types.values():
            type_counts[band_type] = type_counts.get(band_type, 0) + 1
        
        logger.info(f"📊 Band type summary: {type_counts}")
        
        # Check if we have mixed band types (which might need unit conversion)
        unique_types = set(found_band_types.values())
        if len(unique_types) > 1:
            logger.warning(f"⚠️  Mixed band types detected: {unique_types}")
            logger.warning("   This may require unit conversions in processing")
        
        return rhow_bands
    
    def get_processing_status(self) -> ProcessingStatus:
        """Get current processing status with division by zero protection"""
        total = self.processed_count + self.failed_count + self.skipped_count
        if total == 0:
            return ProcessingStatus(0, 0, 0, 0, "", "", 0.0, 0.0, 0.0)
        
        elapsed_time = time.time() - self.start_time
        
        # Calculate ETA with protection against division by zero
        if self.processed_count > 0 and elapsed_time > 0:
            avg_time_per_product = elapsed_time / self.processed_count
            # Estimate based on a typical batch size
            eta_minutes = avg_time_per_product / 60
            processing_speed = (self.processed_count / elapsed_time) * 60  # products per minute
        else:
            eta_minutes = 0.0
            processing_speed = 0.0
        
        # Fix: Ensure no division by zero in progress calculation
        progress_percent = (total / max(total, 1)) * 100 if total > 0 else 0.0
        
        return ProcessingStatus(
            total_products=total,
            processed=self.processed_count,
            failed=self.failed_count,
            skipped=self.skipped_count,
            current_product=self.current_product,
            current_stage=self.current_stage,
            progress_percent=progress_percent,
            eta_minutes=eta_minutes,
            processing_speed=processing_speed
        )
    
    def cleanup(self):
        """Cleanup resources"""
        # Clean up graph files
        graph_files = [getattr(self, 'main_graph_file', None)]
        for graph_file in graph_files:
            if graph_file and os.path.exists(graph_file):
                try:
                    os.remove(graph_file)
                except:
                    pass

# ===== MAIN UNIFIED PROCESSOR =====

class UnifiedS2TSSProcessor:
    """Main processor that coordinates complete S2 processing and TSS estimation"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.s2_processor = None
        self.jiang_processor = None
        
        # Processing statistics
        self.processed_count = 0
        self.failed_count = 0
        self.skipped_count = 0
        self.start_time = time.time()
        
        # System monitoring
        self.system_monitor = SystemMonitor()
        self.system_monitor.start_monitoring()
        
        # Initialize processors based on mode
        self._initialize_processors()
        
        logger.info(f"Initialized Unified S2-TSS Processor - Mode: {config.processing_mode.value}")
    
    def _initialize_processors(self):
        """Initialize processors based on processing mode"""
        mode = self.config.processing_mode
        
        if mode in [ProcessingMode.COMPLETE_PIPELINE, ProcessingMode.S2_PROCESSING_ONLY]:
            self.s2_processor = S2Processor(self.config)
            self.s2_processor.system_monitor = self.system_monitor
        
        if mode in [ProcessingMode.COMPLETE_PIPELINE, ProcessingMode.TSS_PROCESSING_ONLY]:
            self.jiang_processor = JiangTSSProcessor(self.config.jiang_config)
    
    def process_batch(self) -> Dict[str, int]:
        """
        Process all products in the input folder based on selected mode
        
        Returns:
            Processing statistics
        """
        try:
            logger.info("="*80)
            logger.info("STARTING UNIFIED S2-TSS PROCESSING")
            logger.info("="*80)
            
            # Find and validate products
            products = self._find_products()
            if not products:
                logger.error("No compatible products found")
                return {'processed': 0, 'failed': 1, 'skipped': 0}
            
            logger.info(f"Found {len(products)} products to process")
            logger.info(f"Processing mode: {self.config.processing_mode.value}")
            
            # Process each product
            for i, product_path in enumerate(products, 1):
                self._process_single_product(product_path, i, len(products))
            
            # Final summary
            self._print_final_summary()
            
            return {
                'processed': self.processed_count,
                'failed': self.failed_count,
                'skipped': self.skipped_count
            }
            
        except Exception as e:
            error_msg = f"Batch processing error: {str(e)}"
            logger.error(error_msg)
            return {'processed': self.processed_count, 'failed': self.failed_count + 1, 'skipped': self.skipped_count}
    
    def _find_products(self) -> List[str]:
        """Find products based on processing mode"""
        products = ProductDetector.scan_input_folder(self.config.input_folder)
        mode = self.config.processing_mode
        
        # Validate products for current mode
        valid, message, product_list = ProductDetector.validate_processing_mode(products, mode)
        
        if not valid:
            logger.error(f"Product validation failed: {message}")
            return []
        
        logger.info(message)
        return sorted(product_list)
    
    def _process_single_product(self, product_path: str, current: int, total: int):
        """Process single product based on mode - FIXED COMPLETE PIPELINE"""
        processing_start = time.time()
        
        try:
            product_name = self._extract_product_name(product_path)
            
            logger.info(f"\n{'-'*80}")
            logger.info(f"Processing {current}/{total}: {product_name}")
            logger.info(f"Mode: {self.config.processing_mode.value}")
            logger.info(f"{'-'*80}")
            
            # Check if outputs already exist
            if self.config.skip_existing and self._check_outputs_exist(product_name):
                logger.info(f"Outputs exist, skipping: {product_name}")
                self.skipped_count += 1
                return
            
            # Process based on mode
            results = {}
            
            if self.config.processing_mode == ProcessingMode.COMPLETE_PIPELINE:
                # FIXED: Complete pipeline with proper coordination
                # Step 1: S2 Processing (L1C → C2RCC)
                logger.info("Step 1: S2 Processing (L1C → C2RCC)")
                s2_results = self.s2_processor.process_single_product(product_path, self.config.output_folder)
                results.update(s2_results)
                
                # Check if S2 processing succeeded
                if 'error' in s2_results or 's2_error' in s2_results:
                    error_key = 'error' if 'error' in s2_results else 's2_error'
                    logger.error(f"S2 processing failed, skipping TSS: {s2_results[error_key].error_message}")
                    self.failed_count += 1
                    return
                
                # Step 2: TSS Processing (if enabled and processor exists)
                if (self.config.jiang_config.enable_jiang_tss and 
                    hasattr(self, 'jiang_processor') and 
                    self.jiang_processor is not None):
                    
                    logger.info("Step 2: TSS Processing (Jiang)")
                    
                    # Get C2RCC output path from S2 results
                    if 's2_processing' in s2_results:
                        c2rcc_output_path = s2_results['s2_processing'].output_path
                    else:
                        # Find the C2RCC output file
                        c2rcc_output_path = os.path.join(self.config.output_folder, "C2RCC_Products", f"Resampled_{product_name}_Subset_C2RCC.dim")
                    
                    # Process TSS
                    tss_output_folder = os.path.join(self.config.output_folder, "TSS_Products")
                    os.makedirs(tss_output_folder, exist_ok=True)
                    
                    try:
                        s2_result = s2_results.get('s2_processing')
                        tss_results = self.jiang_processor.process_jiang_tss(
                            c2rcc_output_path, tss_output_folder, product_name, s2_result  # <-- Add this
                        )
                        results.update(tss_results)
                        
                        if 'error' in tss_results:
                            logger.error(f"TSS processing failed: {tss_results['error'].error_message}")
                            # Don't mark as completely failed if S2 succeeded
                        else:
                            logger.info("✅ TSS processing completed successfully")
                            
                    except Exception as tss_error:
                        logger.error(f"TSS processing error: {str(tss_error)}")
                        results['tss_error'] = ProcessingResult(False, "", None, str(tss_error))
                
                elif self.config.jiang_config.enable_jiang_tss:
                    logger.warning("⚠️  Jiang TSS enabled but processor not initialized")
                else:
                    logger.info("ℹ️  Jiang TSS disabled - using SNAP TSM/CHL products only")
                
            elif self.config.processing_mode == ProcessingMode.S2_PROCESSING_ONLY:
                # S2 processing only: L1C → C2RCC (with SNAP TSM/CHL)
                results = self.s2_processor.process_single_product(product_path, self.config.output_folder)
                
            elif self.config.processing_mode == ProcessingMode.TSS_PROCESSING_ONLY:
                # TSS processing only: C2RCC → Jiang TSS
                if hasattr(self, 'jiang_processor') and self.jiang_processor is not None:
                    tss_output_folder = os.path.join(self.config.output_folder, "TSS_Products")
                    os.makedirs(tss_output_folder, exist_ok=True)
                    results = self.jiang_processor.process_jiang_tss(product_path, tss_output_folder, product_name)
                else:
                    logger.error("Jiang processor not initialized for TSS_PROCESSING_ONLY mode")
                    results = {'error': ProcessingResult(False, "", None, "Jiang processor not initialized")}
            
            processing_time = time.time() - processing_start
            
            # Check results
            if any('error' in key for key in results.keys()):
                logger.error(f"Processing failed with errors")
                self.failed_count += 1
            else:
                success_count = 0
                for key, result in results.items():
                    if isinstance(result, ProcessingResult) and hasattr(result, 'success'):
                        if result.success:
                            success_count += 1
                    elif result is not None:
                        success_count += 1
                logger.info(f"✓ Processing completed: {success_count} products generated")
                self.processed_count += 1
                
                # Log individual results
                for result_type, result in results.items():
                    if isinstance(result, ProcessingResult) and hasattr(result, 'success') and result.success and result.statistics:
                        stats = result.statistics
                        if 'coverage_percent' in stats:
                            logger.info(f"  {result_type}: {stats.get('coverage_percent', 0):.1f}% coverage, "
                                      f"mean={stats.get('mean', 0):.2f}")
                        elif 'file_size_mb' in stats:
                            logger.info(f"  {result_type}: {stats.get('file_size_mb', 0):.1f}MB, "
                                      f"status={stats.get('status', 'completed')}")
            
            # Enhanced Memory cleanup
            try:
                # Clean up large variables from this processing cycle
                MemoryManager.cleanup_variables(results)
                
                # Also clean up any other large variables that might exist
                if 'bands_data' in locals():
                    del bands_data
                if 'jiang_results' in locals():
                    del jiang_results
                if 'tss_results' in locals():
                    del tss_results
                
                # Force garbage collection
                gc.collect()
                
                # Enhanced memory monitoring
                if MemoryManager.monitor_memory():
                    logger.info("Running enhanced memory cleanup...")
                    MemoryManager.cleanup_variables()
                    gc.collect()
                    
                    # Check memory again after cleanup
                    current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                    logger.info(f"Memory usage after cleanup: {current_memory:.1f}MB")
                    
            except Exception as cleanup_error:
                logger.debug(f"Memory cleanup warning: {cleanup_error}")
            
            # Progress estimation
            if self.processed_count > 0:
                elapsed = time.time() - self.start_time
                avg_time = elapsed / current
                remaining = total - current
                eta_minutes = (avg_time * remaining) / 60
                logger.info(f"→ Progress: {current}/{total} ({(current/total)*100:.1f}%), ETA: {eta_minutes:.1f} minutes")
            
        except Exception as e:
            processing_time = time.time() - processing_start
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            import traceback
            traceback.print_exc()
            self.failed_count += 1
    
    def _extract_product_name(self, product_path: str) -> str:
        """Extract clean product name from path"""
        basename = os.path.basename(product_path)
        
        if basename.endswith('.dim'):
            product_name = basename.replace('.dim', '')
        elif basename.endswith('.zip'):
            product_name = basename.replace('.zip', '')
        elif basename.endswith('.SAFE'):
            product_name = basename.replace('.SAFE', '')
        else:
            product_name = basename
        
        # Clean up common prefixes/suffixes
        if product_name.startswith('Resampled_'):
            product_name = product_name.replace('Resampled_', '')
        if '_Subset_C2RCC' in product_name:
            product_name = product_name.replace('_Subset_C2RCC', '')
        if '_C2RCC' in product_name:
            product_name = product_name.replace('_C2RCC', '')
        
        return product_name
    
    def _check_outputs_exist(self, product_name: str) -> bool:
        """Check if outputs already exist for this product"""
        try:
            mode = self.config.processing_mode
            
            if mode == ProcessingMode.COMPLETE_PIPELINE:
                # Check for C2RCC output
                c2rcc_path = os.path.join(self.config.output_folder, "C2RCC_Products", f"Resampled_{product_name}_Subset_C2RCC.dim")
                if not os.path.exists(c2rcc_path):
                    return False
                
                # Check for TSS output if Jiang is enabled
                if self.config.jiang_config.enable_jiang_tss:
                    tss_path = os.path.join(self.config.output_folder, "TSS_Products", f"{product_name}_Jiang_TSS.tif")
                    if not os.path.exists(tss_path):
                        return False
                
                return True
                
            elif mode == ProcessingMode.S2_PROCESSING_ONLY:
                # Check for C2RCC output
                c2rcc_path = os.path.join(self.config.output_folder, "C2RCC_Products", f"Resampled_{product_name}_Subset_C2RCC.dim")
                return os.path.exists(c2rcc_path)
                
            elif mode == ProcessingMode.TSS_PROCESSING_ONLY:
                # Check for TSS output
                tss_path = os.path.join(self.config.output_folder, "TSS_Products", f"{product_name}_Jiang_TSS.tif")
                return os.path.exists(tss_path)
            
            return False
            
        except Exception as e:
            logger.debug(f"Error checking existing outputs: {e}")
            return False
    
    def _print_final_summary(self):
        """Print final processing summary"""
        total_time = (time.time() - self.start_time) / 60
        
        logger.info(f"\n{'='*80}")
        logger.info("UNIFIED S2-TSS PROCESSING SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Products processed successfully: {self.processed_count}")
        logger.info(f"Products skipped (existing): {self.skipped_count}")
        logger.info(f"Products with errors: {self.failed_count}")
        logger.info(f"Total processing time: {total_time:.2f} minutes")
        
        if self.processed_count > 0:
            avg_time = total_time / self.processed_count
            logger.info(f"Average time per product: {avg_time:.2f} minutes")
        
        # Output summary
        logger.info(f"\nOutput Structure:")
        logger.info(f"├── {self.config.output_folder}/")
        if self.config.processing_mode in [ProcessingMode.COMPLETE_PIPELINE, ProcessingMode.S2_PROCESSING_ONLY]:
            logger.info(f"    ├── Geometric_Products/")
            logger.info(f"    ├── C2RCC_Products/ (with SNAP TSM/CHL + uncertainties)")
        if self.config.processing_mode in [ProcessingMode.COMPLETE_PIPELINE, ProcessingMode.TSS_PROCESSING_ONLY]:
            if self.config.jiang_config.enable_jiang_tss:
                logger.info(f"    ├── TSS_Products/ (Jiang methodology)")
        logger.info(f"    └── Logs/")
    
    def get_processing_status(self) -> ProcessingStatus:
        """Get current processing status"""
        if self.s2_processor:
            return self.s2_processor.get_processing_status()
        else:
            total = self.processed_count + self.failed_count + self.skipped_count
            elapsed_time = time.time() - self.start_time
            
            return ProcessingStatus(
                total_products=total,
                processed=self.processed_count,
                failed=self.failed_count,
                skipped=self.skipped_count,
                current_product="",
                current_stage="",
                progress_percent=(total / max(total, 1)) * 100,
                eta_minutes=0.0,
                processing_speed=(self.processed_count / elapsed_time) * 60 if elapsed_time > 0 else 0.0
            )
    
    def cleanup(self):
        """Cleanup resources"""
        self.system_monitor.stop_monitoring()
        
        if self.s2_processor:
            self.s2_processor.cleanup()

# ===== GEOMETRY UTILITIES =====

def load_geometry_from_file(file_path: str) -> tuple:
    """
    Load geometry from various file formats and convert to WKT
    
    Returns:
        tuple: (wkt_string, info_message, success)
    """
    if not HAS_GEOPANDAS:
        return None, "GeoPandas not installed. Install with: conda install -c conda-forge geopandas", False
    
    try:
        file_extension = file_path.lower().split('.')[-1]
        
        if file_extension == 'shp':
            return load_shapefile_geometry(file_path)
        elif file_extension == 'kml':
            return load_kml_geometry(file_path)
        elif file_extension in ['geojson', 'json']:
            return load_geojson_geometry(file_path)
        else:
            return None, f"Unsupported file format: {file_extension}", False
            
    except Exception as e:
        logger.error(f"Error loading geometry from file: {e}")
        return None, f"Error loading file: {str(e)}", False

def load_shapefile_geometry(shapefile_path: str) -> tuple:
    """
    Load geometry from shapefile using Fiona (bypasses GeoPandas PROJ issues)
    """
    try:
        import os
        print(f"Loading shapefile: {shapefile_path}")
        
        # Method 1: Use Fiona (works based on your test)
        try:
            import fiona
            from shapely.geometry import shape
            
            features = []
            with fiona.open(shapefile_path) as src:
                print(f"✓ Opened with Fiona - CRS: {src.crs}, Features: {len(src)}")
                
                for feature in src:
                    features.append(feature)
            
            if not features:
                return None, "Shapefile contains no features", False
            
            # Convert to shapely geometries
            geometries = []
            for feature in features:
                try:
                    geom = shape(feature['geometry'])
                    geometries.append(geom)
                except Exception as e:
                    print(f"Warning: Skipped invalid geometry: {e}")
            
            if not geometries:
                return None, "No valid geometries found", False
            
            # Combine geometries if multiple
            if len(geometries) > 1:
                from shapely.ops import unary_union
                combined_geometry = unary_union(geometries)
                info_msg = f"Loaded {len(geometries)} features with Fiona, combined into single geometry"
            else:
                combined_geometry = geometries[0]
                info_msg = f"Loaded single feature with Fiona"
            
            print(f"✓ Geometry type: {combined_geometry.geom_type}")
            
        except ImportError:
            print("Fiona not available, trying OGR...")
            raise Exception("Fiona not available")
        
        except Exception as fiona_error:
            print(f"Fiona failed: {fiona_error}, trying OGR...")
            
            # Method 2: Use OGR (also works based on your test)
            try:
                from osgeo import ogr
                from shapely.wkt import loads
                
                driver = ogr.GetDriverByName("ESRI Shapefile")
                datasource = driver.Open(shapefile_path, 0)
                
                if datasource is None:
                    return None, "Could not open shapefile with OGR", False
                
                layer = datasource.GetLayer()
                feature_count = layer.GetFeatureCount()
                print(f"✓ Opened with OGR - Features: {feature_count}")
                
                # Get all geometries
                geometries = []
                for feature in layer:
                    geom = feature.GetGeometryRef()
                    if geom:
                        wkt = geom.ExportToWkt()
                        try:
                            shapely_geom = loads(wkt)
                            geometries.append(shapely_geom)
                        except Exception as e:
                            print(f"Warning: Could not convert geometry: {e}")
                
                if not geometries:
                    return None, "No valid geometries found with OGR", False
                
                # Combine geometries
                if len(geometries) > 1:
                    from shapely.ops import unary_union
                    combined_geometry = unary_union(geometries)
                    info_msg = f"Loaded {len(geometries)} features with OGR, combined into single geometry"
                else:
                    combined_geometry = geometries[0]
                    info_msg = f"Loaded single feature with OGR"
                
                print(f"✓ Geometry type: {combined_geometry.geom_type}")
                
            except Exception as ogr_error:
                return None, f"Both Fiona and OGR failed: {fiona_error} | {ogr_error}", False
        
        # Validate and fix geometry if needed
        if hasattr(combined_geometry, 'is_valid') and not combined_geometry.is_valid:
            print("⚠ Geometry is not valid, attempting to fix...")
            try:
                combined_geometry = combined_geometry.buffer(0)
                info_msg += " (fixed invalid geometry)"
                print("✓ Geometry fixed")
            except Exception as e:
                print(f"⚠ Could not fix geometry: {e}")
                info_msg += " (warning: geometry may be invalid)"
        
        # Convert to WKT
        try:
            wkt_string = combined_geometry.wkt
            print(f"✓ Converted to WKT ({len(wkt_string)} characters)")
        except Exception as e:
            return None, f"Failed to convert geometry to WKT: {str(e)}", False
        
        # Add bounds information
        try:
            bounds = combined_geometry.bounds
            info_msg += f"\nGeometry type: {combined_geometry.geom_type}"
            info_msg += f"\nBounds: W={bounds[0]:.6f}, S={bounds[1]:.6f}, E={bounds[2]:.6f}, N={bounds[3]:.6f}"
            info_msg += f"\nCoordinates: Assumed WGS84 (EPSG:4326)"
        except Exception as e:
            info_msg += f"\nWarning: Could not calculate bounds: {e}"
        
        print(f"✓ Successfully loaded geometry from: {os.path.basename(shapefile_path)}")
        
        return wkt_string, info_msg, True
        
    except Exception as e:
        print(f"Critical error in load_shapefile_geometry: {e}")
        import traceback
        traceback.print_exc()
        return None, f"Critical error loading shapefile: {str(e)}", False


def load_kml_geometry(kml_path: str) -> tuple:
    """
    Load geometry from KML file using Fiona/OGR (bypasses GeoPandas PROJ issues)
    """
    try:
        import os
        print(f"Loading KML file: {kml_path}")
        
        # Method 1: Use Fiona (should work like with shapefiles)
        try:
            import fiona
            from shapely.geometry import shape
            
            # Enable KML driver for fiona
            with fiona.open(kml_path, driver='KML') as src:
                print(f"✓ Opened KML with Fiona - CRS: {src.crs}, Features: {len(src)}")
                
                features = list(src)
            
            if not features:
                return None, "KML file contains no features", False
            
            # Convert to shapely geometries
            geometries = []
            for feature in features:
                try:
                    geom = shape(feature['geometry'])
                    geometries.append(geom)
                except Exception as e:
                    print(f"Warning: Skipped invalid geometry: {e}")
            
            if not geometries:
                return None, "No valid geometries found", False
            
            # Combine geometries if multiple
            if len(geometries) > 1:
                from shapely.ops import unary_union
                combined_geometry = unary_union(geometries)
                info_msg = f"Loaded {len(geometries)} features from KML with Fiona, combined into single geometry"
            else:
                combined_geometry = geometries[0]
                info_msg = f"Loaded single feature from KML with Fiona"
            
            print(f"✓ Geometry type: {combined_geometry.geom_type}")
            
        except Exception as fiona_error:
            print(f"Fiona failed: {fiona_error}, trying OGR...")
            
            # Method 2: Use OGR (KML is well supported by OGR)
            try:
                from osgeo import ogr
                from shapely.wkt import loads
                
                # OGR can handle KML directly
                driver = ogr.GetDriverByName("KML")
                if driver is None:
                    # Try libkml driver if available
                    driver = ogr.GetDriverByName("LIBKML")
                
                if driver is None:
                    return None, "KML driver not available in OGR", False
                
                datasource = driver.Open(kml_path, 0)
                if datasource is None:
                    return None, "Could not open KML file with OGR", False
                
                # KML files can have multiple layers, get all features from all layers
                geometries = []
                layer_count = datasource.GetLayerCount()
                print(f"✓ Opened KML with OGR - Layers: {layer_count}")
                
                total_features = 0
                for layer_idx in range(layer_count):
                    layer = datasource.GetLayer(layer_idx)
                    layer_feature_count = layer.GetFeatureCount()
                    total_features += layer_feature_count
                    print(f"  Layer {layer_idx}: {layer_feature_count} features")
                    
                    # Get geometries from this layer
                    for feature in layer:
                        geom = feature.GetGeometryRef()
                        if geom:
                            wkt = geom.ExportToWkt()
                            try:
                                shapely_geom = loads(wkt)
                                geometries.append(shapely_geom)
                            except Exception as e:
                                print(f"Warning: Could not convert geometry: {e}")
                
                if not geometries:
                    return None, f"No valid geometries found in KML (total features: {total_features})", False
                
                # Combine geometries
                if len(geometries) > 1:
                    from shapely.ops import unary_union
                    combined_geometry = unary_union(geometries)
                    info_msg = f"Loaded {len(geometries)} features from KML with OGR, combined into single geometry"
                else:
                    combined_geometry = geometries[0]
                    info_msg = f"Loaded single feature from KML with OGR"
                
                print(f"✓ Geometry type: {combined_geometry.geom_type}")
                
            except Exception as ogr_error:
                return None, f"Both Fiona and OGR failed for KML: {fiona_error} | {ogr_error}", False
        
        # Validate and fix geometry if needed
        if hasattr(combined_geometry, 'is_valid') and not combined_geometry.is_valid:
            print("⚠ Geometry is not valid, attempting to fix...")
            try:
                combined_geometry = combined_geometry.buffer(0)
                info_msg += " (fixed invalid geometry)"
                print("✓ Geometry fixed")
            except Exception as e:
                print(f"⚠ Could not fix geometry: {e}")
                info_msg += " (warning: geometry may be invalid)"
        
        # Convert to WKT
        try:
            wkt_string = combined_geometry.wkt
            print(f"✓ Converted to WKT ({len(wkt_string)} characters)")
        except Exception as e:
            return None, f"Failed to convert geometry to WKT: {str(e)}", False
        
        # Add bounds information
        try:
            bounds = combined_geometry.bounds
            info_msg += f"\nGeometry type: {combined_geometry.geom_type}"
            info_msg += f"\nBounds: W={bounds[0]:.6f}, S={bounds[1]:.6f}, E={bounds[2]:.6f}, N={bounds[3]:.6f}"
            info_msg += f"\nCoordinates: KML is always in WGS84 (EPSG:4326)"
        except Exception as e:
            info_msg += f"\nWarning: Could not calculate bounds: {e}"
        
        print(f"✓ Successfully loaded geometry from KML: {os.path.basename(kml_path)}")
        
        return wkt_string, info_msg, True
        
    except Exception as e:
        print(f"Critical error in load_kml_geometry: {e}")
        import traceback
        traceback.print_exc()
        return None, f"Critical error loading KML: {str(e)}", False


def load_geojson_geometry(geojson_path: str) -> tuple:
    """
    Load geometry from GeoJSON file using Fiona/OGR (bypasses GeoPandas PROJ issues)
    """
    try:
        import os
        print(f"Loading GeoJSON file: {geojson_path}")
        
        # Method 1: Use Fiona 
        try:
            import fiona
            from shapely.geometry import shape
            
            with fiona.open(geojson_path, driver='GeoJSON') as src:
                print(f"✓ Opened GeoJSON with Fiona - CRS: {src.crs}, Features: {len(src)}")
                
                features = list(src)
            
            if not features:
                return None, "GeoJSON contains no features", False
            
            # Convert to shapely geometries
            geometries = []
            for feature in features:
                try:
                    geom = shape(feature['geometry'])
                    geometries.append(geom)
                except Exception as e:
                    print(f"Warning: Skipped invalid geometry: {e}")
            
            if not geometries:
                return None, "No valid geometries found", False
            
            # Combine geometries if multiple
            if len(geometries) > 1:
                from shapely.ops import unary_union
                combined_geometry = unary_union(geometries)
                info_msg = f"Loaded {len(geometries)} features from GeoJSON with Fiona, combined into single geometry"
            else:
                combined_geometry = geometries[0]
                info_msg = f"Loaded single feature from GeoJSON with Fiona"
            
            print(f"✓ Geometry type: {combined_geometry.geom_type}")
            
        except Exception as fiona_error:
            print(f"Fiona failed: {fiona_error}, trying OGR...")
            
            # Method 2: Use OGR
            try:
                from osgeo import ogr
                from shapely.wkt import loads
                
                driver = ogr.GetDriverByName("GeoJSON")
                if driver is None:
                    return None, "GeoJSON driver not available in OGR", False
                
                datasource = driver.Open(geojson_path, 0)
                if datasource is None:
                    return None, "Could not open GeoJSON file with OGR", False
                
                layer = datasource.GetLayer()
                feature_count = layer.GetFeatureCount()
                print(f"✓ Opened GeoJSON with OGR - Features: {feature_count}")
                
                # Get all geometries
                geometries = []
                for feature in layer:
                    geom = feature.GetGeometryRef()
                    if geom:
                        wkt = geom.ExportToWkt()
                        try:
                            shapely_geom = loads(wkt)
                            geometries.append(shapely_geom)
                        except Exception as e:
                            print(f"Warning: Could not convert geometry: {e}")
                
                if not geometries:
                    return None, "No valid geometries found with OGR", False
                
                # Combine geometries
                if len(geometries) > 1:
                    from shapely.ops import unary_union
                    combined_geometry = unary_union(geometries)
                    info_msg = f"Loaded {len(geometries)} features from GeoJSON with OGR, combined into single geometry"
                else:
                    combined_geometry = geometries[0]
                    info_msg = f"Loaded single feature from GeoJSON with OGR"
                
                print(f"✓ Geometry type: {combined_geometry.geom_type}")
                
            except Exception as ogr_error:
                return None, f"Both Fiona and OGR failed for GeoJSON: {fiona_error} | {ogr_error}", False
        
        # Handle CRS conversion (GeoJSON might not be WGS84)
        # Note: Since we're bypassing GeoPandas, we assume coordinates are already in WGS84
        # If they're not, this would need external conversion
        
        # Validate and fix geometry if needed
        if hasattr(combined_geometry, 'is_valid') and not combined_geometry.is_valid:
            print("⚠ Geometry is not valid, attempting to fix...")
            try:
                combined_geometry = combined_geometry.buffer(0)
                info_msg += " (fixed invalid geometry)"
                print("✓ Geometry fixed")
            except Exception as e:
                print(f"⚠ Could not fix geometry: {e}")
                info_msg += " (warning: geometry may be invalid)"
        
        # Convert to WKT
        try:
            wkt_string = combined_geometry.wkt
            print(f"✓ Converted to WKT ({len(wkt_string)} characters)")
        except Exception as e:
            return None, f"Failed to convert geometry to WKT: {str(e)}", False
        
        # Add bounds information
        try:
            bounds = combined_geometry.bounds
            info_msg += f"\nGeometry type: {combined_geometry.geom_type}"
            info_msg += f"\nBounds: W={bounds[0]:.6f}, S={bounds[1]:.6f}, E={bounds[2]:.6f}, N={bounds[3]:.6f}"
            info_msg += f"\nCoordinates: Assumed WGS84 (GeoJSON default)"
        except Exception as e:
            info_msg += f"\nWarning: Could not calculate bounds: {e}"
        
        print(f"✓ Successfully loaded geometry from GeoJSON: {os.path.basename(geojson_path)}")
        
        return wkt_string, info_msg, True
        
    except Exception as e:
        print(f"Critical error in load_geojson_geometry: {e}")
        import traceback
        traceback.print_exc()
        return None, f"Critical error loading GeoJSON: {str(e)}", False

def validate_wkt_geometry(wkt_string: str) -> tuple:
    """Validate WKT geometry string"""
    try:
        if not HAS_GEOPANDAS:
            return True, "GeoPandas not available for validation"
        
        test_geom = wkt_loads(wkt_string)
        
        if not test_geom.is_valid:
            return False, "WKT geometry is not valid"
        
        bounds = test_geom.bounds
        info_msg = f"Valid {test_geom.geom_type} geometry"
        info_msg += f"\nBounds: W={bounds[0]:.6f}, S={bounds[1]:.6f}, E={bounds[2]:.6f}, N={bounds[3]:.6f}"
        
        return True, info_msg
        
    except Exception as e:
        return False, f"Invalid WKT string: {str(e)}"

def get_geometry_input():
    """Enhanced geometry input dialog with full file format support"""
    
    # Create a simple dialog window
    dialog = tk.Toplevel()
    dialog.title("Geometry Input - Enhanced")
    dialog.geometry("700x500")
    dialog.transient()
    dialog.focus_set()
    
    result = {"geometry": None}
    
    # Main frame
    main_frame = ttk.Frame(dialog, padding="10")
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # Title
    title_label = ttk.Label(main_frame, text="Select Geometry Input Method", 
                           font=("Arial", 12, "bold"))
    title_label.pack(pady=(0, 10))
    
    # Subtitle with file format support
    if HAS_GEOPANDAS:
        subtitle_text = "✓ Full file format support: Shapefile, KML, GeoJSON"
        subtitle_color = "darkgreen"
    else:
        subtitle_text = "⚠ Limited support - install GeoPandas for full functionality"
        subtitle_color = "darkorange"
    
    subtitle_label = ttk.Label(main_frame, text=subtitle_text, 
                              font=("Arial", 9), foreground=subtitle_color)
    subtitle_label.pack(pady=(0, 20))
    
    # Method selection
    method_var = tk.StringVar(value="none")
    
    ttk.Radiobutton(main_frame, text="No spatial subset (process full scene)", 
                   variable=method_var, value="none").pack(anchor=tk.W, pady=5)
    ttk.Radiobutton(main_frame, text="Enter coordinates (North/South/East/West)", 
                   variable=method_var, value="coords").pack(anchor=tk.W, pady=5)
    ttk.Radiobutton(main_frame, text="Enter WKT string", 
                   variable=method_var, value="wkt").pack(anchor=tk.W, pady=5)
    ttk.Radiobutton(main_frame, text="Load from file (Shapefile/KML/GeoJSON)", 
                   variable=method_var, value="file").pack(anchor=tk.W, pady=5)
    
    # Input areas
    input_frame = ttk.LabelFrame(main_frame, text="Input Area", padding="10")
    input_frame.pack(fill=tk.BOTH, expand=True, pady=(20, 0))
    
    # Coordinates input
    coords_frame = ttk.Frame(input_frame)
    
    ttk.Label(coords_frame, text="Coordinate Bounds (WGS84):").pack(anchor=tk.W)
    coord_grid = ttk.Frame(coords_frame)
    coord_grid.pack(fill=tk.X, pady=5)
    
    ttk.Label(coord_grid, text="North:").grid(row=0, column=0, sticky=tk.W, padx=(0,5))
    north_var = tk.StringVar(value="41.35")
    ttk.Entry(coord_grid, textvariable=north_var, width=12).grid(row=0, column=1, padx=5)
    
    ttk.Label(coord_grid, text="South:").grid(row=0, column=2, sticky=tk.W, padx=(10,5))
    south_var = tk.StringVar(value="40.83")
    ttk.Entry(coord_grid, textvariable=south_var, width=12).grid(row=0, column=3, padx=5)
    
    ttk.Label(coord_grid, text="West:").grid(row=1, column=0, sticky=tk.W, padx=(0,5), pady=(5,0))
    west_var = tk.StringVar(value="-9.01")
    ttk.Entry(coord_grid, textvariable=west_var, width=12).grid(row=1, column=1, padx=5, pady=(5,0))
    
    ttk.Label(coord_grid, text="East:").grid(row=1, column=2, sticky=tk.W, padx=(10,5), pady=(5,0))
    east_var = tk.StringVar(value="-7.69")
    ttk.Entry(coord_grid, textvariable=east_var, width=12).grid(row=1, column=3, padx=5, pady=(5,0))
    
    # WKT input
    wkt_frame = ttk.Frame(input_frame)
    ttk.Label(wkt_frame, text="WKT String:").pack(anchor=tk.W)
    wkt_text = tk.Text(wkt_frame, height=4, wrap=tk.WORD, font=("Consolas", 9))
    wkt_text.pack(fill=tk.BOTH, expand=True, pady=5)
    
    # Add WKT validation button
    wkt_button_frame = ttk.Frame(wkt_frame)
    wkt_button_frame.pack(fill=tk.X, pady=(5, 0))
    
    def validate_wkt():
        wkt_string = wkt_text.get(1.0, tk.END).strip()
        if not wkt_string:
            messagebox.showwarning("Warning", "No WKT string to validate", parent=dialog)
            return
        
        is_valid, message = validate_wkt_geometry(wkt_string)
        if is_valid:
            messagebox.showinfo("Validation", f"✓ Valid WKT geometry!\n\n{message}", parent=dialog)
        else:
            messagebox.showerror("Validation", f"✗ Invalid WKT geometry!\n\n{message}", parent=dialog)
    
    ttk.Button(wkt_button_frame, text="Validate WKT", command=validate_wkt).pack(side=tk.LEFT)
    
    # File input
    file_frame = ttk.Frame(input_frame)
    ttk.Label(file_frame, text="Select File:").pack(anchor=tk.W)
    file_path_var = tk.StringVar()
    file_path_frame = ttk.Frame(file_frame)
    file_path_frame.pack(fill=tk.X, pady=5)
    ttk.Entry(file_path_frame, textvariable=file_path_var, state="readonly").pack(side=tk.LEFT, fill=tk.X, expand=True)
    
    def browse_file():
        if HAS_GEOPANDAS:
            filetypes = [
                ("Shapefiles", "*.shp"),
                ("KML files", "*.kml"), 
                ("GeoJSON files", "*.geojson;*.json"),
                ("All supported", "*.shp;*.kml;*.geojson;*.json"),
                ("All files", "*.*")
            ]
        else:
            filetypes = [
                ("Shapefiles", "*.shp"),
                ("All files", "*.*")
            ]
        
        filename = filedialog.askopenfilename(
            title="Select Geometry File",
            filetypes=filetypes,
            parent=dialog
        )
        if filename:
            file_path_var.set(filename)
    
    ttk.Button(file_path_frame, text="Browse...", command=browse_file).pack(side=tk.RIGHT, padx=(5,0))
    
    # File info display
    file_info_frame = ttk.Frame(file_frame)
    file_info_frame.pack(fill=tk.X, pady=(5, 0))
    
    file_info_text = tk.Text(file_info_frame, height=3, wrap=tk.WORD, font=("Arial", 8))
    file_info_text.pack(fill=tk.X)
    file_info_text.config(state=tk.DISABLED)
    
    def update_visibility(*args):
        # Hide all frames
        coords_frame.pack_forget()
        wkt_frame.pack_forget()
        file_frame.pack_forget()
        
        # Show relevant frame
        method = method_var.get()
        if method == "coords":
            coords_frame.pack(fill=tk.X, pady=5)
        elif method == "wkt":
            wkt_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        elif method == "file":
            file_frame.pack(fill=tk.BOTH, expand=True, pady=5)
    
    method_var.trace("w", update_visibility)
    update_visibility()  # Initial call
    
    # Buttons
    button_frame = ttk.Frame(main_frame)
    button_frame.pack(fill=tk.X, pady=(20, 0))
    
    def on_ok():
        method = method_var.get()
        
        if method == "none":
            result["geometry"] = None
            
        elif method == "coords":
            try:
                north = float(north_var.get())
                south = float(south_var.get())
                east = float(east_var.get())
                west = float(west_var.get())
                
                # Validate coordinate ranges
                if not (-90 <= south <= north <= 90):
                    messagebox.showerror("Error", "Invalid latitude values!\nMust be between -90 and 90 degrees.", parent=dialog)
                    return
                
                if not (-180 <= west <= east <= 180):
                    messagebox.showerror("Error", "Invalid longitude values!\nMust be between -180 and 180 degrees.", parent=dialog)
                    return
                
                # Create WKT polygon from coordinates
                wkt = f"POLYGON (({west} {south}, {east} {south}, {east} {north}, {west} {north}, {west} {south}))"
                result["geometry"] = wkt
                
            except ValueError:
                messagebox.showerror("Error", "Invalid coordinates! Please enter numeric values.", parent=dialog)
                return
        
        elif method == "wkt":
            wkt = wkt_text.get(1.0, tk.END).strip()
            if not wkt:
                messagebox.showerror("Error", "Please enter a WKT string!", parent=dialog)
                return
            
            # Validate WKT
            is_valid, message = validate_wkt_geometry(wkt)
            if not is_valid:
                response = messagebox.askyesno("Invalid WKT", 
                                             f"WKT validation failed:\n{message}\n\nUse anyway?", 
                                             parent=dialog)
                if not response:
                    return
            
            result["geometry"] = wkt
        
        elif method == "file":
            file_path = file_path_var.get()
            if not file_path:
                messagebox.showerror("Error", "Please select a file!", parent=dialog)
                return
            
            # Load geometry from the selected file
            try:
                wkt_string, info_message, success = load_geometry_from_file(file_path)
                
                if success and wkt_string:
                    result["geometry"] = wkt_string
                    
                    # Show detailed information about loaded geometry
                    messagebox.showinfo("Success", 
                                      f"Successfully loaded geometry from:\n{os.path.basename(file_path)}\n\n{info_message}", 
                                      parent=dialog)
                else:
                    messagebox.showerror("Error", 
                                       f"Failed to load geometry from:\n{os.path.basename(file_path)}\n\n{info_message}", 
                                       parent=dialog)
                    return
                    
            except Exception as e:
                messagebox.showerror("Error", 
                                   f"Unexpected error loading file:\n{str(e)}", 
                                   parent=dialog)
                return
        
        dialog.destroy()
    
    def on_cancel():
        result["geometry"] = None
        dialog.destroy()
    
    ttk.Button(button_frame, text="OK", command=on_ok).pack(side=tk.LEFT, padx=(0, 5))
    ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side=tk.LEFT)
    
    # Wait for dialog to close
    dialog.wait_window()
    
    return result["geometry"]

def get_area_name(wkt_string: str) -> str:
    """Generates a descriptive area name from a WKT string."""
    if wkt_string:
        try:
            if HAS_GEOPANDAS:
                geom = wkt_loads(wkt_string)
                bounds = geom.bounds
                return f"CustomArea_{bounds[0]:.3f}_{bounds[1]:.3f}_{bounds[2]:.3f}_{bounds[3]:.3f}"
            else:
                return "CustomArea"
        except:
            return "CustomArea"
    return "FullScene"

def bring_window_to_front(window):
    """Enhanced window focus management"""
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
            except:
                pass
                
    except Exception as e:
        logger.warning(f"Could not bring window to front: {e}")

# ===== GUI CLASS FOUNDATION =====

class UnifiedS2TSSGUI:
    """
    Unified GUI for Complete S2 Processing and TSS Estimation Pipeline
    ================================================================
    
    Professional interface that combines S2 pre-processing with TSS estimation,
    featuring automatic SNAP TSM/CHL generation and optional Jiang methodology.
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Unified S2 Processing & TSS Estimation Pipeline v1.0")
        
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
        
        # Configuration objects
        self.resampling_config = ResamplingConfig()
        self.subset_config = SubsetConfig()
        self.c2rcc_config = C2RCCConfig()
        self.jiang_config = JiangTSSConfig()

        # ===== COMPLETE MARINE VISUALIZATION INITIALIZATION =====
        # Ensure marine visualization is fully operational and enabled

        # Step 1: Add missing attributes if they don't exist
        if not hasattr(self.jiang_config, 'enable_marine_visualization'):
            self.jiang_config.enable_marine_visualization = True  # ENABLE by default
            logger.info("Added enable_marine_visualization attribute (ENABLED)")

        if not hasattr(self.jiang_config, 'marine_viz_config'):
            self.jiang_config.marine_viz_config = None
            logger.info("Added marine_viz_config attribute")

        # Step 2: Create MarineVisualizationConfig with full functionality
        if self.jiang_config.enable_marine_visualization and self.jiang_config.marine_viz_config is None:
            try:
                # Import the class if it exists
                self.jiang_config.marine_viz_config = MarineVisualizationConfig()
                logger.info("✓ Marine visualization fully initialized and OPERATIONAL")
            except NameError:
                # Create a complete MarineVisualizationConfig class if it doesn't exist
                logger.info("Creating MarineVisualizationConfig class for full functionality...")
                
                @dataclass
                class MarineVisualizationConfig:
                    """Configuration for marine visualization products - FULL OPERATIONAL VERSION"""
                    
                    # RGB options - ALL ENABLED for maximum output
                    generate_natural_color: bool = True
                    generate_false_color: bool = True
                    generate_water_specific: bool = True
                    generate_research_combinations: bool = True  # Enable research combos too
                    
                    # Spectral indices options - ALL ENABLED
                    generate_water_quality_indices: bool = True
                    generate_chlorophyll_indices: bool = True
                    generate_turbidity_indices: bool = True
                    generate_advanced_indices: bool = True
                    
                    # Output format options
                    rgb_format: str = 'GeoTIFF'
                    export_metadata: bool = True
                    create_overview_images: bool = True
                    
                    # Enhancement options for better visualization
                    apply_contrast_enhancement: bool = True
                    contrast_method: str = 'percentile_stretch'
                    percentile_range: tuple = (2, 98)
                
                # Create the config with all features enabled
                self.jiang_config.marine_viz_config = MarineVisualizationConfig()
                logger.info("✓ Created complete MarineVisualizationConfig - ALL FEATURES ENABLED")

        # Step 3: Ensure advanced algorithms are also operational
        if not hasattr(self.jiang_config, 'enable_advanced_algorithms'):
            self.jiang_config.enable_advanced_algorithms = True
            logger.info("Added enable_advanced_algorithms attribute (ENABLED)")

        if not hasattr(self.jiang_config, 'advanced_config'):
            self.jiang_config.advanced_config = None
            logger.info("Added advanced_config attribute")

        if self.jiang_config.enable_advanced_algorithms and self.jiang_config.advanced_config is None:
            try:
                self.jiang_config.advanced_config = AdvancedAquaticConfig()
                logger.info("✓ Advanced algorithms initialized and OPERATIONAL")
            except NameError:
                # Create AdvancedAquaticConfig if it doesn't exist
                @dataclass
                class AdvancedAquaticConfig:
                    """Configuration for advanced aquatic algorithms - OPERATIONAL VERSION"""
                    
                    # Working algorithms
                    enable_water_clarity: bool = True
                    solar_zenith_angle: float = 30.0
                    
                    enable_hab_detection: bool = True
                    hab_biomass_threshold: float = 20.0
                    hab_extreme_threshold: float = 100.0
                    
                    # Output options
                    save_intermediate_products: bool = True
                    create_classification_maps: bool = True
                    generate_statistics: bool = True
                
                self.jiang_config.advanced_config = AdvancedAquaticConfig()
                logger.info("✓ Created complete AdvancedAquaticConfig - ALL FEATURES ENABLED")

        # Step 4: Log the complete configuration status
        logger.info("=" * 60)
        logger.info("MARINE VISUALIZATION SYSTEM STATUS:")
        logger.info(f"✓ Marine visualization enabled: {self.jiang_config.enable_marine_visualization}")
        logger.info(f"✓ Marine config created: {self.jiang_config.marine_viz_config is not None}")

        if self.jiang_config.marine_viz_config:
            logger.info("ENABLED MARINE VISUALIZATION FEATURES:")
            logger.info(f"  ✓ Natural color RGB: {self.jiang_config.marine_viz_config.generate_natural_color}")
            logger.info(f"  ✓ False color RGB: {self.jiang_config.marine_viz_config.generate_false_color}")
            logger.info(f"  ✓ Water-specific RGB: {self.jiang_config.marine_viz_config.generate_water_specific}")
            logger.info(f"  ✓ Research RGB combinations: {self.jiang_config.marine_viz_config.generate_research_combinations}")
            logger.info(f"  ✓ Water quality indices: {self.jiang_config.marine_viz_config.generate_water_quality_indices}")
            logger.info(f"  ✓ Chlorophyll indices: {self.jiang_config.marine_viz_config.generate_chlorophyll_indices}")
            logger.info(f"  ✓ Turbidity indices: {self.jiang_config.marine_viz_config.generate_turbidity_indices}")
            logger.info(f"  ✓ Advanced indices: {self.jiang_config.marine_viz_config.generate_advanced_indices}")
            logger.info(f"  ✓ Contrast enhancement: {self.jiang_config.marine_viz_config.apply_contrast_enhancement}")

        logger.info(f"✓ Advanced algorithms enabled: {self.jiang_config.enable_advanced_algorithms}")
        logger.info(f"✓ Advanced config created: {self.jiang_config.advanced_config is not None}")
        logger.info("=" * 60)
        logger.info("🎨 MARINE VISUALIZATION SYSTEM READY FOR PRODUCTION! 🎨")
        logger.info("Expected outputs per product:")
        logger.info("  • Multiple RGB composites (natural, false color, water-specific)")
        logger.info("  • Water quality spectral indices (NDWI, NDCI, NDTI, etc.)")
        logger.info("  • Chlorophyll indices (CIG, etc.)")
        logger.info("  • Turbidity indices (TSI, etc.)")
        logger.info("  • Advanced marine indices (FUI, SDD, CDOM, etc.)")
        logger.info("  • All outputs in GeoTIFF format with metadata")
        logger.info("=" * 60)
        
        # GUI state variables
        self.processing_mode = tk.StringVar(value="complete_pipeline")
        self.input_validation_result = {"valid": False, "message": "", "products": []}
        self.system_monitor = SystemMonitor()
        self.system_monitor.start_monitoring()
        
        # Progress tracking
        self.progress_var = tk.DoubleVar()
        self.status_var = tk.StringVar(value="Ready")
        self.eta_var = tk.StringVar(value="")
        
        # Processing state
        self.processor = None
        self.processing_thread = None
        self.processing_active = False
        
        # Tab management
        self.tab_indices = {}
        
        # ===== INPUT/OUTPUT VARIABLES =====
        self.input_dir_var = tk.StringVar()
        self.output_dir_var = tk.StringVar()
        
        # ===== PROCESSING OPTIONS =====
        self.skip_existing_var = tk.BooleanVar(value=True)
        self.test_mode_var = tk.BooleanVar(value=False)
        self.memory_limit_var = tk.StringVar(value="8")
        self.thread_count_var = tk.StringVar(value="4")
        
        # ===== RESAMPLING CONFIGURATION =====
        self.resolution_var = tk.StringVar(value="10")
        self.upsampling_var = tk.StringVar(value="Bilinear")
        self.downsampling_var = tk.StringVar(value="Mean")
        self.flag_downsampling_var = tk.StringVar(value="First")
        self.pyramid_var = tk.BooleanVar(value=True)
        
        # ===== SUBSET CONFIGURATION =====
        self.subset_method_var = tk.StringVar(value="none")
        self.pixel_start_x_var = tk.StringVar()
        self.pixel_start_y_var = tk.StringVar()
        self.pixel_width_var = tk.StringVar()
        self.pixel_height_var = tk.StringVar()
        
        # ===== C2RCC CONFIGURATION =====
        # Neural network and DEM
        self.net_set_var = tk.StringVar(value="C2RCC-Nets")
        self.dem_name_var = tk.StringVar(value="Copernicus 90m Global DEM")
        self.elevation_var = tk.DoubleVar(value=0.0)
        
        # Water and atmospheric parameters (with SNAP defaults)
        self.salinity_var = tk.DoubleVar(value=35.0)
        self.temperature_var = tk.DoubleVar(value=15.0)
        self.ozone_var = tk.DoubleVar(value=330.0)
        self.pressure_var = tk.DoubleVar(value=1000.0)  # SNAP default
        self.use_ecmwf_var = tk.BooleanVar(value=True)  # ENABLED BY DEFAULT as requested
        
        # Essential output products (SNAP defaults + uncertainties enabled)
        self.output_rrs_var = tk.BooleanVar(value=True)
        self.output_rhow_var = tk.BooleanVar(value=True)
        self.output_kd_var = tk.BooleanVar(value=True)
        self.output_uncertainties_var = tk.BooleanVar(value=True)  # Ensures unc_tsm.img and unc_chl.img
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
        
        # ===== JIANG TSS CONFIGURATION =====
        self.enable_jiang_var = tk.BooleanVar(value=True)  # Optional by default
        self.jiang_intermediates_var = tk.BooleanVar(value=True)
        self.jiang_comparison_var = tk.BooleanVar(value=True)
        self.enable_advanced_var = tk.BooleanVar(value=True)
        self.trophic_state_var = tk.BooleanVar(value=True)
        self.water_clarity_var = tk.BooleanVar(value=True)
        self.hab_detection_var = tk.BooleanVar(value=True)
        self.upwelling_detection_var = tk.BooleanVar(value=True)
        self.river_plumes_var = tk.BooleanVar(value=True)
        self.particle_size_var = tk.BooleanVar(value=True)
        self.primary_productivity_var = tk.BooleanVar(value=True)
    
        # NEW: Marine visualization variables
        self.enable_marine_viz_var = tk.BooleanVar(value=True)  # ENABLED BY DEFAULT
        self.natural_color_var = tk.BooleanVar(value=True)
        self.false_color_var = tk.BooleanVar(value=True)
        self.water_specific_var = tk.BooleanVar(value=True)
        self.research_rgb_var = tk.BooleanVar(value=False)
        self.water_quality_indices_var = tk.BooleanVar(value=True)
        self.chlorophyll_indices_var = tk.BooleanVar(value=True)
        self.turbidity_indices_var = tk.BooleanVar(value=True)
        self.advanced_indices_var = tk.BooleanVar(value=False)
        
        # Setup GUI components
        self.setup_gui()
        self.start_gui_updates()
      
    def setup_gui(self):
        """Setup the enhanced GUI interface"""
        try:
            # Create main container with padding
            main_container = ttk.Frame(self.root, padding="10")
            main_container.pack(fill=tk.BOTH, expand=True)
            
            # Title section
            self.setup_title_section(main_container)
            
            # Create notebook for tabbed interface
            self.notebook = ttk.Notebook(main_container)
            self.notebook.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
            
            # Tab 1: Processing Mode & Setup
            self.setup_processing_mode_tab()
            
            # Tab 2: Resampling Configuration
            self.setup_resampling_tab()
            
            # Tab 3: Subset Configuration
            self.setup_subset_tab()
            
            # Tab 4: C2RCC Configuration
            self.setup_c2rcc_tab()
            
            # Tab 5: TSS Configuration (Jiang)
            self.setup_tss_tab()
            
            # Tab 6: System Status & Monitoring
            self.setup_monitoring_tab()
            
            # Status bar and controls
            self.setup_status_bar(main_container)
            self.setup_control_buttons(main_container)
            
            # Update tab visibility based on initial mode
            self.update_tab_visibility()
            
        except Exception as e:
            logger.error(f"GUI setup error: {e}")
            messagebox.showerror("GUI Error", f"Failed to setup GUI: {str(e)}")
    
    def setup_title_section(self, parent):
        """Setup title and system info section"""
        title_frame = ttk.Frame(parent)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Main title
        title_label = ttk.Label(title_frame, text="Unified S2 Processing & TSS Estimation Pipeline", 
                               font=("Arial", 16, "bold"))
        title_label.pack()
        
        # Subtitle
        subtitle_label = ttk.Label(title_frame, text="Complete pipeline: L1C → C2RCC (automatic SNAP TSM/CHL) → Optional Jiang TSS", 
                                  font=("Arial", 10), foreground="gray")
        subtitle_label.pack()
    
    def setup_processing_mode_tab(self):
        """Setup processing mode selection and I/O configuration"""
        frame = ttk.Frame(self.notebook)
        tab_index = self.notebook.add(frame, text="🎯 Processing Mode")
        self.tab_indices['processing'] = tab_index
        
        # Processing mode selection
        mode_frame = ttk.LabelFrame(frame, text="Processing Mode Selection", padding="10")
        mode_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Mode descriptions
        mode_descriptions = {
            "complete_pipeline": "Complete Pipeline: L1C → S2 Processing → C2RCC → Optional Jiang TSS\n• Input: Raw Sentinel-2 L1C products (.zip/.SAFE)\n• Output: C2RCC with SNAP TSM/CHL + optional Jiang TSS",
            "s2_processing_only": "S2 Processing Only: L1C → S2 Processing → C2RCC\n• Input: Raw Sentinel-2 L1C products (.zip/.SAFE)\n• Output: C2RCC with automatic SNAP TSM/CHL generation",
            "tss_processing_only": "TSS Processing Only: C2RCC → Jiang TSS\n• Input: C2RCC products (.dim files)\n• Output: Jiang TSS products only"
        }
        
        for mode, description in mode_descriptions.items():
            radio_frame = ttk.Frame(mode_frame)
            radio_frame.pack(fill=tk.X, pady=2)
            
            ttk.Radiobutton(radio_frame, text="", variable=self.processing_mode, 
                           value=mode, command=self.on_mode_change).pack(side=tk.LEFT)
            
            desc_label = ttk.Label(radio_frame, text=description, font=("Arial", 9), 
                                  wraplength=700, justify=tk.LEFT)
            desc_label.pack(side=tk.LEFT, padx=(5, 0))
        
        # Input/Output configuration
        io_frame = ttk.LabelFrame(frame, text="Input/Output Configuration", padding="10")
        io_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Input directory
        input_frame = ttk.Frame(io_frame)
        input_frame.pack(fill=tk.X, pady=5)
        ttk.Label(input_frame, text="Input Directory:").pack(anchor=tk.W)
        
        input_path_frame = ttk.Frame(input_frame)
        input_path_frame.pack(fill=tk.X, pady=2)
        self.input_entry = ttk.Entry(input_path_frame, textvariable=self.input_dir_var)
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(input_path_frame, text="Browse...", 
                  command=self.browse_input_dir).pack(side=tk.RIGHT, padx=(5,0))
        
        # Input validation display
        self.input_validation_frame = ttk.Frame(input_frame)
        self.input_validation_frame.pack(fill=tk.X, pady=2)
        self.input_validation_label = ttk.Label(self.input_validation_frame, text="", 
                                               foreground="gray", font=("Arial", 9))
        self.input_validation_label.pack(anchor=tk.W)
        
        # Output directory
        output_frame = ttk.Frame(io_frame)
        output_frame.pack(fill=tk.X, pady=5)
        ttk.Label(output_frame, text="Output Directory:").pack(anchor=tk.W)
        
        output_path_frame = ttk.Frame(output_frame)
        output_path_frame.pack(fill=tk.X, pady=2)
        ttk.Entry(output_path_frame, textvariable=self.output_dir_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(output_path_frame, text="Browse...", 
                  command=self.browse_output_dir).pack(side=tk.RIGHT, padx=(5,0))
        
        # Processing options
        options_frame = ttk.LabelFrame(frame, text="Processing Options", padding="10")
        options_frame.pack(fill=tk.X, padx=10, pady=10)
        
        options_grid = ttk.Frame(options_frame)
        options_grid.pack(fill=tk.X)
        
        # Left column
        left_options = ttk.Frame(options_grid)
        left_options.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Checkbutton(left_options, text="Skip existing output files", 
                       variable=self.skip_existing_var).pack(anchor=tk.W, pady=2)
        
        ttk.Checkbutton(left_options, text="Test mode (process only 1 file)", 
                       variable=self.test_mode_var).pack(anchor=tk.W, pady=2)
        
        # Right column - Memory and performance
        right_options = ttk.Frame(options_grid)
        right_options.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        perf_frame = ttk.Frame(right_options)
        perf_frame.pack(anchor=tk.W)
        
        ttk.Label(perf_frame, text="Memory Limit (GB):").pack(side=tk.LEFT)
        memory_spinbox = ttk.Spinbox(perf_frame, from_=4, to=32, width=5, 
                                    textvariable=self.memory_limit_var)
        memory_spinbox.pack(side=tk.LEFT, padx=(5, 20))
        
        ttk.Label(perf_frame, text="Thread Count:").pack(side=tk.LEFT)
        thread_spinbox = ttk.Spinbox(perf_frame, from_=1, to=16, width=5, 
                                    textvariable=self.thread_count_var)
        thread_spinbox.pack(side=tk.LEFT, padx=(5, 0))
        
        # Bind input directory change to validation
        self.input_dir_var.trace("w", self.validate_input_directory)
    
    def setup_resampling_tab(self):
        """Setup S2 Resampling configuration tab"""
        frame = ttk.Frame(self.notebook)
        tab_index = self.notebook.add(frame, text="📐 Resampling")
        self.tab_indices['resampling'] = tab_index
        
        # Title
        title_label = ttk.Label(frame, text="S2 Resampling Configuration", font=("Arial", 14, "bold"))
        title_label.pack(pady=10)
        
        # Resolution selection
        res_frame = ttk.LabelFrame(frame, text="Target Resolution", padding="10")
        res_frame.pack(fill=tk.X, padx=10, pady=5)
        
        res_options = [
            ("10", "10 meters (Default - Best spatial detail)", "Highest resolution, larger file sizes"),
            ("20", "20 meters (Balanced resolution)", "Good balance of detail and file size"),
            ("60", "60 meters (Fastest processing)", "Lowest resolution, smallest files")
        ]
        
        for value, text, desc in res_options:
            radio_frame = ttk.Frame(res_frame)
            radio_frame.pack(fill=tk.X, pady=2)
            
            ttk.Radiobutton(radio_frame, text=text, variable=self.resolution_var, 
                           value=value).pack(anchor=tk.W)
            ttk.Label(radio_frame, text=desc, font=("Arial", 8), 
                     foreground="gray").pack(anchor=tk.W, padx=(20, 0))
        
        # Advanced resampling options
        advanced_frame = ttk.LabelFrame(frame, text="Advanced Resampling Options", padding="10")
        advanced_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Create grid for options
        options_grid = ttk.Frame(advanced_frame)
        options_grid.pack(fill=tk.X)
        
        # Upsampling method
        ttk.Label(options_grid, text="Upsampling Method:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        upsampling_combo = ttk.Combobox(options_grid, textvariable=self.upsampling_var, 
                                       values=["Bilinear", "Bicubic", "Nearest"], state="readonly", width=15)
        upsampling_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Downsampling method
        ttk.Label(options_grid, text="Downsampling Method:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        downsampling_combo = ttk.Combobox(options_grid, textvariable=self.downsampling_var,
                                         values=["Mean", "Median", "Min", "Max", "First", "Last"], 
                                         state="readonly", width=15)
        downsampling_combo.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Flag downsampling
        ttk.Label(options_grid, text="Flag Downsampling:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        flag_combo = ttk.Combobox(options_grid, textvariable=self.flag_downsampling_var,
                                 values=["First", "FlagAnd", "FlagOr", "FlagMedianAnd", "FlagMedianOr"], 
                                 state="readonly", width=15)
        flag_combo.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Pyramid levels
        ttk.Checkbutton(options_grid, text="Resample on pyramid levels (recommended)", 
                       variable=self.pyramid_var).grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
    
    def setup_subset_tab(self):
        """Setup Spatial Subset configuration tab"""
        frame = ttk.Frame(self.notebook)
        tab_index = self.notebook.add(frame, text="✂️ Spatial Subset")
        self.tab_indices['subset'] = tab_index
        
        # Title
        title_label = ttk.Label(frame, text="Spatial Subset Configuration", font=("Arial", 14, "bold"))
        title_label.pack(pady=10)
        
        # Subset method selection
        method_frame = ttk.LabelFrame(frame, text="Subset Method", padding="10")
        method_frame.pack(fill=tk.X, padx=10, pady=5)
        
        subset_options = [
            ("none", "No spatial subset (process full scene)", "Process entire Sentinel-2 tile"),
            ("geometry", "Use geometry (WKT/Shapefile/KML)", "Define area using spatial geometry"),
            ("pixel", "Use pixel coordinates", "Define rectangular area using pixel coordinates")
        ]
        
        for value, text, desc in subset_options:
            radio_frame = ttk.Frame(method_frame)
            radio_frame.pack(fill=tk.X, pady=2)
            
            ttk.Radiobutton(radio_frame, text=text, variable=self.subset_method_var, 
                           value=value, command=self.update_subset_visibility).pack(anchor=tk.W)
            ttk.Label(radio_frame, text=desc, font=("Arial", 8), 
                     foreground="gray").pack(anchor=tk.W, padx=(20, 0))
        
        # Geometry subset
        self.geometry_frame = ttk.LabelFrame(frame, text="Geometry Subset", padding="10")
        self.geometry_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(self.geometry_frame, text="WKT Geometry:").pack(anchor=tk.W, pady=2)
        
        geometry_text_frame = ttk.Frame(self.geometry_frame)
        geometry_text_frame.pack(fill=tk.X, pady=2)
        
        self.geometry_text = tk.Text(geometry_text_frame, height=4, wrap=tk.WORD, font=("Consolas", 9))
        geometry_scrollbar = ttk.Scrollbar(geometry_text_frame, orient=tk.VERTICAL, command=self.geometry_text.yview)
        self.geometry_text.configure(yscrollcommand=geometry_scrollbar.set)
        
        self.geometry_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        geometry_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Geometry buttons
        geometry_btn_frame = ttk.Frame(self.geometry_frame)
        geometry_btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(geometry_btn_frame, text="Load Geometry...", 
                  command=self.load_geometry).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(geometry_btn_frame, text="Clear", 
                  command=lambda: self.geometry_text.delete(1.0, tk.END)).pack(side=tk.LEFT, padx=5)
        ttk.Button(geometry_btn_frame, text="Validate", 
                  command=self.validate_geometry).pack(side=tk.LEFT, padx=5)
        
        # Pixel subset
        self.pixel_frame = ttk.LabelFrame(frame, text="Pixel Subset", padding="10")
        self.pixel_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Pixel coordinates grid
        pixel_grid = ttk.Frame(self.pixel_frame)
        pixel_grid.pack(pady=5)
        
        # Create entry fields in a grid
        coordinates = [
            ("Start X:", "pixel_start_x_var", 0, 0),
            ("Start Y:", "pixel_start_y_var", 0, 2),
            ("Width:", "pixel_width_var", 1, 0),
            ("Height:", "pixel_height_var", 1, 2)
        ]
        
        for label, var_name, row, col in coordinates:
            ttk.Label(pixel_grid, text=label).grid(row=row, column=col, sticky=tk.W, padx=5, pady=2)
            var = getattr(self, var_name)
            ttk.Entry(pixel_grid, textvariable=var, width=10).grid(row=row, column=col+1, padx=5, pady=2)
        
        # Update visibility based on initial selection
        self.update_subset_visibility()
    
    def setup_c2rcc_tab(self):
        """Setup C2RCC configuration tab with ECMWF enabled by default"""
        frame = ttk.Frame(self.notebook)
        tab_index = self.notebook.add(frame, text="🌊 C2RCC Parameters")
        self.tab_indices['c2rcc'] = tab_index
        
        # Create scrollable frame
        canvas = tk.Canvas(frame)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Title
        title_label = ttk.Label(scrollable_frame, text="C2RCC Atmospheric Correction Parameters", 
                            font=("Arial", 14, "bold"))
        title_label.pack(pady=10)
        
        # Important note about automatic SNAP TSM/CHL
        note_frame = ttk.Frame(scrollable_frame, relief=tk.SUNKEN, borderwidth=1)
        note_frame.pack(fill=tk.X, padx=10, pady=5)
        
        note_text = (
            "ℹ️ Automatic SNAP Products:\n"
            "• TSM and CHL concentrations are automatically calculated during C2RCC processing\n"
            "• Uncertainty maps (unc_tsm.img, unc_chl.img) are generated when uncertainties are enabled\n"
            "• Water leaving reflectance (rhow) bands are generated for optional Jiang TSS processing"
        )
        
        note_label = ttk.Label(note_frame, text=note_text,
                              font=("Arial", 9), foreground="darkblue", 
                              wraplength=600, justify=tk.LEFT, padding="5")
        note_label.pack()
        
        # ECMWF Configuration (highlighted as enabled by default)
        ecmwf_frame = ttk.LabelFrame(scrollable_frame, text="ECMWF Auxiliary Data (Enhanced Accuracy)", padding="10")
        ecmwf_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Checkbutton(ecmwf_frame, text="✓ Use ECMWF auxiliary data (ENABLED BY DEFAULT)", 
                    variable=self.use_ecmwf_var, 
                    command=self.on_ecmwf_toggle).pack(anchor=tk.W, pady=2)
        
        ecmwf_info = ttk.Label(ecmwf_frame,
                              text="✨ Uses real atmospheric conditions (ozone, pressure) at acquisition time for superior accuracy",
                              font=("Arial", 9, "bold"), foreground="darkgreen", wraplength=500)
        ecmwf_info.pack(anchor=tk.W, pady=2)
        
        # Water Properties
        water_frame = ttk.LabelFrame(scrollable_frame, text="Water Properties", padding="10")
        water_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Quick presets
        preset_frame = ttk.Frame(water_frame)
        preset_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(preset_frame, text="Quick Presets:").pack(side=tk.LEFT)
        preset_btn_frame = ttk.Frame(preset_frame)
        preset_btn_frame.pack(side=tk.LEFT, padx=(10, 0))
        
        presets = [
            ("Coastal", {"salinity": 35.0, "temperature": 15.0}),
            ("Inland", {"salinity": 0.1, "temperature": 20.0}),
            ("Estuary", {"salinity": 15.0, "temperature": 18.0})
        ]
        
        for name, values in presets:
            ttk.Button(preset_btn_frame, text=name, width=8,
                      command=lambda v=values: self.apply_water_preset(v)).pack(side=tk.LEFT, padx=2)
        
        # Water parameters grid
        water_grid = ttk.Frame(water_frame)
        water_grid.pack(fill=tk.X)
        
        # Salinity
        ttk.Label(water_grid, text="Salinity (PSU):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        salinity_spinbox = ttk.Spinbox(water_grid, from_=0.1, to=42, width=10,
                                      textvariable=self.salinity_var, increment=0.5)
        salinity_spinbox.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Temperature
        ttk.Label(water_grid, text="Temperature (°C):").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        temp_spinbox = ttk.Spinbox(water_grid, from_=0.1, to=35, width=10,
                                  textvariable=self.temperature_var, increment=0.5)
        temp_spinbox.grid(row=0, column=3, sticky=tk.W, padx=5, pady=2)
        
        # Atmospheric parameters
        atmos_frame = ttk.LabelFrame(scrollable_frame, text="Atmospheric Parameters", padding="10")
        atmos_frame.pack(fill=tk.X, padx=10, pady=5)
        
        atmos_grid = ttk.Frame(atmos_frame)
        atmos_grid.pack(fill=tk.X)
        
        # Ozone
        ttk.Label(atmos_grid, text="Ozone (DU):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ozone_spinbox = ttk.Spinbox(atmos_grid, from_=100, to=800, width=10,
                                   textvariable=self.ozone_var, increment=10)
        ozone_spinbox.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Pressure
        ttk.Label(atmos_grid, text="Pressure (hPa):").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        pressure_spinbox = ttk.Spinbox(atmos_grid, from_=850, to=1030, width=10,
                                      textvariable=self.pressure_var, increment=5)
        pressure_spinbox.grid(row=0, column=3, sticky=tk.W, padx=5, pady=2)
        
        # Output Products Configuration
        output_frame = ttk.LabelFrame(scrollable_frame, text="Output Products Configuration", padding="10")
        output_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Essential Outputs (Always recommended)
        essential_frame = ttk.LabelFrame(output_frame, text="Essential Outputs", padding="5")
        essential_frame.pack(fill=tk.X, pady=5)
        
        ttk.Checkbutton(essential_frame, text="✓ Water leaving reflectance (rhow) - Required for TSS", 
                       variable=self.output_rhow_var, 
                       command=self.on_rhow_toggle).pack(anchor=tk.W, pady=2)
        
        ttk.Checkbutton(essential_frame, text="✓ Diffuse attenuation coefficient (Kd)", 
                       variable=self.output_kd_var).pack(anchor=tk.W, pady=2)
        
        ttk.Checkbutton(essential_frame, text="✓ Uncertainty estimates (enables unc_tsm.img & unc_chl.img)", 
                       variable=self.output_uncertainties_var).pack(anchor=tk.W, pady=2)
        
        # Reflectance Products
        reflectance_frame = ttk.LabelFrame(output_frame, text="Reflectance Products", padding="5")
        reflectance_frame.pack(fill=tk.X, pady=5)
        
        ttk.Checkbutton(reflectance_frame, text="✓ Atmospherically corrected reflectance", 
                       variable=self.output_ac_reflectance_var).pack(anchor=tk.W, pady=2)
        
        ttk.Checkbutton(reflectance_frame, text="✓ Top-of-atmosphere reflectance (rtoa)", 
                       variable=self.output_rtoa_var).pack(anchor=tk.W, pady=2)
        
        ttk.Checkbutton(reflectance_frame, text="Remote sensing reflectance (Rrs)", 
                       variable=self.output_rrs_var).pack(anchor=tk.W, pady=2)
        
        # Quick Output Presets
        preset_outputs_frame = ttk.Frame(output_frame)
        preset_outputs_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(preset_outputs_frame, text="Quick Presets:").pack(side=tk.LEFT)
        preset_output_buttons = ttk.Frame(preset_outputs_frame)
        preset_output_buttons.pack(side=tk.LEFT, padx=(10, 0))
        
        ttk.Button(preset_output_buttons, text="Essential", width=12,
                  command=self.apply_essential_outputs).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_output_buttons, text="Scientific", width=12,
                  command=self.apply_scientific_outputs).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_output_buttons, text="SNAP Defaults", width=12,
                  command=self.apply_snap_defaults).pack(side=tk.LEFT, padx=2)
    
    def setup_tss_tab(self):
        """Setup TSS (Jiang + Advanced + Marine Visualization) configuration tab"""
        frame = ttk.Frame(self.notebook)
        tab_index = self.notebook.add(frame, text="🧪 TSS & Visualization")
        self.tab_indices['tss'] = tab_index
        
        # Create scrollable frame for the tab
        canvas = tk.Canvas(frame)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Title
        title_label = ttk.Label(scrollable_frame, text="TSS Estimation & Marine Visualization", font=("Arial", 14, "bold"))
        title_label.pack(pady=10)
        
        # SNAP TSM/CHL note (unchanged)
        snap_note_frame = ttk.Frame(scrollable_frame, relief=tk.SUNKEN, borderwidth=1)
        snap_note_frame.pack(fill=tk.X, padx=10, pady=5)
        
        snap_note_text = (
            "📋 SNAP TSM/CHL Products:\n"
            "• SNAP TSM and CHL concentrations are automatically generated during C2RCC processing\n"
            "• These products include uncertainty maps when uncertainties are enabled\n"
            "• No additional configuration needed - always included in C2RCC output"
        )
        
        snap_note_label = ttk.Label(snap_note_frame, text=snap_note_text,
                                font=("Arial", 9), foreground="darkblue", 
                                wraplength=600, justify=tk.LEFT, padding="5")
        snap_note_label.pack()
        
        # Jiang TSS Configuration (unchanged)
        jiang_frame = ttk.LabelFrame(scrollable_frame, text="Jiang et al. 2023 TSS Methodology (Optional)", padding="10")
        jiang_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Checkbutton(jiang_frame, text="✓ Enable Jiang TSS processing", 
                    variable=self.enable_jiang_var,
                    command=self.update_jiang_visibility).pack(anchor=tk.W, pady=5)
        
        self.jiang_options_frame = ttk.Frame(jiang_frame)
        
        ttk.Checkbutton(self.jiang_options_frame, text="Output intermediate products", 
                    variable=self.jiang_intermediates_var).pack(anchor=tk.W, pady=2)
        
        ttk.Checkbutton(self.jiang_options_frame, text="Generate comparison statistics", 
                    variable=self.jiang_comparison_var).pack(anchor=tk.W, pady=2)
        
        # NEW: Marine Visualization Configuration
        marine_viz_frame = ttk.LabelFrame(scrollable_frame, text="🎨 Marine Visualization (RGB + Indices) - NEW!", padding="10")
        marine_viz_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Enable marine visualization
        ttk.Checkbutton(marine_viz_frame, text="✓ Generate RGB composites and spectral indices (ENABLED BY DEFAULT)", 
                    variable=self.enable_marine_viz_var,
                    command=self.update_marine_viz_visibility).pack(anchor=tk.W, pady=5)
        
        # Marine visualization description
        marine_desc_text = (
            "Automatically generates comprehensive marine visualization products:\n"
            "• RGB Composites: Natural color, false color, turbidity-enhanced, chlorophyll-enhanced, etc.\n"
            "• Spectral Indices: NDTI, NDWI, NDCI, TSI, water quality indices, etc.\n"
            "• Uses actual SNAP C2RCC output files (rho_toa_*.img)\n"
            "• Export formats: GeoTIFF with contrast enhancement for optimal visualization\n"
            "• Perfect for publications, presentations, and visual analysis of your 700+ Douro images"
        )
        
        marine_desc_label = ttk.Label(marine_viz_frame, text=marine_desc_text, font=("Arial", 9),
                                    foreground="darkgreen", wraplength=600, justify=tk.LEFT)
        marine_desc_label.pack(anchor=tk.W, padx=(20, 0), pady=5)
        
        # Marine visualization options frame
        self.marine_viz_options_frame = ttk.Frame(marine_viz_frame)
        
        # RGB options
        rgb_options_frame = ttk.LabelFrame(self.marine_viz_options_frame, text="RGB Composites", padding="5")
        rgb_options_frame.pack(fill=tk.X, pady=5)
        
        ttk.Checkbutton(rgb_options_frame, text="✓ Natural color combinations (true color, enhanced contrast)", 
                    variable=self.natural_color_var).pack(anchor=tk.W, pady=1)
        ttk.Checkbutton(rgb_options_frame, text="✓ False color combinations (infrared, NIR)", 
                    variable=self.false_color_var).pack(anchor=tk.W, pady=1)
        ttk.Checkbutton(rgb_options_frame, text="✓ Water-specific combinations (turbidity, chlorophyll, sediment)", 
                    variable=self.water_specific_var).pack(anchor=tk.W, pady=1)
        ttk.Checkbutton(rgb_options_frame, text="Research combinations (advanced users)", 
                    variable=self.research_rgb_var).pack(anchor=tk.W, pady=1)
        
        # Spectral indices options
        indices_options_frame = ttk.LabelFrame(self.marine_viz_options_frame, text="Spectral Indices", padding="5")
        indices_options_frame.pack(fill=tk.X, pady=5)
        
        ttk.Checkbutton(indices_options_frame, text="✓ Water quality indices (NDWI, MNDWI, water delineation)", 
                    variable=self.water_quality_indices_var).pack(anchor=tk.W, pady=1)
        ttk.Checkbutton(indices_options_frame, text="✓ Chlorophyll indices (NDCI, GNDVI, red edge)", 
                    variable=self.chlorophyll_indices_var).pack(anchor=tk.W, pady=1)
        ttk.Checkbutton(indices_options_frame, text="✓ Turbidity indices (NDTI, TSI, sediment)", 
                    variable=self.turbidity_indices_var).pack(anchor=tk.W, pady=1)
        ttk.Checkbutton(indices_options_frame, text="Advanced indices (FUI, SDD, CDOM)", 
                    variable=self.advanced_indices_var).pack(anchor=tk.W, pady=1)
        
        # Quick preset buttons
        preset_frame = ttk.Frame(self.marine_viz_options_frame)
        preset_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(preset_frame, text="Quick presets:").pack(side=tk.LEFT)
        preset_buttons_frame = ttk.Frame(preset_frame)
        preset_buttons_frame.pack(side=tk.LEFT, padx=(10, 0))
        
        ttk.Button(preset_buttons_frame, text="Essential", width=10,
                command=self.apply_essential_marine_viz).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_buttons_frame, text="Complete", width=10,
                command=self.apply_complete_marine_viz).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_buttons_frame, text="Research", width=10,
                command=self.apply_research_marine_viz).pack(side=tk.LEFT, padx=2)
        
        # Advanced Algorithms Section (unchanged)
        advanced_frame = ttk.LabelFrame(scrollable_frame, text="Advanced Aquatic Algorithms", padding="10")
        advanced_frame.pack(fill=tk.X, padx=10, pady=10)

        self.enable_advanced_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(advanced_frame, text="Enable advanced aquatic algorithms", 
                    variable=self.enable_advanced_var,
                    command=self.update_advanced_visibility).pack(anchor=tk.W, pady=5)

        self.advanced_options_frame = ttk.Frame(advanced_frame)

        working_frame = ttk.LabelFrame(self.advanced_options_frame, text="Available Algorithms", padding="5")
        working_frame.pack(fill=tk.X, pady=5)

        self.water_clarity_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(working_frame, text="✓ Water Clarity Indices", 
                    variable=self.water_clarity_var).pack(anchor=tk.W, pady=2)

        self.hab_detection_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(working_frame, text="✓ Harmful Algal Bloom Detection", 
                    variable=self.hab_detection_var).pack(anchor=tk.W, pady=2)
        
        # Update initial visibility
        self.update_jiang_visibility()
        self.update_marine_viz_visibility()
        self.update_advanced_visibility()

    def update_advanced_visibility(self):
        """Update advanced algorithms options visibility"""
        if self.enable_advanced_var.get():
            self.advanced_options_frame.pack(fill=tk.X, pady=(10, 0))
        else:
            self.advanced_options_frame.pack_forget()
    
    def update_marine_viz_visibility(self):
        """Update marine visualization options visibility"""
        if self.enable_marine_viz_var.get():
            self.marine_viz_options_frame.pack(fill=tk.X, pady=(10, 0))
        else:
            self.marine_viz_options_frame.pack_forget()

    def apply_essential_marine_viz(self):
        """Apply essential marine visualization preset"""
        # RGB
        self.natural_color_var.set(True)
        self.false_color_var.set(False)
        self.water_specific_var.set(True)
        self.research_rgb_var.set(False)
        
        # Indices
        self.water_quality_indices_var.set(True)
        self.chlorophyll_indices_var.set(True)
        self.turbidity_indices_var.set(True)
        self.advanced_indices_var.set(False)
        
        self.status_var.set("Essential marine visualization preset applied")

    def apply_complete_marine_viz(self):
        """Apply complete marine visualization preset"""
        # RGB
        self.natural_color_var.set(True)
        self.false_color_var.set(True)
        self.water_specific_var.set(True)
        self.research_rgb_var.set(False)
        
        # Indices
        self.water_quality_indices_var.set(True)
        self.chlorophyll_indices_var.set(True)
        self.turbidity_indices_var.set(True)
        self.advanced_indices_var.set(True)
        
        self.status_var.set("Complete marine visualization preset applied")

    def apply_research_marine_viz(self):
        """Apply research marine visualization preset"""
        # RGB - everything
        self.natural_color_var.set(True)
        self.false_color_var.set(True)
        self.water_specific_var.set(True)
        self.research_rgb_var.set(True)
        
        # Indices - everything
        self.water_quality_indices_var.set(True)
        self.chlorophyll_indices_var.set(True)
        self.turbidity_indices_var.set(True)
        self.advanced_indices_var.set(True)
        
        self.status_var.set("Research marine visualization preset applied")

    
    def setup_monitoring_tab(self):
        """Setup system monitoring and status tab"""
        frame = ttk.Frame(self.notebook)
        tab_index = self.notebook.add(frame, text="📊 System Monitor")
        self.tab_indices['monitoring'] = tab_index
        
        # Title
        title_label = ttk.Label(frame, text="System Monitoring & Status", font=("Arial", 14, "bold"))
        title_label.pack(pady=10)
        
        # System info frame
        sys_frame = ttk.LabelFrame(frame, text="System Information", padding="10")
        sys_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Create system info labels
        self.cpu_label = ttk.Label(sys_frame, text="CPU: --", font=("Consolas", 10))
        self.cpu_label.pack(anchor=tk.W, pady=2)
        
        self.memory_label = ttk.Label(sys_frame, text="Memory: --", font=("Consolas", 10))
        self.memory_label.pack(anchor=tk.W, pady=2)
        
        self.disk_label = ttk.Label(sys_frame, text="Disk: --", font=("Consolas", 10))
        self.disk_label.pack(anchor=tk.W, pady=2)
        
        self.snap_label = ttk.Label(sys_frame, text="SNAP: --", font=("Consolas", 10))
        self.snap_label.pack(anchor=tk.W, pady=2)
        
        # Processing status frame
        status_frame = ttk.LabelFrame(frame, text="Processing Status", padding="10")
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, 
                                           maximum=100, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # Status labels
        self.current_status_label = ttk.Label(status_frame, textvariable=self.status_var, 
                                             font=("Arial", 10, "bold"))
        self.current_status_label.pack(anchor=tk.W, pady=2)
        
        self.eta_label = ttk.Label(status_frame, textvariable=self.eta_var, 
                                  font=("Arial", 9), foreground="gray")
        self.eta_label.pack(anchor=tk.W, pady=2)
        
        # Processing statistics
        stats_frame = ttk.LabelFrame(frame, text="Processing Statistics", padding="10")
        stats_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.stats_text = tk.Text(stats_frame, height=8, font=("Consolas", 9), 
                                 state=tk.DISABLED, wrap=tk.WORD)
        stats_scrollbar = ttk.Scrollbar(stats_frame, orient=tk.VERTICAL, command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=stats_scrollbar.set)
        
        self.stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        stats_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def setup_status_bar(self, parent):
        """Setup status bar at bottom"""
        self.status_frame = ttk.Frame(parent, relief=tk.SUNKEN, borderwidth=1)
        self.status_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Left side - status
        left_frame = ttk.Frame(self.status_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=2)
        
        self.status_label = ttk.Label(left_frame, textvariable=self.status_var)
        self.status_label.pack(side=tk.LEFT)
        
        # Right side - version info
        right_frame = ttk.Frame(self.status_frame)
        right_frame.pack(side=tk.RIGHT, padx=5, pady=2)
        
        version_label = ttk.Label(right_frame, text="Unified S2-TSS Pipeline v1.0", 
                                 font=("Arial", 8), foreground="gray")
        version_label.pack()
    
    def setup_control_buttons(self, parent):
        """Setup control buttons at bottom"""
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Left side buttons
        left_buttons = ttk.Frame(button_frame)
        left_buttons.pack(side=tk.LEFT)
        
        self.start_button = ttk.Button(left_buttons, text="🚀 Start Processing", 
                                      command=self.start_processing)
        self.start_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.stop_button = ttk.Button(left_buttons, text="⏹️ Stop", 
                                     command=self.stop_processing, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Right side buttons
        right_buttons = ttk.Frame(button_frame)
        right_buttons.pack(side=tk.RIGHT)
        
        ttk.Button(right_buttons, text="💾 Save Config", 
                  command=self.save_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(right_buttons, text="📁 Load Config", 
                  command=self.load_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(right_buttons, text="❌ Exit", 
                  command=self.on_closing).pack(side=tk.LEFT, padx=(10, 0))
    
    # ===== GUI EVENT HANDLERS =====
    
    def on_mode_change(self):
        """Handle processing mode change"""
        self.update_tab_visibility()
        self.validate_input_directory()
        self.status_var.set(f"Mode changed to: {self.processing_mode.get()}")
    
    def update_tab_visibility(self):
        """Update tab visibility based on processing mode"""
        try:
            mode = self.processing_mode.get()
            
            # Show/hide tabs based on mode
            if mode in ["complete_pipeline", "s2_processing_only"]:
                # Show S2 processing tabs
                for tab_name in ['resampling', 'subset', 'c2rcc']:
                    if tab_name in self.tab_indices:
                        try:
                            self.notebook.tab(self.tab_indices[tab_name], state="normal")
                        except tk.TclError:
                            pass
            else:
                # Hide S2 processing tabs for TSS-only mode
                for tab_name in ['resampling', 'subset', 'c2rcc']:
                    if tab_name in self.tab_indices:
                        try:
                            self.notebook.tab(self.tab_indices[tab_name], state="hidden")
                        except tk.TclError:
                            pass
            
            # TSS tab visibility
            if mode in ["complete_pipeline", "tss_processing_only"]:
                if 'tss' in self.tab_indices:
                    try:
                        self.notebook.tab(self.tab_indices['tss'], state="normal")
                    except tk.TclError:
                        pass
            else:
                if 'tss' in self.tab_indices:
                    try:
                        self.notebook.tab(self.tab_indices['tss'], state="hidden")
                    except tk.TclError:
                        pass
                        
        except Exception as e:
            logger.error(f"Error updating tab visibility: {e}")
    
    def update_subset_visibility(self):
        """Update subset frame visibility"""
        method = self.subset_method_var.get()
        
        if method == "geometry":
            self.geometry_frame.pack(fill=tk.X, padx=10, pady=5)
            self.pixel_frame.pack_forget()
        elif method == "pixel":
            self.pixel_frame.pack(fill=tk.X, padx=10, pady=5)
            self.geometry_frame.pack_forget()
        else:
            self.geometry_frame.pack_forget()
            self.pixel_frame.pack_forget()
    
    def update_jiang_visibility(self):
        """Update Jiang options visibility"""
        if self.enable_jiang_var.get():
            self.jiang_options_frame.pack(fill=tk.X, pady=(10, 0))
        else:
            self.jiang_options_frame.pack_forget()
    
    def on_ecmwf_toggle(self):
        """Handle ECMWF toggle with information"""
        if not self.use_ecmwf_var.get():
            result = messagebox.askyesno(
                "ECMWF Disabled", 
                "Disabling ECMWF auxiliary data will reduce atmospheric correction accuracy.\n\n"
                "ECMWF provides real-time ozone and pressure data at acquisition time.\n\n"
                "Continue anyway?",
                parent=self.root
            )
            if not result:
                self.use_ecmwf_var.set(True)
    
    def on_rhow_toggle(self):
        """Handle rhow toggle with warning"""
        if not self.output_rhow_var.get():
            result = messagebox.askyesno(
                "Warning", 
                "Disabling water leaving reflectance (rhow) will prevent Jiang TSS processing.\n\n"
                "This output is required for advanced TSS analysis.\n\n"
                "Continue anyway?",
                parent=self.root
            )
            if not result:
                self.output_rhow_var.set(True)
    
    def validate_input_directory(self, *args):
        """Validate input directory and update display"""
        input_dir = self.input_dir_var.get()
        if not input_dir or not os.path.exists(input_dir):
            self.input_validation_result = {"valid": False, "message": "Please select a valid input directory", "products": []}
            self.input_validation_label.config(text=self.input_validation_result["message"], foreground="red")
            return
        
        # Scan directory for products
        products = ProductDetector.scan_input_folder(input_dir)
        mode = ProcessingMode(self.processing_mode.get())
        
        # Validate products for current mode
        valid, message, product_list = ProductDetector.validate_processing_mode(products, mode)
        
        self.input_validation_result = {
            "valid": valid,
            "message": message,
            "products": product_list
        }
        
        # Update display
        color = "darkgreen" if valid else "red"
        self.input_validation_label.config(text=message, foreground=color)
        
        # Update status
        if valid:
            self.status_var.set(f"Ready: {len(product_list)} products found")
        else:
            self.status_var.set("Input validation failed")
    
    def validate_geometry(self):
        """Validate WKT geometry"""
        wkt_text = self.geometry_text.get(1.0, tk.END).strip()
        if not wkt_text:
            messagebox.showwarning("Warning", "No geometry to validate", parent=self.root)
            return
        
        try:
            # Basic WKT validation
            if not any(wkt_text.upper().startswith(geom) for geom in ['POLYGON', 'POINT', 'LINESTRING']):
                raise ValueError("Invalid WKT format")
            
            messagebox.showinfo("Validation", "Geometry appears valid", parent=self.root)
            self.status_var.set("Geometry validated")
        except Exception as e:
            messagebox.showerror("Validation Error", f"Invalid geometry: {str(e)}", parent=self.root)
    
    def browse_input_dir(self):
        """Browse for input directory"""
        directory = filedialog.askdirectory(title="Select Input Directory", parent=self.root)
        if directory:
            self.input_dir_var.set(directory)
    
    def browse_output_dir(self):
        """Browse for output directory - TRIGGERED BY USER CLICKING BROWSE BUTTON"""
        directory = filedialog.askdirectory(title="Select Output Directory", parent=self.root)
        if directory:
            self.output_dir_var.set(directory)
            
            # ✅ SOLUTION 1: Fix logging when user selects output folder
            global logger
            logger, log_file = setup_enhanced_logging(log_level=logging.INFO, output_folder=directory)
            logger.info(f"🎯 Output directory selected: {directory}")
            logger.info(f"📋 Log file: {log_file}")
            self.status_var.set(f"Output: {directory} | Logging to: {os.path.basename(log_file)}")
    
    def load_geometry(self):
        """Load geometry using geometry utils"""
        try:
            geometry_wkt = get_geometry_input()
            if geometry_wkt:
                self.geometry_text.delete(1.0, tk.END)
                self.geometry_text.insert(1.0, geometry_wkt)
                self.subset_method_var.set("geometry")
                self.update_subset_visibility()
                messagebox.showinfo("Success", "Geometry loaded successfully!", parent=self.root)
                self.status_var.set("Geometry loaded")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load geometry: {str(e)}", parent=self.root)
    
    def apply_water_preset(self, values):
        """Apply water parameter preset"""
        self.salinity_var.set(values["salinity"])
        self.temperature_var.set(values["temperature"])
        self.status_var.set("Water preset applied")
    
    def apply_snap_defaults(self):
        """Apply SNAP default values to all parameters"""
        # Basic parameters
        self.salinity_var.set(35.0)
        self.temperature_var.set(15.0)
        self.ozone_var.set(330.0)
        self.pressure_var.set(1000.0)  # SNAP default
        self.elevation_var.set(0.0)
        
        # Neural network and DEM
        self.net_set_var.set("C2RCC-Nets")
        self.dem_name_var.set("Copernicus 30m Global DEM")
        self.use_ecmwf_var.set(True)  # Keep enabled by default as requested
        
        # Output products (SNAP defaults)
        self.output_rrs_var.set(True)
        self.output_rhow_var.set(True)
        self.output_kd_var.set(True)
        self.output_uncertainties_var.set(True)  # Keep enabled for TSM/CHL uncertainties
        self.output_ac_reflectance_var.set(True)
        self.output_rtoa_var.set(True)
        self.output_rtosa_gc_var.set(False)
        self.output_rtosa_gc_aann_var.set(False)
        self.output_rpath_var.set(False)
        self.output_tdown_var.set(False)
        self.output_tup_var.set(False)
        self.output_oos_var.set(False)
        
        # Advanced parameters
        self.valid_pixel_var.set("B8 > 0 && B8 < 0.1")
        self.threshold_rtosa_oos_var.set(0.05)
        self.threshold_ac_reflec_oos_var.set(0.1)
        self.threshold_cloud_tdown865_var.set(0.955)
        
        # TSM and CHL parameters
        self.tsm_fac_var.set(1.06)
        self.tsm_exp_var.set(0.942)
        self.chl_fac_var.set(21.0)
        self.chl_exp_var.set(1.04)
        
        self.status_var.set("SNAP default values applied")
    
    def apply_essential_outputs(self):
        """Apply essential outputs preset"""
        # Reset all to False first
        self.reset_all_outputs()
        
        # Enable essential outputs
        self.output_rrs_var.set(True)
        self.output_rhow_var.set(True)
        self.output_kd_var.set(True)
        self.output_uncertainties_var.set(True)  # Keep for TSM/CHL uncertainties
        self.output_ac_reflectance_var.set(True)
        
        self.status_var.set("Essential outputs preset applied")
    
    def apply_scientific_outputs(self):
        """Apply scientific outputs preset"""
        # Reset all to False first
        self.reset_all_outputs()
        
        # Enable scientific outputs
        self.output_rrs_var.set(True)
        self.output_rhow_var.set(True)
        self.output_kd_var.set(True)
        self.output_ac_reflectance_var.set(True)
        self.output_rtoa_var.set(True)
        self.output_uncertainties_var.set(True)
        self.output_oos_var.set(True)
        
        self.status_var.set("Scientific outputs preset applied")
    
    def reset_all_outputs(self):
        """Reset all output variables to False"""
        self.output_rhow_var.set(False)
        self.output_kd_var.set(False)
        self.output_ac_reflectance_var.set(False)
        self.output_rtoa_var.set(False)
        self.output_rrs_var.set(False)
        self.output_rtosa_gc_var.set(False)
        self.output_rtosa_gc_aann_var.set(False)
        self.output_rpath_var.set(False)
        self.output_tdown_var.set(False)
        self.output_tup_var.set(False)
        self.output_uncertainties_var.set(False)
        self.output_oos_var.set(False)
    
    def update_configurations(self):
        """Update configuration objects from GUI - COMPLETE VERSION WITH FULL MARINE VISUALIZATION"""
        try:
            # Update resampling config
            self.resampling_config.target_resolution = self.resolution_var.get()
            self.resampling_config.upsampling_method = self.upsampling_var.get()
            self.resampling_config.downsampling_method = self.downsampling_var.get()
            self.resampling_config.flag_downsampling = self.flag_downsampling_var.get()
            self.resampling_config.resample_on_pyramid_levels = self.pyramid_var.get()
            
            # Update subset config
            subset_method = self.subset_method_var.get()
            
            if subset_method == "geometry":
                geometry_text = self.geometry_text.get(1.0, tk.END).strip()
                if geometry_text:
                    self.subset_config.geometry_wkt = geometry_text
                    self.subset_config.pixel_start_x = None
                    self.subset_config.pixel_start_y = None
                    self.subset_config.pixel_size_x = None
                    self.subset_config.pixel_size_y = None
                else:
                    messagebox.showerror("Error", "Geometry method selected but no WKT provided!", parent=self.root)
                    return False
            elif subset_method == "pixel":
                try:
                    start_x = int(self.pixel_start_x_var.get()) if self.pixel_start_x_var.get() else None
                    start_y = int(self.pixel_start_y_var.get()) if self.pixel_start_y_var.get() else None
                    width = int(self.pixel_width_var.get()) if self.pixel_width_var.get() else None
                    height = int(self.pixel_height_var.get()) if self.pixel_height_var.get() else None
                    
                    if all(v is not None for v in [start_x, start_y, width, height]):
                        self.subset_config.pixel_start_x = start_x
                        self.subset_config.pixel_start_y = start_y
                        self.subset_config.pixel_size_x = width
                        self.subset_config.pixel_size_y = height
                        self.subset_config.geometry_wkt = None
                    else:
                        messagebox.showerror("Error", "Pixel method selected but incomplete coordinates!", parent=self.root)
                        return False
                except ValueError:
                    messagebox.showerror("Error", "Invalid pixel coordinates (must be integers)!", parent=self.root)
                    return False
            else:
                # No subset
                self.subset_config.geometry_wkt = None
                self.subset_config.pixel_start_x = None
                self.subset_config.pixel_start_y = None
                self.subset_config.pixel_size_x = None
                self.subset_config.pixel_size_y = None
            
            # Update C2RCC config
            self.c2rcc_config.salinity = self.salinity_var.get()
            self.c2rcc_config.temperature = self.temperature_var.get()
            self.c2rcc_config.ozone = self.ozone_var.get()
            self.c2rcc_config.pressure = self.pressure_var.get()
            self.c2rcc_config.elevation = self.elevation_var.get()
            self.c2rcc_config.net_set = self.net_set_var.get()
            self.c2rcc_config.dem_name = self.dem_name_var.get()
            self.c2rcc_config.use_ecmwf_aux_data = self.use_ecmwf_var.get()
            
            # Output products
            self.c2rcc_config.output_as_rrs = self.output_rrs_var.get()
            self.c2rcc_config.output_rhown = self.output_rhow_var.get()
            self.c2rcc_config.output_kd = self.output_kd_var.get()
            self.c2rcc_config.output_uncertainties = self.output_uncertainties_var.get()
            self.c2rcc_config.output_ac_reflectance = self.output_ac_reflectance_var.get()
            self.c2rcc_config.output_rtoa = self.output_rtoa_var.get()
            self.c2rcc_config.output_rtosa_gc = self.output_rtosa_gc_var.get()
            self.c2rcc_config.output_rtosa_gc_aann = self.output_rtosa_gc_aann_var.get()
            self.c2rcc_config.output_rpath = self.output_rpath_var.get()
            self.c2rcc_config.output_tdown = self.output_tdown_var.get()
            self.c2rcc_config.output_tup = self.output_tup_var.get()
            self.c2rcc_config.output_oos = self.output_oos_var.get()
            
            # Advanced parameters
            self.c2rcc_config.valid_pixel_expression = self.valid_pixel_var.get()
            self.c2rcc_config.threshold_rtosa_oos = self.threshold_rtosa_oos_var.get()
            self.c2rcc_config.threshold_ac_reflec_oos = self.threshold_ac_reflec_oos_var.get()
            self.c2rcc_config.threshold_cloud_tdown865 = self.threshold_cloud_tdown865_var.get()
            
            # TSM and CHL parameters
            self.c2rcc_config.tsm_fac = self.tsm_fac_var.get()
            self.c2rcc_config.tsm_exp = self.tsm_exp_var.get()
            self.c2rcc_config.chl_fac = self.chl_fac_var.get()
            self.c2rcc_config.chl_exp = self.chl_exp_var.get()
            
            # Update Jiang config - BASIC SETTINGS
            self.jiang_config.enable_jiang_tss = self.enable_jiang_var.get()
            self.jiang_config.output_intermediates = self.jiang_intermediates_var.get()
            self.jiang_config.output_comparison_stats = self.jiang_comparison_var.get()
            
            # ===== MARINE VISUALIZATION CONFIGURATION - FULL OPERATIONAL =====
            # Ensure marine visualization attributes exist and are operational
            if not hasattr(self.jiang_config, 'enable_marine_visualization'):
                self.jiang_config.enable_marine_visualization = True
                logger.info("Added enable_marine_visualization (ENABLED)")
                
            if not hasattr(self.jiang_config, 'marine_viz_config'):
                self.jiang_config.marine_viz_config = None
                logger.info("Added marine_viz_config attribute")
            
            # Update marine visualization settings from GUI (if GUI variables exist)
            if hasattr(self, 'enable_marine_viz_var'):
                self.jiang_config.enable_marine_visualization = self.enable_marine_viz_var.get()
            else:
                # Default to enabled if no GUI variable
                self.jiang_config.enable_marine_visualization = True
                logger.info("Marine visualization enabled by default (no GUI variable found)")
            
            # Ensure marine_viz_config is created and operational
            if self.jiang_config.enable_marine_visualization:
                if self.jiang_config.marine_viz_config is None:
                    # Create a complete operational marine visualization config
                    logger.info("Creating operational marine visualization configuration...")
                    
                    @dataclass
                    class MarineVisualizationConfig:
                        """Complete operational marine visualization configuration"""
                        # RGB options - FULLY ENABLED
                        generate_natural_color: bool = True
                        generate_false_color: bool = True
                        generate_water_specific: bool = True
                        generate_research_combinations: bool = True
                        
                        # Spectral indices - FULLY ENABLED
                        generate_water_quality_indices: bool = True
                        generate_chlorophyll_indices: bool = True
                        generate_turbidity_indices: bool = True
                        generate_advanced_indices: bool = True
                        
                        # Output format
                        rgb_format: str = 'GeoTIFF'
                        export_metadata: bool = True
                        create_overview_images: bool = True
                        
                        # Enhancement
                        apply_contrast_enhancement: bool = True
                        contrast_method: str = 'percentile_stretch'
                        percentile_range: tuple = (2, 98)
                    
                    self.jiang_config.marine_viz_config = MarineVisualizationConfig()
                    logger.info("✓ Operational marine visualization config created with ALL FEATURES ENABLED")
                
                # Update settings from GUI variables if they exist
                if hasattr(self, 'natural_color_var'):
                    self.jiang_config.marine_viz_config.generate_natural_color = self.natural_color_var.get()
                if hasattr(self, 'false_color_var'):
                    self.jiang_config.marine_viz_config.generate_false_color = self.false_color_var.get()
                if hasattr(self, 'water_specific_var'):
                    self.jiang_config.marine_viz_config.generate_water_specific = self.water_specific_var.get()
                if hasattr(self, 'research_rgb_var'):
                    self.jiang_config.marine_viz_config.generate_research_combinations = self.research_rgb_var.get()
                if hasattr(self, 'water_quality_indices_var'):
                    self.jiang_config.marine_viz_config.generate_water_quality_indices = self.water_quality_indices_var.get()
                if hasattr(self, 'chlorophyll_indices_var'):
                    self.jiang_config.marine_viz_config.generate_chlorophyll_indices = self.chlorophyll_indices_var.get()
                if hasattr(self, 'turbidity_indices_var'):
                    self.jiang_config.marine_viz_config.generate_turbidity_indices = self.turbidity_indices_var.get()
                if hasattr(self, 'advanced_indices_var'):
                    self.jiang_config.marine_viz_config.generate_advanced_indices = self.advanced_indices_var.get()
                    
                logger.info("✓ Marine visualization settings updated from GUI")
            else:
                self.jiang_config.marine_viz_config = None
                logger.info("Marine visualization disabled by user")
            
            # ===== ADVANCED ALGORITHMS CONFIGURATION - OPERATIONAL =====
            if not hasattr(self.jiang_config, 'enable_advanced_algorithms'):
                self.jiang_config.enable_advanced_algorithms = True
                
            if hasattr(self, 'enable_advanced_var'):
                self.jiang_config.enable_advanced_algorithms = self.enable_advanced_var.get()
            else:
                self.jiang_config.enable_advanced_algorithms = True
                
            # Configure advanced algorithms
            if self.jiang_config.enable_advanced_algorithms:
                if not hasattr(self.jiang_config, 'advanced_config') or self.jiang_config.advanced_config is None:
                    @dataclass
                    class AdvancedAquaticConfig:
                        """Operational advanced aquatic algorithms configuration"""
                        enable_water_clarity: bool = True
                        solar_zenith_angle: float = 30.0
                        enable_hab_detection: bool = True
                        hab_biomass_threshold: float = 20.0
                        hab_extreme_threshold: float = 100.0
                        save_intermediate_products: bool = True
                        create_classification_maps: bool = True
                        generate_statistics: bool = True
                    
                    self.jiang_config.advanced_config = AdvancedAquaticConfig()
                    logger.info("✓ Operational advanced algorithms config created")
                
                # Update from GUI if variables exist
                if hasattr(self, 'water_clarity_var'):
                    self.jiang_config.advanced_config.enable_water_clarity = self.water_clarity_var.get()
                if hasattr(self, 'hab_detection_var'):
                    self.jiang_config.advanced_config.enable_hab_detection = self.hab_detection_var.get()
            else:
                self.jiang_config.advanced_config = None
            
            # Log final configuration status
            logger.info("Configuration update completed - Marine visualization OPERATIONAL!")
            if self.jiang_config.marine_viz_config:
                active_features = []
                if self.jiang_config.marine_viz_config.generate_natural_color:
                    active_features.append("Natural RGB")
                if self.jiang_config.marine_viz_config.generate_false_color:
                    active_features.append("False Color RGB")
                if self.jiang_config.marine_viz_config.generate_water_specific:
                    active_features.append("Water-specific RGB")
                if self.jiang_config.marine_viz_config.generate_water_quality_indices:
                    active_features.append("Water Quality Indices")
                if self.jiang_config.marine_viz_config.generate_chlorophyll_indices:
                    active_features.append("Chlorophyll Indices")
                if self.jiang_config.marine_viz_config.generate_turbidity_indices:
                    active_features.append("Turbidity Indices")
                
                logger.info(f"Active marine visualization features: {', '.join(active_features)}")
            
            return True
            
        except Exception as e:
            error_msg = f"Configuration update error: {e}"
            logger.error(error_msg)
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            messagebox.showerror("Configuration Error", f"Failed to update configurations: {str(e)}", parent=self.root)
            return False


    def patch_jiang_config_for_marine_viz(jiang_config):
        """
        Patch existing JiangTSSConfig instances to include marine visualization support.
        
        This function adds missing marine visualization attributes to JiangTSSConfig
        objects that were created with older versions of the class definition.
        
        Args:
            jiang_config: JiangTSSConfig instance to patch
            
        Returns:
            bool: True if patching was successful, False otherwise
        """
        try:
            # Check if marine visualization attributes exist, if not, add them
            if not hasattr(jiang_config, 'enable_marine_visualization'):
                logger.info("Patching JiangTSSConfig: Adding enable_marine_visualization attribute")
                jiang_config.enable_marine_visualization = True
                
            if not hasattr(jiang_config, 'marine_viz_config'):
                logger.info("Patching JiangTSSConfig: Adding marine_viz_config attribute")
                jiang_config.marine_viz_config = None
                
            # Initialize marine_viz_config if enabled and not set
            if jiang_config.enable_marine_visualization and jiang_config.marine_viz_config is None:
                try:
                    jiang_config.marine_viz_config = MarineVisualizationConfig()
                    logger.info("Patching JiangTSSConfig: Created MarineVisualizationConfig instance")
                except NameError:
                    logger.warning("MarineVisualizationConfig class not found during patching")
                    jiang_config.enable_marine_visualization = False
                    jiang_config.marine_viz_config = None
                    
            logger.info("JiangTSSConfig patching completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to patch JiangTSSConfig: {e}")
            return False


    def initialize_jiang_config_with_marine_viz():
        """
        Create a properly initialized JiangTSSConfig with marine visualization support.
        
        This function ensures that all required attributes are present and properly initialized.
        Use this when creating new JiangTSSConfig instances.
        
        Returns:
            JiangTSSConfig: Properly initialized configuration object
        """
        try:
            # Create basic config
            jiang_config = JiangTSSConfig()
            
            # Ensure marine visualization attributes exist
            if not hasattr(jiang_config, 'enable_marine_visualization'):
                jiang_config.enable_marine_visualization = True
                
            if not hasattr(jiang_config, 'marine_viz_config'):
                jiang_config.marine_viz_config = None
                
            # Initialize marine_viz_config if enabled
            if jiang_config.enable_marine_visualization and jiang_config.marine_viz_config is None:
                try:
                    jiang_config.marine_viz_config = MarineVisualizationConfig()
                    logger.info("Initialized JiangTSSConfig with MarineVisualizationConfig")
                except NameError:
                    logger.warning("MarineVisualizationConfig class not found, disabling marine visualization")
                    jiang_config.enable_marine_visualization = False
                    jiang_config.marine_viz_config = None
                    
            # Ensure advanced config is also properly initialized
            if jiang_config.enable_advanced_algorithms and jiang_config.advanced_config is None:
                try:
                    jiang_config.advanced_config = AdvancedAquaticConfig()
                    logger.info("Initialized JiangTSSConfig with AdvancedAquaticConfig")
                except NameError:
                    logger.warning("AdvancedAquaticConfig class not found, disabling advanced algorithms")
                    jiang_config.enable_advanced_algorithms = False
                    jiang_config.advanced_config = None
                    
            return jiang_config
            
        except Exception as e:
            logger.error(f"Failed to initialize JiangTSSConfig: {e}")
            # Return a basic config as fallback
            return JiangTSSConfig()
           
    def save_config(self):
        """Save configuration to file"""
        try:
            if not self.update_configurations():
                return
            
            config_file = filedialog.asksaveasfilename(
                title="Save Configuration",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                parent=self.root
            )
            
            if config_file:
                config = {
                    'processing_mode': self.processing_mode.get(),
                    'resampling': asdict(self.resampling_config),
                    'subset': asdict(self.subset_config),
                    'c2rcc': asdict(self.c2rcc_config),
                    'jiang': asdict(self.jiang_config),
                    'skip_existing': self.skip_existing_var.get(),
                    'test_mode': self.test_mode_var.get(),
                    'memory_limit': int(self.memory_limit_var.get()),
                    'thread_count': int(self.thread_count_var.get()),
                    'saved_at': datetime.now().isoformat()
                }
                
                with open(config_file, 'w') as f:
                    json.dump(config, f, indent=2)
                
                messagebox.showinfo("Success", f"Configuration saved to:\n{config_file}", parent=self.root)
                self.status_var.set("Configuration saved")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {str(e)}", parent=self.root)
    
    def load_config(self):
        """Load configuration from file - COMPLETE IMPLEMENTATION"""
        try:
            config_file = filedialog.askopenfilename(
                title="Load Configuration",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                parent=self.root
            )
            
            if not config_file:
                return
            
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            logger.info(f"Loading configuration from: {config_file}")
            
            # ===== PROCESSING MODE =====
            if 'processing_mode' in config:
                self.processing_mode.set(config['processing_mode'])
                logger.info(f"✓ Processing mode: {config['processing_mode']}")
            
            # ===== RESAMPLING CONFIGURATION =====
            if 'resampling' in config:
                resampling_data = config['resampling']
                # Reconstruct ResamplingConfig from dictionary
                self.resampling_config = ResamplingConfig(
                    target_resolution=resampling_data.get('target_resolution', '10'),
                    upsampling_method=resampling_data.get('upsampling_method', 'Bilinear'),
                    downsampling_method=resampling_data.get('downsampling_method', 'Mean'),
                    flag_downsampling=resampling_data.get('flag_downsampling', 'First'),
                    resample_on_pyramid_levels=resampling_data.get('resample_on_pyramid_levels', True)
                )
                
                # Update GUI variables
                if hasattr(self, 'target_resolution_var'):
                    self.target_resolution_var.set(self.resampling_config.target_resolution)
                if hasattr(self, 'upsampling_method_var'):
                    self.upsampling_method_var.set(self.resampling_config.upsampling_method)
                if hasattr(self, 'downsampling_method_var'):
                    self.downsampling_method_var.set(self.resampling_config.downsampling_method)
                
                logger.info(f"✓ Resampling config loaded: {self.resampling_config.target_resolution}m resolution")
            
            # ===== SUBSET CONFIGURATION =====
            if 'subset' in config:
                subset_data = config['subset']
                # Reconstruct SubsetConfig from dictionary
                self.subset_config = SubsetConfig(
                    geometry_wkt=subset_data.get('geometry_wkt'),
                    sub_sampling_x=subset_data.get('sub_sampling_x', 1),
                    sub_sampling_y=subset_data.get('sub_sampling_y', 1),
                    full_swath=subset_data.get('full_swath', False),
                    copy_metadata=subset_data.get('copy_metadata', True),
                    pixel_start_x=subset_data.get('pixel_start_x'),
                    pixel_start_y=subset_data.get('pixel_start_y'),
                    pixel_size_x=subset_data.get('pixel_size_x'),
                    pixel_size_y=subset_data.get('pixel_size_y')
                )
                
                # Update GUI variables
                if hasattr(self, 'subset_method_var'):
                    if self.subset_config.geometry_wkt:
                        self.subset_method_var.set("geometry")
                        if hasattr(self, 'geometry_text'):
                            self.geometry_text.delete(1.0, tk.END)
                            self.geometry_text.insert(1.0, self.subset_config.geometry_wkt)
                    elif self.subset_config.pixel_start_x is not None:
                        self.subset_method_var.set("pixel_coordinates")
                        if hasattr(self, 'pixel_start_x_var'):
                            self.pixel_start_x_var.set(str(self.subset_config.pixel_start_x))
                        if hasattr(self, 'pixel_start_y_var'):
                            self.pixel_start_y_var.set(str(self.subset_config.pixel_start_y))
                        if hasattr(self, 'pixel_size_x_var'):
                            self.pixel_size_x_var.set(str(self.subset_config.pixel_size_x))
                        if hasattr(self, 'pixel_size_y_var'):
                            self.pixel_size_y_var.set(str(self.subset_config.pixel_size_y))
                    else:
                        self.subset_method_var.set("no_subset")
                
                logger.info(f"✓ Subset config loaded")
            
            # ===== C2RCC CONFIGURATION =====
            if 'c2rcc' in config:
                c2rcc_data = config['c2rcc']
                # Reconstruct C2RCCConfig from dictionary
                self.c2rcc_config = C2RCCConfig(
                    salinity=c2rcc_data.get('salinity', 35.0),
                    temperature=c2rcc_data.get('temperature', 15.0),
                    ozone=c2rcc_data.get('ozone', 330.0),
                    pressure=c2rcc_data.get('pressure', 1000.0),
                    elevation=c2rcc_data.get('elevation', 0.0),
                    net_set=c2rcc_data.get('net_set', 'C2RCC-Nets'),
                    dem_name=c2rcc_data.get('dem_name', 'Copernicus 30m Global DEM'),
                    use_ecmwf_aux_data=c2rcc_data.get('use_ecmwf_aux_data', True),
                    atmospheric_aux_data_path=c2rcc_data.get('atmospheric_aux_data_path', ''),
                    alternative_nn_path=c2rcc_data.get('alternative_nn_path', ''),
                    output_as_rrs=c2rcc_data.get('output_as_rrs', True),
                    output_rhown=c2rcc_data.get('output_rhown', True),
                    output_kd=c2rcc_data.get('output_kd', True),
                    output_uncertainties=c2rcc_data.get('output_uncertainties', True),
                    output_ac_reflectance=c2rcc_data.get('output_ac_reflectance', True),
                    output_rtoa=c2rcc_data.get('output_rtoa', True),
                    output_rtosa_gc=c2rcc_data.get('output_rtosa_gc', False),
                    output_rtosa_gc_aann=c2rcc_data.get('output_rtosa_gc_aann', False),
                    output_rpath=c2rcc_data.get('output_rpath', False),
                    output_tdown=c2rcc_data.get('output_tdown', False),
                    output_tup=c2rcc_data.get('output_tup', False),
                    output_oos=c2rcc_data.get('output_oos', False),
                    derive_rw_from_path_and_transmittance=c2rcc_data.get('derive_rw_from_path_and_transmittance', False),
                    valid_pixel_expression=c2rcc_data.get('valid_pixel_expression', 'B8 > 0 && B8 < 0.1'),
                    threshold_rtosa_oos=c2rcc_data.get('threshold_rtosa_oos', 0.05),
                    threshold_ac_reflec_oos=c2rcc_data.get('threshold_ac_reflec_oos', 0.1),
                    threshold_cloud_tdown865=c2rcc_data.get('threshold_cloud_tdown865', 0.955),
                    tsm_fac=c2rcc_data.get('tsm_fac', 1.06),
                    tsm_exp=c2rcc_data.get('tsm_exp', 0.942),
                    chl_fac=c2rcc_data.get('chl_fac', 21.0),
                    chl_exp=c2rcc_data.get('chl_exp', 1.04)
                )
                
                # Update GUI variables
                self._update_c2rcc_gui_variables()
                logger.info(f"✓ C2RCC config loaded: {self.c2rcc_config.salinity}‰ salinity, {self.c2rcc_config.temperature}°C")
            
            # ===== JIANG TSS CONFIGURATION - FIXED VERSION =====
            if 'jiang' in config:
                jiang_data = config['jiang']
                
                # STEP 1: Create basic JiangTSSConfig with only essential parameters
                self.jiang_config = JiangTSSConfig(
                    enable_jiang_tss=jiang_data.get('enable_jiang_tss', True),
                    output_intermediates=jiang_data.get('output_intermediates', True),
                    water_mask_threshold=jiang_data.get('water_mask_threshold', 0.01),
                    tss_valid_range=tuple(jiang_data.get('tss_valid_range', [0.01, 10000])),
                    output_comparison_stats=jiang_data.get('output_comparison_stats', True),
                    enable_advanced_algorithms=jiang_data.get('enable_advanced_algorithms', True)
                )
                
                # STEP 2: Add marine visualization attributes safely (after creation)
                if not hasattr(self.jiang_config, 'enable_marine_visualization'):
                    self.jiang_config.enable_marine_visualization = jiang_data.get('enable_marine_visualization', True)
                
                if not hasattr(self.jiang_config, 'marine_viz_config'):
                    self.jiang_config.marine_viz_config = None
                
                # STEP 3: Reconstruct advanced_config if present
                if jiang_data.get('advanced_config') and self.jiang_config.enable_advanced_algorithms:
                    adv_data = jiang_data['advanced_config']
                    self.jiang_config.advanced_config = AdvancedAquaticConfig(
                        enable_water_clarity=adv_data.get('enable_water_clarity', True),
                        solar_zenith_angle=adv_data.get('solar_zenith_angle', 30.0),
                        enable_hab_detection=adv_data.get('enable_hab_detection', True),
                        hab_biomass_threshold=adv_data.get('hab_biomass_threshold', 20.0),
                        hab_extreme_threshold=adv_data.get('hab_extreme_threshold', 100.0),
                        save_intermediate_products=adv_data.get('save_intermediate_products', True),
                        create_classification_maps=adv_data.get('create_classification_maps', True),
                        generate_statistics=adv_data.get('generate_statistics', True)
                    )
                
                # STEP 4: Create marine visualization config safely
                if self.jiang_config.enable_marine_visualization:
                    try:
                        # Create with defaults first
                        self.jiang_config.marine_viz_config = MarineVisualizationConfig()
                        
                        # Update from saved data if available
                        if 'marine_viz_config' in jiang_data and jiang_data['marine_viz_config']:
                            viz_data = jiang_data['marine_viz_config']
                            # Set attributes directly to avoid constructor issues
                            for attr_name, attr_value in viz_data.items():
                                if hasattr(self.jiang_config.marine_viz_config, attr_name):
                                    setattr(self.jiang_config.marine_viz_config, attr_name, attr_value)
                        
                        logger.info("✓ Marine visualization config loaded from file")
                        
                    except Exception as marine_error:
                        logger.warning(f"MarineVisualizationConfig loading failed: {marine_error}")
                        # Fallback: disable marine visualization
                        self.jiang_config.enable_marine_visualization = False
                        self.jiang_config.marine_viz_config = None
                        logger.info("Marine visualization disabled due to loading error")
                else:
                    self.jiang_config.marine_viz_config = None
                
                # Update GUI variables
                self._update_jiang_gui_variables()
                logger.info(f"✓ Jiang TSS config loaded: {self.jiang_config.enable_jiang_tss}")
            
            # ===== SYSTEM SETTINGS =====
            if 'skip_existing' in config:
                self.skip_existing_var.set(config['skip_existing'])
            if 'test_mode' in config:
                self.test_mode_var.set(config['test_mode'])
            if 'memory_limit' in config:
                self.memory_limit_var.set(str(config['memory_limit']))
            if 'thread_count' in config:
                self.thread_count_var.set(str(config['thread_count']))
            
            # ===== UPDATE GUI STATE =====
            self.update_tab_visibility()
            self.update_subset_visibility()
            self.update_jiang_visibility()
            
            # Show success message
            messagebox.showinfo("Success", 
                            f"Configuration loaded successfully!\n\n"
                            f"Processing mode: {self.processing_mode.get()}\n"
                            f"Saved at: {config.get('saved_at', 'Unknown time')}", 
                            parent=self.root)
            self.status_var.set("Configuration loaded successfully")
            logger.info("✅ Configuration loading completed successfully")
            
        except FileNotFoundError:
            messagebox.showerror("Error", "Configuration file not found!", parent=self.root)
        except json.JSONDecodeError as e:
            messagebox.showerror("Error", f"Invalid JSON format: {str(e)}", parent=self.root)
        except Exception as e:
            error_msg = f"Failed to load configuration: {str(e)}"
            logger.error(error_msg)
            messagebox.showerror("Error", error_msg, parent=self.root)

    def _update_c2rcc_gui_variables(self):
        """Update C2RCC GUI variables from config object"""
        try:
            # Basic parameters
            if hasattr(self, 'salinity_var'):
                self.salinity_var.set(self.c2rcc_config.salinity)
            if hasattr(self, 'temperature_var'):
                self.temperature_var.set(self.c2rcc_config.temperature)
            if hasattr(self, 'ozone_var'):
                self.ozone_var.set(self.c2rcc_config.ozone)
            if hasattr(self, 'pressure_var'):
                self.pressure_var.set(self.c2rcc_config.pressure)
            if hasattr(self, 'elevation_var'):
                self.elevation_var.set(self.c2rcc_config.elevation)
            
            # Network and DEM
            if hasattr(self, 'net_set_var'):
                self.net_set_var.set(self.c2rcc_config.net_set)
            if hasattr(self, 'dem_name_var'):
                self.dem_name_var.set(self.c2rcc_config.dem_name)
            if hasattr(self, 'use_ecmwf_var'):
                self.use_ecmwf_var.set(self.c2rcc_config.use_ecmwf_aux_data)
            
            # Output products
            if hasattr(self, 'output_rrs_var'):
                self.output_rrs_var.set(self.c2rcc_config.output_as_rrs)
            if hasattr(self, 'output_rhow_var'):
                self.output_rhow_var.set(self.c2rcc_config.output_rhown)
            if hasattr(self, 'output_kd_var'):
                self.output_kd_var.set(self.c2rcc_config.output_kd)
            if hasattr(self, 'output_uncertainties_var'):
                self.output_uncertainties_var.set(self.c2rcc_config.output_uncertainties)
            if hasattr(self, 'output_ac_reflectance_var'):
                self.output_ac_reflectance_var.set(self.c2rcc_config.output_ac_reflectance)
            if hasattr(self, 'output_rtoa_var'):
                self.output_rtoa_var.set(self.c2rcc_config.output_rtoa)
            
            # Advanced products
            if hasattr(self, 'output_rtosa_gc_var'):
                self.output_rtosa_gc_var.set(self.c2rcc_config.output_rtosa_gc)
            if hasattr(self, 'output_rtosa_gc_aann_var'):
                self.output_rtosa_gc_aann_var.set(self.c2rcc_config.output_rtosa_gc_aann)
            if hasattr(self, 'output_rpath_var'):
                self.output_rpath_var.set(self.c2rcc_config.output_rpath)
            if hasattr(self, 'output_tdown_var'):
                self.output_tdown_var.set(self.c2rcc_config.output_tdown)
            if hasattr(self, 'output_tup_var'):
                self.output_tup_var.set(self.c2rcc_config.output_tup)
            if hasattr(self, 'output_oos_var'):
                self.output_oos_var.set(self.c2rcc_config.output_oos)
            
            # Advanced parameters
            if hasattr(self, 'valid_pixel_var'):
                self.valid_pixel_var.set(self.c2rcc_config.valid_pixel_expression)
            if hasattr(self, 'threshold_rtosa_oos_var'):
                self.threshold_rtosa_oos_var.set(self.c2rcc_config.threshold_rtosa_oos)
            if hasattr(self, 'threshold_ac_reflec_oos_var'):
                self.threshold_ac_reflec_oos_var.set(self.c2rcc_config.threshold_ac_reflec_oos)
            if hasattr(self, 'threshold_cloud_tdown865_var'):
                self.threshold_cloud_tdown865_var.set(self.c2rcc_config.threshold_cloud_tdown865)
            
            # TSM/CHL parameters
            if hasattr(self, 'tsm_fac_var'):
                self.tsm_fac_var.set(self.c2rcc_config.tsm_fac)
            if hasattr(self, 'tsm_exp_var'):
                self.tsm_exp_var.set(self.c2rcc_config.tsm_exp)
            if hasattr(self, 'chl_fac_var'):
                self.chl_fac_var.set(self.c2rcc_config.chl_fac)
            if hasattr(self, 'chl_exp_var'):
                self.chl_exp_var.set(self.c2rcc_config.chl_exp)
            
            logger.debug("✓ C2RCC GUI variables updated from config")
        except Exception as e:
            logger.warning(f"Error updating C2RCC GUI variables: {e}")

    def _update_jiang_gui_variables(self):
        """Update Jiang TSS GUI variables from config object"""
        try:
            # Basic Jiang settings
            if hasattr(self, 'enable_jiang_var'):
                self.enable_jiang_var.set(self.jiang_config.enable_jiang_tss)
            if hasattr(self, 'jiang_intermediates_var'):
                self.jiang_intermediates_var.set(self.jiang_config.output_intermediates)
            if hasattr(self, 'jiang_comparison_var'):
                self.jiang_comparison_var.set(self.jiang_config.output_comparison_stats)
            
            # Advanced algorithms
            if hasattr(self, 'enable_advanced_var'):
                self.enable_advanced_var.set(self.jiang_config.enable_advanced_algorithms)
            
            # Marine visualization
            if hasattr(self, 'enable_marine_viz_var'):
                self.enable_marine_viz_var.set(self.jiang_config.enable_marine_visualization)
            
            # Marine visualization options
            if self.jiang_config.marine_viz_config:
                viz_config = self.jiang_config.marine_viz_config
                if hasattr(self, 'natural_color_var'):
                    self.natural_color_var.set(viz_config.generate_natural_color)
                if hasattr(self, 'false_color_var'):
                    self.false_color_var.set(viz_config.generate_false_color)
                if hasattr(self, 'water_specific_var'):
                    self.water_specific_var.set(viz_config.generate_water_specific)
                if hasattr(self, 'research_rgb_var'):
                    self.research_rgb_var.set(viz_config.generate_research_combinations)
                if hasattr(self, 'water_quality_indices_var'):
                    self.water_quality_indices_var.set(viz_config.generate_water_quality_indices)
                if hasattr(self, 'chlorophyll_indices_var'):
                    self.chlorophyll_indices_var.set(viz_config.generate_chlorophyll_indices)
                if hasattr(self, 'turbidity_indices_var'):
                    self.turbidity_indices_var.set(viz_config.generate_turbidity_indices)
                if hasattr(self, 'advanced_indices_var'):
                    self.advanced_indices_var.set(viz_config.generate_advanced_indices)
            
            # Advanced algorithms options
            if self.jiang_config.advanced_config:
                adv_config = self.jiang_config.advanced_config
                if hasattr(self, 'water_clarity_var'):
                    self.water_clarity_var.set(adv_config.enable_water_clarity)
                if hasattr(self, 'hab_detection_var'):
                    self.hab_detection_var.set(adv_config.enable_hab_detection)
            
            logger.debug("✓ Jiang TSS GUI variables updated from config")
        except Exception as e:
            logger.warning(f"Error updating Jiang TSS GUI variables: {e}")
    
    def start_processing(self):
        """Start processing in background thread"""
        if self.processing_active:
            return
            
        output_folder = self.output_dir_var.get()
        global logger
        logger, log_file = setup_enhanced_logging(log_level=logging.INFO, output_folder=output_folder)
        logger.info("🚀 Processing started - logs redirected to output folder")
        logger.info(f"📋 Log file: {log_file}")
        
        try:
            # Validate configuration
            if not self.input_validation_result["valid"]:
                messagebox.showerror("Error", "Please fix input validation errors first", parent=self.root)
                return
            
            if not self.output_dir_var.get():
                messagebox.showerror("Error", "Please select output directory", parent=self.root)
                return
            
            if not self.update_configurations():
                return
            
            # Create processing configuration
            processing_config = ProcessingConfig(
                processing_mode=ProcessingMode(self.processing_mode.get()),
                input_folder=self.input_dir_var.get(),
                output_folder=self.output_dir_var.get(),
                resampling_config=self.resampling_config,
                subset_config=self.subset_config,
                c2rcc_config=self.c2rcc_config,
                jiang_config=self.jiang_config,
                skip_existing=self.skip_existing_var.get(),
                test_mode=self.test_mode_var.get(),
                memory_limit_gb=int(self.memory_limit_var.get()),
                thread_count=int(self.thread_count_var.get())
            )
            
            # Confirm processing
            products = self.input_validation_result["products"]
            process_count = len(products)
            if processing_config.test_mode:
                process_count = min(2, process_count)
            
            mode_name = processing_config.processing_mode.value.replace('_', ' ').title()
            
            # Build confirmation message
            confirm_msg = f"Start {mode_name} processing?\n\n"
            confirm_msg += f"Products found: {len(products)}\n"
            confirm_msg += f"Will process: {process_count} products\n"
            confirm_msg += f"Mode: {mode_name}\n"
            confirm_msg += f"Output: {processing_config.output_folder}\n"
            confirm_msg += f"ECMWF: {'Enabled' if processing_config.c2rcc_config.use_ecmwf_aux_data else 'Disabled'}\n"
            
            if processing_config.processing_mode in [ProcessingMode.COMPLETE_PIPELINE, ProcessingMode.TSS_PROCESSING_ONLY]:
                if processing_config.jiang_config.enable_jiang_tss:
                    confirm_msg += f"Jiang TSS: Enabled\n"
                else:
                    confirm_msg += f"Jiang TSS: Disabled (SNAP TSM/CHL only)\n"
            
            confirm_msg += f"\nProceed?"
            
            proceed = messagebox.askyesno("Confirm Processing", confirm_msg, parent=self.root)
            
            if not proceed:
                return
            
            # Start processing thread
            self.processing_active = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.status_var.set("Starting processing...")
            
            # Start processing in background thread
            self.processing_thread = threading.Thread(
                target=self.run_processing_thread, 
                args=(processing_config, products),
                daemon=True
            )
            self.processing_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start processing: {str(e)}", parent=self.root)
            self.processing_active = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
    
    def run_processing_thread(self, config, products):
        """Run processing in background thread"""
        try:
            # Create processor
            self.processor = UnifiedS2TSSProcessor(config)
            
            # Limit products for test mode
            if config.test_mode:
                products = products[:1]
                logger.info(f"TEST MODE: Processing only {len(products)} products")
            
            # Process products
            total_products = len(products)
            
            for i, product_path in enumerate(products):
                if not self.processing_active:  # Check for stop signal
                    break
                
                # Update status
                product_name = os.path.basename(product_path)
                self.status_var.set(f"Processing {i+1}/{total_products}: {product_name}")
                
                # Process product
                try:
                    self.processor._process_single_product(product_path, i+1, total_products)
                except Exception as e:
                    logger.error(f"Error processing {product_name}: {e}")
                
                # Update progress
                progress = ((i + 1) / total_products) * 100
                self.progress_var.set(progress)
                
                # Update ETA
                status = self.processor.get_processing_status()
                if status.eta_minutes > 0:
                    self.eta_var.set(f"ETA: {status.eta_minutes:.1f} min | Speed: {status.processing_speed:.1f} products/min")
                else:
                    self.eta_var.set("")
            
            # Processing completed
            self.on_processing_complete()
            
        except Exception as e:
            logger.error(f"Processing thread error: {e}")
            self.status_var.set(f"Processing error: {str(e)}")
        finally:
            self.processing_active = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            if self.processor:
                self.processor.cleanup()
    
    def stop_processing(self):
        """Stop processing"""
        if self.processing_active:
            self.processing_active = False
            self.status_var.set("Stopping processing...")
            self.stop_button.config(state=tk.DISABLED)
    
    def on_processing_complete(self):
        """Handle processing completion"""
        if self.processor:
            status = self.processor.get_processing_status()
            
            # Final status
            self.status_var.set("Processing completed!")
            self.progress_var.set(100)
            
            # Show completion message
            total_time = (time.time() - self.processor.start_time) / 60
            
            completion_msg = (
                f"Processing completed!\n\n"
                f"Successfully processed: {status.processed}\n"
                f"Failed: {status.failed}\n"
                f"Skipped: {status.skipped}\n"
                f"Total time: {total_time:.1f} minutes\n\n"
                f"Outputs saved to:\n{self.output_dir_var.get()}\n\n"
                f"Check log file for details."
            )
            
            messagebox.showinfo("Processing Complete", completion_msg, parent=self.root)
    
    def start_gui_updates(self):
        """Start GUI update loop"""
        self.update_system_info()
        self.update_processing_stats()
        self.root.after(2000, self.start_gui_updates)  # Update every 2 seconds
    
    def update_system_info(self):
        """Update system information display with error protection"""
        try:
            info = self.system_monitor.get_current_info()
            
            # Update system labels
            self.cpu_label.config(text=f"CPU: {info['cpu_percent']:.1f}%")
            
            # Fix: Protect against division by zero
            if info['memory_total_gb'] > 0:
                memory_percent = (info['memory_used_gb'] / info['memory_total_gb']) * 100
            else:
                memory_percent = 0.0
                
            self.memory_label.config(text=f"Memory: {info['memory_used_gb']:.1f}/{info['memory_total_gb']:.1f} GB ({memory_percent:.1f}%)")
            
            self.disk_label.config(text=f"Disk Free: {info['disk_free_gb']:.1f} GB")
            
            # SNAP status
            snap_home = os.environ.get('SNAP_HOME', 'Not set')
            self.snap_label.config(text=f"SNAP: {snap_home}")
            
            # Color coding for warnings
            if memory_percent > 90:
                self.memory_label.config(foreground="red")
            elif memory_percent > 75:
                self.memory_label.config(foreground="orange")
            else:
                self.memory_label.config(foreground="black")
            
            if info['disk_free_gb'] < 10:
                self.disk_label.config(foreground="red")
            elif info['disk_free_gb'] < 50:
                self.disk_label.config(foreground="orange")
            else:
                self.disk_label.config(foreground="black")
                
        except Exception as e:
            logger.warning(f"GUI update error: {e}")
            # Set default values to prevent repeated errors
            try:
                self.cpu_label.config(text="CPU: --")
                self.memory_label.config(text="Memory: --")
                self.disk_label.config(text="Disk: --")
            except:
                pass
    
    def update_processing_stats(self):
        """Update processing statistics display"""
        if self.processor and self.processing_active:
            try:
                status = self.processor.get_processing_status()
                
                # Update stats text
                stats_text = (
                    f"Processing Statistics:\n"
                    f"{'='*30}\n"
                    f"Total Products: {status.total_products}\n"
                    f"Processed: {status.processed}\n"
                    f"Failed: {status.failed}\n"
                    f"Skipped: {status.skipped}\n"
                    f"Progress: {status.progress_percent:.1f}%\n"
                    f"Current: {status.current_product}\n"
                    f"Stage: {status.current_stage}\n"
                    f"ETA: {status.eta_minutes:.1f} minutes\n"
                    f"Speed: {status.processing_speed:.2f} products/min\n"
                    f"\nSystem Health:\n"
                    f"{'='*15}\n"
                )
                
                healthy, warnings = self.system_monitor.check_system_health()
                if healthy:
                    stats_text += "✓ System healthy\n"
                else:
                    stats_text += "⚠ System warnings:\n"
                    for warning in warnings:
                        stats_text += f"  - {warning}\n"
                
                # Update text widget
                self.stats_text.config(state=tk.NORMAL)
                self.stats_text.delete(1.0, tk.END)
                self.stats_text.insert(1.0, stats_text)
                self.stats_text.config(state=tk.DISABLED)
                
            except Exception as e:
                logger.warning(f"Stats update error: {e}")
    
    def on_closing(self):
        """Handle application closing"""
        if self.processing_active:
            result = messagebox.askyesno(
                "Confirm Exit", 
                "Processing is active. Stop processing and exit?",
                parent=self.root
            )
            if result:
                self.stop_processing()
                time.sleep(1)  # Give time to stop
            else:
                return
        
        # Cleanup
        try:
            self.system_monitor.stop_monitoring()
            if self.processor:
                self.processor.cleanup()
        except:
            pass
        
        self.root.destroy()
    
    def run(self):
        """Run the GUI application"""
        try:
            # Set default directories if available
            default_paths = [
                (r"D:\GSEU_WP5\Imagens\2019", "input"),
                (r"D:\GSEU_WP5\Processed", "output"),
                (r"D:\GSEU_WP5\C2RCC", "input"),  # For TSS-only mode
                (r"D:\GSEU_WP5\C2RCC_Output", "output")
            ]
            
            for path, path_type in default_paths:
                if os.path.exists(path):
                    if path_type == "input" and not self.input_dir_var.get():
                        self.input_dir_var.set(path)
                    elif path_type == "output" and not self.output_dir_var.get():
                        self.output_dir_var.set(path)
                    break
            
            # Protocol for window closing
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            
            # Start main loop
            self.root.mainloop()
            
        except Exception as e:
            logger.error(f"GUI error: {str(e)}")
            messagebox.showerror("Application Error", f"GUI error: {str(e)}")
        finally:
            # Ensure cleanup
            try:
                self.system_monitor.stop_monitoring()
            except:
                pass

# ===== CONFIGURATION HELPER =====

def create_default_config(input_folder: str, output_folder: str, 
                         processing_mode: str = "complete_pipeline") -> ProcessingConfig:
    """Create default processing configuration"""
    
    # Validate processing mode
    try:
        mode = ProcessingMode(processing_mode.lower())
    except ValueError:
        logger.warning(f"Invalid processing mode '{processing_mode}', using 'complete_pipeline'")
        mode = ProcessingMode.COMPLETE_PIPELINE
    
    # Create configurations with defaults
    resampling_config = ResamplingConfig()
    subset_config = SubsetConfig()
    c2rcc_config = C2RCCConfig()  # ECMWF enabled by default
    jiang_config = JiangTSSConfig()
    jiang_config.enable_advanced_algorithms = True  # Ensure this is set
    jiang_config.advanced_config = AdvancedAquaticConfig()  # Ensure this is set
    
    return ProcessingConfig(
        processing_mode=mode,
        input_folder=input_folder,
        output_folder=output_folder,
        resampling_config=resampling_config,
        subset_config=subset_config,
        c2rcc_config=c2rcc_config,
        jiang_config=jiang_config,  # Use the properly initialized config
        skip_existing=True,
        test_mode=False,
        memory_limit_gb=8,
        thread_count=4
    )

# ===== COMMAND LINE INTERFACE =====

def cli_main():
    """Command line interface for batch processing"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Unified S2 Processing & TSS Estimation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python unified_s2_tss_pipeline.py -i D:/L1C_Products -o D:/Results
  python unified_s2_tss_pipeline.py -i /path/to/l1c -o /path/to/results --mode s2_processing_only
  python unified_s2_tss_pipeline.py -i ./c2rcc_products -o ./tss_results --mode tss_processing_only --enable-jiang
        """
    )
    
    parser.add_argument("-i", "--input", required=True,
                       help="Input folder containing L1C products (.zip/.SAFE) or C2RCC products (.dim)")
    
    parser.add_argument("-o", "--output", required=True,
                       help="Output folder for results")
    
    parser.add_argument("--mode", choices=["complete_pipeline", "s2_processing_only", "tss_processing_only"], 
                       default="complete_pipeline",
                       help="Processing mode (default: complete_pipeline)")
    
    parser.add_argument("--no-skip", action="store_true",
                       help="Process all products (don't skip existing outputs)")
    
    parser.add_argument("--enable-jiang", action="store_true",
                       help="Enable Jiang TSS methodology (in addition to automatic SNAP TSM/CHL)")
    
    parser.add_argument("--no-ecmwf", action="store_true",
                       help="Disable ECMWF auxiliary data (reduces accuracy)")
    
    parser.add_argument("--test", action="store_true",
                       help="Test mode (process only 1 product)")
    
    parser.add_argument("--memory-limit", type=int, default=8,
                       help="Memory limit in GB (default: 8)")
    
    parser.add_argument("--threads", type=int, default=4,
                       help="Number of processing threads (default: 4)")
    
    parser.add_argument("--resolution", choices=["10", "20", "60"], default="10",
                       help="Target resolution in meters (default: 10)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.input):
        print(f"ERROR: Input folder does not exist: {args.input}")
        return False
    
    # Create output folder
    try:
        os.makedirs(args.output, exist_ok=True)
    except Exception as e:
        print(f"ERROR: Cannot create output folder: {e}")
        return False
    
    # Create configuration from arguments
    c2rcc_config = C2RCCConfig()
    c2rcc_config.use_ecmwf_aux_data = not args.no_ecmwf  # ECMWF enabled by default unless disabled
    
    jiang_config = JiangTSSConfig()
    jiang_config.enable_jiang_tss = args.enable_jiang
    
    resampling_config = ResamplingConfig()
    resampling_config.target_resolution = args.resolution
    
    config = ProcessingConfig(
        processing_mode=ProcessingMode(args.mode),
        input_folder=args.input,
        output_folder=args.output,
        resampling_config=resampling_config,
        subset_config=SubsetConfig(),
        c2rcc_config=c2rcc_config,
        jiang_config=jiang_config,
        skip_existing=not args.no_skip,
        test_mode=args.test,
        memory_limit_gb=args.memory_limit,
        thread_count=args.threads
    )
    
    # Print configuration
    print("="*80)
    print("UNIFIED S2 PROCESSING & TSS ESTIMATION PIPELINE")
    print("="*80)
    print(f"Input folder: {config.input_folder}")
    print(f"Output folder: {config.output_folder}")
    print(f"Processing mode: {config.processing_mode.value}")
    print(f"Resolution: {config.resampling_config.target_resolution}m")
    print(f"ECMWF: {'Enabled' if config.c2rcc_config.use_ecmwf_aux_data else 'Disabled'}")
    print(f"Jiang TSS: {'Enabled' if config.jiang_config.enable_jiang_tss else 'Disabled'}")
    print(f"Skip existing: {config.skip_existing}")
    print(f"Test mode: {config.test_mode}")
    print(f"Memory limit: {config.memory_limit_gb} GB")
    print()
    
    # Run processing
    try:
        processor = UnifiedS2TSSProcessor(config)
        results = processor.process_batch()
        
        # Print results
        print("\n" + "="*80)
        print("PROCESSING COMPLETED")
        print("="*80)
        print(f"Successfully processed: {results['processed']}")
        print(f"Skipped (existing): {results['skipped']}")
        print(f"Failed: {results['failed']}")
        
        processor.cleanup()
        return results['failed'] == 0
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        return False
    except Exception as e:
        print(f"\nProcessing failed: {str(e)}")
        logger.error(f"CLI processing failed: {str(e)}")
        return False

# ===== MAIN FUNCTION =====

def main():
    """Main entry point with enhanced error handling"""
    try:
        # Check Python version
        if sys.version_info < (3, 6):
            print("ERROR: Python 3.6 or higher is required!")
            sys.exit(1)
        
        # Check for required dependencies
        missing_deps = []
        try:
            import numpy as np
            print("✓ NumPy available")
        except ImportError:
            missing_deps.append("numpy")
        
        try:
            from osgeo import gdal
            print("✓ GDAL available")
        except ImportError:
            missing_deps.append("gdal")
        
        try:
            import psutil
            print("✓ psutil available")
        except ImportError:
            missing_deps.append("psutil")
        
        if missing_deps:
            print(f"\nERROR: Missing required dependencies: {missing_deps}")
            print("Install with:")
            for dep in missing_deps:
                if dep == "gdal":
                    print(f"  conda install {dep}")
                else:
                    print(f"  pip install {dep}")
            sys.exit(1)
        
        # Set environment variable if not set (Windows default)
        if not os.environ.get('SNAP_HOME'):
            default_snap_paths = [
                r"C:\Program Files\esa-snap",
                r"C:\Program Files (x86)\esa-snap",
                r"D:\Program Files\esa-snap"
            ]
            
            snap_found = False
            for snap_path in default_snap_paths:
                if os.path.exists(snap_path):
                    os.environ['SNAP_HOME'] = snap_path
                    logger.info(f"Auto-detected SNAP_HOME: {snap_path}")
                    snap_found = True
                    break
            
            if not snap_found:
                logger.error("SNAP_HOME not set and SNAP installation not found!")
                logger.error("Please install SNAP or set SNAP_HOME environment variable")
                
                # Show error dialog if running GUI
                if len(sys.argv) == 1:  # No command line arguments = GUI mode
                    root = tk.Tk()
                    root.withdraw()
                    messagebox.showerror(
                        "SNAP Not Found",
                        "SNAP installation not found!\n\n"
                        "Please:\n"
                        "1. Install SNAP from https://step.esa.int/\n"
                        "2. Or set SNAP_HOME environment variable\n"
                        "3. Restart this application"
                    )
                sys.exit(1)
        
        # Verify SNAP installation
        snap_home = os.environ.get('SNAP_HOME')
        gpt_path = os.path.join(snap_home, 'bin', 'gpt.exe' if sys.platform.startswith('win') else 'gpt')
        
        if not os.path.exists(gpt_path):
            logger.error(f"GPT not found at: {gpt_path}")
            logger.error("Please check your SNAP installation")
            
            if len(sys.argv) == 1:  # GUI mode
                root = tk.Tk()
                root.withdraw()
                messagebox.showerror(
                    "SNAP Configuration Error",
                    f"GPT executable not found at:\n{gpt_path}\n\n"
                    "Please check your SNAP installation"
                )
            sys.exit(1)
        
        # Log startup information
        logger.info("="*80)
        logger.info("UNIFIED S2 PROCESSING & TSS ESTIMATION PIPELINE v1.0")
        logger.info("="*80)
        logger.info(f"SNAP_HOME: {snap_home}")
        logger.info(f"GPT: {gpt_path}")
        logger.info(f"Python: {sys.version}")
        logger.info(f"Platform: {sys.platform}")
        logger.info(f"Working Directory: {os.getcwd()}")
        logger.info("="*80)
        
        # Check if command line arguments provided
        if len(sys.argv) > 1:
            # Run CLI interface
            success = cli_main()
            sys.exit(0 if success else 1)
        else:
            # Run GUI interface
            logger.info("Starting GUI application...")
            app = UnifiedS2TSSGUI()
            app.run()
        
        logger.info("Application finished successfully")
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Critical application error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Show critical error dialog for GUI mode
        try:
            if len(sys.argv) == 1:  # GUI mode
                root = tk.Tk()
                root.withdraw()
                messagebox.showerror(
                    "Critical Error",
                    f"A critical error occurred:\n\n{str(e)}\n\n"
                    "Check the log file for details."
                )
        except:
            pass
        
        sys.exit(1)

# ===== ENTRY POINT =====
if __name__ == "__main__":
    main()