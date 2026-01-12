"""
Product detection and system monitoring utilities.

Provides smart product type detection and real-time system monitoring.
"""

import os
import threading
import time
import logging
from typing import Dict, List, Tuple

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from ..config.enums import ProductType, ProcessingMode

logger = logging.getLogger('sentinel2_tss_pipeline')


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
        if not PSUTIL_AVAILABLE:
            logger.warning("psutil not available, system monitoring disabled")
            return
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

        # Handle case where monitoring hasn't started
        if info['memory_total_gb'] == 0:
            return True, []

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
