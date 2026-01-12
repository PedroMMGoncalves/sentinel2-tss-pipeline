"""
Memory management utilities for Sentinel-2 TSS Pipeline.
"""

import gc
import logging

logger = logging.getLogger('sentinel2_tss_pipeline')

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class MemoryManager:
    """Memory management utilities"""

    @staticmethod
    def cleanup_variables(*variables):
        """
        Clean up variables and force garbage collection.

        Args:
            *variables: Variables to clean up
        """
        for var in variables:
            if var is not None:
                try:
                    del var
                except:
                    pass
        gc.collect()

    @staticmethod
    def monitor_memory(threshold_mb=8000):
        """
        Monitor memory usage and warn if above threshold.

        Args:
            threshold_mb: Warning threshold in MB

        Returns:
            True if memory usage is above threshold
        """
        if not PSUTIL_AVAILABLE:
            return False

        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024

            if memory_mb > threshold_mb:
                logger.warning(f"High memory usage: {memory_mb:.1f} MB")
                return True
            return False
        except:
            return False

    @staticmethod
    def get_memory_usage_mb():
        """
        Get current memory usage in MB.

        Returns:
            Memory usage in MB, or -1 if not available
        """
        if not PSUTIL_AVAILABLE:
            return -1

        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return -1
