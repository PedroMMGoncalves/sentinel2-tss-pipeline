"""
Memory management utilities for Sentinel-2 TSS Pipeline.
"""

import gc
import logging

logger = logging.getLogger('ocean_rs')

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class MemoryManager:
    """Memory management utilities"""

    # Cache the psutil.Process() handle for the current process to avoid
    # re-creating it on every monitoring call.
    _process = None

    @classmethod
    def _get_process(cls):
        """Return cached psutil.Process() for the current process."""
        if cls._process is None and PSUTIL_AVAILABLE:
            cls._process = psutil.Process()
        return cls._process

    @classmethod
    def monitor_memory(cls, threshold_mb=8000):
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
            process = cls._get_process()
            memory_mb = process.memory_info().rss / 1024 / 1024

            if memory_mb > threshold_mb:
                logger.warning(f"High memory usage: {memory_mb:.1f} MB")
                return True
            return False
        except Exception as e:
            logger.debug(f"Memory monitoring unavailable: {e}")
            return False

    @classmethod
    def get_memory_usage_mb(cls):
        """
        Get current memory usage in MB.

        Returns:
            Memory usage in MB, or -1 if not available
        """
        if not PSUTIL_AVAILABLE:
            return -1

        try:
            process = cls._get_process()
            return process.memory_info().rss / 1024 / 1024
        except Exception as e:
            logger.debug(f"Memory monitoring unavailable: {e}")
            return -1
