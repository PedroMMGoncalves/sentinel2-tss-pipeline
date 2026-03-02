"""
Download and credential configuration for SAR data acquisition.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DownloadConfig:
    """Configuration for SAR scene download."""
    download_directory: str = ""
    max_concurrent: int = 2
    retry_count: int = 3
    resume_downloads: bool = True
    timeout_seconds: int = 300


@dataclass
class SearchConfig:
    """Configuration for SAR scene search."""
    platform: str = "Sentinel-1"
    beam_mode: str = "IW"
    polarization: str = "VV+VH"
    orbit_direction: str = ""
    processing_level: str = "SLC"
    aoi_wkt: Optional[str] = None
    start_date: str = ""
    end_date: str = ""
