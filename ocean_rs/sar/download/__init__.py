"""
Download module for OceanRS SAR Bathymetry Toolkit.

Provides scene discovery (ASF), batch download, and credential management.
"""

from .credentials import CredentialManager, CredentialError
from .scene_discovery import SceneMetadata, search_scenes
from .batch_downloader import BatchDownloader

__all__ = [
    'CredentialManager',
    'CredentialError',
    'SceneMetadata',
    'search_scenes',
    'BatchDownloader',
]
