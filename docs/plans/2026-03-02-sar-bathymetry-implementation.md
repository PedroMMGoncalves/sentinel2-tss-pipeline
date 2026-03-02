# OceanRS SAR Bathymetry Toolkit — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build Phase 1 of the SAR Bathymetry Toolkit — scene search/download, S1 preprocessing, FFT swell extraction, depth inversion, and a 4-tab tkinter GUI.

**Architecture:** Hybrid domain structure within `ocean_rs/sar/` using OceanRS patterns (config dataclasses, shared logging, GUI tab factories). The bathymetry pipeline is sensor-agnostic via the `OceanImage` contract — sensor adapters produce `OceanImage`, the pipeline consumes it. GUI follows the same patterns as `ocean_rs/optical/gui/`.

**Tech Stack:** Python 3.8+, numpy, scipy, asf_search, python-dotenv, requests, GDAL/rasterio, SNAP GPT, tkinter

**Reference files for patterns:**
- Config dataclass: `ocean_rs/optical/config/processing_config.py`
- Enum: `ocean_rs/optical/config/enums.py`
- Main entry: `ocean_rs/optical/main.py`
- GUI class: `ocean_rs/optical/gui/unified_gui.py`
- Theme: `ocean_rs/optical/gui/theme.py`
- Tab factory: `ocean_rs/optical/gui/tabs/processing_tab.py`
- Handlers: `ocean_rs/optical/gui/handlers.py`
- Config I/O: `ocean_rs/optical/gui/config_io.py`
- Processing controller: `ocean_rs/optical/gui/processing_controller.py`
- Shared utils: `ocean_rs/shared/__init__.py`

---

## Task 1: Data Models & Enums

**Files:**
- Create: `ocean_rs/sar/core/data_models.py`
- Create: `ocean_rs/sar/core/__init__.py`

**Step 1: Create `ocean_rs/sar/core/data_models.py`**

```python
"""
Data models for SAR Bathymetry Toolkit.

Sensor-agnostic containers for SAR imagery, swell fields, and bathymetry results.
The OceanImage contract decouples sensor adapters from the bathymetry pipeline.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np


class ImageType(Enum):
    """SAR image type hierarchy. Higher value = better for bathymetry."""
    SIGMA0 = 1          # Intensity (any SAR sensor)
    PSEUDO_ALPHA = 2    # Dual-pol decomposition (Sentinel-1, ALOS-2)
    ALPHA = 3           # Quad-pol decomposition (RADARSAT-2, NISAR)


@dataclass
class GeoTransform:
    """Geospatial reference for raster data."""
    origin_x: float          # Top-left X coordinate
    origin_y: float          # Top-left Y coordinate
    pixel_size_x: float      # Pixel width (meters or degrees)
    pixel_size_y: float      # Pixel height (negative for north-up)
    crs_wkt: str              # Coordinate reference system as WKT
    rows: int = 0
    cols: int = 0


@dataclass
class OceanImage:
    """Sensor-agnostic SAR image container.

    All sensor adapters produce OceanImage instances.
    The bathymetry pipeline consumes them without knowing the source sensor.
    """
    data: np.ndarray              # 2D array (rows x cols)
    image_type: ImageType         # ALPHA | PSEUDO_ALPHA | SIGMA0
    geo: GeoTransform             # Spatial reference
    metadata: dict = field(default_factory=dict)  # sensor, orbit, datetime, etc.
    pixel_spacing_m: float = 10.0  # Ground resolution in meters


@dataclass
class SwellField:
    """FFT-derived swell parameters per tile.

    Each element corresponds to one spatial tile from the FFT analysis.
    """
    wavelength: np.ndarray        # Dominant wavelength per tile (meters)
    direction: np.ndarray         # Wave direction per tile (degrees from north)
    confidence: np.ndarray        # Spectral peak SNR per tile (0-1)
    tile_centers_x: np.ndarray    # Tile center X coordinates (geo)
    tile_centers_y: np.ndarray    # Tile center Y coordinates (geo)
    tile_size_m: float = 512.0    # Tile size used (meters)
    geo: Optional[GeoTransform] = None


@dataclass
class BathymetryResult:
    """Inverted depth map with uncertainty.

    Depth is positive downward (oceanographic convention).
    """
    depth: np.ndarray             # Depth in meters (positive down)
    uncertainty: np.ndarray       # Depth uncertainty in meters
    method: str = "linear_dispersion"
    wave_period: float = 0.0      # Wave period used (seconds)
    wave_period_source: str = "wavewatch3"  # "wavewatch3" | "manual"
    geo: Optional[GeoTransform] = None
    metadata: dict = field(default_factory=dict)
```

**Step 2: Create `ocean_rs/sar/core/__init__.py`**

```python
"""
Core module for OceanRS SAR Bathymetry Toolkit.

Contains data models and the main bathymetry pipeline orchestrator.
"""

from .data_models import (
    ImageType,
    GeoTransform,
    OceanImage,
    SwellField,
    BathymetryResult,
)

__all__ = [
    'ImageType',
    'GeoTransform',
    'OceanImage',
    'SwellField',
    'BathymetryResult',
]
```

**Step 3: Verify syntax**

Run: `python -m py_compile ocean_rs/sar/core/data_models.py`
Run: `python -m py_compile ocean_rs/sar/core/__init__.py`
Expected: No errors

**Step 4: Commit**

```bash
git add ocean_rs/sar/core/
git commit -m "feat(sar): add core data models (OceanImage, SwellField, BathymetryResult)"
```

---

## Task 2: Configuration Dataclasses

**Files:**
- Create: `ocean_rs/sar/config/sar_config.py`
- Create: `ocean_rs/sar/config/download_config.py`
- Create: `ocean_rs/sar/config/__init__.py`

**Step 1: Create `ocean_rs/sar/config/download_config.py`**

```python
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
    orbit_direction: str = ""  # "" = any, "ASCENDING", "DESCENDING"
    processing_level: str = "SLC"
    aoi_wkt: Optional[str] = None
    start_date: str = ""
    end_date: str = ""
```

**Step 2: Create `ocean_rs/sar/config/sar_config.py`**

```python
"""
SAR Processing Configuration.

Master configuration dataclass following OceanRS pattern
(see ocean_rs/optical/config/processing_config.py).
"""

from dataclasses import dataclass
from .download_config import DownloadConfig, SearchConfig


@dataclass
class FFTConfig:
    """FFT swell extraction parameters."""
    tile_size_m: float = 512.0
    overlap: float = 0.5
    min_wavelength_m: float = 50.0
    max_wavelength_m: float = 600.0
    confidence_threshold: float = 0.3
    window_function: str = "hann"


@dataclass
class DepthInversionConfig:
    """Depth inversion parameters."""
    max_depth_m: float = 100.0
    wave_period_source: str = "wavewatch3"  # "wavewatch3" | "manual"
    manual_wave_period: float = 10.0        # seconds (used if source="manual")
    gravity: float = 9.81
    max_iterations: int = 50
    convergence_tol: float = 1e-6


@dataclass
class CompositingConfig:
    """Multi-temporal compositing parameters."""
    enabled: bool = True
    method: str = "weighted_median"  # "weighted_median" | "weighted_mean"


@dataclass
class SARProcessingConfig:
    """Complete SAR processing configuration.

    Follows the OceanRS pattern: master config contains nested config objects.
    """
    search_config: SearchConfig = None
    download_config: DownloadConfig = None
    fft_config: FFTConfig = None
    depth_config: DepthInversionConfig = None
    compositing_config: CompositingConfig = None
    snap_gpt_path: str = ""
    output_directory: str = ""
    export_geotiff: bool = True
    export_png: bool = True
    memory_limit_gb: int = 8

    def __post_init__(self):
        if self.search_config is None:
            self.search_config = SearchConfig()
        if self.download_config is None:
            self.download_config = DownloadConfig()
        if self.fft_config is None:
            self.fft_config = FFTConfig()
        if self.depth_config is None:
            self.depth_config = DepthInversionConfig()
        if self.compositing_config is None:
            self.compositing_config = CompositingConfig()
```

**Step 3: Create `ocean_rs/sar/config/__init__.py`**

```python
"""
Configuration module for OceanRS SAR Bathymetry Toolkit.
"""

from .download_config import DownloadConfig, SearchConfig
from .sar_config import (
    FFTConfig,
    DepthInversionConfig,
    CompositingConfig,
    SARProcessingConfig,
)

__all__ = [
    'DownloadConfig',
    'SearchConfig',
    'FFTConfig',
    'DepthInversionConfig',
    'CompositingConfig',
    'SARProcessingConfig',
]
```

**Step 4: Verify syntax**

Run: `python -m py_compile ocean_rs/sar/config/download_config.py`
Run: `python -m py_compile ocean_rs/sar/config/sar_config.py`
Run: `python -m py_compile ocean_rs/sar/config/__init__.py`

**Step 5: Commit**

```bash
git add ocean_rs/sar/config/
git commit -m "feat(sar): add configuration dataclasses"
```

---

## Task 3: Credential Manager

**Files:**
- Create: `ocean_rs/sar/download/__init__.py`
- Create: `ocean_rs/sar/download/credentials.py`

**Step 1: Create `ocean_rs/sar/download/credentials.py`**

```python
"""
Credential management for satellite data access.

Security rules:
- Credentials NEVER stored in config JSON files
- Priority: env vars > .env file > GUI prompt > error
- .env file is in .gitignore
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger('ocean_rs')


class CredentialError(Exception):
    """Raised when credentials cannot be obtained."""
    pass


class CredentialManager:
    """Manage satellite data access credentials securely.

    Priority chain:
        1. Environment variables (EARTHDATA_USERNAME, EARTHDATA_PASSWORD)
        2. .env file in project root
        3. Programmatic set (from GUI dialog)
        4. Raise CredentialError with setup instructions
    """

    EARTHDATA_USER_ENV = "EARTHDATA_USERNAME"
    EARTHDATA_PASS_ENV = "EARTHDATA_PASSWORD"

    def __init__(self):
        self._username: Optional[str] = None
        self._password: Optional[str] = None
        self._load_dotenv()

    def _load_dotenv(self):
        """Load credentials from .env file if available."""
        try:
            from dotenv import load_dotenv
            # Search up from current directory for .env
            env_path = self._find_env_file()
            if env_path:
                load_dotenv(env_path)
                logger.debug(f"Loaded .env from: {env_path}")
        except ImportError:
            logger.debug("python-dotenv not installed, skipping .env loading")

    def _find_env_file(self) -> Optional[Path]:
        """Find .env file searching up from cwd."""
        current = Path.cwd()
        for parent in [current] + list(current.parents):
            env_file = parent / ".env"
            if env_file.exists():
                return env_file
        return None

    def set_credentials(self, username: str, password: str):
        """Set credentials programmatically (from GUI dialog)."""
        self._username = username
        self._password = password

    def get_earthdata_credentials(self) -> Tuple[str, str]:
        """Get NASA Earthdata credentials.

        Returns:
            Tuple of (username, password)

        Raises:
            CredentialError: If credentials cannot be found
        """
        # Priority 1: Programmatically set (from GUI)
        if self._username and self._password:
            return self._username, self._password

        # Priority 2: Environment variables (includes .env via dotenv)
        username = os.environ.get(self.EARTHDATA_USER_ENV)
        password = os.environ.get(self.EARTHDATA_PASS_ENV)

        if username and password:
            return username, password

        # Priority 3: Raise with instructions
        raise CredentialError(
            "NASA Earthdata credentials not found.\n\n"
            "Set them using one of these methods:\n"
            "  1. Environment variables:\n"
            f"     {self.EARTHDATA_USER_ENV}=your_username\n"
            f"     {self.EARTHDATA_PASS_ENV}=your_password\n\n"
            "  2. Create a .env file in the project root:\n"
            f"     {self.EARTHDATA_USER_ENV}=your_username\n"
            f"     {self.EARTHDATA_PASS_ENV}=your_password\n\n"
            "  3. Enter credentials in the GUI Download tab.\n\n"
            "Register at: https://urs.earthdata.nasa.gov/users/new"
        )

    def save_to_dotenv(self, username: str, password: str,
                       directory: Optional[str] = None):
        """Save credentials to .env file (gitignored).

        Args:
            username: Earthdata username
            password: Earthdata password
            directory: Directory for .env file (default: cwd)
        """
        target_dir = Path(directory) if directory else Path.cwd()
        env_path = target_dir / ".env"

        # Read existing content (preserve other variables)
        existing_lines = []
        if env_path.exists():
            with open(env_path, 'r') as f:
                existing_lines = [
                    line for line in f.readlines()
                    if not line.startswith(self.EARTHDATA_USER_ENV)
                    and not line.startswith(self.EARTHDATA_PASS_ENV)
                ]

        with open(env_path, 'w') as f:
            for line in existing_lines:
                f.write(line)
            f.write(f"{self.EARTHDATA_USER_ENV}={username}\n")
            f.write(f"{self.EARTHDATA_PASS_ENV}={password}\n")

        logger.info(f"Credentials saved to: {env_path}")

    def test_connection(self) -> Tuple[bool, str]:
        """Test Earthdata credentials by authenticating with ASF.

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            username, password = self.get_earthdata_credentials()
        except CredentialError as e:
            return False, str(e)

        try:
            import asf_search as asf
            session = asf.ASFSession()
            session.auth_with_creds(username, password)
            return True, "Authentication successful"
        except Exception as e:
            return False, f"Authentication failed: {str(e)}"
```

**Step 2: Create `ocean_rs/sar/download/__init__.py`**

```python
"""
Download module for OceanRS SAR Bathymetry Toolkit.

Provides scene discovery (ASF), batch download, and credential management.
"""

from .credentials import CredentialManager, CredentialError

__all__ = [
    'CredentialManager',
    'CredentialError',
]
```

**Step 3: Verify syntax**

Run: `python -m py_compile ocean_rs/sar/download/credentials.py`
Run: `python -m py_compile ocean_rs/sar/download/__init__.py`

**Step 4: Commit**

```bash
git add ocean_rs/sar/download/
git commit -m "feat(sar): add credential manager (env vars, .env, GUI fallback)"
```

---

## Task 4: Scene Discovery (asf_search)

**Files:**
- Create: `ocean_rs/sar/download/scene_discovery.py`
- Modify: `ocean_rs/sar/download/__init__.py`

**Step 1: Create `ocean_rs/sar/download/scene_discovery.py`**

```python
"""
Scene discovery using ASF (Alaska Satellite Facility) search API.

Wraps asf_search to find SAR scenes intersecting an AOI with date/sensor filters.
Results are returned as SceneMetadata for GUI display and download selection.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger('ocean_rs')


@dataclass
class SceneMetadata:
    """Metadata for a discovered SAR scene."""
    granule_id: str
    platform: str               # "Sentinel-1A", "Sentinel-1B", etc.
    beam_mode: str              # "IW", "EW", "SM"
    polarization: str           # "VV+VH", "HH+HV", etc.
    orbit_direction: str        # "ASCENDING" or "DESCENDING"
    acquisition_date: str       # ISO format
    frame_number: int = 0
    path_number: int = 0
    size_mb: float = 0.0
    download_url: str = ""
    footprint_wkt: str = ""
    processing_level: str = "SLC"
    _asf_result: object = field(default=None, repr=False)


def search_scenes(aoi_wkt: str,
                  start_date: str,
                  end_date: str,
                  platform: str = "Sentinel-1",
                  beam_mode: str = "IW",
                  polarization: Optional[str] = None,
                  orbit_direction: Optional[str] = None,
                  processing_level: str = "SLC",
                  max_results: int = 250) -> List[SceneMetadata]:
    """Search ASF DAAC for SAR scenes.

    Args:
        aoi_wkt: Area of interest as WKT polygon
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        platform: Satellite platform (default: Sentinel-1)
        beam_mode: Beam mode (default: IW)
        polarization: Polarization filter (optional)
        orbit_direction: ASCENDING or DESCENDING (optional, None=any)
        processing_level: Processing level (default: SLC)
        max_results: Maximum results to return

    Returns:
        List of SceneMetadata sorted by acquisition date (newest first)

    Raises:
        ImportError: If asf_search is not installed
        RuntimeError: If search fails
    """
    try:
        import asf_search as asf
    except ImportError:
        raise ImportError(
            "asf_search is required for scene discovery.\n"
            "Install with: pip install asf_search"
        )

    logger.info(f"Searching ASF: platform={platform}, beam={beam_mode}, "
                f"dates={start_date} to {end_date}")

    # Build search parameters
    search_params = {
        'intersectsWith': aoi_wkt,
        'start': datetime.strptime(start_date, "%Y-%m-%d"),
        'end': datetime.strptime(end_date, "%Y-%m-%d"),
        'maxResults': max_results,
    }

    # Platform mapping
    platform_map = {
        "Sentinel-1": asf.PLATFORM.SENTINEL1,
    }
    if platform in platform_map:
        search_params['platform'] = platform_map[platform]

    # Beam mode
    if beam_mode:
        search_params['beamMode'] = [beam_mode]

    # Processing level
    if processing_level:
        search_params['processingLevel'] = [processing_level]

    # Orbit direction
    if orbit_direction and orbit_direction.upper() in ("ASCENDING", "DESCENDING"):
        search_params['flightDirection'] = orbit_direction.upper()

    try:
        results = asf.search(**search_params)
    except Exception as e:
        raise RuntimeError(f"ASF search failed: {str(e)}")

    logger.info(f"Found {len(results)} scenes")

    # Convert to SceneMetadata
    scenes = []
    for r in results:
        props = r.properties
        try:
            scene = SceneMetadata(
                granule_id=props.get('sceneName', props.get('fileID', 'unknown')),
                platform=props.get('platform', platform),
                beam_mode=props.get('beamModeType', beam_mode),
                polarization=props.get('polarization', ''),
                orbit_direction=props.get('flightDirection', ''),
                acquisition_date=props.get('startTime', ''),
                frame_number=int(props.get('frameNumber', 0)),
                path_number=int(props.get('pathNumber', 0)),
                size_mb=float(props.get('bytes', 0)) / (1024 * 1024),
                download_url=props.get('url', ''),
                footprint_wkt=props.get('wkt', ''),
                processing_level=processing_level,
                _asf_result=r,
            )
            scenes.append(scene)
        except Exception as e:
            logger.warning(f"Failed to parse scene result: {e}")

    # Sort by date (newest first)
    scenes.sort(key=lambda s: s.acquisition_date, reverse=True)

    return scenes
```

**Step 2: Update `ocean_rs/sar/download/__init__.py`**

Add scene_discovery exports:

```python
"""
Download module for OceanRS SAR Bathymetry Toolkit.

Provides scene discovery (ASF), batch download, and credential management.
"""

from .credentials import CredentialManager, CredentialError
from .scene_discovery import SceneMetadata, search_scenes

__all__ = [
    'CredentialManager',
    'CredentialError',
    'SceneMetadata',
    'search_scenes',
]
```

**Step 3: Verify syntax**

Run: `python -m py_compile ocean_rs/sar/download/scene_discovery.py`

**Step 4: Commit**

```bash
git add ocean_rs/sar/download/
git commit -m "feat(sar): add ASF scene discovery (asf_search wrapper)"
```

---

## Task 5: Batch Downloader

**Files:**
- Create: `ocean_rs/sar/download/batch_downloader.py`
- Modify: `ocean_rs/sar/download/__init__.py`

**Step 1: Create `ocean_rs/sar/download/batch_downloader.py`**

```python
"""
Batch downloader for SAR scenes from ASF DAAC.

Handles authenticated downloads with resume support and progress reporting.
"""

import os
import logging
from pathlib import Path
from typing import List, Callable, Optional

from .scene_discovery import SceneMetadata
from .credentials import CredentialManager

logger = logging.getLogger('ocean_rs')


class BatchDownloader:
    """Download SAR scenes from ASF with authentication and retry.

    Uses asf_search download API with NASA Earthdata Login.
    """

    def __init__(self, credential_manager: CredentialManager):
        self.creds = credential_manager
        self._cancel_requested = False

    def cancel(self):
        """Request download cancellation."""
        self._cancel_requested = True

    def download_scenes(self,
                       scenes: List[SceneMetadata],
                       output_dir: str,
                       progress_callback: Optional[Callable] = None
                       ) -> List[Path]:
        """Download selected scenes to output directory.

        Args:
            scenes: List of scenes to download
            output_dir: Directory to save downloaded files
            progress_callback: Optional callback(scene_index, total, status_msg)

        Returns:
            List of paths to downloaded files

        Raises:
            ImportError: If asf_search not installed
            CredentialError: If credentials not available
        """
        try:
            import asf_search as asf
        except ImportError:
            raise ImportError("asf_search required: pip install asf_search")

        self._cancel_requested = False
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Authenticate
        username, password = self.creds.get_earthdata_credentials()
        session = asf.ASFSession()
        session.auth_with_creds(username, password)

        downloaded = []
        total = len(scenes)

        for i, scene in enumerate(scenes):
            if self._cancel_requested:
                logger.info("Download cancelled by user")
                break

            logger.info(f"Downloading [{i+1}/{total}]: {scene.granule_id}")
            if progress_callback:
                progress_callback(i, total, f"Downloading: {scene.granule_id}")

            # Check if already downloaded
            expected_file = output_path / f"{scene.granule_id}.zip"
            if expected_file.exists() and expected_file.stat().st_size > 0:
                logger.info(f"Already downloaded: {scene.granule_id}")
                downloaded.append(expected_file)
                continue

            try:
                asf_result = scene._asf_result
                if asf_result is None:
                    logger.warning(f"No ASF result for {scene.granule_id}, skipping")
                    continue

                asf_result.download(
                    path=str(output_path),
                    session=session
                )

                # Find the downloaded file
                for ext in ['.zip', '.SAFE']:
                    candidate = output_path / f"{scene.granule_id}{ext}"
                    if candidate.exists():
                        downloaded.append(candidate)
                        logger.info(f"Downloaded: {candidate.name} "
                                   f"({candidate.stat().st_size / 1e6:.1f} MB)")
                        break
                else:
                    # Check for any new file matching granule ID
                    matches = list(output_path.glob(f"*{scene.granule_id}*"))
                    if matches:
                        downloaded.append(matches[0])
                        logger.info(f"Downloaded: {matches[0].name}")
                    else:
                        logger.warning(f"Download completed but file not found: "
                                      f"{scene.granule_id}")

            except Exception as e:
                logger.error(f"Download failed for {scene.granule_id}: {e}")
                if progress_callback:
                    progress_callback(i, total, f"FAILED: {scene.granule_id}")

        if progress_callback:
            progress_callback(total, total, "Download complete")

        logger.info(f"Downloaded {len(downloaded)}/{total} scenes")
        return downloaded
```

**Step 2: Update `ocean_rs/sar/download/__init__.py` — add BatchDownloader**

```python
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
```

**Step 3: Verify syntax**

Run: `python -m py_compile ocean_rs/sar/download/batch_downloader.py`

**Step 4: Commit**

```bash
git add ocean_rs/sar/download/
git commit -m "feat(sar): add batch downloader with auth and resume support"
```

---

## Task 6: Sensor Adapter (Sentinel-1)

**Files:**
- Create: `ocean_rs/sar/sensors/__init__.py`
- Create: `ocean_rs/sar/sensors/base.py`
- Create: `ocean_rs/sar/sensors/sentinel1.py`

**Step 1: Create `ocean_rs/sar/sensors/base.py`**

```python
"""
Base sensor adapter for SAR preprocessing.

All sensor adapters inherit from SensorAdapter and produce OceanImage instances.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from ..core.data_models import OceanImage


class SensorAdapter(ABC):
    """Abstract base class for SAR sensor adapters.

    Each sensor adapter knows how to preprocess raw data from a specific
    SAR sensor into an OceanImage ready for bathymetry analysis.
    """

    @property
    @abstractmethod
    def sensor_name(self) -> str:
        """Human-readable sensor name."""
        ...

    @abstractmethod
    def preprocess(self, input_path: Path, output_dir: Path,
                   snap_gpt_path: Optional[str] = None) -> OceanImage:
        """Preprocess raw SAR data to calibrated, geocoded OceanImage.

        Args:
            input_path: Path to raw SAR product (.zip, .SAFE, etc.)
            output_dir: Directory for intermediate files
            snap_gpt_path: Path to SNAP GPT executable

        Returns:
            OceanImage ready for bathymetry analysis
        """
        ...

    @abstractmethod
    def can_process(self, input_path: Path) -> bool:
        """Check if this adapter can process the given input.

        Args:
            input_path: Path to check

        Returns:
            True if this adapter handles this file type
        """
        ...
```

**Step 2: Create `ocean_rs/sar/sensors/sentinel1.py`**

```python
"""
Sentinel-1 sensor adapter.

Preprocesses Sentinel-1 SLC products using SNAP GPT:
    Apply-Orbit-File -> Thermal-Noise-Removal -> Calibration -> Terrain-Correction

Produces a sigma0 OceanImage for bathymetry analysis.
"""

import os
import sys
import shutil
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from ..core.data_models import OceanImage, ImageType, GeoTransform
from .base import SensorAdapter

logger = logging.getLogger('ocean_rs')


class Sentinel1Adapter(SensorAdapter):
    """Preprocess Sentinel-1 SLC to calibrated sigma0 via SNAP GPT."""

    @property
    def sensor_name(self) -> str:
        return "Sentinel-1"

    def can_process(self, input_path: Path) -> bool:
        """Check if file is a Sentinel-1 product."""
        name = input_path.name.upper()
        return name.startswith("S1") and (
            name.endswith(".ZIP") or name.endswith(".SAFE")
        )

    def preprocess(self, input_path: Path, output_dir: Path,
                   snap_gpt_path: Optional[str] = None) -> OceanImage:
        """Run SNAP GPT preprocessing chain on Sentinel-1 SLC.

        Processing chain:
            1. Apply-Orbit-File
            2. Thermal-Noise-Removal
            3. Calibration (to Sigma0)
            4. Terrain-Correction (Range-Doppler, to UTM)

        Args:
            input_path: Path to S1 SLC product (.zip or .SAFE)
            output_dir: Directory for intermediate and output files
            snap_gpt_path: Path to SNAP GPT (auto-detected if None)

        Returns:
            OceanImage with calibrated sigma0 data
        """
        gpt = self._find_gpt(snap_gpt_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        scene_name = input_path.stem.replace('.SAFE', '')
        output_file = output_dir / f"{scene_name}_sigma0.dim"

        if output_file.exists():
            logger.info(f"Preprocessed file exists, loading: {output_file.name}")
            return self._load_snap_output(output_file)

        # Create SNAP GPT XML graph
        graph_xml = self._create_processing_graph(
            str(input_path), str(output_file)
        )

        # Write graph to temp file
        graph_path = output_dir / f"{scene_name}_graph.xml"
        with open(graph_path, 'w') as f:
            f.write(graph_xml)

        # Execute SNAP GPT
        logger.info(f"Running SNAP GPT preprocessing: {scene_name}")
        cmd = [gpt, str(graph_path)]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"SNAP GPT failed (exit {result.returncode}):\n"
                    f"{result.stderr[-500:]}"
                )
            logger.info(f"SNAP GPT completed: {scene_name}")
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"SNAP GPT timed out after 2 hours: {scene_name}")

        # Clean up graph file
        graph_path.unlink(missing_ok=True)

        return self._load_snap_output(output_file)

    def _find_gpt(self, snap_gpt_path: Optional[str] = None) -> str:
        """Find SNAP GPT executable."""
        if snap_gpt_path and os.path.exists(snap_gpt_path):
            return snap_gpt_path

        snap_home = os.environ.get('SNAP_HOME', '')
        if snap_home:
            gpt_name = 'gpt.exe' if sys.platform.startswith('win') else 'gpt'
            gpt_path = os.path.join(snap_home, 'bin', gpt_name)
            if os.path.exists(gpt_path):
                return gpt_path

        gpt_name = 'gpt.exe' if sys.platform.startswith('win') else 'gpt'
        gpt_on_path = shutil.which(gpt_name)
        if gpt_on_path:
            return gpt_on_path

        raise FileNotFoundError(
            "SNAP GPT not found. Set SNAP_HOME environment variable "
            "or provide snap_gpt_path parameter."
        )

    def _create_processing_graph(self, input_path: str,
                                  output_path: str) -> str:
        """Create SNAP GPT XML graph for S1 preprocessing."""
        return f"""<graph id="S1-Preprocessing">
  <version>1.0</version>
  <node id="Read">
    <operator>Read</operator>
    <sources/>
    <parameters>
      <file>{input_path}</file>
    </parameters>
  </node>
  <node id="Apply-Orbit-File">
    <operator>Apply-Orbit-File</operator>
    <sources>
      <sourceProduct refid="Read"/>
    </sources>
    <parameters>
      <orbitType>Sentinel Precise (Auto Download)</orbitType>
      <polyDegree>3</polyDegree>
      <continueOnFail>true</continueOnFail>
    </parameters>
  </node>
  <node id="ThermalNoiseRemoval">
    <operator>ThermalNoiseRemoval</operator>
    <sources>
      <sourceProduct refid="Apply-Orbit-File"/>
    </sources>
    <parameters>
      <removeThermalNoise>true</removeThermalNoise>
    </parameters>
  </node>
  <node id="Calibration">
    <operator>Calibration</operator>
    <sources>
      <sourceProduct refid="ThermalNoiseRemoval"/>
    </sources>
    <parameters>
      <outputSigmaBand>true</outputSigmaBand>
      <selectedPolarisations>VV</selectedPolarisations>
    </parameters>
  </node>
  <node id="Terrain-Correction">
    <operator>Terrain-Correction</operator>
    <sources>
      <sourceProduct refid="Calibration"/>
    </sources>
    <parameters>
      <demName>SRTM 1Sec HGT</demName>
      <pixelSpacingInMeter>10.0</pixelSpacingInMeter>
      <mapProjection>AUTO:42001</mapProjection>
    </parameters>
  </node>
  <node id="Write">
    <operator>Write</operator>
    <sources>
      <sourceProduct refid="Terrain-Correction"/>
    </sources>
    <parameters>
      <file>{output_path}</file>
      <formatName>BEAM-DIMAP</formatName>
    </parameters>
  </node>
</graph>"""

    def _load_snap_output(self, dim_path: Path) -> OceanImage:
        """Load SNAP BEAM-DIMAP output as OceanImage using GDAL."""
        from ocean_rs.shared.raster_io import RasterIO

        data_dir = dim_path.with_suffix('.data')
        # Find sigma0 band
        sigma0_files = list(data_dir.glob("Sigma0_VV*.img"))
        if not sigma0_files:
            sigma0_files = list(data_dir.glob("Sigma0*.img"))
        if not sigma0_files:
            raise FileNotFoundError(
                f"No Sigma0 band found in: {data_dir}"
            )

        band_file = sigma0_files[0]
        data, geo_info = RasterIO.read_band(str(band_file))

        geo = GeoTransform(
            origin_x=geo_info['origin_x'],
            origin_y=geo_info['origin_y'],
            pixel_size_x=geo_info['pixel_size_x'],
            pixel_size_y=geo_info['pixel_size_y'],
            crs_wkt=geo_info.get('crs_wkt', ''),
            rows=data.shape[0],
            cols=data.shape[1],
        )

        return OceanImage(
            data=data,
            image_type=ImageType.SIGMA0,
            geo=geo,
            metadata={
                'sensor': 'Sentinel-1',
                'source_file': str(dim_path),
                'band': band_file.name,
            },
            pixel_spacing_m=abs(geo.pixel_size_x),
        )
```

**Step 3: Create `ocean_rs/sar/sensors/__init__.py`**

```python
"""
Sensor adapters for OceanRS SAR Bathymetry Toolkit.

Each adapter preprocesses raw SAR data into the OceanImage contract.
"""

from .base import SensorAdapter
from .sentinel1 import Sentinel1Adapter

__all__ = [
    'SensorAdapter',
    'Sentinel1Adapter',
]
```

**Step 4: Verify syntax**

Run: `python -m py_compile ocean_rs/sar/sensors/base.py`
Run: `python -m py_compile ocean_rs/sar/sensors/sentinel1.py`

**Step 5: Commit**

```bash
git add ocean_rs/sar/sensors/
git commit -m "feat(sar): add Sentinel-1 adapter with SNAP GPT preprocessing"
```

---

## Task 7: FFT Swell Extraction

**Files:**
- Create: `ocean_rs/sar/bathymetry/__init__.py`
- Create: `ocean_rs/sar/bathymetry/fft_extractor.py`

**Step 1: Create `ocean_rs/sar/bathymetry/fft_extractor.py`**

```python
"""
FFT-based swell wavelength and direction extraction from SAR imagery.

Tiles the image, computes 2D FFT per tile, and finds the dominant spectral
peak to determine swell wavelength and propagation direction.
"""

import logging
import numpy as np
from typing import Optional

from ..core.data_models import OceanImage, SwellField, GeoTransform

logger = logging.getLogger('ocean_rs')


def extract_swell(image: OceanImage,
                  tile_size_m: float = 512.0,
                  overlap: float = 0.5,
                  min_wavelength_m: float = 50.0,
                  max_wavelength_m: float = 600.0,
                  confidence_threshold: float = 0.3) -> SwellField:
    """Extract dominant swell wavelength and direction from SAR image.

    Algorithm:
        1. Tile image with configurable overlap
        2. Per tile: apply Hanning window, compute 2D FFT
        3. Compute power spectrum, mask to wavelength range
        4. Find dominant peak -> wavelength + direction
        5. Compute confidence from spectral peak SNR

    Args:
        image: OceanImage to analyze
        tile_size_m: Tile size in meters (default 512)
        overlap: Tile overlap fraction (0-1, default 0.5)
        min_wavelength_m: Minimum wavelength to detect (default 50m)
        max_wavelength_m: Maximum wavelength to detect (default 600m)
        confidence_threshold: Minimum confidence to keep (default 0.3)

    Returns:
        SwellField with wavelength, direction, and confidence per tile
    """
    data = image.data
    pixel_m = image.pixel_spacing_m

    # Calculate tile size in pixels
    tile_px = int(tile_size_m / pixel_m)
    # Round to power of 2 for FFT efficiency
    tile_px = _next_power_of_2(tile_px)
    step_px = int(tile_px * (1 - overlap))

    rows, cols = data.shape
    logger.info(f"FFT extraction: image={rows}x{cols}px, tile={tile_px}px, "
                f"step={step_px}px, pixel={pixel_m}m")

    # Generate tile positions
    row_starts = list(range(0, rows - tile_px + 1, step_px))
    col_starts = list(range(0, cols - tile_px + 1, step_px))

    if not row_starts or not col_starts:
        raise ValueError(
            f"Image too small ({rows}x{cols}px) for tile size {tile_px}px"
        )

    n_tiles = len(row_starts) * len(col_starts)
    logger.info(f"Processing {n_tiles} tiles ({len(row_starts)}x{len(col_starts)})")

    # Pre-compute Hanning window
    window = np.outer(np.hanning(tile_px), np.hanning(tile_px))

    # Frequency arrays for wavelength conversion
    freqs = np.fft.fftfreq(tile_px, d=pixel_m)
    fx, fy = np.meshgrid(freqs, freqs)
    freq_magnitude = np.sqrt(fx**2 + fy**2)
    # Avoid division by zero
    wavelength_map = np.zeros_like(freq_magnitude)
    nonzero = freq_magnitude > 0
    wavelength_map[nonzero] = 1.0 / freq_magnitude[nonzero]

    # Wavelength mask
    valid_wavelength = (wavelength_map >= min_wavelength_m) & \
                       (wavelength_map <= max_wavelength_m)

    # Process tiles
    wavelengths = []
    directions = []
    confidences = []
    centers_x = []
    centers_y = []

    for r0 in row_starts:
        for c0 in col_starts:
            tile = data[r0:r0+tile_px, c0:c0+tile_px].astype(np.float64)

            # Skip tiles with too many NaN/zero
            valid_frac = np.sum(np.isfinite(tile) & (tile != 0)) / tile.size
            if valid_frac < 0.5:
                continue

            # Replace NaN with tile mean
            tile_mean = np.nanmean(tile)
            tile = np.where(np.isfinite(tile), tile, tile_mean)

            # Detrend and window
            tile -= tile_mean
            tile *= window

            # 2D FFT
            fft2 = np.fft.fft2(tile)
            power = np.abs(np.fft.fftshift(fft2))**2

            # Shift frequency arrays to match fftshift
            wl_shifted = np.fft.fftshift(wavelength_map)
            valid_shifted = np.fft.fftshift(valid_wavelength)
            fx_shifted = np.fft.fftshift(fx)
            fy_shifted = np.fft.fftshift(fy)

            # Mask to valid wavelength range
            masked_power = power * valid_shifted

            if np.max(masked_power) == 0:
                continue

            # Find dominant peak
            peak_idx = np.unravel_index(np.argmax(masked_power), power.shape)
            peak_power = masked_power[peak_idx]

            # Wavelength at peak
            wl = wl_shifted[peak_idx]

            # Direction at peak (degrees from north, clockwise)
            peak_fx = fx_shifted[peak_idx]
            peak_fy = fy_shifted[peak_idx]
            direction = np.degrees(np.arctan2(peak_fx, peak_fy)) % 360

            # Confidence: SNR of peak relative to mean power
            mean_power = np.mean(masked_power[masked_power > 0])
            snr = peak_power / mean_power if mean_power > 0 else 0
            confidence = min(1.0, snr / 10.0)  # Normalize to 0-1

            if confidence >= confidence_threshold:
                wavelengths.append(wl)
                directions.append(direction)
                confidences.append(confidence)

                # Tile center in geo coordinates
                cx = image.geo.origin_x + (c0 + tile_px/2) * image.geo.pixel_size_x
                cy = image.geo.origin_y + (r0 + tile_px/2) * image.geo.pixel_size_y
                centers_x.append(cx)
                centers_y.append(cy)

    n_valid = len(wavelengths)
    logger.info(f"FFT complete: {n_valid}/{n_tiles} tiles above confidence threshold")

    if n_valid == 0:
        raise ValueError(
            "No valid swell detected. Try adjusting wavelength range "
            "or lowering confidence threshold."
        )

    return SwellField(
        wavelength=np.array(wavelengths),
        direction=np.array(directions),
        confidence=np.array(confidences),
        tile_centers_x=np.array(centers_x),
        tile_centers_y=np.array(centers_y),
        tile_size_m=tile_size_m,
        geo=image.geo,
    )


def _next_power_of_2(n: int) -> int:
    """Round up to next power of 2."""
    p = 1
    while p < n:
        p *= 2
    return p
```

**Step 2: Create `ocean_rs/sar/bathymetry/__init__.py`**

```python
"""
Bathymetry module for OceanRS SAR Toolkit.

FFT swell extraction, wave period retrieval, and depth inversion.
"""

from .fft_extractor import extract_swell

__all__ = [
    'extract_swell',
]
```

**Step 3: Verify syntax**

Run: `python -m py_compile ocean_rs/sar/bathymetry/fft_extractor.py`

**Step 4: Commit**

```bash
git add ocean_rs/sar/bathymetry/
git commit -m "feat(sar): add FFT swell extraction (2D FFT per tile)"
```

---

## Task 8: Wave Period Retrieval + Depth Inversion

**Files:**
- Create: `ocean_rs/sar/bathymetry/wave_period.py`
- Create: `ocean_rs/sar/bathymetry/depth_inversion.py`
- Modify: `ocean_rs/sar/bathymetry/__init__.py`

**Step 1: Create `ocean_rs/sar/bathymetry/wave_period.py`**

```python
"""
Wave period retrieval from WaveWatch III via NOAA ERDDAP.

ERDDAP is free, no authentication required.
"""

import logging
from typing import Optional
from functools import lru_cache

logger = logging.getLogger('ocean_rs')


@lru_cache(maxsize=32)
def get_wave_period(lon: float, lat: float, datetime_str: str) -> float:
    """Get dominant wave period from WaveWatch III.

    Uses NOAA ERDDAP to query the nearest WaveWatch III grid point.
    Results are cached to avoid repeated API calls.

    Args:
        lon: Longitude (degrees East)
        lat: Latitude (degrees North)
        datetime_str: UTC datetime (YYYY-MM-DDTHH:MM:SSZ)

    Returns:
        Dominant wave period in seconds

    Raises:
        RuntimeError: If ERDDAP query fails
    """
    import requests

    # NOAA ERDDAP WaveWatch III global model
    # Dataset: NWW3 global 0.5 degree
    base_url = "https://coastwatch.pfeg.noaa.gov/erddap/griddap"
    dataset = "NWW3_Global_Best"

    # Query peak wave period (perpw)
    url = (
        f"{base_url}/{dataset}.json?"
        f"perpw[({datetime_str}):1:({datetime_str})]"
        f"[({lat}):1:({lat})]"
        f"[({lon}):1:({lon})]"
    )

    logger.info(f"Querying WaveWatch III: lon={lon}, lat={lat}, time={datetime_str}")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        rows = data.get('table', {}).get('rows', [])
        if rows and len(rows[0]) >= 4:
            period = float(rows[0][3])
            if period > 0:
                logger.info(f"WaveWatch III peak period: {period:.1f}s")
                return period

        raise ValueError("No valid wave period in response")

    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            f"ERDDAP query failed: {e}\n"
            "Use manual wave period entry as fallback."
        )
    except (ValueError, KeyError, IndexError) as e:
        raise RuntimeError(
            f"Failed to parse ERDDAP response: {e}\n"
            "Use manual wave period entry as fallback."
        )
```

**Step 2: Create `ocean_rs/sar/bathymetry/depth_inversion.py`**

```python
"""
Depth inversion using the linear wave dispersion relation.

Solves omega^2 = g * k * tanh(k * h) for depth h using Newton-Raphson.

Reference:
    The dispersion relation relates wavelength to depth:
    waves slow down and shorten as they enter shallow water.
    If we know the wavelength and wave period, we can solve for depth.
"""

import logging
import numpy as np
from typing import Optional

from ..core.data_models import SwellField, BathymetryResult, GeoTransform

logger = logging.getLogger('ocean_rs')


def invert_depth(swell: SwellField,
                 wave_period: float,
                 max_depth_m: float = 100.0,
                 gravity: float = 9.81,
                 max_iterations: int = 50,
                 convergence_tol: float = 1e-6) -> BathymetryResult:
    """Invert depth from swell wavelength using linear dispersion relation.

    Solves: omega^2 = g * k * tanh(k * h)
    Where: omega = 2*pi/T, k = 2*pi/L, h = depth

    Newton-Raphson iteration:
        f(h) = omega^2 - g*k*tanh(k*h) = 0
        f'(h) = -g*k^2 / cosh^2(k*h)
        h_new = h - f(h)/f'(h)

    Args:
        swell: SwellField from FFT extraction
        wave_period: Dominant wave period (seconds)
        max_depth_m: Maximum depth limit (meters)
        gravity: Gravitational acceleration (m/s^2)
        max_iterations: Newton-Raphson max iterations
        convergence_tol: Convergence tolerance (meters)

    Returns:
        BathymetryResult with depth and uncertainty arrays
    """
    wavelengths = swell.wavelength
    omega = 2 * np.pi / wave_period
    k = 2 * np.pi / wavelengths

    logger.info(f"Depth inversion: {len(wavelengths)} points, T={wave_period:.1f}s, "
                f"wavelength range: {wavelengths.min():.0f}-{wavelengths.max():.0f}m")

    # Deep water wavelength for reference
    L_deep = gravity * wave_period**2 / (2 * np.pi)
    logger.info(f"Deep water wavelength: {L_deep:.0f}m")

    # Initial guess: deep water depth (half wavelength)
    h = wavelengths / 2.0

    # Newton-Raphson iteration
    for iteration in range(max_iterations):
        kh = k * h

        # Clamp kh to avoid overflow in cosh/tanh
        kh = np.clip(kh, 0, 20)

        tanh_kh = np.tanh(kh)
        cosh_kh = np.cosh(kh)

        # f(h) = omega^2 - g*k*tanh(k*h)
        f = omega**2 - gravity * k * tanh_kh

        # f'(h) = -g*k^2 / cosh^2(k*h)
        f_prime = -gravity * k**2 / (cosh_kh**2)

        # Avoid division by zero
        valid = np.abs(f_prime) > 1e-12
        delta = np.zeros_like(h)
        delta[valid] = f[valid] / f_prime[valid]

        h -= delta

        # Enforce positive depth
        h = np.maximum(h, 0.1)

        max_delta = np.max(np.abs(delta))
        if max_delta < convergence_tol:
            logger.info(f"Converged after {iteration + 1} iterations "
                       f"(max delta: {max_delta:.2e}m)")
            break
    else:
        logger.warning(f"Newton-Raphson did not converge after {max_iterations} "
                      f"iterations (max delta: {max_delta:.2e}m)")

    # Clip to max depth
    h = np.clip(h, 0, max_depth_m)

    # Uncertainty estimate: sensitivity of depth to wavelength error
    # dh/dL ~ 1/(2*pi) * L/tanh(kh) * (1 - kh/sinh(kh)*cosh(kh))
    # Simplified: assume 10% wavelength uncertainty
    wavelength_uncertainty = 0.1 * wavelengths
    kh_final = np.clip(k * h, 0, 20)
    # Depth sensitivity to wavelength
    dh_dL = h / wavelengths  # First-order approximation
    depth_uncertainty = np.abs(dh_dL * wavelength_uncertainty)

    # Cap uncertainty
    depth_uncertainty = np.clip(depth_uncertainty, 0.5, max_depth_m * 0.5)

    logger.info(f"Depth range: {h.min():.1f} - {h.max():.1f}m "
               f"(mean uncertainty: {depth_uncertainty.mean():.1f}m)")

    return BathymetryResult(
        depth=h,
        uncertainty=depth_uncertainty,
        method="linear_dispersion",
        wave_period=wave_period,
        wave_period_source="",  # Set by caller
        geo=swell.geo,
        metadata={
            'n_points': len(h),
            'wave_period': wave_period,
            'deep_water_wavelength': L_deep,
            'iterations': min(iteration + 1, max_iterations),
        },
    )
```

**Step 3: Update `ocean_rs/sar/bathymetry/__init__.py`**

```python
"""
Bathymetry module for OceanRS SAR Toolkit.

FFT swell extraction, wave period retrieval, and depth inversion.
"""

from .fft_extractor import extract_swell
from .wave_period import get_wave_period
from .depth_inversion import invert_depth

__all__ = [
    'extract_swell',
    'get_wave_period',
    'invert_depth',
]
```

**Step 4: Verify syntax**

Run: `python -m py_compile ocean_rs/sar/bathymetry/wave_period.py`
Run: `python -m py_compile ocean_rs/sar/bathymetry/depth_inversion.py`

**Step 5: Commit**

```bash
git add ocean_rs/sar/bathymetry/
git commit -m "feat(sar): add wave period retrieval and depth inversion"
```

---

## Task 9: Multi-temporal Compositor

**Files:**
- Create: `ocean_rs/sar/bathymetry/compositor.py`
- Modify: `ocean_rs/sar/bathymetry/__init__.py`

**Step 1: Create `ocean_rs/sar/bathymetry/compositor.py`**

```python
"""
Multi-temporal bathymetry compositing.

Combines multiple BathymetryResult from different SAR acquisitions
into a single robust depth estimate, weighted by confidence.
"""

import logging
import numpy as np
from typing import List

from ..core.data_models import BathymetryResult

logger = logging.getLogger('ocean_rs')


def composite_bathymetry(results: List[BathymetryResult],
                         method: str = "weighted_median") -> BathymetryResult:
    """Combine multiple temporal bathymetry observations.

    Args:
        results: List of BathymetryResult from different acquisitions
        method: "weighted_median" or "weighted_mean"

    Returns:
        Composite BathymetryResult with reduced uncertainty
    """
    if len(results) == 0:
        raise ValueError("No bathymetry results to composite")

    if len(results) == 1:
        logger.info("Single result, no compositing needed")
        return results[0]

    logger.info(f"Compositing {len(results)} bathymetry results using {method}")

    # Stack all depths and uncertainties
    all_depths = np.array([r.depth for r in results])
    all_uncertainties = np.array([r.uncertainty for r in results])

    # Weights = inverse uncertainty squared
    weights = 1.0 / (all_uncertainties**2 + 1e-10)

    if method == "weighted_mean":
        weight_sum = np.sum(weights, axis=0)
        depth = np.sum(all_depths * weights, axis=0) / weight_sum
        # Combined uncertainty
        uncertainty = 1.0 / np.sqrt(weight_sum)
    else:
        # weighted_median (default) - more robust to outliers
        depth = _weighted_median(all_depths, weights)
        # Uncertainty: MAD-based estimate
        residuals = np.abs(all_depths - depth[np.newaxis, :])
        uncertainty = 1.4826 * _weighted_median(residuals, weights)

    logger.info(f"Composite depth range: {depth.min():.1f} - {depth.max():.1f}m "
               f"(mean uncertainty: {uncertainty.mean():.1f}m)")

    return BathymetryResult(
        depth=depth,
        uncertainty=uncertainty,
        method=f"composite_{method}",
        wave_period=np.mean([r.wave_period for r in results]),
        wave_period_source="composite",
        geo=results[0].geo,
        metadata={
            'n_observations': len(results),
            'compositing_method': method,
        },
    )


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute weighted median along first axis.

    Args:
        values: Array of shape (n_obs, n_points)
        weights: Array of shape (n_obs, n_points)

    Returns:
        Weighted median of shape (n_points,)
    """
    n_obs, n_points = values.shape
    result = np.zeros(n_points)

    for j in range(n_points):
        v = values[:, j]
        w = weights[:, j]
        sorted_idx = np.argsort(v)
        v_sorted = v[sorted_idx]
        w_sorted = w[sorted_idx]
        cumsum = np.cumsum(w_sorted)
        cutoff = cumsum[-1] / 2.0
        idx = np.searchsorted(cumsum, cutoff)
        result[j] = v_sorted[min(idx, len(v_sorted) - 1)]

    return result
```

**Step 2: Update `ocean_rs/sar/bathymetry/__init__.py`**

Add compositor export:

```python
"""
Bathymetry module for OceanRS SAR Toolkit.

FFT swell extraction, wave period retrieval, depth inversion, and compositing.
"""

from .fft_extractor import extract_swell
from .wave_period import get_wave_period
from .depth_inversion import invert_depth
from .compositor import composite_bathymetry

__all__ = [
    'extract_swell',
    'get_wave_period',
    'invert_depth',
    'composite_bathymetry',
]
```

**Step 3: Verify syntax**

Run: `python -m py_compile ocean_rs/sar/bathymetry/compositor.py`

**Step 4: Commit**

```bash
git add ocean_rs/sar/bathymetry/
git commit -m "feat(sar): add multi-temporal bathymetry compositor"
```

---

## Task 10: Bathymetry Pipeline Orchestrator

**Files:**
- Create: `ocean_rs/sar/core/bathymetry_pipeline.py`
- Modify: `ocean_rs/sar/core/__init__.py`

**Step 1: Create `ocean_rs/sar/core/bathymetry_pipeline.py`**

```python
"""
Bathymetry pipeline orchestrator.

Coordinates the full pipeline: preprocess -> FFT -> wave period -> depth inversion.
Follows the pattern of ocean_rs/optical/core/unified_processor.py.
"""

import gc
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Callable

from ..core.data_models import OceanImage, BathymetryResult
from ..config.sar_config import SARProcessingConfig
from ..sensors.sentinel1 import Sentinel1Adapter
from ..bathymetry.fft_extractor import extract_swell
from ..bathymetry.wave_period import get_wave_period
from ..bathymetry.depth_inversion import invert_depth
from ..bathymetry.compositor import composite_bathymetry
from ocean_rs.shared import RasterIO, MemoryManager

logger = logging.getLogger('ocean_rs')


class BathymetryPipeline:
    """Main orchestrator for SAR bathymetry processing.

    Pipeline: Preprocess -> FFT -> Wave Period -> Depth Inversion -> Export
    """

    def __init__(self, config: SARProcessingConfig):
        self.config = config
        self.adapter = Sentinel1Adapter()

        # Processing statistics
        self.processed_count = 0
        self.failed_count = 0
        self.start_time = None

        # Cancel flag
        self._cancelled = False

    def cancel(self):
        """Request processing cancellation."""
        self._cancelled = True

    def process_scenes(self,
                      scene_paths: List[Path],
                      progress_callback: Optional[Callable] = None
                      ) -> Optional[BathymetryResult]:
        """Process multiple SAR scenes through the bathymetry pipeline.

        Args:
            scene_paths: Paths to downloaded SAR products
            progress_callback: Optional callback(step, total_steps, message)

        Returns:
            Composite BathymetryResult, or None if all failed
        """
        self._cancelled = False
        self.start_time = time.time()
        total = len(scene_paths)
        results = []

        output_dir = Path(self.config.output_directory)
        intermediate_dir = output_dir / "Intermediate"
        intermediate_dir.mkdir(parents=True, exist_ok=True)

        for i, scene_path in enumerate(scene_paths):
            if self._cancelled:
                logger.info("Processing cancelled by user")
                break

            scene_name = scene_path.stem
            logger.info(f"{'='*60}")
            logger.info(f"Processing [{i+1}/{total}]: {scene_name}")
            logger.info(f"{'='*60}")

            if progress_callback:
                progress_callback(i, total, f"Processing: {scene_name}")

            try:
                result = self._process_single_scene(scene_path, intermediate_dir)
                if result is not None:
                    results.append(result)
                    self.processed_count += 1
                else:
                    self.failed_count += 1
            except Exception as e:
                logger.error(f"Failed to process {scene_name}: {e}")
                self.failed_count += 1

            # Memory cleanup between scenes
            gc.collect()

        if not results:
            logger.warning("No scenes produced valid bathymetry results")
            return None

        # Composite if multiple results
        if len(results) > 1 and self.config.compositing_config.enabled:
            logger.info(f"Compositing {len(results)} results...")
            final = composite_bathymetry(
                results,
                method=self.config.compositing_config.method
            )
        else:
            final = results[0]

        # Export
        self._export_results(final, output_dir)

        elapsed = time.time() - self.start_time
        logger.info(f"Pipeline complete: {self.processed_count} processed, "
                   f"{self.failed_count} failed, {elapsed:.0f}s elapsed")

        if progress_callback:
            progress_callback(total, total, "Processing complete")

        return final

    def _process_single_scene(self, scene_path: Path,
                               intermediate_dir: Path) -> Optional[BathymetryResult]:
        """Process a single SAR scene through the pipeline."""
        # Step 1: Preprocess
        logger.info("Step 1/4: Preprocessing (SNAP GPT)...")
        image = self.adapter.preprocess(
            scene_path,
            intermediate_dir,
            snap_gpt_path=self.config.snap_gpt_path or None
        )

        # Step 2: FFT swell extraction
        logger.info("Step 2/4: FFT swell extraction...")
        fft_cfg = self.config.fft_config
        swell = extract_swell(
            image,
            tile_size_m=fft_cfg.tile_size_m,
            overlap=fft_cfg.overlap,
            min_wavelength_m=fft_cfg.min_wavelength_m,
            max_wavelength_m=fft_cfg.max_wavelength_m,
            confidence_threshold=fft_cfg.confidence_threshold,
        )

        # Step 3: Wave period
        logger.info("Step 3/4: Wave period retrieval...")
        depth_cfg = self.config.depth_config
        if depth_cfg.wave_period_source == "manual":
            wave_period = depth_cfg.manual_wave_period
            period_source = "manual"
        else:
            try:
                # Get center coordinates from image
                cx = image.geo.origin_x + (image.geo.cols / 2) * image.geo.pixel_size_x
                cy = image.geo.origin_y + (image.geo.rows / 2) * image.geo.pixel_size_y
                acq_time = image.metadata.get('datetime', '')
                wave_period = get_wave_period(cx, cy, acq_time)
                period_source = "wavewatch3"
            except Exception as e:
                logger.warning(f"WaveWatch III failed: {e}. Using manual period.")
                wave_period = depth_cfg.manual_wave_period
                period_source = "manual_fallback"

        # Step 4: Depth inversion
        logger.info("Step 4/4: Depth inversion...")
        result = invert_depth(
            swell,
            wave_period=wave_period,
            max_depth_m=depth_cfg.max_depth_m,
            gravity=depth_cfg.gravity,
            max_iterations=depth_cfg.max_iterations,
            convergence_tol=depth_cfg.convergence_tol,
        )
        result.wave_period_source = period_source

        return result

    def _export_results(self, result: BathymetryResult, output_dir: Path):
        """Export bathymetry result as GeoTIFF."""
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.config.export_geotiff and result.geo is not None:
            tiff_path = output_dir / "bathymetry_depth.tif"
            RasterIO.write_geotiff(
                str(tiff_path),
                result.depth,
                geo_transform=(
                    result.geo.origin_x,
                    result.geo.pixel_size_x,
                    0,
                    result.geo.origin_y,
                    0,
                    result.geo.pixel_size_y,
                ),
                crs_wkt=result.geo.crs_wkt,
                nodata=-9999.0,
            )
            logger.info(f"Exported depth GeoTIFF: {tiff_path}")

            # Uncertainty
            unc_path = output_dir / "bathymetry_uncertainty.tif"
            RasterIO.write_geotiff(
                str(unc_path),
                result.uncertainty,
                geo_transform=(
                    result.geo.origin_x,
                    result.geo.pixel_size_x,
                    0,
                    result.geo.origin_y,
                    0,
                    result.geo.pixel_size_y,
                ),
                crs_wkt=result.geo.crs_wkt,
                nodata=-9999.0,
            )
            logger.info(f"Exported uncertainty GeoTIFF: {unc_path}")
```

**Step 2: Update `ocean_rs/sar/core/__init__.py`**

```python
"""
Core module for OceanRS SAR Bathymetry Toolkit.

Contains data models and the main bathymetry pipeline orchestrator.
"""

from .data_models import (
    ImageType,
    GeoTransform,
    OceanImage,
    SwellField,
    BathymetryResult,
)
from .bathymetry_pipeline import BathymetryPipeline

__all__ = [
    'ImageType',
    'GeoTransform',
    'OceanImage',
    'SwellField',
    'BathymetryResult',
    'BathymetryPipeline',
]
```

**Step 3: Verify syntax**

Run: `python -m py_compile ocean_rs/sar/core/bathymetry_pipeline.py`

**Step 4: Commit**

```bash
git add ocean_rs/sar/core/
git commit -m "feat(sar): add bathymetry pipeline orchestrator"
```

---

## Task 11: GUI Theme

**Files:**
- Create: `ocean_rs/sar/gui/__init__.py`
- Create: `ocean_rs/sar/gui/theme.py`

**Step 1: Create `ocean_rs/sar/gui/theme.py`**

Copy the pattern from `ocean_rs/optical/gui/theme.py` but with ocean/teal palette. Same `ThemeManager` class, same `FONTS` and `ICONS`, different `COLORS`.

```python
"""
Theme Manager for SAR Bathymetry Toolkit GUI.

Ocean/teal palette distinct from the optical GUI's blue palette.
Same ThemeManager pattern as ocean_rs/optical/gui/theme.py.
"""

import tkinter as tk
from tkinter import ttk
import logging

logger = logging.getLogger('ocean_rs')


class ThemeManager:
    """Manages SAR GUI theming and styling."""

    COLORS = {
        # Primary colors - teal/ocean
        'primary': '#0e7490',
        'primary_hover': '#0c6478',
        'primary_light': '#cffafe',

        # Status colors (shared with optical)
        'success': '#16a34a',
        'success_light': '#dcfce7',
        'warning': '#d97706',
        'warning_light': '#fef3c7',
        'error': '#dc2626',
        'error_light': '#fee2e2',
        'info': '#0891b2',
        'info_light': '#cffafe',

        # Neutral colors
        'bg_main': '#f0fdfa',
        'bg_card': '#ffffff',
        'bg_hover': '#f0f9ff',
        'bg_active': '#e0f2fe',

        # Text colors
        'text_primary': '#1e293b',
        'text_secondary': '#64748b',
        'text_muted': '#94a3b8',
        'text_inverse': '#ffffff',

        # Border colors
        'border': '#e2e8f0',
        'border_focus': '#0e7490',
        'border_error': '#dc2626',

        # Section colors
        'section_search': '#ecfeff',
        'section_download': '#f0f9ff',
        'section_processing': '#f0fdf4',
        'section_results': '#fefce8',
    }

    FONTS = {
        'title': ('Calibri', 18, 'bold'),
        'subtitle': ('Calibri', 13, 'bold'),
        'heading': ('Calibri', 12, 'bold'),
        'body': ('Calibri', 11),
        'small': ('Calibri', 10),
        'mono': ('Consolas', 10),
    }

    ICONS = {
        'expand': '\u25BC',
        'collapse': '\u25B6',
        'check': '\u2713',
        'cross': '\u2717',
        'search': '\u2315',
        'download': '\u2B07',
        'process': '\u2699',
        'map': '\u2316',
    }

    def __init__(self, root):
        """Apply theme to root window."""
        self.root = root
        self._configure_styles()

    def _configure_styles(self):
        """Configure ttk styles with SAR theme."""
        style = ttk.Style()
        style.theme_use('clam')

        self.root.configure(bg=self.COLORS['bg_main'])

        # Button styles
        style.configure('Primary.TButton',
                        background=self.COLORS['primary'],
                        foreground=self.COLORS['text_inverse'],
                        font=self.FONTS['body'],
                        padding=(15, 8))
        style.map('Primary.TButton',
                  background=[('active', self.COLORS['primary_hover'])])

        style.configure('Success.TButton',
                        background=self.COLORS['success'],
                        foreground=self.COLORS['text_inverse'],
                        font=self.FONTS['body'],
                        padding=(15, 8))

        style.configure('Danger.TButton',
                        background=self.COLORS['error'],
                        foreground=self.COLORS['text_inverse'],
                        font=self.FONTS['body'],
                        padding=(15, 8))

        # Frame styles
        style.configure('Card.TFrame',
                        background=self.COLORS['bg_card'])
        style.configure('Main.TFrame',
                        background=self.COLORS['bg_main'])

        # Label styles
        style.configure('Title.TLabel',
                        background=self.COLORS['bg_main'],
                        foreground=self.COLORS['text_primary'],
                        font=self.FONTS['title'])
        style.configure('Heading.TLabel',
                        background=self.COLORS['bg_card'],
                        foreground=self.COLORS['text_primary'],
                        font=self.FONTS['heading'])
        style.configure('Body.TLabel',
                        background=self.COLORS['bg_card'],
                        foreground=self.COLORS['text_primary'],
                        font=self.FONTS['body'])
        style.configure('Status.TLabel',
                        background=self.COLORS['bg_main'],
                        foreground=self.COLORS['text_secondary'],
                        font=self.FONTS['small'])

        # LabelFrame
        style.configure('TLabelframe',
                        background=self.COLORS['bg_card'])
        style.configure('TLabelframe.Label',
                        background=self.COLORS['bg_card'],
                        foreground=self.COLORS['text_primary'],
                        font=self.FONTS['heading'])

        # Notebook
        style.configure('TNotebook',
                        background=self.COLORS['bg_main'])
        style.configure('TNotebook.Tab',
                        background=self.COLORS['bg_card'],
                        foreground=self.COLORS['text_primary'],
                        font=self.FONTS['body'],
                        padding=(12, 6))
        style.map('TNotebook.Tab',
                  background=[('selected', self.COLORS['primary_light'])],
                  foreground=[('selected', self.COLORS['primary'])])

        # Progressbar
        style.configure('TProgressbar',
                        background=self.COLORS['primary'],
                        troughcolor=self.COLORS['bg_main'])

        # Treeview (for search results table)
        style.configure('Treeview',
                        background=self.COLORS['bg_card'],
                        foreground=self.COLORS['text_primary'],
                        font=self.FONTS['body'],
                        rowheight=25)
        style.configure('Treeview.Heading',
                        background=self.COLORS['primary_light'],
                        foreground=self.COLORS['primary'],
                        font=self.FONTS['heading'])
```

**Step 2: Create `ocean_rs/sar/gui/__init__.py`**

```python
"""
GUI module for OceanRS SAR Bathymetry Toolkit.

4-tab interface:
    1. Search & Select - AOI, date range, sensor filters, results table
    2. Download & Credentials - Auth, download queue, progress
    3. Processing - SNAP, FFT params, depth inversion options
    4. Results & Monitor - System info, log, results, export
"""

__all__ = []
```

**Step 3: Verify syntax**

Run: `python -m py_compile ocean_rs/sar/gui/theme.py`
Run: `python -m py_compile ocean_rs/sar/gui/__init__.py`

**Step 4: Commit**

```bash
git add ocean_rs/sar/gui/
git commit -m "feat(sar): add GUI theme (ocean/teal palette)"
```

---

## Task 12: GUI Tabs

**Files:**
- Create: `ocean_rs/sar/gui/tabs/__init__.py`
- Create: `ocean_rs/sar/gui/tabs/search_tab.py`
- Create: `ocean_rs/sar/gui/tabs/download_tab.py`
- Create: `ocean_rs/sar/gui/tabs/processing_tab.py`
- Create: `ocean_rs/sar/gui/tabs/results_tab.py`

Each tab follows the factory function pattern from optical: `create_XXX_tab(gui, notebook) -> tab_index`.

These are large files. Implement each tab as a separate step, verify syntax after each.

The full tab code will be written during implementation. The pattern for each is:

```python
def create_XXX_tab(gui, notebook):
    frame = ttk.Frame(notebook)
    tab_index = notebook.add(frame, text="Tab Name")
    # ... widgets ...
    return tab_index
```

**Step 1: Create `ocean_rs/sar/gui/tabs/__init__.py`**

```python
"""
GUI Tab Modules for SAR Bathymetry Toolkit.

4 tabs, each as a factory function receiving the parent GUI instance.
"""

from .search_tab import create_search_tab
from .download_tab import create_download_tab
from .processing_tab import create_processing_tab
from .results_tab import create_results_tab

__all__ = [
    'create_search_tab',
    'create_download_tab',
    'create_processing_tab',
    'create_results_tab',
]
```

**Step 2-5: Create each tab file**

Each tab file is substantial (100-300 lines). During implementation, create them one at a time following the optical tab pattern. Key widgets per tab:

- **search_tab.py**: AOI text + browse button, date entries, sensor combo boxes, Search button, Treeview results table, Select All/Deselect All, scene count label
- **download_tab.py**: Username/password entries, Test Connection button, status label, Save to .env button, download dir selector, download queue Treeview, progress bars, Start/Stop buttons
- **processing_tab.py**: SNAP GPT path entry + browse, FFT parameter sliders/spinboxes, wave period source radio buttons + manual entry, depth max spinbox, compositing toggle + method, output dir selector, Start/Stop buttons
- **results_tab.py**: System info labels (CPU/RAM/disk), scrolled text log, results summary labels, export buttons

**Step 6: Verify syntax for all tabs**

Run: `python -m py_compile ocean_rs/sar/gui/tabs/search_tab.py`
(repeat for each tab)

**Step 7: Commit**

```bash
git add ocean_rs/sar/gui/tabs/
git commit -m "feat(sar): add 4 GUI tabs (search, download, processing, results)"
```

---

## Task 13: GUI Handlers, Config I/O, Processing Controller

**Files:**
- Create: `ocean_rs/sar/gui/handlers.py`
- Create: `ocean_rs/sar/gui/config_io.py`
- Create: `ocean_rs/sar/gui/processing_controller.py`

These follow the exact same patterns as their optical counterparts. Key differences:

- **handlers.py**: search button handler, download handlers, credential validation
- **config_io.py**: save/load SAR config JSON (never save credentials)
- **processing_controller.py**: start_processing launches BathymetryPipeline in daemon thread

**Step 1: Create each file following the optical pattern**

**Step 2: Verify syntax**

**Step 3: Commit**

```bash
git add ocean_rs/sar/gui/handlers.py ocean_rs/sar/gui/config_io.py ocean_rs/sar/gui/processing_controller.py
git commit -m "feat(sar): add GUI handlers, config I/O, and processing controller"
```

---

## Task 14: Main GUI Class

**Files:**
- Create: `ocean_rs/sar/gui/unified_gui.py`
- Modify: `ocean_rs/sar/gui/__init__.py`

**Step 1: Create `ocean_rs/sar/gui/unified_gui.py`**

Follow the pattern from `ocean_rs/optical/gui/unified_gui.py`:

```python
"""
Unified GUI for SAR Bathymetry Toolkit.

4-tab layout:
    1. Search & Select
    2. Download & Credentials
    3. Processing
    4. Results & Monitor
"""

import sys
import logging
import tkinter as tk
from tkinter import ttk

from ..config import SARProcessingConfig, SearchConfig, DownloadConfig, FFTConfig, DepthInversionConfig
from .theme import ThemeManager
from .tabs import create_search_tab, create_download_tab, create_processing_tab, create_results_tab

logger = logging.getLogger('ocean_rs')


def bring_window_to_front(window):
    """Bring window to front (same as optical)."""
    try:
        window.lift()
        window.attributes('-topmost', True)
        window.focus_force()
        window.update_idletasks()
        window.update()
        window.after(100, lambda: window.attributes('-topmost', False))
        if sys.platform.startswith('win'):
            try:
                import ctypes
                hwnd = ctypes.windll.user32.GetActiveWindow()
                ctypes.windll.user32.FlashWindow(hwnd, True)
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"Could not bring window to front: {e}")


class UnifiedSARGUI:
    """Unified GUI for SAR Bathymetry Toolkit."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("OceanRS \u2014 SAR Bathymetry Toolkit v0.1")

        self.theme = ThemeManager(self.root)

        # Window sizing
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = 1000
        window_height = max(800, int(screen_height * 0.8))
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")

        bring_window_to_front(self.root)

        self._init_configurations()
        self._init_state_variables()
        self._init_tk_variables()
        self._setup_gui()

    def _init_configurations(self):
        self.config = SARProcessingConfig()

    def _init_state_variables(self):
        self.processing_active = False
        self.download_active = False
        self.processing_thread = None
        self.download_thread = None
        self.search_results = []
        self.selected_scenes = []

    def _init_tk_variables(self):
        # Search variables
        self.aoi_var = tk.StringVar()
        self.start_date_var = tk.StringVar()
        self.end_date_var = tk.StringVar()
        self.platform_var = tk.StringVar(value="Sentinel-1")
        self.beam_mode_var = tk.StringVar(value="IW")
        self.polarization_var = tk.StringVar(value="VV+VH")
        self.orbit_dir_var = tk.StringVar(value="")

        # Credential variables
        self.username_var = tk.StringVar()
        self.password_var = tk.StringVar()
        self.download_dir_var = tk.StringVar()

        # Processing variables
        self.snap_gpt_var = tk.StringVar()
        self.tile_size_var = tk.DoubleVar(value=512.0)
        self.overlap_var = tk.DoubleVar(value=0.5)
        self.min_wavelength_var = tk.DoubleVar(value=50.0)
        self.max_wavelength_var = tk.DoubleVar(value=600.0)
        self.confidence_var = tk.DoubleVar(value=0.3)
        self.wave_source_var = tk.StringVar(value="wavewatch3")
        self.manual_period_var = tk.DoubleVar(value=10.0)
        self.max_depth_var = tk.DoubleVar(value=100.0)
        self.compositing_var = tk.BooleanVar(value=True)
        self.compositing_method_var = tk.StringVar(value="weighted_median")
        self.output_dir_var = tk.StringVar()

        # Status variables
        self.status_var = tk.StringVar(value="Ready")
        self.progress_var = tk.DoubleVar(value=0.0)

    def _setup_gui(self):
        # Notebook
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create tabs
        self.tab_indices = {}
        self.tab_indices['search'] = create_search_tab(self, self.notebook)
        self.tab_indices['download'] = create_download_tab(self, self.notebook)
        self.tab_indices['processing'] = create_processing_tab(self, self.notebook)
        self.tab_indices['results'] = create_results_tab(self, self.notebook)

        # Status bar
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        ttk.Label(status_frame, textvariable=self.status_var,
                  style='Status.TLabel').pack(side=tk.LEFT, padx=5)
        ttk.Progressbar(status_frame, variable=self.progress_var,
                       maximum=100, length=200).pack(side=tk.RIGHT, padx=5)

        # Close handler
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _on_closing(self):
        if self.processing_active or self.download_active:
            from tkinter import messagebox
            if not messagebox.askokcancel("Quit", "Processing is active. Quit anyway?"):
                return
        self.root.destroy()

    def run(self):
        self.root.mainloop()
```

**Step 2: Update `ocean_rs/sar/gui/__init__.py`**

```python
"""
GUI module for OceanRS SAR Bathymetry Toolkit.
"""

from .unified_gui import UnifiedSARGUI, bring_window_to_front

__all__ = [
    'UnifiedSARGUI',
    'bring_window_to_front',
]
```

**Step 3: Verify syntax**

Run: `python -m py_compile ocean_rs/sar/gui/unified_gui.py`

**Step 4: Commit**

```bash
git add ocean_rs/sar/gui/
git commit -m "feat(sar): add main GUI class (UnifiedSARGUI)"
```

---

## Task 15: Entry Points (main.py, __main__.py, __init__.py, run_sar_gui.py)

**Files:**
- Create: `ocean_rs/sar/main.py`
- Modify: `ocean_rs/sar/__main__.py`
- Modify: `ocean_rs/sar/__init__.py`
- Create: `run_sar_gui.py`

**Step 1: Create `ocean_rs/sar/main.py`**

Follow `ocean_rs/optical/main.py` pattern:

```python
"""
Main entry point for OceanRS SAR — SAR Bathymetry Toolkit.

Usage:
    GUI mode (default):
        python -m ocean_rs.sar

    CLI mode:
        python -m ocean_rs.sar --aoi "POLYGON(...)" --start 2024-01-01 --end 2024-06-01 -o results/
"""

import os
import sys
import logging
import argparse
import tkinter as tk
from tkinter import messagebox

from ocean_rs.shared import setup_enhanced_logging

logger = logging.getLogger('ocean_rs')


def _check_dependencies():
    """Check for required dependencies."""
    missing = []
    for pkg in ['numpy', 'scipy']:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    try:
        from osgeo import gdal
    except ImportError:
        missing.append('gdal')
    return missing


def cli_main():
    """CLI interface for batch bathymetry processing."""
    parser = argparse.ArgumentParser(
        description="OceanRS SAR — SAR Bathymetry Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m ocean_rs.sar --aoi "POLYGON((-9.5 38.5, -9.0 38.5, -9.0 39.0, -9.5 39.0, -9.5 38.5))" --start 2024-01-01 --end 2024-06-01 -o results/
  python -m ocean_rs.sar --help
        """
    )
    parser.add_argument("--aoi", required=True, help="AOI as WKT polygon")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("--platform", default="Sentinel-1", help="SAR platform")
    parser.add_argument("--beam-mode", default="IW", help="Beam mode")
    parser.add_argument("--wave-period", type=float, help="Manual wave period (seconds)")
    parser.add_argument("--max-depth", type=float, default=100.0, help="Max depth (m)")

    args = parser.parse_args()

    from .config import SARProcessingConfig, SearchConfig, DepthInversionConfig
    from .download import search_scenes, BatchDownloader, CredentialManager
    from .core import BathymetryPipeline

    config = SARProcessingConfig()
    config.search_config.aoi_wkt = args.aoi
    config.search_config.start_date = args.start
    config.search_config.end_date = args.end
    config.search_config.platform = args.platform
    config.search_config.beam_mode = args.beam_mode
    config.output_directory = args.output
    config.depth_config.max_depth_m = args.max_depth

    if args.wave_period:
        config.depth_config.wave_period_source = "manual"
        config.depth_config.manual_wave_period = args.wave_period

    os.makedirs(args.output, exist_ok=True)

    # Search
    print(f"Searching for {args.platform} scenes...")
    scenes = search_scenes(
        aoi_wkt=args.aoi,
        start_date=args.start,
        end_date=args.end,
        platform=args.platform,
        beam_mode=args.beam_mode,
    )
    print(f"Found {len(scenes)} scenes")

    if not scenes:
        print("No scenes found. Adjust search parameters.")
        return False

    # Download
    creds = CredentialManager()
    downloader = BatchDownloader(creds)
    download_dir = os.path.join(args.output, "Downloads")
    paths = downloader.download_scenes(scenes, download_dir)

    if not paths:
        print("No scenes downloaded successfully.")
        return False

    # Process
    pipeline = BathymetryPipeline(config)
    result = pipeline.process_scenes(paths)

    return result is not None


def main():
    """Main entry point."""
    try:
        missing = _check_dependencies()
        if missing:
            print(f"Missing dependencies: {missing}")
            print("Install with: conda install " + " ".join(missing))
            sys.exit(1)

        if len(sys.argv) > 1:
            success = cli_main()
            sys.exit(0 if success else 1)
        else:
            logger.info("Starting SAR Bathymetry Toolkit GUI...")
            from .gui import UnifiedSARGUI
            app = UnifiedSARGUI()
            app.run()

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Critical error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        try:
            if len(sys.argv) == 1:
                root = tk.Tk()
                root.withdraw()
                messagebox.showerror("Critical Error", f"{e}\n\nCheck log for details.")
        except Exception:
            pass
        sys.exit(1)
```

**Step 2: Update `ocean_rs/sar/__main__.py`**

```python
"""
Package entry point for OceanRS SAR Bathymetry Toolkit.

    python -m ocean_rs.sar
"""

from .main import main

if __name__ == "__main__":
    main()
```

**Step 3: Update `ocean_rs/sar/__init__.py`**

```python
"""
OceanRS SAR — SAR Bathymetry Toolkit

SAR-based nearshore bathymetry estimation using swell wave analysis.
"""

__version__ = "0.1.0"
__author__ = "Pedro Goncalves"

from .config import (
    SARProcessingConfig,
    SearchConfig,
    DownloadConfig,
    FFTConfig,
    DepthInversionConfig,
    CompositingConfig,
)
from .core import (
    ImageType,
    GeoTransform,
    OceanImage,
    SwellField,
    BathymetryResult,
    BathymetryPipeline,
)
from .download import (
    CredentialManager,
    CredentialError,
    SceneMetadata,
    search_scenes,
    BatchDownloader,
)
from .sensors import (
    SensorAdapter,
    Sentinel1Adapter,
)
from .bathymetry import (
    extract_swell,
    get_wave_period,
    invert_depth,
    composite_bathymetry,
)

__all__ = [
    # Config
    'SARProcessingConfig', 'SearchConfig', 'DownloadConfig',
    'FFTConfig', 'DepthInversionConfig', 'CompositingConfig',
    # Core
    'ImageType', 'GeoTransform', 'OceanImage', 'SwellField',
    'BathymetryResult', 'BathymetryPipeline',
    # Download
    'CredentialManager', 'CredentialError', 'SceneMetadata',
    'search_scenes', 'BatchDownloader',
    # Sensors
    'SensorAdapter', 'Sentinel1Adapter',
    # Bathymetry
    'extract_swell', 'get_wave_period', 'invert_depth', 'composite_bathymetry',
]
```

**Step 4: Create `run_sar_gui.py`**

```python
"""
Launch OceanRS SAR Bathymetry Toolkit GUI.

Open this file in Spyder and press F5 (Run) to start the application.
"""

from ocean_rs.sar.main import main

main()
```

**Step 5: Verify syntax**

Run: `python -m py_compile ocean_rs/sar/main.py`
Run: `python -m py_compile ocean_rs/sar/__main__.py`
Run: `python -m py_compile ocean_rs/sar/__init__.py`
Run: `python -m py_compile run_sar_gui.py`

**Step 6: Commit**

```bash
git add ocean_rs/sar/ run_sar_gui.py
git commit -m "feat(sar): add entry points (main.py, __main__.py, run_sar_gui.py)"
```

---

## Task 16: Final Verification & Integration

**Step 1: Verify all files compile**

```bash
find ocean_rs/sar -name "*.py" -exec python -m py_compile {} \;
```

**Step 2: Import test**

```bash
python -c "from ocean_rs.sar import BathymetryPipeline, SARProcessingConfig, extract_swell; print('SAR imports OK')"
```

**Step 3: Entry point test**

```bash
python -m ocean_rs.sar --help
```

**Step 4: GUI launch test**

```bash
python run_sar_gui.py
```

**Step 5: Stage all changes**

```bash
git add ocean_rs/sar/ run_sar_gui.py docs/plans/
git status
```

Prepare commit message for user (per CLAUDE.md rule 10 — do NOT commit):

```
feat(sar): implement SAR Bathymetry Toolkit Phase 1

- Core data models (OceanImage, SwellField, BathymetryResult)
- Config dataclasses (SARProcessingConfig, FFTConfig, etc.)
- Credential manager (env vars, .env, GUI fallback)
- ASF scene discovery (asf_search wrapper)
- Batch downloader with auth and resume
- Sentinel-1 adapter (SNAP GPT preprocessing)
- FFT swell extraction (2D FFT per tile)
- WaveWatch III wave period retrieval
- Depth inversion (linear dispersion, Newton-Raphson)
- Multi-temporal compositor
- Pipeline orchestrator
- 4-tab tkinter GUI (search, download, processing, results)
- CLI entry point (python -m ocean_rs.sar)
```
