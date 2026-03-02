# OceanRS SAR Bathymetry Toolkit — Design Document

**Date:** 2026-03-02
**Author:** Pedro Goncalves
**Package:** `ocean_rs.sar`
**Version:** 0.1.0 (Phase 1)

---

## Overview

SAR-based nearshore bathymetry estimation using swell wave analysis. Extracts dominant swell wavelength from SAR imagery via 2D FFT, retrieves wave period from WaveWatch III, and inverts depth using the linear dispersion relation. All operations driven from a tkinter GUI or CLI.

**Core physics:** `omega^2 = g * k * tanh(k * h)` — waves "feel" the bottom when depth < lambda/2.

---

## Package Structure (Phase 1)

```
ocean_rs/sar/
├── __init__.py              # Package exports, __version__ = "0.1.0"
├── __main__.py              # python -m ocean_rs.sar
├── main.py                  # CLI + GUI entry point
│
├── config/                  # Configuration dataclasses
│   ├── __init__.py
│   ├── sar_config.py        # SARProcessingConfig (master config)
│   └── download_config.py   # DownloadConfig, CredentialConfig
│
├── core/                    # Data models + orchestrator
│   ├── __init__.py
│   ├── data_models.py       # OceanImage, SwellField, BathymetryResult
│   └── bathymetry_pipeline.py  # Main orchestrator
│
├── sensors/                 # Sensor-specific adapters
│   ├── __init__.py
│   ├── base.py              # SensorAdapter ABC
│   └── sentinel1.py         # Sentinel1Adapter (SNAP GPT preprocessing)
│
├── download/                # Data discovery + download
│   ├── __init__.py
│   ├── scene_discovery.py   # ASF search wrapper (asf_search)
│   ├── batch_downloader.py  # Authenticated download with retry
│   └── credentials.py       # CredentialManager (env vars -> .env fallback)
│
├── bathymetry/              # Core science
│   ├── __init__.py
│   ├── fft_extractor.py     # 2D FFT -> dominant wavelength/direction
│   ├── wave_period.py       # WaveWatch III ERDDAP lookup
│   ├── depth_inversion.py   # Linear dispersion relation solver
│   └── compositor.py        # Multi-temporal depth compositing
│
└── gui/                     # Tkinter GUI
    ├── __init__.py
    ├── unified_gui.py       # UnifiedSARGUI main class
    ├── handlers.py          # Event handlers
    ├── config_io.py         # Config save/load JSON
    ├── processing_controller.py  # Background thread control
    ├── theme.py             # ThemeManager (ocean/teal palette)
    └── tabs/
        ├── __init__.py
        ├── search_tab.py    # AOI, date range, sensor, results table
        ├── download_tab.py  # Credentials, download queue, progress
        ├── processing_tab.py # SNAP, FFT params, depth inversion
        └── results_tab.py   # Map preview, statistics, export
```

---

## Data Models

### OceanImage (sensor-agnostic container)

```python
@dataclass
class OceanImage:
    data: np.ndarray              # 2D array
    image_type: ImageType         # ALPHA | PSEUDO_ALPHA | SIGMA0
    geo: GeoTransform             # origin, pixel_size, crs
    metadata: dict                # sensor, orbit, datetime
    pixel_spacing_m: float        # ground resolution
```

### ImageType hierarchy

```python
class ImageType(Enum):
    ALPHA = 3        # Quad-pol: best for bathymetry
    PSEUDO_ALPHA = 2 # Dual-pol: good
    SIGMA0 = 1       # Intensity: baseline
```

### SwellField (FFT output)

```python
@dataclass
class SwellField:
    wavelength: np.ndarray        # dominant wavelength (m)
    direction: np.ndarray         # wave direction (degrees)
    confidence: np.ndarray        # spectral peak SNR
    tile_centers: np.ndarray      # (row, col) centers
```

### BathymetryResult (final output)

```python
@dataclass
class BathymetryResult:
    depth: np.ndarray             # depth (m, positive down)
    uncertainty: np.ndarray       # depth uncertainty (m)
    method: str                   # "linear_dispersion"
    wave_period_source: str       # "wavewatch3" | "manual"
    geo: GeoTransform
```

---

## Processing Pipeline

```
1. Discover  -->  2. Download  -->  3. Preprocess  -->  4. FFT
(asf_search)     (batch+auth)      (SNAP GPT)          (swell)
                                                          |
7. Export    <--  6. Composite <--  5. Invert     <------+
(GeoTIFF)        (temporal)        (dispersion)
```

### Step 1: Scene Discovery

- Uses `asf_search` Python API to search ASF DAAC
- Filters: platform, beam mode, polarization, orbit direction, date range
- AOI from WKT or shapefile (reuse `ocean_rs.shared.geometry_utils`)
- Results displayed in GUI Treeview table for user selection

### Step 2: Batch Download

- NASA Earthdata Login authentication
- Resume support for interrupted downloads
- Progress callbacks for GUI integration
- Concurrent downloads (configurable, default 2)

### Step 3: Sentinel-1 Preprocessing (SNAP GPT)

Processing chain:
1. Apply-Orbit-File — precise orbit state vectors
2. Thermal-Noise-Removal — remove thermal noise floor
3. Calibration — to sigma0 (or complex for polarimetric)
4. Terrain-Correction — Range-Doppler geocoding to UTM

Output: `OceanImage` with `image_type=SIGMA0`

### Step 4: FFT Swell Extraction

- Tile image with overlap (default 512m tiles, 50% overlap)
- Apply Hanning window + 2D FFT per tile
- Compute power spectrum, mask to wavelength range (50-600m)
- Find dominant peak -> wavelength + direction
- Compute confidence from spectral peak SNR

### Step 5: Wave Period Retrieval

- Primary: WaveWatch III via NOAA ERDDAP (free, no auth)
- Fallback: Manual entry in GUI
- Cache results to avoid repeated API calls

### Step 6: Depth Inversion

- Linear dispersion relation: `omega^2 = g * k * tanh(k * h)`
- Newton-Raphson iteration to solve for depth h
- Uncertainty from wavelength measurement precision
- Max depth limit (default 100m)

### Step 7: Multi-temporal Compositing

- Combine multiple temporal observations
- Weighted by confidence score
- Median reduces outlier influence
- Methods: weighted_median (default), weighted_mean

### Step 8: Export

- GeoTIFF with proper NoData, CRS, metadata
- PNG preview with colorbar
- Uses `ocean_rs.shared.RasterIO`

---

## GUI Design

### Window

Title: `"OceanRS — SAR Bathymetry Toolkit v0.1"`
Launch: `python -m ocean_rs.sar` (or `python run_sar_gui.py` from Spyder)

### Tab 1: Search & Select

- AOI input: WKT text field, or browse for shapefile, or map draw
- Date range: start/end date pickers
- Sensor filters: platform, beam mode, polarization, orbit direction
- **"Search" button** -> populates results Treeview table
- Results table: Scene ID, Date, Orbit, Polarization, Pass, Size
- Row checkboxes for selection, Select All / Deselect All
- Scene count + total download size

### Tab 2: Download & Credentials

- Username + password fields (masked)
- "Test Connection" button with green/red status
- "Save to .env" button (writes to gitignored .env)
- Download directory selector
- Download queue with per-scene status
- Per-scene + overall progress bars
- Start/Stop download buttons

### Tab 3: Processing

- SNAP GPT path (auto-detect or browse)
- Target CRS / resolution
- FFT: tile size, overlap, wavelength range, confidence threshold
- Wave period: auto (WaveWatch III) vs manual entry
- Depth: max depth limit, compositing method
- Output directory
- Start/Stop processing buttons

### Tab 4: Results & Monitor

- System info (CPU, RAM, disk)
- Processing log (scrolled text, colored)
- Results summary (scenes processed, depth range, coverage)
- Map preview (if tkintermapview available)
- Export: GeoTIFF, PNG
- Statistics: min/max/mean depth, uncertainty

### Theme

Ocean/teal palette distinct from optical's blue:

```python
COLORS = {
    'primary': '#0e7490',
    'bg_main': '#f0fdfa',
    'section_search': '#ecfeff',
    'section_download': '#f0f9ff',
    'section_processing': '#f0fdf4',
    'section_results': '#fefce8',
}
```

---

## Credential Security

```
Priority 1: Environment variables (EARTHDATA_USERNAME, EARTHDATA_PASSWORD)
Priority 2: .env file in project root (gitignored)
Priority 3: GUI prompt (offers "Save to .env" checkbox)
Priority 4: Raise CredentialError with setup instructions
```

- Config JSON files store processing parameters only, never credentials
- `.env` is in `.gitignore`
- GUI "Test Connection" verifies via `asf_search.ASFSession`

---

## Configuration (JSON)

```json
{
    "version": "1.0",
    "search": {
        "platform": "Sentinel-1",
        "beam_mode": "IW",
        "polarization": "VV+VH",
        "aoi_wkt": "POLYGON(...)"
    },
    "processing": {
        "tile_size_m": 512,
        "overlap": 0.5,
        "wavelength_range": [50, 600],
        "confidence_threshold": 0.3,
        "wave_period_source": "wavewatch3",
        "max_depth_m": 100,
        "compositing_method": "weighted_median"
    },
    "output": {
        "output_directory": "",
        "export_geotiff": true,
        "export_png": true
    }
}
```

---

## Shared Utilities (from ocean_rs.shared)

| Utility | SAR Usage |
|---|---|
| RasterIO | Read/write GeoTIFF bathymetry outputs |
| setup_enhanced_logging | Structured logging with step tracking |
| SafeMathNumPy | Safe division in FFT/inversion |
| MemoryManager | Large SAR scenes need memory tracking |
| proj_fix | PROJ/GDAL environment setup |
| geometry_utils | AOI loading from WKT/shapefile |

---

## Dependencies (Phase 1)

| Package | Purpose | License |
|---|---|---|
| numpy | Arrays, FFT | BSD |
| scipy | Newton-Raphson, signal processing | BSD |
| asf_search | ASF DAAC scene discovery/download | BSD |
| python-dotenv | .env credential loading | BSD |
| requests | HTTP (ERDDAP, downloads) | Apache |
| rasterio / GDAL | GeoTIFF I/O | BSD/MIT |
| psutil | System monitoring | BSD |
| tkinter | GUI (stdlib) | PSF |
| SNAP GPT | Sentinel-1 preprocessing | GPL (standalone) |

---

## Phase Roadmap

| Phase | Scope | Status |
|---|---|---|
| **1** | **Bathymetry core: S1 search/download/preprocess, FFT, depth inversion, GUI** | **This design** |
| 2 | Polarimetric decomposition (H/A/alpha), EMODNET validation, advanced compositing | Planned |
| 3 | NISAR + ALOS sensor adapters | Planned |
| 4 | InSAR wrappers (PyGMTSAR/ISCE2/ISCE3/MintPy) | Planned |
| 5 | Advanced: Kalman filter, uncertainty propagation, ensemble methods | Planned |

---

## Entry Points

```bash
# GUI mode (default)
python -m ocean_rs.sar

# CLI mode
python -m ocean_rs.sar --aoi "POLYGON(...)" --start 2024-01-01 --end 2024-06-01 -o results/

# Help
python -m ocean_rs.sar --help

# Spyder launcher
python run_sar_gui.py
```
