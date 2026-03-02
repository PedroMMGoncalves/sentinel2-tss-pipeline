# OceanRS — Ocean Remote Sensing Toolkit

## Global Claude Instructions

These rules apply to every project.

### Root Cause
No quick fixes. Always diagnose to the root cause and devise proper solutions. Never apply patches or workarounds unless the user explicitly asks.

### Security & Secrets
- Never hardcode secrets or commit them to git
- Use separate API tokens/credentials for dev, staging, and prod environments
- Validate all input server-side — never trust client data
- Add rate limiting on auth and write operations

### Architecture & Code Quality
- Design architecture before building — don't let it emerge from spaghetti
- Break up large view controllers/components early
- Wrap external API calls in a clean service layer (easier to cache, swap, or extend later)
- Version database schema changes through proper migrations
- Use real feature flags, not commented-out code

### Observability
- Add crash reporting from day one
- Implement persistent logging (not just console output)
- Include a /health endpoint for every service

### Environments & Deployment
- Maintain a real staging environment that mirrors production
- Set CORS to specific origins, never *
- Set up CI/CD early — deploys come from the pipeline, not a laptop
- Document how to run, build, and deploy the project

### Testing & Resilience
- Test unhappy paths: network failures, unexpected API responses, malformed data
- Test backup restores at least once — don't wait for an emergency
- Don't assume the happy path is sufficient

### Time Handling
- Store all timestamps in UTC
- Convert to local time only on display

### Discipline
- Fix hacky code now or create a tracked ticket with a deadline — "later" never comes
- Don't skip fundamentals just because the project is small

---

## Project Rules

1. First think through the problem, read the codebase for relevant files
2. Before implementing, check in with me to verify the approach
3. Work on items one at a time, marking each complete immediately
4. Give high-level explanations of changes at each step
5. SIMPLICITY IS KING: Minimal code impact, no over-engineering
6. NO LAZINESS: Find root causes, no temporary fixes, no placeholders
7. Always verify syntax with `python -m py_compile` after edits
8. For ArcGIS tools: Follow the existing toolbox pattern exactly
9. For raster operations: Always consider memory limits (256GB available)
10. GIT: Prepare commit message and stage changes - DO NOT commit (user commits manually)
11. No silent fallbacks - if a method fails, report error clearly
12. No backward-compatibility aliases - old imports must fail with ImportError

---

## Project Overview

OceanRS is an ocean remote sensing toolkit with two main components:

- **Optical** (`ocean_rs.optical`): Sentinel-2 water quality and TSS estimation using Jiang et al. (2021)
- **SAR** (`ocean_rs.sar`): SAR-based bathymetry estimation (planned)

**Author:** Pedro Goncalves (LNEG)
**Version:** 3.0.0
**License:** Research use

## Scientific References

### Primary Algorithm (Optical)
```
Jiang, D., Matsushita, B., Pahlevan, N., et al. (2021).
"Remotely Estimating Total Suspended Solids Concentration in Clear to
Extremely Turbid Waters Using a Novel Semi-Analytical Method."
Remote Sensing of Environment, 258, 112386.
DOI: https://doi.org/10.1016/j.rse.2021.112386
```

### Water Type Classification
- **Type I** (Clear): Rrs(490) > Rrs(560) - uses 560nm
- **Type II** (Moderately turbid): Rrs(490) > Rrs(620) - uses 665nm
- **Type III** (Highly turbid): Default - uses 740nm
- **Type IV** (Extremely turbid): Rrs(740) > Rrs(490) AND Rrs(740) > 0.010 - uses 865nm

### SNAP C2RCC Formulas
- **TSM:** `TSM = 1.06 * (bpart + bwit)^0.942`
- **CHL:** `CHL = apig^1.04 * 21.0`

### Trophic State Index (Carlson 1977)
- **TSI:** `TSI = 9.81 * ln(CHL) + 30.6`
- Scale: <40 Oligotrophic, 40-50 Mesotrophic, 50-70 Eutrophic, >70 Hypereutrophic

## Project Structure

```
ocean-rs/                            # Repository root
├── ocean_rs/                        # Umbrella package
│   ├── __init__.py                  # OceanRS v3.0.0
│   │
│   ├── shared/                      # Shared utilities (both optical & SAR)
│   │   ├── __init__.py
│   │   ├── logging_utils.py         # ColoredFormatter, StepTracker
│   │   ├── math_utils.py            # SafeMathNumPy
│   │   ├── memory_manager.py        # MemoryManager
│   │   ├── raster_io.py             # RasterIO (GDAL wrapper)
│   │   ├── geometry_utils.py        # load_geometry, validate_wkt
│   │   └── proj_fix.py              # PROJ/GDAL environment config
│   │
│   ├── optical/                     # Sentinel-2 TSS Pipeline
│   │   ├── __init__.py              # Package exports
│   │   ├── __main__.py              # python -m ocean_rs.optical
│   │   ├── main.py                  # CLI and GUI entry point
│   │   ├── config/                  # Configuration dataclasses
│   │   │   ├── enums.py             # ProcessingMode, ProductType
│   │   │   ├── s2_config.py         # ResamplingConfig, SubsetConfig, C2RCCConfig
│   │   │   ├── tss_config.py        # TSSConfig
│   │   │   ├── output_categories.py # OutputCategoryConfig (6 toggles)
│   │   │   ├── water_quality_config.py
│   │   │   └── processing_config.py
│   │   ├── utils/                   # Optical-specific utilities
│   │   │   ├── product_detector.py  # ProductDetector, SystemMonitor
│   │   │   └── output_structure.py  # OutputStructure
│   │   ├── processors/              # Processing modules
│   │   │   ├── tsm_chl_calculator.py
│   │   │   ├── tss_processor.py     # TSSConstants, TSSProcessor
│   │   │   ├── water_quality_processor.py
│   │   │   ├── visualization_processor.py  # RGB + 12 spectral indices
│   │   │   └── c2rcc_processor.py
│   │   ├── core/
│   │   │   └── unified_processor.py # UnifiedS2TSSProcessor
│   │   └── gui/                     # 5-tab tkinter GUI
│   │       ├── unified_gui.py
│   │       ├── handlers.py
│   │       ├── config_io.py
│   │       ├── processing_controller.py
│   │       ├── theme.py
│   │       ├── widgets/
│   │       └── tabs/
│   │
│   └── sar/                         # SAR Bathymetry Toolkit (scaffold)
│       ├── __init__.py
│       └── __main__.py
│
├── run_gui.py                       # Spyder launcher
├── CLAUDE.md                        # This file
└── README.md
```

## Output Category System (6 toggles)

| Category | Default | Products | Folder |
|----------|---------|----------|--------|
| TSS | ON | TSS, Absorption, Backscattering, ReferenceBand, WaterTypes, ValidMask, Legend | TSS/ |
| RGB | ON | 15 unique composites (deduplicated) | RGB/ |
| Indices | ON | NDWI, MNDWI, NDTI, NDMI, AWEI, WI, WRI, NDCI, CHL_RED_EDGE, GNDVI, TSI_Turbidity, CDOM | Indices/ |
| WaterClarity | OFF | SecchiDepth, Kd, ClarityIndex, EuphoticDepth, BeamAttenuation, RelativeTurbidity | WaterClarity/ |
| HAB | OFF | NDCI/MCI bloom detection, probability, risk level, potential bloom, biomass alerts | HAB/ |
| TrophicState | OFF | TSI_Chlorophyll, TSI_Secchi, TrophicClass | TrophicState/ |

## Naming Conventions

### File → Class Mapping (under ocean_rs/optical/)

| File | Classes |
|------|---------|
| config/tss_config.py | TSSConfig |
| config/output_categories.py | OutputCategoryConfig |
| processors/tss_processor.py | TSSConstants, TSSProcessor |
| processors/c2rcc_processor.py | C2RCCProcessor, ProcessingStatus |
| processors/tsm_chl_calculator.py | TSMCHLCalculator, ProcessingResult |
| processors/visualization_processor.py | VisualizationProcessor |
| processors/water_quality_processor.py | WaterQualityConstants, WaterQualityProcessor |
| core/unified_processor.py | UnifiedS2TSSProcessor |

## How to Run

### Package Entry Point (GUI)
```bash
python -m ocean_rs.optical
```

### CLI Mode
```bash
python -m ocean_rs.optical -i /path/to/L1C -o /path/to/results
python -m ocean_rs.optical --help
```

### Import Test
```bash
python -c "from ocean_rs.optical import UnifiedS2TSSProcessor, C2RCCProcessor, TSSProcessor; print('OK')"
```

### SAR Toolkit (scaffold)
```bash
python -m ocean_rs.sar
```

## Processing Modes

| Mode | Input | Output |
|------|-------|--------|
| COMPLETE_PIPELINE | L1C .SAFE/.zip | C2RCC + TSS + Visualizations |
| S2_PROCESSING_ONLY | L1C .SAFE/.zip | C2RCC products |
| TSS_PROCESSING_ONLY | C2RCC .dim | TSS + Visualizations |

## Output Structure

```
output_folder/
├── <scene_name>/
│   ├── TSS/           (TSS products)
│   ├── RGB/           (RGB composites)
│   ├── Indices/       (Spectral indices)
│   ├── WaterClarity/  (if enabled)
│   ├── HAB/           (if enabled)
│   └── TrophicState/  (if enabled)
├── Intermediate/
│   ├── Geometric/     (Resampled L1C)
│   └── C2RCC/         (C2RCC products)
└── Logs/
```

## Water Masking

- **Auto NDWI+NIR** (default ON): `water = (NDWI > 0) AND (NIR(865nm) < 0.03)`
- Applied to: TSS, WaterClarity, HAB, TrophicState
- NOT applied to: RGB, Indices (full scene coverage)
- Override: user can provide shapefile for exact coastline control

## C2RCC NN Presets

| NN Selection | B8 Threshold | RTOSA OOS | AC Reflec OOS | Use Case |
|-------------|-------------|-----------|---------------|----------|
| C2RCC-Nets | B8 < 0.1 | 0.05 | 0.1 | Standard open water |
| C2X-Nets | B8 < 0.15 | 0.08 | 0.15 | Extended turbid water |
| C2X-COMPLEX-Nets | B8 < 0.2 | 0.1 | 0.2 | Surf zone, extreme turbidity |

Auto-applied via `C2RCCConfig.apply_nn_presets()` when configuration is saved.

## Config Versioning

- Current version: `2.0`
- Old v1.x configs are rejected with clear error message
- Config saved as JSON with `version` field

## Dependencies

### Required
- Python 3.8+
- numpy
- GDAL (osgeo)
- tkinter (GUI)
- psutil (system monitoring)

### Required for Processing
- SNAP GPT (ESA SNAP Graph Processing Tool)
- SNAP C2RCC module

### Optional
- geopandas (geometry loading)
- fiona (shapefile support)
- shapely (geometry operations)
- tkintermapview (interactive map in Spatial tab)
- tqdm (progress bars)

## Development Guidelines

### Import Patterns
- Shared utilities: `from ocean_rs.shared.raster_io import RasterIO`
- Optical-specific: `from ..utils.product_detector import ProductDetector` (relative)
- Within optical: `from ..config import ProcessingConfig` (relative)
- Logger name: `ocean_rs`

### Code Style
- Use dataclasses for configuration
- Type hints for all functions
- Docstrings with scientific references
- Logger name: `ocean_rs` (single logger for all modules)
- No `except: pass` - all errors must be logged or raised

### Testing
```bash
# Import test
python -c "from ocean_rs.optical import *; print('All imports OK')"

# Config test
python -c "from ocean_rs.optical.config import OutputCategoryConfig, TSSConfig, C2RCCConfig; c = C2RCCConfig(); c.net_set = 'C2X-COMPLEX-Nets'; c.apply_nn_presets(); print(f'B8 threshold: {c.valid_pixel_expression}'); print('OK')"

# Entry point test
python -m ocean_rs.optical --help
```

## Scientific Validation

### Verified Correct
- Jiang et al. 2021 - 4 water types classification
- QAA v6.0 formulas (560, 665, 740, 865nm)
- Sentinel-2 band mapping (B1=443nm, B2=490nm, etc.)
- Pure water constants (Pope & Fry, 1997)
- NDCI for HAB (Mishra & Mishra, 2012)
- MCI (Gitelson et al., 2008)
- Secchi Depth (Gordon, 1989)
- SNAP TSM/CHL coefficients
- TSI (Carlson 1977)

### TSS Ranges by Water Type
- Type I (clear): TSS < 10 g/m³
- Type II (moderately turbid): 10-50 g/m³
- Type III (highly turbid): 50-200 g/m³
- Type IV (extremely turbid): > 200 g/m³

## Key Files for Reference

| File | Purpose |
|------|---------|
| ocean_rs/optical/processors/tss_processor.py:1-80 | TSSConstants with water type thresholds |
| ocean_rs/optical/processors/water_quality_processor.py:1-50 | WaterQualityConstants with index formulas |
| ocean_rs/optical/processors/c2rcc_processor.py:1-80 | C2RCCProcessor SNAP graph creation |
| ocean_rs/optical/core/unified_processor.py | Main orchestrator |
| ocean_rs/optical/config/s2_config.py | C2RCCConfig with apply_nn_presets() |
| ocean_rs/optical/config/output_categories.py | OutputCategoryConfig (6 toggles) |
| ocean_rs/shared/logging_utils.py | StepTracker, ColoredFormatter |
| ocean_rs/shared/raster_io.py | RasterIO (GDAL wrapper) |

## Contact

For issues or questions about this toolkit, please open an issue on GitHub.
