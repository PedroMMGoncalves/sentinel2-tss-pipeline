# Legacy Files

This folder contains the original monolithic implementation before modularization.

## Files

- `sentinel2_tss_pipeline_original.py` - Original ~10,150 line monolithic file containing all classes

## Why Keep This?

1. **Reference** - Compare old vs new implementation
2. **Rollback** - If needed, can restore original behavior
3. **Documentation** - Shows the evolution of the codebase

## Current Modular Structure

The codebase has been reorganized into:

```
src/sentinel2_tss_pipeline/
├── config/           # Configuration classes
├── utils/            # Utility functions
├── processors/       # Processing modules
│   ├── s2/          # Sentinel-2 SNAP processing
│   ├── tss/         # TSS estimation (Jiang)
│   ├── water_quality/  # Water quality parameters
│   └── visualization/  # RGB composites & indices
├── core/            # Core unified processor
└── gui/             # GUI components
```

## Date

Legacy snapshot created: 2026-01-13
