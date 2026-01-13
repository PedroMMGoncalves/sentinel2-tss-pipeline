# Refactoring Proposal: Function Names and Module Organization

## Current Issues

1. **Vague function names** that don't clearly indicate what they do
2. **Mixed responsibilities** in single modules
3. **Inconsistent naming conventions** across modules
4. **Large monolithic classes** that should be split

---

## Current Structure Analysis

### processors/water_quality_processor.py

| Current Name | Issue | Proposed Name |
|--------------|-------|---------------|
| `create_advanced_processor()` | Too vague - what is "advanced"? | `create_water_quality_processor()` |
| `integrate_with_existing_pipeline()` | Unclear - which pipeline? | `process_water_quality_from_results()` |
| `calculate_water_clarity()` | OK but could be more specific | `calculate_secchi_depth_and_kd()` |
| `detect_harmful_algal_blooms()` | Good name | Keep |
| `calculate_trophic_state()` | Good name | Keep |

### processors/marine_viz.py

| Current Name | Issue | Proposed Name |
|--------------|-------|---------------|
| `S2MarineRGBGenerator` | OK but unclear scope | `RGBCompositeDefinitions` |
| `S2MarineVisualizationProcessor` | Too long, unclear | `VisualizationProcessor` |
| `process_marine_visualizations()` | Good | Keep |
| `_load_bands_from_geometric_products()` | Confusing - what's "geometric"? | `_load_bands_from_resampled_product()` |
| `_load_available_bands()` | Good | Keep |
| `_generate_rgb_composites()` | Good | Keep |
| `_generate_spectral_indices()` | Good | Keep |
| `_calculate_spectral_index()` | Good | Keep |
| `_create_robust_rgb_composite()` | Redundant "robust" | `_create_rgb_composite()` |
| `_apply_additional_contrast_enhancement()` | Redundant "additional" | `_apply_contrast_enhancement()` |
| `_save_rgb_geotiff()` | Good | Keep |
| `_save_single_band_geotiff()` | Good | Keep |
| `_create_visualization_summary()` | Good | Keep |
| `_cleanup_geometric_products()` | Confusing name | `_cleanup_intermediate_products()` |
| `_calculate_rgb_statistics()` | Good | Keep |
| `_calculate_band_statistics()` | Good | Keep |

### processors/jiang_processor.py

| Current Name | Issue | Proposed Name |
|--------------|-------|---------------|
| `JiangTSSProcessor` | Good | Keep |
| `process_jiang_tss()` | Good | Keep |
| `_load_bands_data()` | Good | Keep |
| `_apply_simple_unit_conversion()` | What's "simple"? | `_convert_rhow_to_rrs()` |
| `_validate_rrs_ranges()` | Good | Keep |
| `_update_band_mapping_for_mixed_types()` | Too long | `_update_band_mapping()` |
| `_process_advanced_algorithms()` | Too vague | `_process_water_quality_algorithms()` |
| `_extract_snap_chlorophyll()` | Good | Keep |
| `_apply_full_jiang_methodology()` | What's "full"? | `_apply_jiang_tss_algorithm()` |
| `_create_valid_pixel_mask()` | Good | Keep |
| `_process_valid_pixels()` | Good | Keep |
| `_estimate_tss_single_pixel()` | Good | Keep |
| `_estimate_rrs620_from_rrs665()` | Good | Keep |
| `_qaa_560()` | Unclear - add wavelength context | `_qaa_type1_560nm()` |
| `_qaa_665()` | Unclear | `_qaa_type2_665nm()` |
| `_qaa_740()` | Unclear | `_qaa_type3_740nm()` |
| `_qaa_865()` | Unclear | `_qaa_type4_865nm()` |
| `_save_complete_results()` | What's "complete"? | `_save_tss_results()` |
| `_create_water_type_legend()` | Good | Keep |
| `_create_product_index()` | Good | Keep |
| `_log_processing_summary()` | Good | Keep |

### processors/s2_processor.py

| Current Name | Issue | Proposed Name |
|--------------|-------|---------------|
| `S2Processor` | Good | Keep |
| `validate_snap_installation()` | Good | Keep |
| `get_gpt_command()` | Good | Keep |
| `setup_processing_graphs()` | Good | Keep |
| `create_s2_graph_with_subset()` | Good | Keep |
| `create_s2_graph_no_subset()` | Good | Keep |
| `_get_subset_parameters()` | Good | Keep |
| `_get_c2rcc_parameters()` | Good | Keep |
| `_get_geometric_output_path()` | Confusing "geometric" | `_get_resampled_output_path()` |
| `_extract_clean_product_name()` | Good | Keep |
| `get_output_filename()` | Good | Keep |
| `process_single_product()` | Good | Keep |
| `_run_s2_processing()` | Good | Keep |
| `_verify_c2rcc_output()` | Good | Keep |
| `_verify_file_integrity()` | Good | Keep |
| `_calculate_missing_btot()` | Good | Keep |
| `_get_file_size_kb()` | Good | Keep |
| `_check_rhow_bands_availability()` | Good | Keep |
| `get_processing_status()` | Good | Keep |
| `cleanup()` | Good | Keep |

### processors/snap_calculator.py

| Current Name | Issue | Proposed Name |
|--------------|-------|---------------|
| `SNAPTSMCHLCalculator` | Too abbreviated | `TSMChlorophyllCalculator` |
| `calculate_snap_tsm_chl()` | Redundant "snap" | `calculate_tsm_and_chlorophyll()` |
| `calculate_uncertainties()` | Good | Keep |

---

## Proposed Module Reorganization

### Option A: Split by Scientific Domain (Recommended)

```
processors/
├── __init__.py
│
├── s2/                          # Sentinel-2 Processing
│   ├── __init__.py
│   ├── processor.py             # S2Processor (SNAP GPT orchestration)
│   └── snap_calculator.py       # TSM/CHL calculation
│
├── tss/                         # TSS Estimation
│   ├── __init__.py
│   ├── constants.py             # JiangTSSConstants
│   ├── processor.py             # JiangTSSProcessor
│   └── qaa.py                   # QAA algorithms (Type I-IV)
│
├── water_quality/               # Water Quality Parameters
│   ├── __init__.py
│   ├── constants.py             # WaterQualityConstants
│   ├── clarity.py               # Secchi depth, Kd calculations
│   ├── trophic.py               # TSI calculation
│   └── hab.py                   # HAB detection
│
├── visualization/               # Visualization Products
│   ├── __init__.py
│   ├── rgb_definitions.py       # RGB composite definitions
│   ├── rgb_generator.py         # RGB composite generation
│   └── indices.py               # Spectral indices calculation
│
└── common/                      # Shared utilities
    ├── __init__.py
    ├── results.py               # ProcessingResult, ProcessingStatus
    └── band_loader.py           # Band loading utilities
```

### Option B: Minimal Changes (Keep Current Structure)

Just rename files and classes for clarity:

```
processors/
├── __init__.py
├── s2_processor.py              # Keep (S2 SNAP processing)
├── tsm_calculator.py            # Rename from snap_calculator.py
├── tss_processor.py             # Rename from jiang_processor.py
├── water_quality.py             # Rename from water_quality_processor.py
├── rgb_composites.py            # Rename from marine_viz.py
└── spectral_indices.py          # NEW: Split from marine_viz.py
```

---

## Class Renaming Proposals

### Current → Proposed

| Current | Proposed | Reason |
|---------|----------|--------|
| `S2MarineRGBGenerator` | `RGBCompositeDefinitions` | Not just "marine", defines all RGB combos |
| `S2MarineVisualizationProcessor` | `VisualizationProcessor` | Shorter, clearer |
| `SNAPTSMCHLCalculator` | `TSMChlorophyllCalculator` | Less abbreviation |
| `WaterQualityProcessor` | Keep | Already clear |
| `JiangTSSProcessor` | Keep | Already clear |
| `S2Processor` | Keep | Already clear |

---

## Splitting marine_viz.py

The `marine_viz.py` file (~1800 lines) should be split into:

### 1. rgb_definitions.py (~400 lines)
Contains: `RGBCompositeDefinitions` class with all RGB combo definitions

### 2. rgb_generator.py (~600 lines)
Contains:
- `_create_rgb_composite()`
- `_apply_contrast_enhancement()`
- `_save_rgb_geotiff()`
- `_generate_rgb_composites()`

### 3. spectral_indices.py (~600 lines)
Contains:
- `_calculate_spectral_index()`
- `_generate_spectral_indices()`
- `_save_single_band_geotiff()`
- Index formulas (NDWI, NDCI, FLH, etc.)

### 4. visualization_processor.py (~200 lines)
Contains: Main `VisualizationProcessor` orchestrator

---

## Priority Actions

### High Priority (Should Do)
1. Rename `create_advanced_processor()` → `create_water_quality_processor()`
2. Rename `integrate_with_existing_pipeline()` → `process_water_quality_from_results()`
3. Add `__pycache__` to `.gitignore` ✅ DONE
4. Rename `_cleanup_geometric_products()` → `_cleanup_intermediate_products()`

### Medium Priority (Nice to Have)
5. Rename `SNAPTSMCHLCalculator` → `TSMChlorophyllCalculator`
6. Rename QAA methods to include water type: `_qaa_type1_560nm()`
7. Split `marine_viz.py` into smaller modules

### Low Priority (Future)
8. Full module reorganization (Option A)
9. Create shared `band_loader.py` utility

---

## Immediate Fixes

Here are the specific renames to do now:

```python
# water_quality_processor.py
create_advanced_processor → create_water_quality_processor
integrate_with_existing_pipeline → process_water_quality_from_results

# jiang_processor.py
_apply_simple_unit_conversion → _convert_rhow_to_rrs
_process_advanced_algorithms → _process_water_quality_algorithms
_apply_full_jiang_methodology → _apply_jiang_tss_algorithm
_save_complete_results → _save_tss_results
_qaa_560 → _qaa_type1_560nm
_qaa_665 → _qaa_type2_665nm
_qaa_740 → _qaa_type3_740nm
_qaa_865 → _qaa_type4_865nm

# marine_viz.py
_load_bands_from_geometric_products → _load_bands_from_resampled_product
_create_robust_rgb_composite → _create_rgb_composite
_apply_additional_contrast_enhancement → _apply_contrast_enhancement
_cleanup_geometric_products → _cleanup_intermediate_products

# s2_processor.py
_get_geometric_output_path → _get_resampled_output_path
```

---

## Questions for User

1. **Do you want Option A (full reorganization) or Option B (minimal renames)?**
2. **Should we split marine_viz.py now or later?**
3. **Any other naming conventions you prefer?**
