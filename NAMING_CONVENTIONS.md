# Naming Conventions and Complete Function Analysis

## Naming Convention Options

### Option 1: Domain-Prefixed (Recommended)
Uses prefixes that indicate the scientific domain:
- `tss_` - TSS estimation functions
- `rgb_` - RGB composite functions
- `idx_` - Spectral index functions
- `wq_` - Water quality functions
- `s2_` - Sentinel-2 processing functions

**Example:** `_qaa_560()` → `_tss_qaa_type1_clear()`

### Option 2: Action-Object
Follows verb_object pattern consistently:
- `calculate_X` - Compute something
- `generate_X` - Create output product
- `load_X` - Read data
- `save_X` - Write data
- `validate_X` - Check validity
- `process_X` - Full processing workflow

**Example:** `_apply_simple_unit_conversion()` → `convert_rhow_to_rrs()`

### Option 3: Scientific Reference
Uses scientific paper/method names:
- QAA methods reference water type
- Indices reference original author
- Algorithms reference publication

**Example:** `_qaa_560()` → `_qaa_jiang2021_type1()`

---

## Complete Function Analysis by Module

### 1. processors/jiang_processor.py (TSS Estimation)

| Line | Current Name | Issue | Option 1 | Option 2 | Option 3 |
|------|--------------|-------|----------|----------|----------|
| 133 | `_load_bands_data` | OK | Keep | Keep | Keep |
| 158 | `_apply_simple_unit_conversion` | "simple" is vague | `_tss_convert_reflectance` | `convert_rhow_to_rrs` | `_convert_rhow_rrs_pi` |
| 211 | `_validate_rrs_ranges` | OK | Keep | Keep | Keep |
| 240 | `_update_band_mapping_for_mixed_types` | Too long | `_tss_update_band_mapping` | `update_band_paths` | Keep shorter |
| 302 | `process_jiang_tss` | OK | Keep | Keep | Keep |
| 507 | `_process_advanced_algorithms` | "advanced" vague | `_tss_process_water_quality` | `process_water_quality_products` | Keep |
| 597 | `_extract_snap_chlorophyll` | OK | Keep | `load_chlorophyll_from_c2rcc` | Keep |
| 615 | `_apply_full_jiang_methodology` | "full" redundant | `_tss_apply_methodology` | `estimate_tss_all_pixels` | `_apply_jiang2021_algorithm` |
| 690 | `_create_valid_pixel_mask` | OK | Keep | Keep | Keep |
| 705 | `_process_valid_pixels` | OK | Keep | Keep | Keep |
| 737 | `_estimate_tss_single_pixel` | OK | Keep | Keep | Keep |
| 768 | `_estimate_rrs620_from_rrs665` | OK | Keep | `interpolate_rrs_620nm` | Keep |
| 775 | `_qaa_560` | Unclear water type | `_tss_qaa_type1_clear` | `_qaa_clear_water_560nm` | `_qaa_jiang_type1` |
| 807 | `_qaa_665` | Unclear water type | `_tss_qaa_type2_moderate` | `_qaa_moderate_water_665nm` | `_qaa_jiang_type2` |
| 836 | `_qaa_740` | Unclear water type | `_tss_qaa_type3_turbid` | `_qaa_turbid_water_740nm` | `_qaa_jiang_type3` |
| 863 | `_qaa_865` | Unclear water type | `_tss_qaa_type4_extreme` | `_qaa_extreme_water_865nm` | `_qaa_jiang_type4` |
| 890 | `_save_complete_results` | "complete" vague | `_tss_save_results` | `save_tss_products` | Keep |
| 1091 | `_create_water_type_legend` | OK | Keep | Keep | Keep |
| 1140 | `_create_product_index` | OK | Keep | Keep | Keep |
| 1177 | `_log_processing_summary` | OK | Keep | Keep | Keep |

### 2. processors/marine_viz.py (RGB & Indices)

| Line | Current Name | Issue | Option 1 | Option 2 | Option 3 |
|------|--------------|-------|----------|----------|----------|
| 489 | `process_marine_visualizations` | OK | `rgb_process_visualizations` | Keep | Keep |
| 664 | `_load_bands_from_geometric_products` | "geometric" confusing | `_rgb_load_from_resampled` | `load_bands_from_resampled_product` | Keep |
| 710 | `_load_available_bands` | OK | Keep | Keep | Keep |
| 762 | `_load_bands_data_from_paths` | OK | Keep | Keep | Keep |
| 800 | `_generate_rgb_composites` | OK | Keep | Keep | Keep |
| 994 | `_generate_spectral_indices` | OK | `_idx_generate_all` | Keep | Keep |
| 1212 | `_calculate_spectral_index` | OK | `_idx_calculate_single` | Keep | Keep |
| 1321 | `_create_robust_rgb_composite` | "robust" redundant | `_rgb_create_composite` | `create_rgb_array` | Keep |
| 1360 | `_apply_additional_contrast_enhancement` | "additional" redundant | `_rgb_enhance_contrast` | `apply_contrast_stretch` | Keep |
| 1401 | `_save_rgb_geotiff` | OK | Keep | Keep | Keep |
| 1470 | `_save_single_band_geotiff` | OK | `_idx_save_geotiff` | Keep | Keep |
| 1538 | `_create_visualization_summary` | OK | Keep | Keep | Keep |
| 1656 | `_cleanup_geometric_products` | "geometric" confusing | `_rgb_cleanup_temp_files` | `cleanup_intermediate_products` | Keep |
| 1782 | `_calculate_rgb_statistics` | OK | Keep | Keep | Keep |
| 1812 | `_calculate_band_statistics` | OK | Keep | Keep | Keep |

### 3. processors/s2_processor.py (SNAP Processing)

| Line | Current Name | Issue | Option 1 | Option 2 | Option 3 |
|------|--------------|-------|----------|----------|----------|
| 66 | `validate_snap_installation` | OK | Keep | Keep | Keep |
| 95 | `get_gpt_command` | OK | Keep | Keep | Keep |
| 103 | `setup_processing_graphs` | OK | Keep | Keep | Keep |
| 115 | `create_s2_graph_with_subset` | OK | Keep | Keep | Keep |
| 206 | `create_s2_graph_no_subset` | OK | Keep | Keep | Keep |
| 268 | `_get_subset_parameters` | OK | Keep | Keep | Keep |
| 280 | `_get_c2rcc_parameters` | OK | Keep | Keep | Keep |
| 327 | `_get_geometric_output_path` | "geometric" confusing | `_s2_get_resampled_path` | `get_resampled_output_path` | Keep |
| 345 | `_extract_clean_product_name` | OK | Keep | Keep | Keep |
| 362 | `get_output_filename` | OK | Keep | Keep | Keep |
| 398 | `process_single_product` | OK | Keep | Keep | Keep |
| 559 | `_run_s2_processing` | OK | Keep | `execute_gpt_graph` | Keep |
| 635 | `_verify_c2rcc_output` | OK | Keep | Keep | Keep |
| 797 | `_verify_file_integrity` | OK | Keep | Keep | Keep |
| 813 | `_calculate_missing_btot` | OK | Keep | `compute_total_backscatter` | Keep |
| 888 | `_get_file_size_kb` | OK | Keep | Keep | Keep |
| 897 | `_check_rhow_bands_availability` | OK | Keep | Keep | Keep |
| 963 | `get_processing_status` | OK | Keep | Keep | Keep |
| 996 | `cleanup` | OK | Keep | Keep | Keep |

### 4. processors/water_quality_processor.py

| Line | Current Name | Issue | Option 1 | Option 2 | Option 3 |
|------|--------------|-------|----------|----------|----------|
| 58 | `calculate_water_clarity` | OK | `wq_calculate_clarity` | Keep | Keep |
| 126 | `detect_harmful_algal_blooms` | OK | `wq_detect_hab` | Keep | Keep |
| 319 | `calculate_trophic_state` | OK | `wq_calculate_tsi` | Keep | Keep |
| 404 | `create_advanced_processor` | "advanced" vague | `wq_create_processor` | `create_water_quality_processor` | Keep |
| 423 | `integrate_with_existing_pipeline` | Very vague | `wq_process_from_results` | `process_water_quality_from_c2rcc` | Keep |

### 5. processors/snap_calculator.py

| Line | Current Name | Issue | Option 1 | Option 2 | Option 3 |
|------|--------------|-------|----------|----------|----------|
| 48 | `calculate_snap_tsm_chl` | "snap" redundant | `calculate_tsm_chlorophyll` | Keep | Keep |
| 206 | `calculate_uncertainties` | OK | Keep | Keep | Keep |

### 6. core/unified_processor.py

| Line | Current Name | Issue | Option 1 | Option 2 | Option 3 |
|------|--------------|-------|----------|----------|----------|
| 60 | `_initialize_processors` | OK | Keep | Keep | Keep |
| 71 | `process_batch` | OK | Keep | Keep | Keep |
| 110 | `_find_products` | OK | `_discover_input_products` | Keep | Keep |
| 125 | `_process_single_product` | OK | Keep | Keep | Keep |
| 279 | `_extract_product_name` | OK | Keep | Keep | Keep |
| 302 | `_check_outputs_exist` | OK | Keep | Keep | Keep |
| 337 | `_print_final_summary` | OK | Keep | Keep | Keep |
| 364 | `get_processing_status` | OK | Keep | Keep | Keep |
| 384 | `cleanup` | OK | Keep | Keep | Keep |

### 7. utils/ modules

| File | Line | Current Name | Issue | Proposed |
|------|------|--------------|-------|----------|
| logging_utils.py | 39 | `setup_enhanced_logging` | "enhanced" vague | `setup_logging` |
| memory_manager.py | 21 | `cleanup_variables` | OK | Keep |
| memory_manager.py | 37 | `monitor_memory` | OK | Keep |
| memory_manager.py | 62 | `get_memory_usage_mb` | OK | Keep |
| raster_io.py | 26 | `read_raster` | OK | Keep |
| raster_io.py | 68 | `write_raster` | OK | Keep |
| raster_io.py | 181 | `calculate_statistics` | OK | Keep |
| raster_io.py | 210 | `load_bands_safely` | OK | Keep |
| math_utils.py | 18 | `safe_divide` | OK | Keep |
| math_utils.py | 48 | `safe_sqrt` | OK | Keep |
| math_utils.py | 70 | `safe_log` | OK | Keep |
| math_utils.py | 98 | `safe_power` | OK | Keep |
| product_detector.py | 28 | `detect_product_type` | OK | Keep |
| product_detector.py | 47 | `scan_input_folder` | OK | Keep |
| product_detector.py | 81 | `validate_processing_mode` | OK | Keep |

---

## Class Renaming Analysis

| Current Class | Module | Issue | Proposed Name |
|---------------|--------|-------|---------------|
| `S2MarineRGBGenerator` | marine_viz.py | Not just "marine", misleading | `RGBCompositeDefinitions` |
| `S2MarineVisualizationProcessor` | marine_viz.py | Too long, "marine" misleading | `VisualizationProcessor` |
| `SNAPTSMCHLCalculator` | snap_calculator.py | Too abbreviated | `TSMChlorophyllCalculator` |
| `JiangTSSConstants` | jiang_processor.py | OK | Keep |
| `JiangTSSProcessor` | jiang_processor.py | OK | Keep |
| `WaterQualityConstants` | water_quality_processor.py | OK | Keep |
| `WaterQualityProcessor` | water_quality_processor.py | OK | Keep |
| `S2Processor` | s2_processor.py | OK | Keep |
| `ProcessingResult` | snap_calculator.py | OK | Keep |
| `ProcessingStatus` | s2_processor.py | OK | Keep |
| `UnifiedS2TSSProcessor` | unified_processor.py | OK | Keep or `PipelineOrchestrator` |

---

## Recommended Module Structure (Option A)

```
sentinel2_tss_pipeline/
├── __init__.py
├── __main__.py
├── main.py
│
├── config/                      # ✅ Keep as is
│   ├── __init__.py
│   ├── enums.py
│   ├── s2_config.py
│   ├── jiang_config.py
│   ├── water_quality_config.py
│   ├── marine_config.py
│   └── processing_config.py
│
├── utils/                       # ✅ Keep as is
│   ├── __init__.py
│   ├── logging_utils.py
│   ├── math_utils.py
│   ├── memory_manager.py
│   ├── raster_io.py
│   └── product_detector.py
│
├── core/                        # ✅ Keep as is
│   ├── __init__.py
│   └── pipeline.py              # Rename from unified_processor.py
│
├── gui/                         # ✅ Keep as is
│   ├── __init__.py
│   └── unified_gui.py
│
├── s2/                          # NEW: Sentinel-2 Processing
│   ├── __init__.py
│   ├── processor.py             # S2Processor (from s2_processor.py)
│   └── tsm_calculator.py        # TSMChlorophyllCalculator (from snap_calculator.py)
│
├── tss/                         # NEW: Jiang TSS Estimation
│   ├── __init__.py
│   ├── constants.py             # JiangTSSConstants
│   ├── processor.py             # JiangTSSProcessor
│   └── qaa/                     # QAA algorithms subfolder
│       ├── __init__.py
│       ├── type1_clear.py       # _qaa_type1_clear (560nm)
│       ├── type2_moderate.py    # _qaa_type2_moderate (665nm)
│       ├── type3_turbid.py      # _qaa_type3_turbid (740nm)
│       └── type4_extreme.py     # _qaa_type4_extreme (865nm)
│
├── water_quality/               # NEW: Water Quality
│   ├── __init__.py
│   ├── constants.py             # WaterQualityConstants
│   ├── clarity.py               # Secchi depth, Kd, turbidity
│   ├── trophic.py               # TSI calculation
│   └── hab.py                   # HAB detection (NDCI, FLH, MCI)
│
└── visualization/               # NEW: RGB & Spectral Indices
    ├── __init__.py
    ├── rgb/
    │   ├── __init__.py
    │   ├── definitions.py       # RGB composite definitions
    │   ├── generator.py         # RGB creation logic
    │   └── enhancement.py       # Contrast enhancement
    ├── indices/
    │   ├── __init__.py
    │   ├── water_quality.py     # NDWI, NDTI, etc.
    │   ├── chlorophyll.py       # CIG, NDCI, etc.
    │   ├── turbidity.py         # Turbidity indices
    │   └── advanced.py          # FUI, CDOM, etc.
    └── processor.py             # VisualizationProcessor orchestrator
```

---

## Summary: Functions to Rename (Priority Order)

### HIGH PRIORITY (Vague/Misleading Names)

| Current | Proposed (Option 2 recommended) |
|---------|--------------------------------|
| `create_advanced_processor` | `create_water_quality_processor` |
| `integrate_with_existing_pipeline` | `process_water_quality_from_c2rcc` |
| `_apply_simple_unit_conversion` | `convert_rhow_to_rrs` |
| `_process_advanced_algorithms` | `process_water_quality_products` |
| `_apply_full_jiang_methodology` | `estimate_tss_all_pixels` |
| `_save_complete_results` | `save_tss_products` |
| `_cleanup_geometric_products` | `cleanup_intermediate_products` |
| `_load_bands_from_geometric_products` | `load_bands_from_resampled_product` |
| `_get_geometric_output_path` | `get_resampled_output_path` |

### MEDIUM PRIORITY (Unclear but Functional)

| Current | Proposed |
|---------|----------|
| `_qaa_560` | `_qaa_type1_clear_560nm` |
| `_qaa_665` | `_qaa_type2_moderate_665nm` |
| `_qaa_740` | `_qaa_type3_turbid_740nm` |
| `_qaa_865` | `_qaa_type4_extreme_865nm` |
| `_create_robust_rgb_composite` | `create_rgb_array` |
| `_apply_additional_contrast_enhancement` | `apply_contrast_stretch` |
| `calculate_snap_tsm_chl` | `calculate_tsm_chlorophyll` |
| `setup_enhanced_logging` | `setup_logging` |

### LOW PRIORITY (Minor Improvements)

| Current | Proposed |
|---------|----------|
| `_update_band_mapping_for_mixed_types` | `update_band_paths` |
| `_extract_snap_chlorophyll` | `load_chlorophyll_from_c2rcc` |
| `_run_s2_processing` | `execute_gpt_graph` |

---

## My Recommendation

**For naming convention: Option 2 (Action-Object)** because:
1. Most readable for developers
2. Consistent with Python conventions (PEP 8)
3. Self-documenting - you understand what it does from the name
4. Doesn't require domain knowledge to understand

**For structure: Option A (Domain modules)** because:
1. Easier to maintain - find related code quickly
2. Easier to test - each module is independent
3. Easier to extend - add new indices in one place
4. Clear separation of concerns

---

## Next Steps

1. **Decide on naming convention** (Option 1, 2, or 3)
2. **Decide on structure** (keep current vs full reorganization)
3. I will implement the changes systematically
4. Update all imports and tests
