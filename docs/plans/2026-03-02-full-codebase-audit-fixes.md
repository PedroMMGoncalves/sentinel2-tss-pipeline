# Full Codebase Audit Fixes — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix all 103 issues (19 critical, 44 important, 41 minor) identified in the full codebase audit.

**Architecture:** 6 parallel agents working on non-overlapping file groups. Each agent reads its files, applies all fixes, and verifies syntax with `python -m py_compile`. No new files created except potentially a shared XML escape helper.

**Tech Stack:** Python 3.8+, GDAL, tkinter, xml.sax.saxutils (stdlib)

**Verification:** After all agents complete: `python -c "from ocean_rs.optical import *; from ocean_rs.sar import *; print('OK')"`

---

## Agent A: Shared Modules (6 files)

### Files:
- `ocean_rs/shared/raster_io.py`
- `ocean_rs/shared/geometry_utils.py`
- `ocean_rs/shared/logging_utils.py`
- `ocean_rs/shared/proj_fix.py`
- `ocean_rs/shared/__init__.py`
- `ocean_rs/shared/memory_manager.py`

### Fixes:

**raster_io.py:**
1. [Critical #11] Line 197: Change `'sentinel2_tss_pipeline v2.0'` → `'OceanRS v3.0.0'`
2. [Important #44] `write_raster` returns False on failure — callers ignore it. Change to raise `RuntimeError` on failure instead of returning False. Keep `logger.error` calls, add `raise` after each.

**geometry_utils.py:**
3. [Critical #17] Line 20: Remove unused `import sys`

**logging_utils.py:**
4. [Minor #103] Line 189: Update docstring example from `"SENTINEL-2 TSS PIPELINE v2.0"` to `"OCEANRS v3.0.0"`

**proj_fix.py:**
5. [Important #57] Line 134: Update old import path in docstring from `sentinel2_tss_pipeline.utils.proj_fix` → `ocean_rs.shared.proj_fix`

**shared/__init__.py:**
6. [Important #58] Add missing export: `find_proj_lib_paths` from `proj_fix`

**memory_manager.py:**
7. [Minor #89] Line 32-33: Add `logger.debug(f"cleanup error: {e}")` instead of bare `pass`
8. [Minor #90] Lines 58, 75: Add `logger.debug` on psutil error returns

---

## Agent B: Optical Processors (5 files)

### Files:
- `ocean_rs/optical/processors/tss_processor.py`
- `ocean_rs/optical/processors/visualization_processor.py`
- `ocean_rs/optical/processors/c2rcc_processor.py`
- `ocean_rs/optical/processors/tsm_chl_calculator.py`
- `ocean_rs/optical/processors/water_quality_processor.py`

### Fixes:

**tss_processor.py:**
1. [Critical #14] Lines 1059-1213: Delete 6 dead single-pixel QAA methods (~155 lines): `_estimate_tss_single_pixel`, `_estimate_rrs620_from_rrs665`, `_qaa_type1_clear_560nm`, `_qaa_type2_moderate_665nm`, `_qaa_type3_turbid_740nm`, `_qaa_type4_extreme_865nm`
2. [Critical #16] Line 1474: Change `"Processing v2.0"` → `"OceanRS v3.0.0"`
3. [Important #20] Lines 1354-1364: Add `dtype="uint8"` to `write_raster()` calls for classification products (WaterTypes, HAB risk, Trophic classes)
4. [Important #53] Line 395: Delete dead method `_update_band_mapping_for_mixed_types()`
5. [Important #53] Line 1512: Delete dead method `_log_processing_summary()`
6. [Important #49] Line 1087: Change `logger.debug` → `logger.warning` for per-pixel Jiang algorithm errors
7. [Minor #64] Line 484: Remove the immediate overwrite `intermediate_paths = {}` and fix the parameter usage or remove the parameter

**visualization_processor.py:**
8. [Critical #12] Lines 984-985 in `_save_single_band_geotiff`: Before writing, replace NaN with nodata value: `data = np.where(np.isnan(data), nodata_val, data)` where `nodata_val = metadata.get('nodata', -9999)`
9. [Important #27] `_save_rgb_geotiff`: Add `finally` block to ensure `dataset = None` on exception
10. [Important #28] `_save_single_band_geotiff`: Add `finally` block to ensure `dataset = None` on exception
11. [Important #29] Lines 373-374: Remove redundant double nodata masking (read_raster already handles this)
12. [Important #32] Lines 938, 1001: Change provenance strings from `'Unified S2 TSS Pipeline - Marine Visualization'` → `'OceanRS v3.0.0'`
13. [Minor #94] Lines 1153-1156, 1178-1179: Add `logger.debug` to cleanup exception handlers

**c2rcc_processor.py:**
14. [Critical #6] Line 72: Improve error message: Include common paths checked and suggest setting SNAP_HOME
15. [Critical #7] Line 79: Include the path that was checked in the RuntimeError message
16. [Critical #8] Line 88: Include `result.stderr` in the RuntimeError message
17. [Important #22-24] Lines 314, 316, 333, 336: Use `xml.sax.saxutils.escape()` for all f-string values inserted into XML
18. [Important #46] Lines 201-202, 275-276: Wrap XML file writing in try/except with contextual error
19. [Important #54] Line 919: Delete dead method `_get_file_size_kb()`
20. [Minor #95] Line 925-926: Already covered by deleting dead method
21. [Minor #70] Line 632: Acceptable — keep debug logging of GPT command

**tsm_chl_calculator.py:**
22. [Important #30] Lines 232, 265: Pass explicit `nodata=-9999` to `write_raster()` calls

**water_quality_processor.py:**
23. [Important #45] Lines 134-136, 292-294, 380-382: Change from returning `{}` on error to `logger.error` + `raise` (or at minimum `logger.warning` + return `{}` with clear message that processing failed, not just "no data")

---

## Agent C: Optical GUI (10 files)

### Files:
- `ocean_rs/optical/gui/unified_gui.py`
- `ocean_rs/optical/gui/handlers.py`
- `ocean_rs/optical/gui/processing_controller.py`
- `ocean_rs/optical/gui/theme.py`
- `ocean_rs/optical/gui/tabs/monitoring_tab.py`
- `ocean_rs/optical/gui/tabs/spatial_tab.py`
- `ocean_rs/optical/gui/tabs/c2rcc_tab.py`
- `ocean_rs/optical/gui/tabs/outputs_tab.py`
- `ocean_rs/optical/gui/tabs/processing_tab.py`
- `ocean_rs/optical/gui/config_io.py`

### Fixes:

**unified_gui.py:**
1. [Critical #18] Line 35: Remove unused `WaterQualityConfig` from import
2. [Minor #92] Lines 87-88: Add `logger.debug` to window flash exception

**handlers.py:**
3. [Important #52] Lines 69-309: Delete 14 dead functions: `update_subset_visibility`, `update_tss_visibility`, `on_ecmwf_toggle`, `on_rhow_toggle`, `validate_geometry`, `browse_input_dir`, `browse_output_dir`, `apply_water_preset`, `apply_snap_defaults`, `apply_essential_outputs`, `apply_scientific_outputs`, `reset_all_outputs`
4. [Important #35] Lines 321-324: Replace `time.sleep(1)` with `root.after(100, check_thread_and_destroy)` pattern
5. [Minor #91] Lines 334-335: Add `logger.debug` to cleanup exception
6. [Minor #93] Lines 40-63: Add comments explaining why TclError is expected for tab visibility

**processing_controller.py:**
7. [Important #33] Line 205: Use `gui.root.after(0, lambda: setattr(gui, 'processing_active', False))` to set flag from main thread
8. [Important #34] Line 158: Move `gui.processor` assignment to main thread via `root.after`
9. [Important #38] Lines 30-31: Move `gui.processing_active = True` immediately after the guard check, before any validation
10. [Important #40] Lines 208-209: Schedule cleanup on main thread via `root.after`
11. [Important #43] Line 266: Add guard `if not gui.root.winfo_exists(): return` before recursive scheduling
12. [Minor #96] Lines 331-332: Add `logger.debug` to system info exception

**theme.py:**
13. [Important #56] Line 7: Change `"Part of the sentinel2_tss_pipeline package."` → `"Part of the OceanRS toolkit (ocean_rs.optical)."`

**monitoring_tab.py:**
14. [Important #55] Lines 197, 221: Delete dead functions `update_statistics()` and `clear_statistics()`
15. [Important #50] Lines 152-153: Add `logger.warning(f"Cannot read disk usage for {output_dir}: {e}")` instead of silent swallow
16. [Minor #99] Lines 217-218, 231-232: Change `logger.debug` → `logger.warning` for statistics errors

**spatial_tab.py:**
17. [Minor #98] Lines 407-408: Change `logger.debug` → `logger.warning` for zoom error

**processing_tab.py:**
18. [Minor #65] Lines 186-188: Capture return value from `setup_enhanced_logging()` or add comment explaining why it's intentionally discarded

---

## Agent D: Optical Config, Core, Main, __init__ (9 files)

### Files:
- `ocean_rs/optical/config/enums.py`
- `ocean_rs/optical/config/s2_config.py`
- `ocean_rs/optical/config/tss_config.py`
- `ocean_rs/optical/config/output_categories.py`
- `ocean_rs/optical/config/water_quality_config.py`
- `ocean_rs/optical/config/processing_config.py`
- `ocean_rs/optical/core/unified_processor.py`
- `ocean_rs/optical/main.py`
- `ocean_rs/optical/__init__.py`

### Fixes:

**6 config files (enums, s2_config, tss_config, output_categories, water_quality_config, processing_config):**
1. [Important #56] Change docstring `"Part of the sentinel2_tss_pipeline package."` → `"Part of the OceanRS toolkit (ocean_rs.optical)."` in all 6 files

**unified_processor.py:**
2. [Important #21] Lines 244-246: Fix `process_tss()` call — pass `None` instead of `s2_result` for the `intermediate_paths` parameter (or remove the parameter from the call since it's immediately overwritten)
3. [Minor #97] Line 424: Change `logger.debug` → `logger.warning` for existing output check error

**main.py:**
4. [Critical #15] Line 289: Change `"PIPELINE v2.0"` → `"OceanRS v3.0.0"` or use `from ocean_rs import __version__`

**optical/__init__.py:**
5. [Minor #62] Remove `SafeMathNumPy` from exports (unused dead export) — or keep if it's part of public API
6. [Important #59] Add `OutputStructure` to exports

---

## Agent E: SAR GUI (8 files)

### Files:
- `ocean_rs/sar/gui/processing_controller.py`
- `ocean_rs/sar/gui/tabs/download_tab.py`
- `ocean_rs/sar/gui/handlers.py`
- `ocean_rs/sar/gui/unified_gui.py`
- `ocean_rs/sar/gui/tabs/search_tab.py`
- `ocean_rs/sar/gui/tabs/results_tab.py`
- `ocean_rs/sar/gui/tabs/processing_tab.py`
- `ocean_rs/sar/gui/config_io.py`
- `ocean_rs/sar/main.py`

### Fixes:

**sar/gui/processing_controller.py:**
1. [Critical #1] Lines 46-49: Wrap progress_callback GUI updates in `gui.root.after(0, lambda: ...)` for both `progress_var.set()` and `status_var.set()`
2. [Important #36] Line 74: Use `gui.root.after(0, ...)` to set `processing_active = False` from main thread
3. [Important #39] Lines 19-20: Move `gui.processing_active = True` immediately after the guard check
4. [Important #41] Line 85-88: Add safety check for `gui.pipeline` existence
5. [Minor #86] Line 87: No change needed — `stop_processing` correctly delegates to pipeline.cancel()

**sar/gui/tabs/download_tab.py:**
6. [Critical #2] Lines 165-169: Wrap download progress_callback GUI updates in `gui.root.after(0, lambda: ...)`
7. [Critical #3] Lines 161-162: Add `if gui.download_active: return` guard at top of `_start_download()`
8. [Important #37] Line 181: Use `gui.root.after(0, ...)` to set `download_active = False`

**sar/gui/handlers.py:**
9. [Critical #4] Lines 10-17: Add cancellation of active processing/download before `root.destroy()`. Call `stop_processing()`, set cancelled flags, use `root.after` to delay destroy.
10. [Important #51] Lines 33-34: Show "psutil not available" message instead of silent `pass`
11. [Important #84] Lines 41-43: Add `if not gui.root.winfo_exists(): return` guard

**sar/gui/unified_gui.py:**
12. [Minor #92] Lines 39-40: Add `logger.debug` to window flash exception

**sar/gui/tabs/search_tab.py:**
13. [Important #42] Lines 174-176: Move `_do_search()` network call to background thread with `root.after` for GUI updates
14. [Minor #87] Lines 238-240: Add bounds check for Treeview index lookup

**sar/gui/tabs/results_tab.py:**
15. [Minor #67] Line 74: Add `os.path.isdir(output)` check before `subprocess.Popen`
16. [Minor #101] Lines 67-78: Wrap `subprocess.Popen` in try/except

**sar/gui/tabs/processing_tab.py:**
17. [Important #60] Scrollable frame duplication — note: extracting to shared is optional, but add TODO comment

**sar/gui/config_io.py:**
18. [Minor #69] Line 72: Add comment documenting that SearchConfig has no credential fields

**sar/main.py:**
19. [Minor #100] Lines 139-140: Add `print(f"Fatal error: {e}", file=sys.stderr)` as fallback

---

## Agent F: SAR Core, Download, Sensors (5 files)

### Files:
- `ocean_rs/sar/core/bathymetry_pipeline.py`
- `ocean_rs/sar/download/scene_discovery.py`
- `ocean_rs/sar/download/batch_downloader.py`
- `ocean_rs/sar/download/credentials.py`
- `ocean_rs/sar/sensors/sentinel1.py`

### Fixes:

**bathymetry_pipeline.py:**
1. [Critical #13] Lines 195-211: Check `write_raster()` return value (or handle the RuntimeError now that Agent A changes it to raise). Wrap in try/except and log + raise on failure.

**scene_discovery.py:**
2. [Critical #5] Line 97: Change to `raise RuntimeError(...) from e` to preserve exception chain
3. [Critical #10] Lines 94-97: Add specific handling for `requests.ConnectionError`, `requests.Timeout`, `requests.HTTPError` with meaningful messages before the generic `Exception` catch

**batch_downloader.py:**
4. [Critical #9] Lines 98-99: Implement retry logic using existing `DownloadConfig.retry_count`. Add loop with exponential backoff for transient errors (ConnectionError, Timeout). Distinguish transient vs permanent (401/403) errors.

**credentials.py:**
5. [Important #26] Lines 115-119: Add `os.chmod(env_path, 0o600)` after writing .env (with platform check for Windows)
6. [Important #48] Lines 108-119: Wrap file operations in try/except with contextual error messages

**sentinel1.py:**
7. [Important #25] Lines 125-185: Use `xml.sax.saxutils.escape()` for `input_path`, `output_path`, and `polarization` in XML graph
8. [Important #47] Line 71: Wrap XML file writing in try/except with contextual error message

---

## Verification Steps

After all agents complete:

1. Syntax check all modified files:
   ```bash
   find ocean_rs -name "*.py" -exec python -m py_compile {} +
   ```

2. Import test:
   ```bash
   python -c "from ocean_rs.optical import *; from ocean_rs.sar import *; print('All imports OK')"
   ```

3. Entry point tests:
   ```bash
   python -m ocean_rs.optical --help
   python -m ocean_rs.sar --help
   ```
