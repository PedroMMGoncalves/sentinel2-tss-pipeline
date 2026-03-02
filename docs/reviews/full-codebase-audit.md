# OceanRS Full Codebase Audit Report

**Date:** 2026-03-02
**Audited by:** 7 parallel Claude Opus auditor agents
**Scope:** All 76 Python files under `ocean_rs/` (optical + SAR + shared)
**Methodology:** Each auditor read every file in its scope and checked against specific rules. Findings classified as Critical (will crash/corrupt at runtime), Important (incorrect behavior/risk), or Minor (cosmetic/style).

---

## Summary

| Auditor | Critical | Important | Minor |
|---------|----------|-----------|-------|
| 1. API Contract | 0 | 2 | 3 |
| 2. Scientific Formulas | 0 | 0 | 1 |
| 3. Security | 0 | 6 | 4 |
| 4. GDAL / Raster I/O | 3 | 7 | 11 |
| 5. GUI Thread Safety | 4 | 11 | 6 |
| 6. Error Handling | 6 | 8 | 15 |
| 7. Dead Code & Consistency | 6 | 10 | 1 |
| **TOTAL** | **19** | **44** | **41** |

> **Note:** Some issues were independently discovered by multiple auditors (e.g., provenance stamp, uint8 classification). These are listed once below under the most relevant auditor, with cross-references noted.

---

## All Critical Issues (19 — must fix)

### GUI Thread Safety (Auditor 5) — 4 issues

1. **[ocean_rs/sar/gui/processing_controller.py:46-49]** SAR `progress_callback` calls `gui.progress_var.set()` and `gui.status_var.set()` directly from background daemon thread without `root.after()`. Can cause Tcl crashes. The `_log_processing` call on the same file correctly uses `root.after(0, ...)`, but these two `.set()` calls do not.

2. **[ocean_rs/sar/gui/tabs/download_tab.py:165-169]** Download `progress_callback` calls `gui.download_status_label.config(text=msg)` directly from background download thread. Direct `.config()` on a widget from a non-main thread is a textbook tkinter thread safety violation that can cause segfaults.

3. **[ocean_rs/sar/gui/tabs/download_tab.py:161-162]** No double-click protection on `_start_download()`. Unlike the optical module, there is no `if gui.download_active: return` guard. A fast double-click can launch two concurrent download threads.

4. **[ocean_rs/sar/gui/handlers.py:10-17]** SAR `on_closing()` does not cancel active processing/download before destroying root. Daemon threads continue running and call `root.after()` on a destroyed root, causing `TclError` exceptions.

### Error Handling (Auditor 6) — 6 issues

5. **[ocean_rs/sar/download/scene_discovery.py:97]** Exception chain lost — `raise RuntimeError(f"ASF search failed: {str(e)}")` discards original traceback. Should be `raise RuntimeError(...) from e`.

6. **[ocean_rs/optical/processors/c2rcc_processor.py:72]** SNAP `RuntimeError("SNAP installation not found")` missing search context — no guidance on expected paths or `SNAP_HOME`. Compare to `sar/sensors/sentinel1.py:110-113` which includes actionable instructions.

7. **[ocean_rs/optical/processors/c2rcc_processor.py:79]** GPT `RuntimeError("GPT executable not found")` — the path that was checked is logged but not included in the exception message. Users see bare error with no actionable information.

8. **[ocean_rs/optical/processors/c2rcc_processor.py:88]** GPT validation `RuntimeError("GPT validation failed")` — `result.stderr` is logged but not included in the raised exception. GUI displays bare message with no diagnostic detail.

9. **[ocean_rs/sar/download/batch_downloader.py:98-99]** No retry logic for download failures. Single `except Exception` catches all errors (timeout, auth, disk full) with no retry, no transient/permanent distinction, no partial file cleanup. `DownloadConfig.retry_count=3` is defined but never used.

10. **[ocean_rs/sar/download/scene_discovery.py:94-97]** No network-specific error handling for ASF API. All errors (timeout, 401, 403, 429, network unreachable) collapsed to generic `RuntimeError`.

### GDAL / Raster I/O (Auditor 4) — 3 issues

11. **[ocean_rs/shared/raster_io.py:197]** Stale metadata provenance stamp: `PROCESSING_SOFTWARE` set to `'sentinel2_tss_pipeline v2.0'` — should be `'ocean_rs v3.0.0'`. Every GeoTIFF written carries wrong software tag. *(Also found by Auditor 7)*

12. **[ocean_rs/optical/processors/visualization_processor.py:984-985]** NoData header/pixel mismatch in `_save_single_band_geotiff`. Metadata from `read_raster` sets nodata to `-9999`, but pixel array contains literal NaN values. The nodata sentinel in the file header (`-9999`) does not match actual nodata values (`NaN`) in pixels. Downstream readers masking on `-9999` will miss NaN pixels entirely.

13. **[ocean_rs/sar/core/bathymetry_pipeline.py:195-211]** `write_raster()` return value silently ignored for both `bathymetry_depth.tif` and `bathymetry_uncertainty.tif`. Pipeline logs "Exported" even if write fails.

### Dead Code & Consistency (Auditor 7) — 6 issues

14. **[ocean_rs/optical/processors/tss_processor.py:1059-1213]** 6 dead single-pixel QAA methods (~155 lines): `_estimate_tss_single_pixel()`, `_estimate_rrs620_from_rrs665()`, `_qaa_type1_clear_560nm()`, `_qaa_type2_moderate_665nm()`, `_qaa_type3_turbid_740nm()`, `_qaa_type4_extreme_865nm()`. All superseded by vectorized `_process_valid_pixels()`.

15. **[ocean_rs/optical/main.py:289]** Hardcoded `"PIPELINE v2.0"` — should be v3.0.0 or use `__version__`.

16. **[ocean_rs/optical/processors/tss_processor.py:1474]** Hardcoded `"Processing v2.0"` in summary file — should be v3.0.0 or use `__version__`.

17. **[ocean_rs/shared/geometry_utils.py:20]** Unused `import sys`.

18. **[ocean_rs/optical/gui/unified_gui.py:35]** Unused import `WaterQualityConfig`.

19. **[ocean_rs/shared/raster_io.py:197]** Old branding `'sentinel2_tss_pipeline v2.0'` actively written into output file metadata. *(Same as #11)*

> **De-duplicated total: 18 unique critical issues** (items 11 and 19 are the same)

---

## All Important Issues (44 — should fix)

### API Contract (Auditor 1) — 2 issues

20. **[ocean_rs/optical/processors/tss_processor.py:1354-1364]** Classification products cast to `uint8` then written via `write_raster()` without `dtype="uint8"`. Default `dtype="float32"` causes re-cast, wasting 4x disk space and losing integer semantics. *(Also found by Auditor 4)*

21. **[ocean_rs/optical/core/unified_processor.py:244-246]** `process_tss()` 4th argument passes `ProcessingResult` object where `Dict[str, str]` is expected. Currently harmless (parameter immediately overwritten at line 484), but indicates dead/confused code.

### Security (Auditor 3) — 6 issues

22. **[ocean_rs/optical/processors/c2rcc_processor.py:333]** `atmospheric_aux_data_path` inserted into XML via f-string without XML entity escaping. Paths with `<`, `>`, or `&` produce malformed XML.

23. **[ocean_rs/optical/processors/c2rcc_processor.py:336]** `alternative_nn_path` inserted into XML without escaping. Same risk.

24. **[ocean_rs/optical/processors/c2rcc_processor.py:314,316]** `net_set` and `dem_name` strings inserted into XML without escaping. Values from user-supplied JSON configs could contain XML special characters.

25. **[ocean_rs/sar/sensors/sentinel1.py:125-185]** `input_path`, `output_path`, and `polarization` inserted into SAR XML graph via f-string without escaping.

26. **[ocean_rs/sar/download/credentials.py:115-119]** `save_to_dotenv()` writes `.env` file without restrictive permissions (`chmod 600`). Credentials world-readable on Linux/macOS.

### GDAL / Raster I/O (Auditor 4) — 6 issues (excluding duplicate #20)

27. **[ocean_rs/optical/processors/visualization_processor.py:889-950]** Dataset not closed on exception in `_save_rgb_geotiff`. No `finally` block — GDAL dataset leaks on error, causing Windows file locks.

28. **[ocean_rs/optical/processors/visualization_processor.py:952-1014]** Dataset not closed on exception in `_save_single_band_geotiff`. Same issue.

29. **[ocean_rs/optical/processors/visualization_processor.py:373-374]** Double nodata masking — `read_raster` already converts nodata to NaN, then caller re-applies same check. Redundant but benign unless float precision differs.

30. **[ocean_rs/optical/processors/tsm_chl_calculator.py:232,265]** Uncertainty files written without explicit nodata parameter. Relies on implicit defaults — fragile.

31. **[ocean_rs/optical/processors/tss_processor.py:1347-1354]** Classification `uint8` data passed to `write_raster` with default `float32` dtype. *(Same as #20)*

32. **[ocean_rs/optical/processors/visualization_processor.py:938,1001]** Metadata provenance strings still reference `'Unified S2 TSS Pipeline - Marine Visualization'` instead of `'ocean_rs'`.

### GUI Thread Safety (Auditor 5) — 11 issues

33. **[ocean_rs/optical/gui/processing_controller.py:205]** `processing_active` flag set from background thread without synchronization. TOCTOU race with main thread readers.

34. **[ocean_rs/optical/gui/processing_controller.py:158]** `gui.processor` assigned from background thread. Main thread reads it in `update_processing_stats()` — potential race.

35. **[ocean_rs/optical/gui/handlers.py:321-324]** `on_closing()` calls `time.sleep(1)` on main thread, freezing GUI event loop for 1 second. Should use `root.after()` polling.

36. **[ocean_rs/sar/gui/processing_controller.py:74]** SAR `processing_active` flag set from background thread without synchronization.

37. **[ocean_rs/sar/gui/tabs/download_tab.py:181]** `download_active` flag set from download background thread without lock.

38. **[ocean_rs/optical/gui/processing_controller.py:30-31]** Optical double-click protection non-atomic — 59 lines between `if processing_active: return` check and `processing_active = True` set.

39. **[ocean_rs/sar/gui/processing_controller.py:19-20]** SAR double-click protection same non-atomic pattern.

40. **[ocean_rs/optical/gui/processing_controller.py:208-209]** `gui.processor.cleanup()` called from background thread. Could race with main thread's periodic `get_processing_status()` calls.

41. **[ocean_rs/sar/gui/processing_controller.py:85-88]** `stop_processing()` accesses `gui.pipeline` which may be assigned from background thread at any time.

42. **[ocean_rs/sar/gui/tabs/search_tab.py:174-176]** `_do_search()` blocks main thread with synchronous network call. GUI freezes during search.

43. **[ocean_rs/optical/gui/processing_controller.py:266]** Recursive `root.after` scheduling — if root destroyed while pending, `TclError` raised.

### Error Handling (Auditor 6) — 8 issues

44. **[ocean_rs/shared/raster_io.py:85-217]** `write_raster` returns `False` on failure instead of raising. Callers don't check return value, so write failures are silently ignored.

45. **[ocean_rs/optical/processors/water_quality_processor.py:134-136,292-294,380-382]** Three processor methods return empty dict on error. Masks real errors (OOM, corrupt data) as "no products".

46. **[ocean_rs/optical/processors/c2rcc_processor.py:201-202,275-276]** SNAP graph XML writing with no file I/O error handling. Raw `PermissionError`/`OSError` propagates.

47. **[ocean_rs/sar/sensors/sentinel1.py:71]** SAR graph XML writing with no file I/O error handling. Same pattern.

48. **[ocean_rs/sar/download/credentials.py:108-119]** `save_to_dotenv` writes credentials with no error handling for permission/disk errors.

49. **[ocean_rs/optical/processors/tss_processor.py:1087]** Error in per-pixel Jiang algorithm logged at DEBUG level. Many pixel failures could produce mostly-NaN output with no visible warning.

50. **[ocean_rs/optical/gui/tabs/monitoring_tab.py:152]** Disk usage exception silently swallowed — shows "Disk: --" with no indication path is invalid.

51. **[ocean_rs/sar/gui/handlers.py:33-34]** psutil `ImportError` silently swallowed with `except ImportError: pass`. System info labels remain "--" forever with no indication psutil needed.

### Dead Code & Consistency (Auditor 7) — 10 issues

52. **[ocean_rs/optical/gui/handlers.py:69-309]** 14 dead functions (~240 lines) superseded by local tab implementations: `update_subset_visibility`, `update_tss_visibility`, `on_ecmwf_toggle`, `on_rhow_toggle`, `validate_geometry`, `browse_input_dir`, `browse_output_dir`, `apply_water_preset`, `apply_snap_defaults`, `apply_essential_outputs`, `apply_scientific_outputs`, `reset_all_outputs`.

53. **[ocean_rs/optical/processors/tss_processor.py:395,1512]** 2 dead methods: `_update_band_mapping_for_mixed_types()`, `_log_processing_summary()`.

54. **[ocean_rs/optical/processors/c2rcc_processor.py:919]** Dead method `_get_file_size_kb()` — file size computed inline instead.

55. **[ocean_rs/optical/gui/tabs/monitoring_tab.py:197,221]** 2 dead functions: `update_statistics()`, `clear_statistics()`.

56. **[7 config files + theme.py]** Old branding `"Part of the sentinel2_tss_pipeline package."` in docstrings:
    - `enums.py:4`, `s2_config.py:4`, `tss_config.py:12`, `output_categories.py:8`, `water_quality_config.py:4`, `processing_config.py:4`, `theme.py:7`

57. **[ocean_rs/shared/proj_fix.py:134]** Old import path in docstring: `"from sentinel2_tss_pipeline.utils.proj_fix import ..."`.

58. **[ocean_rs/shared/__init__.py]** Missing export: `find_proj_lib_paths` (in `proj_fix.__all__` but not re-exported by shared).

59. **[ocean_rs/optical/__init__.py]** Missing export: `OutputStructure` (in `optical/utils/__init__.__all__` but not available via `from ocean_rs.optical import OutputStructure`).

60. **[4 tab files]** Duplicated scrollable frame setup (~12 identical lines in `c2rcc_tab.py`, `outputs_tab.py`, `spatial_tab.py`, `sar/processing_tab.py`). Should extract to shared utility.

61. **[optical/gui/unified_gui.py, sar/gui/unified_gui.py]** Duplicated `bring_window_to_front()` (~15 identical lines). Should be in shared.

---

## All Minor Issues (41 — nice to have)

### API Contract (Auditor 1)

62. `SafeMathNumPy` re-exported from `optical/__init__.py` but never called anywhere in codebase — dead export.
63. `find_proj_lib_paths` not re-exported by `shared/__init__.py` — inconsistency with sibling functions.
64. `intermediate_paths` parameter in `process_tss` immediately overwritten — misleading signature.
65. `setup_enhanced_logging()` return value not captured in `processing_tab.py:186-188`.

### Scientific Formulas (Auditor 2)

66. CLAUDE.md approximate pure water absorption values (aw(740)~2.38) differ from S2-SRF-interpolated code values (2.711). Documentation should note these are band-center interpolated from Jiang R code.

### Security (Auditor 3)

67. `[sar/gui/tabs/results_tab.py:74]` `subprocess.Popen(['explorer', output])` — no validation that path exists before passing to file manager.
68. `[c2rcc_processor.py:140-144]` Resampling config strings in XML without escaping (from GUI dropdowns/loaded JSON).
69. `[sar/gui/config_io.py:72]` `asdict(gui.config.search_config)` serializes full dataclass — safe now but fragile if credential field ever added.
70. `[c2rcc_processor.py:632]` Full GPT command logged at DEBUG, could expose directory structures.

### GDAL / Raster I/O (Auditor 4)

71-81. GeoTransform order verified correct (3 locations). Band indices verified 1-based. Dataset closure verified correct in `raster_io.py`. CRS preservation verified. Default GeoTransform correct. RGB band indexing correct. (11 items verified — no issues, listed as passing checks.)

### GUI Thread Safety (Auditor 5)

82. `[processing_controller.py:196]` Lambda closure pattern in `eta_var.set` — correct but worth documenting.
83. `[processing_controller.py:266]` Recursive `root.after` scheduling no guard against root destruction.
84. `[sar/gui/handlers.py:41-43]` SAR `start_gui_updates` same recursive scheduling concern.
85. `[processing_controller.py:342]` `gui.processor` read without lock — not dangerous due to GIL but not formally safe.
86. `[sar/gui/processing_controller.py:87]` `stop_processing()` doesn't reset `processing_active` flag.
87. `[search_tab.py / sar search_tab.py]` Treeview index-based lookup of `search_results` could go stale if list mutated.

### Error Handling (Auditor 6)

88. `[logging_utils.py:351-352,392-393]` psutil sampling errors silently swallowed — add `logger.debug`.
89. `[memory_manager.py:32-33]` `del var` wrapped in `except Exception: pass` — unnecessarily broad.
90. `[memory_manager.py:58,75]` psutil monitoring returns sentinel values on error — add `logger.debug`.
91. `[optical/gui/handlers.py:334-335]` `on_closing` cleanup exception swallowed.
92. `[optical/gui/unified_gui.py:87-88, sar/gui/unified_gui.py:39-40]` Window flash exception swallowed.
93. `[optical/gui/handlers.py:40-41,48-49,56-57,62-63]` TclError in tab visibility — expected but could use comment.
94. `[visualization_processor.py:1153-1156,1178-1179]` Cleanup exceptions silently swallowed.
95. `[c2rcc_processor.py:925-926]` `_get_file_size_kb` returns 0.0 on error.
96. `[processing_controller.py:331-332]` System info update exception swallowed.
97. `[unified_processor.py:424]` Existing output check logged at DEBUG instead of WARNING.
98. `[spatial_tab.py:407-408]` Zoom to geometry error at DEBUG — should be WARNING.
99. `[monitoring_tab.py:217-218,231-232]` Statistics update/clear errors at DEBUG.
100. `[sar/main.py:139-140]` Error dialog failure swallowed — could at least `print` to stderr.
101. `[sar/gui/tabs/results_tab.py:67-78]` `subprocess.Popen` with no error handling.
102. `[wave_period.py]` No retry logic for transient ERDDAP failures.

### Dead Code & Consistency (Auditor 7)

103. `[logging_utils.py:189]` Outdated `"SENTINEL-2 TSS PIPELINE v2.0"` in docstring example.

---

## Positive Findings

Several areas passed audit with no issues:

- **Scientific formulas:** All 42 formulas verified correct (Jiang et al. 2021, QAA v6.0, SNAP TSM/CHL, TSI Carlson 1977, linear dispersion, FFT, MAD, all 12 spectral indices, SafeMathNumPy guards)
- **No hardcoded secrets:** Zero matches across all 76 files
- **No bare `except:` clauses** anywhere in the codebase
- **Subprocess safety:** All `subprocess.run()`/`Popen()` calls use list form, no `shell=True`
- **Credential serialization:** Both `config_io.py` files correctly exclude credentials from JSON output
- **Password masking:** `download_tab.py` correctly uses `show="*"`
- **.gitignore:** `.env`, `*.env`, and `credentials.*` patterns all present
- **GDAL band indexing:** Correct 1-based indexing everywhere
- **GeoTransform order:** Correct at all 3 construction/destructuring sites
- **CRS preservation:** Projection strings carried through read/write cycle correctly
- **All cross-module API calls** match actual method signatures
- **All dataclass field accesses** verified against declarations
- **All `__init__.py` exports** resolve to real source-module symbols

---

## Recommended Fix Priority

### Phase 1: Runtime Safety (Critical)
1. Fix SAR GUI thread safety (items 1-4) — Tcl crashes, segfaults
2. Fix nodata header/pixel mismatch (item 12) — corrupts output
3. Add `from e` to exception chains (item 5)
4. Improve SNAP error messages (items 6-8)
5. Implement download retry logic using existing `retry_count` (item 9)

### Phase 2: Correctness (Important)
6. Update provenance stamp to `'ocean_rs v3.0.0'` (item 11)
7. Add XML entity escaping for SNAP graph parameters (items 22-25)
8. Add `finally` blocks for GDAL dataset closure (items 27-28)
9. Fix uint8 classification dtype (item 20)
10. Make `write_raster` raise on failure or ensure callers check return (item 44)

### Phase 3: Cleanup (Important/Minor)
11. Remove ~400 lines of dead code (items 14, 52-55)
12. Update old branding in 7+ files (items 56-57)
13. Extract duplicated code to shared utilities (items 60-61)
14. Upgrade error logging from DEBUG to WARNING where appropriate (items 49, 97-99)
15. Add missing `__init__.py` exports (items 58-59)
