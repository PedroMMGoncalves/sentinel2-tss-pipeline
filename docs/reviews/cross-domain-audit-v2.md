# OceanRS Cross-Domain Audit Report v2

**Date:** 2026-03-02
**Auditors:** 5 domain-expert agents (Geospatial Engineer, Scientific Computing, ML/Statistical Modeler, Python Systems Engineer, Domain Scientist)
**Scope:** Full `ocean_rs/` codebase — POST-FIX audit (after 85 fixes from v1 audit)
**Methodology:** Each auditor independently reviewed all files in their domain, then findings were cross-referenced and deduplicated.

---

## Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 1 |
| HIGH | 9 |
| MEDIUM | 18 |
| LOW | 14 |
| **Total** | **42** |

*After deduplication — raw total across 5 auditors was ~56 findings; duplicates merged with severity elevated to highest.*

---

## Findings (sorted by severity)

### CRITICAL

```
ID: C2-1
SEVERITY: CRITICAL
AUDITORS: Scientific Computing, ML/Statistical Modeler
FILE: ocean_rs/optical/processors/tss_processor.py:1028
PROBLEM: Type II QAA denominator has no division-by-zero guard.
  ratio = vp[665][m] / (vp[443][m] + vp[490][m])
  The valid pixel mask allows vp[443] and vp[490] down to -0.001 (relaxed threshold).
  If vp[443] + vp[490] ≈ 0, this produces inf values. Type I has np.maximum(denom, 1e-10) guard; Type II does not.
IMPACT: For moderately turbid water where blue Rrs nearly cancel, produces inf absorption → inf bbp → corrupted TSS. Silently drops pixels.
FIX: denominator = vp[443][m] + vp[490][m]; denominator = np.maximum(denominator, 1e-10); ratio = vp[665][m] / denominator
CROSS-DOMAIN: Data quality (Geospatial), Classification (ML)
```

---

### HIGH

```
ID: H2-1
SEVERITY: HIGH
AUDITORS: Scientific Computing
FILE: ocean_rs/optical/processors/tss_processor.py:954-966
PROBLEM: Borderline zero-Rrs pixels silently produce NaN TSS.
  rrs[wl] = Rrs_val / (0.52 + 1.7 * Rrs_val)
  When Rrs=0 (borderline valid), rrs=0, then u computation requires rrs>0, producing NaN.
  These pixels pass valid_mask but output NaN TSS with no diagnostic.
IMPACT: Coverage percentage misleading — counts pixels from valid_mask but some produce NaN.
FIX: Add epsilon guard: rrs[wl] = Rrs_val / (0.52 + 1.7 * np.maximum(Rrs_val, 0)); Add diagnostic log for NaN u counts.
CROSS-DOMAIN: None
```

```
ID: H2-2
SEVERITY: HIGH
AUDITORS: Scientific Computing
FILE: ocean_rs/sar/bathymetry/compositor.py:106-134
PROBLEM: _weighted_median corrupted by NaN values in input.
  np.argsort places NaN at end; NaN weights cause cumsum to become NaN;
  np.searchsorted(cumsum, NaN) returns len(cumsum), producing wrong result.
IMPACT: When compositing bathymetry where some observations have NaN (deep water in some passes), composite depth is incorrect.
FIX: Filter NaN values/weights before sorting: valid = np.isfinite(v) & np.isfinite(w) & (w > 0); operate only on valid subset; default to NaN when no valid points.
CROSS-DOMAIN: SAR bathymetry (Domain Scientist)
```

```
ID: H2-3
SEVERITY: HIGH
AUDITORS: ML/Statistical Modeler, Domain Scientist
FILE: ocean_rs/optical/processors/water_quality_processor.py:204,242
PROBLEM: HAB detection accumulates NaN-contaminated contributions into zero-initialized array.
  0 + NaN = NaN, so pixels where only one algorithm has valid data get NaN hab_score
  even though a partial result is available. Uses scalar n_algorithms instead of per-pixel count.
IMPACT: Valid bloom detections silently lost when one algorithm has NaN but the other has a valid score.
FIX: Use per-pixel algorithm counting: n_valid = np.zeros(shape); for each algorithm, increment n_valid where result is finite; hab_score = hab_score_accum / np.maximum(n_valid, 1); set hab_score[n_valid == 0] = np.nan.
CROSS-DOMAIN: Scientific validity (Domain Scientist)
```

```
ID: H2-4
SEVERITY: HIGH
AUDITORS: Geospatial Engineer
FILE: ocean_rs/shared/geometry_utils.py:193-309
PROBLEM: Fiona path skips CRS reprojection when OGR unavailable.
  When Fiona is available but OGR is not, loading a projected-CRS shapefile returns
  UTM coordinates as WKT. SNAP geoRegion expects WGS84.
IMPACT: SNAP subset covers entire globe or fails silently with projected coordinates interpreted as degrees.
FIX: After Fiona path succeeds, if not HAS_OGR and CRS is projected: return error message requiring WGS84 shapefile or GDAL installation.
CROSS-DOMAIN: GUI/handlers (spatial tab)
```

```
ID: H2-5
SEVERITY: HIGH
AUDITORS: Geospatial Engineer
FILE: ocean_rs/sar/bathymetry/depth_inversion.py:46-52,125-128
PROBLEM: Wavelength filtering (removing non-positive values) shortens arrays, breaking spatial correspondence.
  wavelengths = wavelengths[positive_mask] produces shorter array than swell.tile_centers.
  BathymetryResult.geo still references original tile layout.
IMPACT: Exported GeoTIFF has wrong spatial extent; compositor may crash on shape mismatch.
FIX: Replace filtering with NaN-masking: wavelengths[~positive_mask] = np.nan; handle NaN in k computation.
CROSS-DOMAIN: SAR pipeline architecture
```

```
ID: H2-6
SEVERITY: HIGH
AUDITORS: Geospatial Engineer
FILE: ocean_rs/sar/core/bathymetry_pipeline.py:229-276
PROBLEM: _export_results writes 1D swell array to RasterIO.write_raster which requires 2D.
  SwellField stores tile-level results as 1D arrays, not gridded 2D rasters.
  write_raster raises RuntimeError("Data must be 2D array").
IMPACT: SAR pipeline cannot produce its final GeoTIFF output — always crashes.
FIX: Add gridding step (scipy.interpolate.griddata) from tile centers to regular grid, or log warning and skip export until gridding is implemented.
CROSS-DOMAIN: SAR pipeline architecture
```

```
ID: H2-7
SEVERITY: HIGH
AUDITORS: Python Systems Engineer
FILE: ocean_rs/optical/processors/c2rcc_processor.py:289-299
PROBLEM: _get_graph_file_path creates tempfile.mkdtemp() directories that are never cleaned up.
  cleanup() only removes the XML file, not the temp directory.
IMPACT: Accumulates orphan ocean_rs_snap_* directories in system temp folder across processing runs.
FIX: Track temp dirs in self._temp_dirs list; clean them up with shutil.rmtree in cleanup().
CROSS-DOMAIN: None
```

```
ID: H2-8
SEVERITY: HIGH
AUDITORS: Python Systems Engineer
FILE: ocean_rs/optical/processors/c2rcc_processor.py:1057-1058
PROBLEM: get_processing_status progress always shows 100%.
  progress_percent = (total / max(total, 1)) * 100 = total/total*100 = 100%.
IMPACT: GUI progress bar jumps to 100% after first product; ETA meaningless.
FIX: Use self.total_products as denominator: progress_percent = (total / max(self.total_products, 1)) * 100
CROSS-DOMAIN: GUI correctness
```

```
ID: H2-9
SEVERITY: HIGH
AUDITORS: Domain Scientist
FILE: ocean_rs/sar/config/sar_config.py:14
PROBLEM: Default FFT tile_size_m=512 but max_wavelength_m=600 — Nyquist violation in config defaults.
  512m tile can only fit <1 cycle of 600m swell; Nyquist requires >=2 cycles.
  fft_extractor.py defaults were updated to 1024m, but sar_config.py still has 512m.
IMPACT: If config is loaded from sar_config.py (overriding fft_extractor defaults), swell >256m is aliased.
FIX: Change default tile_size_m to 1024 in sar_config.py. Or: max_wavelength_m = tile_size_m / 2.
CROSS-DOMAIN: Scientific Computing (FFT)
```

---

### MEDIUM

```
ID: M2-1
SEVERITY: MEDIUM
AUDITORS: Scientific Computing
FILE: ocean_rs/optical/processors/water_quality_processor.py:122-129
PROBLEM: Beam attenuation saturates at clip limit (50 m^-1) for turbid water without warning.
  bb/0.0183 amplifies backscattering ~55x; turbid water (bb=0.5) produces beam_att=27+, frequently clipping.
IMPACT: Beam attenuation product uninformative for Type III/IV water. Users unaware of saturation.
FIX: Add log warning when n_saturated > 0.
CROSS-DOMAIN: None
```

```
ID: M2-2
SEVERITY: MEDIUM
AUDITORS: Scientific Computing
FILE: ocean_rs/sar/bathymetry/fft_extractor.py:101,141
PROBLEM: Inconsistent valid-pixel criteria between tile gate and confidence penalty.
  Gate (line 101): isfinite AND nonzero. Penalty (line 141): only ~isnan. Zero-fill tiles get inflated confidence.
IMPACT: Tiles with significant zero-fill get higher confidence than warranted.
FIX: Make criteria consistent: valid_data_frac = np.sum(np.isfinite(tile_original) & (tile_original != 0)) / tile_original.size
CROSS-DOMAIN: None
```

```
ID: M2-3
SEVERITY: MEDIUM
AUDITORS: Scientific Computing
FILE: ocean_rs/sar/bathymetry/depth_inversion.py:120-122
PROBLEM: dh/dL = h/L uncertainty approximation inaccurate for intermediate depth (kh~0.5-3).
  True sensitivity from dispersion relation differs by factor 2-3x in this regime.
IMPACT: Uncertainty estimates off by 2-3x in the regime where satellite bathymetry is most useful.
FIX: Use proper dispersion derivative or numerical perturbation method.
CROSS-DOMAIN: Domain Scientist
```

```
ID: M2-4
SEVERITY: MEDIUM
AUDITORS: Scientific Computing
FILE: ocean_rs/optical/processors/tss_processor.py:970-972
PROBLEM: Cubic polynomial for rrs620 can produce negative values for certain Rrs665 ranges.
  Negative rrs620 makes Type II condition trivially true, misclassifying turbid pixels.
IMPACT: Systematic classification error for pixels where polynomial yields negative rrs620.
FIX: rrs620 = np.maximum(rrs620, 0.0)
CROSS-DOMAIN: ML/Statistical (classification)
```

```
ID: M2-5
SEVERITY: MEDIUM
AUDITORS: Geospatial Engineer
FILE: ocean_rs/shared/raster_io.py:13-43
PROBLEM: GDAL import ordering — _configure_gdal() forward-references GDAL_AVAILABLE defined later.
IMPACT: No crash today, but fragile ordering.
FIX: Move try/except import block above function definition.
CROSS-DOMAIN: None
```

```
ID: M2-6
SEVERITY: MEDIUM
AUDITORS: Geospatial Engineer
FILE: ocean_rs/optical/processors/visualization_processor.py:1008-1013
PROBLEM: Spectral index nodata inherited from source metadata (may be 0).
  If source nodata=0, legitimate index values of 0 become indistinguishable from nodata.
IMPACT: Water-land boundary pixels at index=0 masked as nodata in GIS.
FIX: Always use nodata_val = -9999 for spectral indices.
CROSS-DOMAIN: None
```

```
ID: M2-7
SEVERITY: MEDIUM
AUDITORS: Geospatial Engineer
FILE: ocean_rs/shared/geometry_utils.py:443-552
PROBLEM: GeoJSON loader assumes WGS84 without verifying CRS. Projected CRS files pass through unchecked.
IMPACT: Projected GeoJSON coordinates interpreted as WGS84, wildly incorrect SNAP subset.
FIX: Add bounds sanity check: if any bound exceeds 360, reject with error.
CROSS-DOMAIN: None
```

```
ID: M2-8
SEVERITY: MEDIUM
AUDITORS: Geospatial Engineer
FILE: ocean_rs/optical/processors/tss_processor.py:247-259
PROBLEM: _rasterize_shapefile_mask missing SetAxisMappingStrategy for GDAL 3+.
  CRS comparison may have X/Y swapped without explicit axis order setting.
IMPACT: Water mask incorrect (coordinates swapped) depending on GDAL version.
FIX: Add shp_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER) on both SRS objects.
CROSS-DOMAIN: None
```

```
ID: M2-9
SEVERITY: MEDIUM
AUDITORS: Geospatial Engineer
FILE: ocean_rs/shared/raster_io.py:163-166,223
PROBLEM: NoData out of range for integer dtypes (e.g., nodata=-9999 for uint8 max=255).
IMPACT: GDAL stores invalid nodata metadata. No crash today (callers pass correct values).
FIX: Add range check and warning; clamp to dtype range.
CROSS-DOMAIN: None
```

```
ID: M2-10
SEVERITY: MEDIUM
AUDITORS: Geospatial Engineer
FILE: ocean_rs/sar/bathymetry/compositor.py:98
PROBLEM: No CRS consistency check across composited results.
  Inherits geo from results[0] without verifying all results share the same CRS.
IMPACT: Multi-zone compositing produces spatially incoherent GeoTIFF.
FIX: Compare CRS WKT across all results; raise ValueError on mismatch.
CROSS-DOMAIN: None
```

```
ID: M2-11
SEVERITY: MEDIUM
AUDITORS: Geospatial Engineer
FILE: ocean_rs/sar/core/bathymetry_pipeline.py:194-196
PROBLEM: _get_wgs84_center returns raw pixel coordinates as WGS84 when CRS is missing.
IMPACT: WW3 API receives pixel coordinates instead of lon/lat.
FIX: Add coordinate range sanity check; raise if not plausible WGS84.
CROSS-DOMAIN: None
```

```
ID: M2-12
SEVERITY: MEDIUM
AUDITORS: Python Systems Engineer
FILE: ocean_rs/shared/logging_utils.py:50-71
PROBLEM: ColoredFormatter.format() mutates record.levelname with ANSI codes permanently.
  File handler receives mutated record, producing garbled ANSI codes in log files.
IMPACT: Log files contain [32mINFO [0m instead of clean text.
FIX: Work on a copy: record = logging.makeLogRecord(record.__dict__) before mutation.
CROSS-DOMAIN: Logging correctness
```

```
ID: M2-13
SEVERITY: MEDIUM
AUDITORS: Python Systems Engineer
FILE: ocean_rs/optical/gui/handlers.py:125-130
PROBLEM: on_closing() runs cleanup() before processing thread fully stops.
  SNAP graph XML could be deleted while GPT subprocess still reads it.
IMPACT: Potential SNAP failure if graph file deleted mid-read.
FIX: Move cleanup() into _check_and_destroy() so it runs after thread stops.
CROSS-DOMAIN: Process lifecycle
```

```
ID: M2-14
SEVERITY: MEDIUM
AUDITORS: Python Systems Engineer
FILE: ocean_rs/optical/processors/tss_processor.py:236-244
PROBLEM: OGR datasource handle leaked on early return when layer is None.
IMPACT: Shapefile locked on Windows until GC collects the handle.
FIX: Add shp = None before return None on the error path.
CROSS-DOMAIN: Resource management
```

```
ID: M2-15
SEVERITY: MEDIUM
AUDITORS: Python Systems Engineer
FILE: ocean_rs/optical/gui/processing_controller.py:215
PROBLEM: Cleanup lambda captures gui.processor at evaluation time, not definition time.
  gui.processor could be None or already cleaned up when lambda fires.
IMPACT: Potential AttributeError or double-cleanup.
FIX: Capture reference under lock: processor = gui.processor; lambda p=processor: p.cleanup()
CROSS-DOMAIN: Thread safety
```

```
ID: M2-16
SEVERITY: MEDIUM
AUDITORS: Domain Scientist
FILE: ocean_rs/sar/sensors/sentinel1.py:239
PROBLEM: 5x5 Lee Sigma speckle filter attenuates short-wavelength swell (50-100m) before FFT.
IMPACT: Degraded FFT confidence for wavelengths approaching filter kernel size.
FIX: Increase filter to 7x7 or 9x9 for IW mode; add comment documenting trade-off.
CROSS-DOMAIN: Scientific Computing (FFT SNR)
```

```
ID: M2-17
SEVERITY: MEDIUM
AUDITORS: Scientific Computing
FILE: ocean_rs/sar/bathymetry/depth_inversion.py:128
PROBLEM: np.maximum(confidence, 0.1) returns NaN when confidence is NaN.
IMPACT: NaN uncertainties produced silently for points with NaN confidence.
FIX: safe_confidence = np.where(np.isfinite(confidence), np.maximum(confidence, 0.1), 0.1)
CROSS-DOMAIN: None
```

```
ID: M2-18
SEVERITY: MEDIUM
AUDITORS: Scientific Computing
FILE: ocean_rs/optical/processors/water_quality_processor.py:112,118
PROBLEM: Secchi/euphotic depth epsilon guards produce meaningless intermediate values for Kd≈0.
  1.7 / 1e-8 = 1.7e8 meters before clip catches it.
IMPACT: Correct after clip, but epsilon guard gives false confidence in code robustness.
FIX: Use conditional: safe_kd = kd_valid > 0.001; compute only on safe pixels.
CROSS-DOMAIN: None
```

---

### LOW

```
ID: L2-1
SEVERITY: LOW
AUDITORS: Scientific Computing
FILE: ocean_rs/shared/raster_io.py:279
PROBLEM: np.std() uses ddof=0 (population std) instead of ddof=1 (sample std).
IMPACT: Negligible for millions of pixels, but technically wrong statistic.
FIX: np.std(valid_data, ddof=1)
```

```
ID: L2-2
SEVERITY: LOW
AUDITORS: Geospatial Engineer
FILE: ocean_rs/optical/processors/visualization_processor.py:949-951
PROBLEM: RGB nodata=0 masks legitimate deep-water pixels (very low reflectance → 0 after normalization).
IMPACT: Deep water appears transparent in GIS mosaics. Cosmetic, not scientific.
FIX: Remove nodata from RGB composites or use alpha band.
```

```
ID: L2-3
SEVERITY: LOW
AUDITORS: Geospatial Engineer
FILE: ocean_rs/shared/raster_io.py:193-205
PROBLEM: write_raster does not create parent directories.
IMPACT: Fails with opaque error if parent dir missing. Mitigated by OutputStructure.
FIX: Add os.makedirs(os.path.dirname(output_path), exist_ok=True)
```

```
ID: L2-4
SEVERITY: LOW
AUDITORS: Geospatial Engineer
FILE: ocean_rs/optical/processors/c2rcc_processor.py:903,950
PROBLEM: _calculate_missing_btot writes GeoTIFF with .img extension (format mismatch).
IMPACT: SNAP may not read the file. Current pipeline reads via GDAL (works fine).
FIX: Change to iop_btot.tif
```

```
ID: L2-5
SEVERITY: LOW
AUDITORS: Python Systems Engineer
FILE: ocean_rs/optical/processors/visualization_processor.py:955,1028
PROBLEM: datetime.now() without timezone (local time instead of UTC).
  raster_io.py correctly uses datetime.now(timezone.utc).
IMPACT: Inconsistent timestamps across output GeoTIFFs. Violates CLAUDE.md UTC rule.
FIX: datetime.now(timezone.utc).isoformat()
```

```
ID: L2-6
SEVERITY: LOW
AUDITORS: Python Systems Engineer
FILE: ocean_rs/optical/config/tss_config.py:42
PROBLEM: water_quality_config typed as Optional[object] instead of Optional[WaterQualityConfig].
IMPACT: No autocomplete or type checking. No runtime bug.
FIX: Use TYPE_CHECKING import for proper type annotation.
```

```
ID: L2-7
SEVERITY: LOW
AUDITORS: Python Systems Engineer
FILE: ocean_rs/shared/memory_manager.py:22-29
PROBLEM: _process class-level cache without lock (benign race in CPython).
IMPACT: Worst case: redundant psutil.Process() object created.
FIX: Accept with comment or add threading.Lock.
```

```
ID: L2-8
SEVERITY: LOW
AUDITORS: Python Systems Engineer
FILE: ocean_rs/optical/core/unified_processor.py:562-567
PROBLEM: cleanup() does not clean up TSSProcessor (no cleanup method exists yet).
IMPACT: Maintenance hazard if TSSProcessor later manages resources.
FIX: Add defensive call: if self.tss_processor and hasattr(self.tss_processor, 'cleanup'): self.tss_processor.cleanup()
```

```
ID: L2-9
SEVERITY: LOW
AUDITORS: Python Systems Engineer
FILE: ocean_rs/sar/gui/processing_controller.py:22
PROBLEM: SAR processing_active set True before validation (same bug previously fixed in optical).
IMPACT: Early exception leaves Start button permanently disabled.
FIX: Move processing_active = True to after all validation.
```

```
ID: L2-10
SEVERITY: LOW
AUDITORS: Python Systems Engineer
FILE: ocean_rs/shared/raster_io.py:80-106
PROBLEM: GDAL dataset not explicitly closed on exception path (relies on GC).
IMPACT: File handle locked on Windows until garbage collected.
FIX: Add finally block: dataset = None
```

```
ID: L2-11
SEVERITY: LOW
AUDITORS: Python Systems Engineer
FILE: ocean_rs/shared/logging_utils.py:354-359,406
PROBLEM: psutil exceptions silently swallowed with bare except Exception: pass.
IMPACT: Resource stats silently empty with no indication of failure.
FIX: except Exception as e: logger.debug(f"Resource sampling failed: {e}")
```

```
ID: L2-12
SEVERITY: LOW
AUDITORS: Domain Scientist
FILE: ocean_rs/optical/processors/tss_processor.py:1293
PROBLEM: Legend TSS ranges inconsistent with CLAUDE.md and Jiang 2021 documentation.
IMPACT: User confusion in result interpretation.
FIX: Align legend ranges with documented values.
```

```
ID: L2-13
SEVERITY: LOW
AUDITORS: Scientific Computing
FILE: ocean_rs/sar/bathymetry/depth_inversion.py:83-100
PROBLEM: Newton-Raphson convergence report counts stalled deep-water points as "converged".
IMPACT: Misleading metadata — functionally correct (post-filter catches them).
FIX: Add debug log for n_stalled points per iteration.
```

```
ID: L2-14
SEVERITY: LOW
AUDITORS: Python Systems Engineer
FILE: ocean_rs/optical/processors/c2rcc_processor.py:729-736
PROBLEM: Double-close of stdout/stderr log files (inner finally + outer except).
IMPACT: Harmless (Python file.close() is idempotent), but unclear control flow.
FIX: Remove redundant close() from TimeoutExpired handler.
```

---

## Cross-Cutting Concerns

### XC2-1: SAR Pipeline Cannot Produce GeoTIFF Output (H2-5, H2-6)
**Auditors:** Geospatial Engineer, Scientific Computing
**Files:** depth_inversion.py, bathymetry_pipeline.py
The depth inversion returns 1D tile-level arrays, but the export function requires 2D grids. Additionally, wavelength filtering breaks the spatial correspondence between depth arrays and tile center coordinates. The SAR pipeline needs a gridding step between inversion and export.

### XC2-2: Type II QAA Path Vulnerable (C2-1, M2-4)
**Auditors:** Scientific Computing, ML/Statistical Modeler
**Files:** tss_processor.py
Type II water processing has two related vulnerabilities: the denominator (vp[443]+vp[490]) can be zero (C2-1), and the rrs620 polynomial can produce negative values causing misclassification (M2-4). Both affect moderately turbid water — the transitional regime between Types I and III.

### XC2-3: HAB Score NaN Propagation (H2-3)
**Auditors:** ML/Statistical Modeler, Domain Scientist
**Files:** water_quality_processor.py
The HAB detection uses a scalar algorithm count instead of per-pixel counting, dropping valid single-algorithm detections when the other algorithm has NaN. This is a systematic data loss issue.

### XC2-4: Log File Corruption (M2-12)
**Auditors:** Python Systems Engineer
**Files:** logging_utils.py
The ColoredFormatter mutates the shared LogRecord object, leaking ANSI escape codes into file logs. This affects all file-based logging output.

---

## Fix Priority Roadmap

### Phase 1: Critical + Blocking (fix immediately)
- C2-1: Type II QAA denominator guard
- H2-5 + H2-6: SAR export (NaN-mask instead of filter + gridding/skip)
- H2-8: Progress always 100%

### Phase 2: Data Quality (fix before next release)
- H2-1: Borderline zero-Rrs NaN guard
- H2-2: weighted_median NaN handling
- H2-3: HAB per-pixel algorithm counting
- H2-7: Temp directory leak
- M2-4: rrs620 clamp to non-negative
- M2-8: Axis mapping strategy for GDAL 3+

### Phase 3: Robustness (fix soon)
- H2-4: Fiona CRS reprojection fallback
- H2-9: SAR config tile_size default
- M2-5 through M2-18: Medium findings

### Phase 4: Polish
- L2-1 through L2-14: Low findings
