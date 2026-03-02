# OceanRS Cross-Domain Audit Report

**Date:** 2026-03-02
**Auditors:** 5 domain-expert agents (Geospatial Engineer, Scientific Computing, ML/Statistical Modeler, Python Systems Engineer, Domain Scientist)
**Scope:** Full `ocean_rs/` codebase (optical + SAR + shared)
**Methodology:** Each auditor independently reviewed all files in their domain, then findings were cross-referenced for multi-domain issues.

---

## Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 4 |
| HIGH | 20 |
| MEDIUM | 27 |
| LOW | 34 |
| **Total** | **85** |

*After deduplication — raw total across 5 auditors (with verification runs) was 120+ findings; duplicates were merged and severity elevated to the highest across auditors.*

---

## Findings (sorted by severity)

### CRITICAL

```
ID: C-1
SEVERITY: CRITICAL
AUDITORS: Python Systems Engineer, Scientific Computing
FILE: ocean_rs/shared/memory_manager.py:21-33
PROBLEM: cleanup_variables() deletes local parameter binding, not caller's reference — entire memory cleanup strategy is a no-op
IMPACT: Multi-scene batch processing accumulates memory without release. Only gc.collect() runs, which cannot reclaim arrays still referenced by callers.
FIX: Remove cleanup_variables(). At each call site (e.g., unified_processor.py:321), explicitly `del` the actual variables, then call gc.collect().
CROSS-DOMAIN: Geospatial Engineer — large raster arrays never freed between scenes
```

```
ID: C-2
SEVERITY: CRITICAL
AUDITORS: Python Systems Engineer
FILE: ocean_rs/optical/processors/tss_processor.py:394-624
PROBLEM: ~30 full-resolution float32 arrays held simultaneously without explicit cleanup (~14 GB for a 10m S2 tile)
IMPACT: Combined with C-1 (no-op cleanup), processing multiple scenes will exhaust memory. _convert_rhow_to_rrs creates unnecessary .copy() for rrs bands, doubling band memory.
FIX: (1) Remove .copy() for rrs/unknown bands in _convert_rhow_to_rrs. (2) del bands_data after conversion. (3) del converted_bands_data after jiang_results. (4) del jiang_results after merging. (5) gc.collect() after each major deletion.
CROSS-DOMAIN: None
```

```
ID: C-3
SEVERITY: CRITICAL
AUDITORS: ML/Statistical Modeler
FILE: ocean_rs/optical/processors/tsm_chl_calculator.py:219-230
PROBLEM: Uncertainty model (TSM*0.20+0.1, CHL*0.30+0.05) is fabricated with no empirical or theoretical basis — not uncertainty propagation
IMPACT: Users relying on uncertainty bands for data assimilation or decision support will have scientifically unsupported confidence intervals. The linear error model does not reflect heteroscedastic NN-derived IOP errors.
FIX: (a) Propagate actual C2RCC unc_* bands if available, (b) use published validation statistics (Brockmann et al. 2016), or (c) clearly label as "heuristic placeholder" in metadata and log a warning.
CROSS-DOMAIN: Domain Scientist — misrepresents measurement quality
```

```
ID: C-4
SEVERITY: CRITICAL
AUDITORS: ML/Statistical Modeler, Scientific Computing
FILE: ocean_rs/sar/bathymetry/compositor.py:49
PROBLEM: MAD-based uncertainty with N=2 scenes reduces to min(|residual|), systematically underestimating spread by 2x or more. No minimum sample size check.
IMPACT: 2-scene composites (common case) report artificially low uncertainty, giving false confidence in depth estimates.
FIX: When N<3, fall back to weighted_mean (well-defined uncertainty for N=2) or use half-range. Log warning when N<5 that MAD-based uncertainty is unreliable.
CROSS-DOMAIN: None
```

---

### HIGH

```
ID: H-1
SEVERITY: HIGH
AUDITORS: Scientific Computing
FILE: ocean_rs/optical/processors/tss_processor.py:935-940
PROBLEM: QAA Type I water path has 3 unguarded numerical operations: (1) division by rrs[490] near zero, (2) division by composed denominator, (3) 10.0** overflow for large negative x
IMPACT: For clear water (Type I — most common in open ocean), near-zero Rrs(490) causes denominator explosion → ratio→0 → x→-inf → 10^(large positive) → inf absorption → corrupted TSS. Pixels silently dropped.
FIX: (1) Add epsilon to rrs[490]: rrs[490][m] + 1e-10. (2) Guard denominator > 1e-10 before dividing. (3) Clamp x to [-2, 2] based on physical QAA ranges before computing 10**exponent.
CROSS-DOMAIN: Domain Scientist — affects most common water type
```

```
ID: H-2
SEVERITY: HIGH
AUDITORS: ML/Statistical Modeler, Domain Scientist
FILE: ocean_rs/optical/processors/tss_processor.py:862-876
PROBLEM: Valid pixel mask requires ALL bands (443-865nm) > 0, but turbid waters often have negative blue-band Rrs from atmospheric correction artifacts
IMPACT: Type III/IV pixels (most interesting for TSS) are disproportionately excluded because blue bands go negative in turbid water. Creates systematic clear-water bias.
FIX: Require Rrs > 0 only for bands actually used by each water type, or relax to Rrs > -0.001 for classification-only bands.
CROSS-DOMAIN: None
```

```
ID: H-3
SEVERITY: HIGH
AUDITORS: ML/Statistical Modeler
FILE: ocean_rs/optical/processors/tss_processor.py:920-923
PROBLEM: Water type classification at exact boundaries (e.g., Rrs(490)==Rrs(560)) causes discontinuous TSS jumps of 20-40% due to different reference wavelengths per type
IMPACT: Small floating-point perturbations flip pixels between water types. No boundary uncertainty flag or fuzzy classification.
FIX: (a) Document tie-breaking convention. (b) Add boundary uncertainty flag for pixels within 5% of thresholds. (c) Consider fuzzy blending at boundaries.
CROSS-DOMAIN: None
```

```
ID: H-4
SEVERITY: HIGH
AUDITORS: ML/Statistical Modeler
FILE: ocean_rs/optical/processors/water_quality_processor.py:193-201
PROBLEM: HAB "probability" is an uncalibrated heuristic score (NDCI+MCI in [0,1]) but named/presented as a probability. Single-algorithm pixels max out at 0.5, making high-risk (>0.7) unreachable.
IMPACT: Users will misinterpret as calibrated statistical probability. Severe blooms detected by only one algorithm can never trigger high-risk alert.
FIX: (a) Rename to hab_score/hab_index. (b) Scale single-algorithm output to [0,1]. (c) Make ramp thresholds configurable. (d) Add metadata disclaimer.
CROSS-DOMAIN: Domain Scientist — scientific claim mismatch
```

```
ID: H-5
SEVERITY: HIGH
AUDITORS: ML/Statistical Modeler, Scientific Computing, Domain Scientist
FILE: ocean_rs/sar/bathymetry/depth_inversion.py:74-77
PROBLEM: Depth uncertainty uses simplified dh/dL=h/L with assumed 10% wavelength uncertainty, ignoring actual FFT confidence values entirely
IMPACT: Low-confidence FFT tiles get same uncertainty as high-confidence ones. Shallow-water uncertainty underestimated (nonlinear dispersion). Deep-water overestimated.
FIX: (a) Scale wavelength uncertainty by FFT confidence: base_uncertainty/confidence. (b) Use analytical dispersion relation partial derivatives for proper propagation.
CROSS-DOMAIN: None
```

```
ID: H-6
SEVERITY: HIGH
AUDITORS: ML/Statistical Modeler
FILE: ocean_rs/sar/bathymetry/compositor.py:40
PROBLEM: Inverse-variance weighting assumes independent errors, but overlapping SAR scenes have correlated errors (tide, currents). Epsilon 1e-10 provides no regularization.
IMPACT: Composite uncertainty systematically underestimated. Near-zero uncertainty points dominate the composite.
FIX: (a) Increase epsilon to 0.25 (=0.5m^2 minimum uncertainty). (b) Document independence assumption. (c) Down-weight temporally correlated acquisitions.
CROSS-DOMAIN: None
```

```
ID: H-7
SEVERITY: HIGH
AUDITORS: Domain Scientist
FILE: ocean_rs/sar/sensors/sentinel1.py:39-95
PROBLEM: Terrain Correction applied BEFORE FFT extraction — resampling acts as low-pass filter, attenuating wave signals and biasing toward longer wavelengths
IMPACT: Systematic depth overestimation in shallow areas due to corrupted wave spectrum from interpolation artifacts.
FIX: Split preprocessing: (1) Orbit→Thermal-Noise→Calibration (no TC) for FFT input, (2) apply TC only to final bathymetry output for georeferencing.
CROSS-DOMAIN: Python Systems Engineer — pipeline architecture change
```

```
ID: H-8
SEVERITY: HIGH
AUDITORS: Domain Scientist
FILE: ocean_rs/sar/core/bathymetry_pipeline.py:152-155
PROBLEM: UTM projected coordinates (e.g., 500000, 4000000) passed to WaveWatch III ERDDAP API which expects WGS84 lon/lat degrees
IMPACT: WW3 query will fail (coordinates outside valid range) or return wrong-location data. Always falls back to manual wave period.
FIX: Convert UTM center coordinates to WGS84 using CRS info from image.geo.crs_wkt before ERDDAP query.
CROSS-DOMAIN: Geospatial Engineer — coordinate system mismatch
```

```
ID: H-9
SEVERITY: HIGH
AUDITORS: Domain Scientist
FILE: ocean_rs/sar/bathymetry/depth_inversion.py:36-37
PROBLEM: Single wave period used for entire scene (250km x 250km). Period varies significantly near coast due to shoaling, refraction, multiple swell systems.
IMPACT: Depth errors grow nonlinearly — 1s period error → several meters depth error in shallow water.
FIX: (a) Query WW3 at multiple points, (b) derive per-tile period from deep-water FFT tiles, or (c) document limitation and recommend small AOIs.
CROSS-DOMAIN: None
```

```
ID: H-10
SEVERITY: HIGH
AUDITORS: Python Systems Engineer
FILE: ocean_rs/sar/sensors/sentinel1.py:81-92
PROBLEM: subprocess.run() with 2-hour timeout does not kill child process on TimeoutExpired — SNAP GPT becomes zombie
IMPACT: After timeout, GPT continues consuming CPU/memory/disk. User sees error but zombie persists until manual kill.
FIX: Switch to subprocess.Popen with explicit process.kill() + process.wait() in timeout handler (matching c2rcc_processor.py pattern).
CROSS-DOMAIN: None
```

```
ID: H-11
SEVERITY: HIGH
AUDITORS: Python Systems Engineer
FILE: ocean_rs/optical/gui/processing_controller.py:158-208
PROBLEM: gui.processor accessed from background thread without synchronization; get_processing_status() reads multiple fields that could be mid-update
IMPACT: Processing stats display may show inconsistent progress numbers or incorrect ETA.
FIX: Use threading.Lock to protect shared processor state, or pass status via thread-safe queue.
CROSS-DOMAIN: None
```

```
ID: H-12
SEVERITY: HIGH
AUDITORS: Geospatial Engineer
FILE: ocean_rs/shared/geometry_utils.py:146-232
PROBLEM: Shapefile geometry not reprojected to WGS84 before returning WKT — projected CRS coordinates (UTM) passed directly to SNAP geoRegion
IMPACT: Shapefiles in UTM produce WKT with projected coordinates interpreted as degrees — subsetting fails silently or produces wildly incorrect results.
FIX: Check CRS and reproject to EPSG:4326 if not already geographic before converting to WKT.
CROSS-DOMAIN: Domain Scientist — affects SNAP spatial subsetting
```

```
ID: H-13
SEVERITY: HIGH
AUDITORS: Geospatial Engineer
FILE: ocean_rs/optical/processors/tss_processor.py:400-416
PROBLEM: When reference band not found, metadata populated with None geotransform/projection — processing continues silently
IMPACT: All TSS output rasters lose spatial reference (appear at 0,0 with 1-unit pixels). Unusable in GIS.
FIX: Raise RuntimeError if reference band not found. Try alternative bands (rrs_B3, rrs_B2) before failing.
CROSS-DOMAIN: None
```

```
ID: H-14
SEVERITY: HIGH
AUDITORS: Python Systems Engineer
FILE: ocean_rs/optical/processors/c2rcc_processor.py:646-698
PROBLEM: subprocess.Popen pipes buffer entire SNAP GPT stdout/stderr in memory for multi-hour runs
IMPACT: Memory spike from large SNAP output buffers during long processing jobs.
FIX: Redirect stdout/stderr to temporary log files instead of PIPE, read after completion.
CROSS-DOMAIN: None
```

```
ID: H-15
SEVERITY: HIGH
AUDITORS: Domain Scientist
FILE: ocean_rs/sar/sensors/sentinel1.py:129-189
PROBLEM: SAR preprocessing chain missing speckle filter — multiplicative speckle noise creates spurious FFT peaks
IMPACT: Unreliable wavelength estimates from noise-dominated spectral peaks, leading to incorrect depth inversions.
FIX: Add Lee Sigma speckle filter node between Calibration and Terrain-Correction in SNAP graph. Use 5x5 filter, 7x7 window, sigma=0.9.
CROSS-DOMAIN: Scientific Computing — affects FFT SNR
```

```
ID: H-16
SEVERITY: HIGH
AUDITORS: Python Systems Engineer
FILE: ocean_rs/optical/gui/handlers.py:120
PROBLEM: on_closing calls gui.stop_processing() which does not exist — crashes on exit during processing
IMPACT: AttributeError prevents clean shutdown. Processing thread never stopped, window stays open.
FIX: Import and call stop_processing(gui) from processing_controller module.
CROSS-DOMAIN: None
```

```
ID: H-17
SEVERITY: HIGH
AUDITORS: Python Systems Engineer
FILE: ocean_rs/optical/gui/processing_controller.py:31-33
PROBLEM: processing_active set True before validation — early returns leave it stuck True permanently
IMPACT: After any validation failure, Start button permanently disabled. GUI must be restarted.
FIX: Move processing_active=True to after all validation, before thread start.
CROSS-DOMAIN: None
```

```
ID: H-18
SEVERITY: HIGH
AUDITORS: Python Systems Engineer
FILE: ocean_rs/optical/core/unified_processor.py:549
PROBLEM: get_processing_status() computes time.time() - self.start_time but start_time is None before process_batch()
IMPACT: TypeError crash in GUI update loop (called every 2 seconds by start_gui_updates).
FIX: Guard: elapsed_time = time.time() - self.start_time if self.start_time else 0.0
CROSS-DOMAIN: None
```

```
ID: H-19
SEVERITY: HIGH
AUDITORS: Scientific Computing
FILE: ocean_rs/sar/bathymetry/fft_extractor.py:46-48
PROBLEM: tile_px and step_px can be zero for low-resolution data (pixel_m > tile_size_m) — causes infinite loop or ValueError
IMPACT: Pipeline hangs or crashes on low-resolution SAR data.
FIX: Add: tile_px = max(int(tile_size_m / pixel_m), 8); step_px = max(int(tile_px * (1 - overlap)), 1).
CROSS-DOMAIN: None
```

```
ID: H-20
SEVERITY: HIGH
AUDITORS: Scientific Computing
FILE: ocean_rs/sar/bathymetry/depth_inversion.py:36-37
PROBLEM: No validation on wave_period or wavelengths being positive before division — zero values produce inf
IMPACT: Entire depth inversion corrupted if wave_period=0 (from failed WW3 query returning default 0).
FIX: Add: if wave_period <= 0: raise ValueError(f"Wave period must be positive, got {wave_period}")
CROSS-DOMAIN: None
```

---

### MEDIUM (continued with additional findings)

```
ID: M-23
SEVERITY: MEDIUM
AUDITORS: Domain Scientist
FILE: ocean_rs/optical/processors/tss_processor.py:934-935
PROBLEM: QAA ratio uses below-water rrs for all components, but standard QAA v6.0 uses above-water Rrs — needs verification against Jiang R code
IMPACT: If Jiang's R code uses Rrs (not rrs), absorption estimates change by 5-10%.
FIX: Verify against original R code. If Rrs intended, change to vp[443], vp[490], vp[560], vp[665].
CROSS-DOMAIN: None
```

```
ID: M-24
SEVERITY: MEDIUM
AUDITORS: Geospatial Engineer
FILE: ocean_rs/sar/core/bathymetry_pipeline.py:183-189
PROBLEM: pixel_size_y sign not validated — positive value produces vertically flipped output GeoTIFF
IMPACT: All GIS overlays incorrect if OceanImage stores pixel_size_y as positive.
FIX: Validate: pixel_size_y = result.geo.pixel_size_y if result.geo.pixel_size_y < 0 else -result.geo.pixel_size_y
CROSS-DOMAIN: None
```

```
ID: M-25
SEVERITY: MEDIUM
AUDITORS: Geospatial Engineer
FILE: ocean_rs/optical/processors/tsm_chl_calculator.py:122-127
PROBLEM: Writes GeoTIFF with .img extension into BEAM-DIMAP .data folder — format mismatch
IMPACT: SNAP won't recognize files as valid BEAM-DIMAP bands. OceanRS reads them fine via GDAL.
FIX: Use ENVI driver for .img files, or save to separate location with .tif extension.
CROSS-DOMAIN: None
```

```
ID: M-26
SEVERITY: MEDIUM
AUDITORS: Domain Scientist
FILE: ocean_rs/sar/sensors/sentinel1.py:174
PROBLEM: Terrain Correction pixelSpacingInMeter=10.0 hardcoded — only appropriate for IW mode (EW=40m, SM=5m)
IMPACT: EW data oversampled 4x (creates false spatial patterns in FFT). SM data undersampled.
FIX: Make pixelSpacingInMeter configurable. Validate against sensor mode.
CROSS-DOMAIN: None
```

```
ID: M-27
SEVERITY: MEDIUM
AUDITORS: Python Systems Engineer
FILE: ocean_rs/optical/gui/handlers.py:133-139
PROBLEM: _check_and_destroy polls with root.after(100ms) but has no timeout — window never closes if processing thread hangs
IMPACT: Application cannot be exited if processing thread is stuck (e.g., waiting for SNAP).
FIX: Add max attempts counter (e.g., 100 = 10s), force destroy after timeout.
CROSS-DOMAIN: None
```

---

### MEDIUM

```
ID: M-1
SEVERITY: MEDIUM
AUDITORS: Scientific Computing
FILE: ocean_rs/optical/processors/tss_processor.py:948
PROBLEM: safe mask for bbp only checks denom_u, not a560 validity — NaN a560 propagates into "safe" pixels
IMPACT: Some Type I pixels have valid denom_u but NaN a560, producing NaN TSS with no diagnostic distinguishing failure causes.
FIX: Add a560_valid = ~np.isnan(a560) to safe mask. Log pixel counts lost at each guard stage.
CROSS-DOMAIN: None
```

```
ID: M-2
SEVERITY: MEDIUM
AUDITORS: Scientific Computing, ML/Statistical Modeler
FILE: ocean_rs/optical/processors/water_quality_processor.py:93-99
PROBLEM: Kd clipped to [0,20] AFTER Secchi depth, euphotic depth, and clarity index computed from unclipped values
IMPACT: Negative Kd → negative Secchi (clips to 0) but saved Kd shows 0 — internal inconsistency. Products not mutually consistent.
FIX: Move Kd clipping (lines 108-111) BEFORE Secchi/euphotic/clarity computation (lines 93-99).
CROSS-DOMAIN: None
```

```
ID: M-3
SEVERITY: MEDIUM
AUDITORS: Scientific Computing
FILE: ocean_rs/optical/processors/water_quality_processor.py:89
PROBLEM: No NaN filtering on absorption/backscattering inputs — NaN propagates silently through all water clarity products
IMPACT: Upstream TSS failures silently produce NaN across all derived water clarity products with no diagnostic.
FIX: Add valid_mask = np.isfinite(absorption) & np.isfinite(backscattering) at start. Initialize outputs as NaN. Compute only on valid pixels. Log valid pixel count.
CROSS-DOMAIN: None
```

```
ID: M-4
SEVERITY: MEDIUM
AUDITORS: ML/Statistical Modeler
FILE: ocean_rs/optical/processors/tss_processor.py:783-785
PROBLEM: TSS clipping to (0.01, 10000) is silent — no count of clipped pixels, no quality flag
IMPACT: Negative backscattering artifacts clip to 0.01, hiding algorithm failures. Users cannot assess data quality.
FIX: Log number of pixels clipped at each bound. Add tss_quality_flag band marking clipped pixels.
CROSS-DOMAIN: None
```

```
ID: M-5
SEVERITY: MEDIUM
AUDITORS: ML/Statistical Modeler
FILE: ocean_rs/sar/bathymetry/compositor.py:49
PROBLEM: MAD-to-sigma factor 1.4826 assumes Gaussian residuals; coastal bathymetry residuals are often skewed (tidal bias, currents)
IMPACT: Uncertainty estimate biased for non-normal distributions. Underestimates in heavy-tailed, overestimates in light-tailed.
FIX: Document Gaussian assumption. Consider unscaled MAD or IQR/1.349 as alternative. Add normality test for N>10.
CROSS-DOMAIN: None
```

```
ID: M-6
SEVERITY: MEDIUM
AUDITORS: Scientific Computing
FILE: ocean_rs/sar/bathymetry/depth_inversion.py:54
PROBLEM: Newton-Raphson oscillation for deep-water tiles (kh>10) where dispersion function is very flat
IMPACT: Slow convergence consuming iterations without meaningful depth resolution. Max_iterations=50 catches it but wastes compute.
FIX: Flag deep_water = kh > 10 and set those depths to NaN directly, skipping iteration.
CROSS-DOMAIN: None
```

```
ID: M-7
SEVERITY: MEDIUM
AUDITORS: Scientific Computing
FILE: ocean_rs/sar/bathymetry/fft_extractor.py:88-93
PROBLEM: NaN-replaced tiles (mean-fill for coastline pixels) bias FFT peak detection when NaN concentrated in one region
IMPACT: Wavelength estimates near coastlines may be biased. Confidence metric doesn't reflect reduced effective tile size.
FIX: Scale confidence by valid_frac: confidence *= valid_frac. Reduces weight of partially valid tiles in compositor.
CROSS-DOMAIN: None
```

```
ID: M-8
SEVERITY: MEDIUM
AUDITORS: Scientific Computing
FILE: ocean_rs/sar/bathymetry/fft_extractor.py:107
PROBLEM: max(masked_power) == 0 uses exact float comparison — very low variance tiles with noise peaks pass the check
IMPACT: Noise peaks from uniform backscatter tiles could produce arbitrary wavelength estimates.
FIX: Replace == 0 with < 1e-20 threshold.
CROSS-DOMAIN: None
```

```
ID: M-9
SEVERITY: MEDIUM
AUDITORS: ML/Statistical Modeler
FILE: ocean_rs/sar/bathymetry/fft_extractor.py:119-121
PROBLEM: FFT confidence = min(1.0, snr/10.0) is an uncalibrated heuristic with no statistical interpretation
IMPACT: Threshold 0.3 may accept false peaks or reject valid ones depending on sea state.
FIX: Document as heuristic. Consider spectral contrast ratio (peak/second-peak) as more discriminating metric. Make mapping configurable.
CROSS-DOMAIN: None
```

```
ID: M-10
SEVERITY: MEDIUM
AUDITORS: Domain Scientist
FILE: ocean_rs/optical/processors/tss_processor.py:1015
PROBLEM: Log says "NDWI water mask" but code only uses NIR threshold (< 0.03). NDWI component documented in CLAUDE.md is missing from implementation.
IMPACT: Misleading diagnostics. Water mask less restrictive than documented — may miss bright land surfaces NDWI would catch.
FIX: Either add NDWI > 0 condition, or update docs/logs to "NIR-only mask". NIR-only is more robust for turbid water (recommend option b).
CROSS-DOMAIN: None
```

```
ID: M-11
SEVERITY: MEDIUM
AUDITORS: Domain Scientist
FILE: ocean_rs/optical/processors/tss_processor.py:162-200
PROBLEM: NIR threshold 0.03 too conservative for Type IV (extremely turbid) water where NIR Rrs can exceed 0.03
IMPACT: Valid turbid water pixels masked as land — data gaps in scientifically interesting high-sediment areas.
FIX: (a) Increase threshold to 0.05-0.08 for C2X-COMPLEX preset, (b) make adaptive per NN preset, or (c) log warning when high % of pixels masked.
CROSS-DOMAIN: None
```

```
ID: M-12
SEVERITY: MEDIUM
AUDITORS: Domain Scientist
FILE: ocean_rs/sar/sensors/sentinel1.py:163-166
PROBLEM: Calibration to Sigma0 includes incidence angle normalization that suppresses wave modulation signal
IMPACT: Reduced SNR of ocean wave patterns in FFT, more tiles below confidence threshold.
FIX: Consider Beta0 option for FFT analysis (preserves intensity modulation). Document trade-off.
CROSS-DOMAIN: None
```

```
ID: M-13
SEVERITY: MEDIUM
AUDITORS: Domain Scientist
FILE: ocean_rs/sar/bathymetry/fft_extractor.py:18-22
PROBLEM: Default 512m tile can only fit ~1 cycle of max-wavelength (600m) swell — Nyquist requires >= 2 cycles
IMPACT: Long-period swells (T>15s, L>350m) undetectable, biasing toward short-period waves and limiting depth range.
FIX: Increase default to 1024-2048m or validate tile_size_m >= 2 * max_wavelength_m. Add warning if violated.
CROSS-DOMAIN: None
```

```
ID: M-14
SEVERITY: MEDIUM
AUDITORS: Domain Scientist
FILE: ocean_rs/sar/sensors/sentinel1.py:191-233
PROBLEM: Acquisition datetime not extracted from S1 product metadata — image.metadata['datetime'] always empty
IMPACT: WW3 ERDDAP query always fails (requires valid datetime), forcing fallback to manual wave period.
FIX: Parse datetime from S1 filename (S1A_IW_..._YYYYMMDDTHHMMSS_...) and store in metadata['datetime'].
CROSS-DOMAIN: Python Systems Engineer — missing data extraction
```

```
ID: M-15
SEVERITY: MEDIUM
AUDITORS: ML/Statistical Modeler
FILE: ocean_rs/optical/processors/water_quality_processor.py:92-96
PROBLEM: clarity_index = 1/(1+kd) and relative_turbidity = bb*20 are custom heuristics saved alongside scientifically referenced products without differentiation
IMPACT: Users may treat heuristic indices as scientifically validated products.
FIX: Prefix heuristic products with "custom_" or "proxy_" in output filenames. Document as non-published formulas.
CROSS-DOMAIN: Domain Scientist — scientific transparency
```

```
ID: M-16
SEVERITY: MEDIUM
AUDITORS: ML/Statistical Modeler
FILE: ocean_rs/optical/processors/visualization_processor.py:756-843
PROBLEM: Spectral indices saved without output range clipping — normalized indices should be [-1,1], non-normalized are unbounded
IMPACT: Edge cases produce out-of-range values. GIS color stretching/classification issues.
FIX: Clip normalized indices to [-1,1]. Add physical range limits for non-normalized indices. Log warnings for out-of-range.
CROSS-DOMAIN: None
```

```
ID: M-17
SEVERITY: MEDIUM
AUDITORS: Python Systems Engineer
FILE: ocean_rs/optical/processors/c2rcc_processor.py:201-209
PROBLEM: SNAP graph XML written to CWD with hardcoded name — fails on read-only CWD, race condition on concurrent runs
IMPACT: Pipeline fails with obscure OS error if CWD is not writable (OneDrive, network drive).
FIX: Write to tempfile.mkdtemp() or output folder with unique name (include PID or UUID).
CROSS-DOMAIN: None
```

```
ID: M-18
SEVERITY: MEDIUM
AUDITORS: Python Systems Engineer
FILE: ocean_rs/optical/__init__.py:67
PROBLEM: Top-level import of GUI module triggers tkinter import even in CLI/library mode
IMPACT: Package fails on headless Linux servers with TclError: no display name. Even --help fails.
FIX: Make GUI import lazy — remove from __init__.py, import only in main.py when GUI mode selected.
CROSS-DOMAIN: None
```

```
ID: M-19
SEVERITY: MEDIUM
AUDITORS: Geospatial Engineer
FILE: ocean_rs/optical/processors/visualization_processor.py:912-916, 977-981
PROBLEM: Direct GDAL writes bypass RasterIO — no warning when geotransform/projection missing from metadata
IMPACT: Silently produces non-georeferenced GeoTIFFs if metadata dict incomplete.
FIX: Log warning when geotransform or projection missing. Consider raising error.
CROSS-DOMAIN: None
```

```
ID: M-20
SEVERITY: MEDIUM
AUDITORS: Geospatial Engineer
FILE: ocean_rs/shared/raster_io.py:74
PROBLEM: Default nodata=-9999 fabricated when source raster has no nodata set — not applied to data pixels on read
IMPACT: Metadata claims nodata=-9999 but data pixels with that value are not masked to NaN. Inconsistency between header and pixel values.
FIX: Return nodata as None when source has no nodata. Let callers decide.
CROSS-DOMAIN: Scientific Computing — may silently affect statistics
```

```
ID: M-21
SEVERITY: MEDIUM
AUDITORS: Python Systems Engineer
FILE: ocean_rs/optical/processors/visualization_processor.py:371
PROBLEM: data.astype(np.float32) creates unnecessary copy when data already float32 (from RasterIO)
IMPACT: Doubles band memory during loading.
FIX: Use np.asarray(data, dtype=np.float32) which returns same array if already correct dtype.
CROSS-DOMAIN: None
```

```
ID: M-22
SEVERITY: MEDIUM
AUDITORS: Python Systems Engineer
FILE: ocean_rs/shared/geometry_utils.py:175-215
PROBLEM: OGR datasource not explicitly closed in shapefile OGR fallback path
IMPACT: File handle leak on shapefile until garbage collection.
FIX: Add datasource = None in finally block after OGR section.
CROSS-DOMAIN: None
```

---

### LOW

```
ID: L-1
SEVERITY: LOW
AUDITORS: Scientific Computing
FILE: ocean_rs/shared/math_utils.py (entire module)
PROBLEM: SafeMathNumPy is defined but never imported or used anywhere in the codebase — dead code
IMPACT: Misleading architecture — developers may think safe math is being used when it is not.
FIX: Either integrate into actual computation code, or remove entirely.
CROSS-DOMAIN: Python Systems Engineer — dead code
```

```
ID: L-2
SEVERITY: LOW
AUDITORS: ML/Statistical Modeler
FILE: ocean_rs/shared/raster_io.py:240
PROBLEM: calculate_statistics does not exclude np.inf/-np.inf values
IMPACT: Statistics report inf as max or nan as mean if infinite values present.
FIX: Add & (~np.isinf(data)) to valid data filter.
CROSS-DOMAIN: None
```

```
ID: L-3
SEVERITY: LOW
AUDITORS: Scientific Computing
FILE: ocean_rs/optical/processors/tss_processor.py:609
PROBLEM: Division by total_count without zero guard in success rate logging
IMPACT: ZeroDivisionError if no products generated.
FIX: Guard with if total_count > 0.
CROSS-DOMAIN: None
```

```
ID: L-4
SEVERITY: LOW
AUDITORS: Scientific Computing
FILE: ocean_rs/optical/processors/tsm_chl_calculator.py:101
PROBLEM: CHL values not clamped to physical range — np.power(apig, 1.04)*21 can exceed 2500 mg/m3
IMPACT: Unrealistic CHL values in output products (max realistic ~1000 mg/m3).
FIX: Add np.clip(chl_values, 0, 1000) after calculation.
CROSS-DOMAIN: None
```

```
ID: L-5
SEVERITY: LOW
AUDITORS: Scientific Computing
FILE: ocean_rs/sar/bathymetry/depth_inversion.py:93
PROBLEM: iteration variable undefined if wavelengths array is empty — NameError
IMPACT: Crash on empty input.
FIX: Initialize iteration = 0 before the loop.
CROSS-DOMAIN: None
```

```
ID: L-6
SEVERITY: LOW
AUDITORS: Scientific Computing
FILE: ocean_rs/sar/bathymetry/compositor.py:81-90
PROBLEM: _weighted_median uses per-point Python loop — O(n_points * n_obs * log(n_obs))
IMPACT: Performance bottleneck for large datasets.
FIX: Vectorize using np.apply_along_axis or sorted index arrays.
CROSS-DOMAIN: None
```

```
ID: L-7
SEVERITY: LOW
AUDITORS: ML/Statistical Modeler
FILE: ocean_rs/sar/bathymetry/compositor.py:58
PROBLEM: Composite wave_period is unweighted mean across scenes — poor estimates contribute equally
IMPACT: Inaccurate composite metadata (does not affect depth values).
FIX: Use weighted mean by valid tile count, or median.
CROSS-DOMAIN: None
```

```
ID: L-8
SEVERITY: LOW
AUDITORS: ML/Statistical Modeler
FILE: ocean_rs/optical/processors/water_quality_processor.py:276
PROBLEM: HAB statistics count includes land pixels (hab_probability initialized to 0, not NaN)
IMPACT: Risk percentages diluted by land pixels in logging only.
FIX: Use water/valid mask for statistics computation.
CROSS-DOMAIN: None
```

```
ID: L-9
SEVERITY: LOW
AUDITORS: Domain Scientist
FILE: ocean_rs/sar/bathymetry/compositor.py:37-38
PROBLEM: np.array([r.depth for r in results]) assumes same-shaped arrays — different scenes have different tile grids
IMPACT: Multi-scene compositing crashes with shape mismatch.
FIX: Interpolate all results to common grid using tile center coordinates.
CROSS-DOMAIN: Python Systems Engineer — runtime crash
```

```
ID: L-10
SEVERITY: LOW
AUDITORS: Domain Scientist
FILE: ocean_rs/sar/bathymetry/wave_period.py:35-36
PROBLEM: ERDDAP dataset ID NWW3_Global_Best may be renamed/restructured over time
IMPACT: WW3 queries silently fail, always falling back to manual period.
FIX: Add startup validation querying ERDDAP catalog. Consider fallback data sources.
CROSS-DOMAIN: None
```

```
ID: L-11
SEVERITY: LOW
AUDITORS: Domain Scientist
FILE: ocean_rs/optical/processors/water_quality_processor.py:92-93
PROBLEM: Secchi depth constant 1.7 is the in-situ disc constant (Tyler 1968) — satellite-appropriate value is ~1.0 (Doron 2011)
IMPACT: Secchi depth overestimated by ~70% compared to satellite-appropriate formulations.
FIX: Document the specific formulation. Consider offering Lee et al. (2015) satellite model as alternative.
CROSS-DOMAIN: None
```

```
ID: L-12
SEVERITY: LOW
AUDITORS: Domain Scientist
FILE: ocean_rs/optical/processors/water_quality_processor.py:102-104
PROBLEM: Fixed backscatter ratio 0.0183 (Petzold 1972 coastal average) — varies 0.005-0.03 by water type
IMPACT: Beam attenuation biased: overestimated in open ocean, underestimated in mineral-turbid water.
FIX: Make configurable, or derive from spectral slope (Boss et al. 2001). Document assumption.
CROSS-DOMAIN: None
```

```
ID: L-13
SEVERITY: LOW
AUDITORS: Domain Scientist
FILE: ocean_rs/optical/processors/tss_processor.py:308-363
PROBLEM: rhow-to-Rrs conversion relies entirely on filename prefix — fragile if SNAP naming conventions change
IMPACT: If detection fails, rhow used as Rrs → TSS overestimated by factor of pi (~3.14).
FIX: Add range check: typical rhow 0-0.25, Rrs 0-0.08. Apply conversion based on data range regardless of filename.
CROSS-DOMAIN: None
```

```
ID: L-14
SEVERITY: LOW
AUDITORS: Geospatial Engineer
FILE: ocean_rs/shared/raster_io.py:200-213
PROBLEM: PROCESSING_DATE uses local time (datetime.now()) instead of UTC
IMPACT: Inconsistent timestamps across time zones. Violates CLAUDE.md UTC rule.
FIX: Use datetime.now(timezone.utc).isoformat().
CROSS-DOMAIN: None
```

```
ID: L-15
SEVERITY: LOW
AUDITORS: Geospatial Engineer
FILE: ocean_rs/shared/raster_io.py:62
PROBLEM: ReadAsArray forces float32 — silent precision loss for float64 source data
IMPACT: Acceptable for S2 pipeline (float32 adequate) but undocumented design choice.
FIX: Document in docstring. Optionally add force_float32=True parameter.
CROSS-DOMAIN: None
```

```
ID: L-16
SEVERITY: LOW
AUDITORS: Geospatial Engineer
FILE: ocean_rs/shared/raster_io.py:66-67
PROBLEM: NoData comparison uses exact float equality — fails for non-exactly-representable nodata values
IMPACT: Unlikely for -9999 (exactly representable) but could affect edge cases.
FIX: Use np.isclose() or document that only exactly representable nodata supported.
CROSS-DOMAIN: None
```

```
ID: L-17
SEVERITY: LOW
AUDITORS: Geospatial Engineer
FILE: ocean_rs/optical/processors/visualization_processor.py:906, 965
PROBLEM: Direct GDAL writes omit PREDICTOR in GTiff options (PREDICTOR=2 for uint8, PREDICTOR=3 for float32)
IMPACT: Output files 2-4x larger than necessary. No data corruption.
FIX: Add PREDICTOR=2 for RGB (uint8), PREDICTOR=3 for indices (float32).
CROSS-DOMAIN: None
```

```
ID: L-18
SEVERITY: LOW
AUDITORS: Geospatial Engineer
FILE: ocean_rs/optical/processors/visualization_processor.py:887-951
PROBLEM: RGB composites use 0 for both "no data" and "black pixels" — GIS cannot distinguish
IMPACT: Missing regions appear black instead of transparent in mosaics.
FIX: Set NoDataValue(0) on each band, or use 255 as nodata with valid range [1,254].
CROSS-DOMAIN: None
```

```
ID: L-19
SEVERITY: LOW
AUDITORS: Python Systems Engineer
FILE: ocean_rs/optical/processors/logging_utils.py:40, 256-267
PROBLEM: ColoredFormatter._in_box is class-level mutable shared across all instances without synchronization
IMPACT: If optical+SAR run in same process, log formatting corrupted.
FIX: Move _in_box to StepTracker instance. Pass to formatter via thread-local or filter.
CROSS-DOMAIN: None
```

```
ID: L-20
SEVERITY: LOW
AUDITORS: Python Systems Engineer
FILE: ocean_rs/optical/processors/tss_processor.py:101-118
PROBLEM: Config mutated in TSSProcessor.__init__ — shared config modification persists across reuses
IMPACT: Low in current flow (new config each run) but fragile pattern.
FIX: Remove redundant mutation at line 118, or use copy.deepcopy(config) in constructor.
CROSS-DOMAIN: None
```

```
ID: L-21
SEVERITY: LOW
AUDITORS: Python Systems Engineer
FILE: ocean_rs/shared/proj_fix.py:176-197
PROBLEM: _proj_configured global checked/set without thread safety
IMPACT: Benign race — configure_proj_environment() is idempotent.
FIX: Accept benign race or add threading.Lock.
CROSS-DOMAIN: None
```

```
ID: L-22
SEVERITY: LOW
AUDITORS: Python Systems Engineer
FILE: ocean_rs/shared/raster_io.py:14
PROBLEM: os.environ['CPL_LOG'] set at import time — side effect of importing module
IMPACT: All GDAL-using code in process has CPL logging suppressed silently.
FIX: Move to configuration function or document clearly.
CROSS-DOMAIN: None
```

```
ID: L-23
SEVERITY: LOW
AUDITORS: Python Systems Engineer
FILE: ocean_rs/optical/processors/tss_processor.py:402
PROBLEM: str.replace('.dim', '.data') replaces ALL occurrences in path, not just extension
IMPACT: Incorrect path if ".dim" appears in directory names (e.g., C:/dimension_project/).
FIX: Use pathlib.Path(path).with_suffix('.data').
CROSS-DOMAIN: None
```

```
ID: L-24
SEVERITY: LOW
AUDITORS: Python Systems Engineer
FILE: ocean_rs/shared/memory_manager.py:50-60
PROBLEM: psutil.Process() created on every call to monitor_memory()
IMPACT: Negligible — designed to be lightweight.
FIX: Cache as class-level variable.
CROSS-DOMAIN: None
```

```
ID: L-25
SEVERITY: LOW
AUDITORS: Python Systems Engineer
FILE: ocean_rs/shared/logging_utils.py:200-204
PROBLEM: _batch_cpu_samples and _batch_ram_samples grow unboundedly across batch
IMPACT: Negligible for typical batches, could reach several MB for 1000+ scenes.
FIX: Use rolling window or running statistics instead of raw sample storage.
CROSS-DOMAIN: None
```

```
ID: L-26
SEVERITY: LOW
AUDITORS: Python Systems Engineer
FILE: ocean_rs/sar/download/batch_downloader.py:78
PROBLEM: Accessing private _asf_result attribute with object type annotation
IMPACT: Coupling to ASF library internals. Type annotation lacks specificity.
FIX: Type as Optional[Any] with comment, or create wrapper method in SceneMetadata.
CROSS-DOMAIN: None
```

```
ID: L-27
SEVERITY: LOW
AUDITORS: Python Systems Engineer
FILE: ocean_rs/optical/processors/visualization_processor.py:1027-1046
PROBLEM: Visualization summary written to output root, not per-scene folder
IMPACT: Summaries accumulate in root instead of being organized per scene.
FIX: Write to OutputStructure.get_scene_folder().
CROSS-DOMAIN: None
```

```
ID: L-28
SEVERITY: LOW
AUDITORS: Domain Scientist
FILE: ocean_rs/optical/processors/visualization_processor.py:831-835
PROBLEM: CDOM index required_bands order [443,560] inconsistent with formula comment "B3/B1"
IMPACT: No computational error — result correct. Confusing for maintenance.
FIX: Reorder required_bands to [560,443] to match formula, or fix variable names.
CROSS-DOMAIN: None
```

```
ID: L-29
SEVERITY: LOW
AUDITORS: Domain Scientist
FILE: ocean_rs/optical/processors/tss_processor.py:916-917
PROBLEM: Variable rrs665_orig misleadingly named — actually holds above-water Rrs(665), not below-water rrs
IMPACT: No computational error (polynomial derived from Rrs). Future maintenance confusion.
FIX: Rename to Rrs665 or rrs665_above_water.
CROSS-DOMAIN: None
```

```
ID: L-30
SEVERITY: LOW
AUDITORS: Domain Scientist
FILE: ocean_rs/optical/processors/water_quality_processor.py:211
PROBLEM: MCI detection requires band 865nm but only uses 665, 705, 740 — unnecessary exclusion
IMPACT: MCI computation prevented when 865nm band unavailable despite not being needed.
FIX: Remove 865 from the required bands check.
CROSS-DOMAIN: None
```

```
ID: L-31
SEVERITY: LOW
AUDITORS: Domain Scientist
FILE: ocean_rs/optical/processors/c2rcc_processor.py:822
PROBLEM: Dictionary key typo 'ready_for_tss_tss' should be 'ready_for_tss'
IMPACT: Stats reporting only — no processing impact.
FIX: Fix the key name.
CROSS-DOMAIN: None
```

```
ID: L-32
SEVERITY: LOW
AUDITORS: Scientific Computing
FILE: ocean_rs/shared/math_utils.py:113-118
PROBLEM: safe_power allows negative bases with non-integer exponents — produces undetected NaN
IMPACT: np.power(-0.5, 2.3) silently produces NaN, not replaced by default_value.
FIX: For non-integer exponents, add valid_mask = valid_mask & (base > 0).
CROSS-DOMAIN: None
```

```
ID: L-33
SEVERITY: LOW
AUDITORS: Python Systems Engineer
FILE: ocean_rs/optical/main.py:247-248
PROBLEM: Python version check uses 3.6 but CLAUDE.md says Python 3.8+
IMPACT: Users with 3.6/3.7 pass version check but hit syntax errors later.
FIX: Change to sys.version_info < (3, 8).
CROSS-DOMAIN: None
```

```
ID: L-34
SEVERITY: LOW
AUDITORS: ML/Statistical Modeler
FILE: ocean_rs/optical/config/water_quality_config.py:27-28
PROBLEM: HAB config thresholds (biomass 20.0, extreme 100.0) defined but never used in detection algorithm
IMPACT: Users may change these values expecting them to affect detection — they have no effect.
FIX: Wire thresholds into detect_harmful_algal_blooms, or remove from config with documentation.
CROSS-DOMAIN: None
```

---

## Cross-Cutting Concerns

Issues that span multiple auditor domains, requiring coordinated fixes:

### XC-1: SAR Bathymetry Pipeline Chain is Fundamentally Flawed

**Auditors:** Domain Scientist, Geospatial Engineer, Scientific Computing
**Files:** sentinel1.py, bathymetry_pipeline.py, depth_inversion.py, fft_extractor.py
**Findings:** H-7, H-8, H-9, M-6, M-12, M-13, M-14

The SAR bathymetry pipeline has a cascade of issues that compound:
1. Terrain Correction before FFT corrupts wave spectra (H-7)
2. UTM coordinates passed to WW3 instead of WGS84 (H-8)
3. Datetime not extracted from S1 metadata, so WW3 always fails (M-14)
4. Single wave period for entire 250km scene (H-9)
5. FFT tile size too small for long swells (M-13)
6. Sigma0 normalization suppresses wave signal (M-12)

**Net effect:** The SAR bathymetry results are likely to have systematic biases from multiple compounding errors. The WW3 wave period query will always fail (coordinates + missing datetime), forcing manual period fallback. The TC resampling corrupts wave spectra. The tile size limits depth range.

**Recommendation:** Before fixing individual issues, redesign the SAR pipeline architecture:
1. Split preprocessing into pre-FFT (no TC) and post-FFT (TC for output)
2. Fix coordinate projection for WW3 queries
3. Extract datetime from S1 filenames
4. Increase default tile size to >= 2 * max_wavelength_m

---

### XC-2: Memory Management is Cosmetic

**Auditors:** Python Systems Engineer, Scientific Computing
**Files:** memory_manager.py, tss_processor.py, unified_processor.py, visualization_processor.py
**Findings:** C-1, C-2, M-21

The entire memory management strategy is based on `cleanup_variables()` which is a no-op (deletes local parameter bindings, not caller's references). Combined with unnecessary `.copy()` calls and `.astype()` copies, peak memory usage for a single S2 scene is ~14 GB when it could be ~5 GB with proper cleanup. Multi-scene batch processing will accumulate memory without release.

**Recommendation:**
1. Delete `cleanup_variables()` entirely
2. At each call site, explicitly `del` variables and call `gc.collect()`
3. Remove unnecessary `.copy()` in `_convert_rhow_to_rrs`
4. Use `np.asarray()` instead of `.astype()` when dtype already matches

---

### XC-3: Uncertainty Reporting is Unreliable Across Both Pipelines

**Auditors:** ML/Statistical Modeler, Scientific Computing, Domain Scientist
**Files:** tsm_chl_calculator.py, compositor.py, depth_inversion.py
**Findings:** C-3, C-4, H-5, H-6, M-5

Both optical and SAR pipelines have uncertainty products that misrepresent actual measurement quality:
- **Optical:** TSM/CHL uncertainty is fabricated (constant % of estimate, no propagation from C2RCC NN errors)
- **SAR:** Depth uncertainty ignores FFT confidence, uses simplified sensitivity, assumes Gaussian errors
- **SAR compositor:** MAD with N=2 underestimates, inverse-variance assumes independence

**Recommendation:**
1. Either propagate real uncertainties or clearly label as "heuristic placeholder"
2. For SAR: scale uncertainty by FFT confidence, increase minimum epsilon
3. For compositor: require N>=3 for MAD, fall back to range-based for N=2
4. Add metadata flag: `uncertainty_method: "heuristic"` vs `"propagated"`

---

### XC-4: Turbid Water Pixels Systematically Excluded

**Auditors:** ML/Statistical Modeler, Domain Scientist
**Files:** tss_processor.py
**Findings:** H-2, M-11

Two independent mechanisms exclude turbid water pixels:
1. Valid pixel mask requires ALL bands > 0 (blue bands go negative in turbid water)
2. NIR water mask threshold 0.03 is too conservative for Type IV water

**Net effect:** The very pixels that are most scientifically interesting for TSS estimation (highly turbid, Type III/IV) are most likely to be excluded. This creates a systematic clear-water bias in the output.

**Recommendation:**
1. Require positivity only for bands used by each water type
2. Make NIR threshold adaptive per NN preset (0.03 → 0.05-0.08 for C2X-COMPLEX)
3. Log coverage statistics per water type to surface the bias

---

### XC-5: SAR Preprocessing Missing Speckle Filter

**Auditors:** Domain Scientist, Scientific Computing
**Files:** sentinel1.py
**Findings:** H-7 (TC before FFT), H-15 (missing speckle filter)

The SAR preprocessing chain omits speckle filtering entirely. Without speckle filtering, multiplicative noise creates spurious peaks in the FFT power spectrum. Combined with TC resampling before FFT (which acts as a low-pass filter), the wave spectrum is both contaminated by noise AND smoothed by interpolation.

**Recommendation:** Add Lee Sigma speckle filter between Calibration and Terrain-Correction in the SNAP graph. Use 5x5 filter with 7x7 window and sigma=0.9 for coastal applications.

---

### XC-6: Optical GUI Has Multiple Crash Paths

**Auditors:** Python Systems Engineer
**Files:** handlers.py, processing_controller.py, unified_processor.py
**Findings:** H-16, H-17, H-18

Three independent crash paths in the optical GUI:
1. `on_closing` calls `gui.stop_processing()` which doesn't exist (AttributeError)
2. `processing_active` set True before validation — early returns leave it stuck True permanently
3. `get_processing_status()` crashes with TypeError when `start_time` is None (called every 2 seconds)

**Recommendation:** Fix all three — these are straightforward bugs that will crash the GUI during normal use.

---

### XC-7: Geometry/CRS Not Reprojected for SNAP

**Auditors:** Geospatial Engineer, Domain Scientist
**Files:** geometry_utils.py, bathymetry_pipeline.py
**Findings:** H-12, H-8

Two separate components pass coordinates in the wrong CRS:
1. Shapefiles in projected CRS (UTM) produce WKT with projected coordinates for SNAP geoRegion (expects WGS84)
2. UTM scene center coordinates passed to WW3 ERDDAP API (expects degrees)

**Recommendation:** Add a centralized coordinate transformation utility in `shared/geometry_utils.py` that ensures all outgoing coordinates are in the expected CRS. Use `osr.CoordinateTransformation` or `pyproj`.

---

## Fix Priority Roadmap

### Phase 1 — Critical (fix immediately)
- C-1: Remove cleanup_variables no-op
- C-2: Add explicit memory cleanup in tss_processor
- C-3: Label uncertainty as heuristic or propagate real C2RCC uncertainties
- C-4: Add minimum sample size check in compositor

### Phase 2 — High: GUI crashes (fix immediately)
- H-16: Fix on_closing AttributeError (calls non-existent method)
- H-17: Fix processing_active stuck True after validation failure
- H-18: Guard get_processing_status against None start_time

### Phase 3 — High: Scientific/numerical (fix before production use)
- H-1: Guard QAA Type I numerical operations (3 unguarded divisions)
- H-2 + M-11: Fix turbid water pixel exclusion (XC-4)
- H-5: Scale depth uncertainty by FFT confidence
- H-12: Add CRS reprojection in geometry loader
- H-13: Abort on missing reference band
- H-19: Validate tile_px/step_px > 0 in FFT extractor
- H-20: Validate wave_period > 0 before depth inversion

### Phase 4 — High: SAR pipeline redesign (fix for SAR production)
- H-7 + H-15: Split preprocessing (no TC/speckle before FFT)
- H-8 + M-14: Fix UTM→WGS84 for WW3 + extract S1 datetime
- H-9: Support per-region wave period
- H-10: Fix subprocess zombie on timeout
- M-12 + M-13 + M-26: Configurable calibration/tile-size/pixel-spacing

### Phase 5 — Medium (fix for robustness)
- M-2: Reorder Kd clipping before derived quantities
- M-3: Add NaN input filtering to water clarity
- M-4: Log clipped TSS pixel counts
- M-17: Move SNAP graph XML to temp directory
- M-18: Make GUI import lazy
- M-10: Fix water mask documentation/implementation mismatch
- M-23: Verify QAA ratio uses correct Rrs vs rrs
- M-24: Validate pixel_size_y sign
- M-25: Fix GeoTIFF/.img format mismatch
- M-27: Add timeout to _check_and_destroy polling
- H-4 + M-15: Rename HAB "probability" to "score", fix single-algorithm scaling
- H-6 + M-5: Increase compositor epsilon, document assumptions
- H-11 + H-14: Thread safety and subprocess memory for long runs

### Phase 6 — Low (fix for polish)
- L-1: Remove dead SafeMathNumPy or integrate
- L-9: Fix compositor shape mismatch for multi-scene
- L-14: Fix timestamp to UTC
- L-17: Add PREDICTOR to direct GDAL writes
- L-30: Remove unnecessary 865nm requirement from MCI
- L-31: Fix 'ready_for_tss_tss' typo
- L-33: Fix Python version check to 3.8+
- Remaining LOW items as time permits
