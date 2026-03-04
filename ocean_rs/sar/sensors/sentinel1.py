"""
Sentinel-1 sensor adapter.

Preprocesses Sentinel-1 SLC products using SNAP GPT:
    FFT path:   Apply-Orbit-File -> Thermal-Noise-Removal -> Calibration -> Speckle-Filter (NO TC)
    Georef path: Apply-Orbit-File -> Thermal-Noise-Removal -> Calibration -> Speckle-Filter -> TC
"""

import os
import re
import sys
import shutil
import logging
import subprocess
from pathlib import Path
from typing import Optional
from xml.sax.saxutils import escape as xml_escape

import numpy as np

from ..core.data_models import (
    OceanImage, SLCImage, ImageType, GeoTransform, OrbitStateVector,
)
from .base import SensorAdapter

logger = logging.getLogger('ocean_rs')


from ocean_rs.shared.raster_io import check_memory_for_array


_VALID_POLARIZATIONS = frozenset(('VV', 'VH', 'HH', 'HV'))


class Sentinel1Adapter(SensorAdapter):
    """Preprocess Sentinel-1 SLC to calibrated sigma0 via SNAP GPT."""

    @staticmethod
    def _validate_polarization(polarization: str) -> str:
        """Validate and normalize the polarization parameter.

        Args:
            polarization: Polarization string to validate.

        Returns:
            Uppercase polarization string.

        Raises:
            ValueError: If polarization is not one of VV, VH, HH, HV.
        """
        pol = polarization.upper()
        if pol not in _VALID_POLARIZATIONS:
            raise ValueError(
                f"Invalid polarization '{polarization}'. "
                f"Must be one of: {sorted(_VALID_POLARIZATIONS)}"
            )
        return pol

    def __init__(self, pixel_spacing_m: float = 10.0):
        """Initialize Sentinel-1 adapter.

        Args:
            pixel_spacing_m: Output pixel spacing in meters.
                Default 10.0m for IW mode. EW=40m, SM=5m.
        """
        self.pixel_spacing_m = pixel_spacing_m

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
                   snap_gpt_path: Optional[str] = None,
                   polarization: str = "VV") -> OceanImage:
        """Run SNAP GPT preprocessing chain on Sentinel-1 SLC (no Terrain Correction).

        H-7: This method produces output suitable for FFT analysis.
        Terrain Correction is excluded because it resamples the image grid,
        corrupting the spatial frequency content needed for swell extraction.

        Processing chain:
            1. Apply-Orbit-File
            2. Thermal-Noise-Removal
            3. Calibration (to Sigma0)
            4. Lee Sigma Speckle Filter

        Args:
            input_path: Path to Sentinel-1 product (.SAFE or .zip).
            output_dir: Directory for intermediate outputs.
            snap_gpt_path: Optional explicit path to SNAP GPT executable.
            polarization: Polarization to calibrate (VV, VH, HH, HV).
        """
        polarization = self._validate_polarization(polarization)
        return self._run_snap_graph(
            input_path, output_dir, snap_gpt_path, polarization,
            include_tc=False, suffix="_sigma0_fft"
        )

    def preprocess_for_georef(self, input_path: Path, output_dir: Path,
                               snap_gpt_path: Optional[str] = None,
                               polarization: str = "VV") -> OceanImage:
        """Run full SNAP GPT preprocessing including Terrain Correction.

        H-7: This method produces georeferenced output suitable for final
        map overlay and export, but NOT for FFT analysis.

        Processing chain:
            1. Apply-Orbit-File
            2. Thermal-Noise-Removal
            3. Calibration (to Sigma0)
            4. Lee Sigma Speckle Filter
            5. Terrain-Correction (Range-Doppler, to UTM)

        Args:
            input_path: Path to Sentinel-1 product (.SAFE or .zip).
            output_dir: Directory for intermediate outputs.
            snap_gpt_path: Optional explicit path to SNAP GPT executable.
            polarization: Polarization to calibrate (VV, VH, HH, HV).
        """
        polarization = self._validate_polarization(polarization)
        return self._run_snap_graph(
            input_path, output_dir, snap_gpt_path, polarization,
            include_tc=True, suffix="_sigma0"
        )

    def _run_snap_graph(self, input_path: Path, output_dir: Path,
                        snap_gpt_path: Optional[str],
                        polarization: str,
                        include_tc: bool,
                        suffix: str) -> OceanImage:
        """Run a SNAP GPT graph and load the result."""
        gpt = self._find_gpt(snap_gpt_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        scene_name = input_path.stem.replace('.SAFE', '')
        output_file = output_dir / f"{scene_name}{suffix}.dim"

        if output_file.exists():
            logger.info(f"Preprocessed file exists, loading: {output_file.name}")
            image = self._load_snap_output(output_file, polarization)
            # M-14: Parse datetime from filename
            image.metadata['datetime'] = self._parse_datetime(scene_name)
            return image

        graph_xml = self._create_processing_graph(
            str(input_path), str(output_file),
            polarization=polarization, include_tc=include_tc
        )

        graph_path = output_dir / f"{scene_name}_graph.xml"
        try:
            with open(graph_path, 'w') as f:
                f.write(graph_xml)
        except OSError as e:
            raise RuntimeError(f"Failed to write SNAP graph to '{graph_path}': {e}") from e

        logger.info(f"Running SNAP GPT preprocessing: {scene_name}")
        cmd = [gpt, str(graph_path)]

        # H-10: Use Popen with explicit kill on timeout to prevent zombie processes
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        try:
            stdout, stderr = process.communicate(timeout=7200)
            if process.returncode != 0:
                raise RuntimeError(
                    f"SNAP GPT failed (exit {process.returncode}):\n"
                    f"{stderr[-500:]}"
                )
            logger.info(f"SNAP GPT completed: {scene_name}")
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise RuntimeError(f"SNAP GPT timed out after 2 hours: {scene_name}")

        graph_path.unlink(missing_ok=True)

        image = self._load_snap_output(output_file, polarization)
        # M-14: Parse datetime from filename
        image.metadata['datetime'] = self._parse_datetime(scene_name)
        return image

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
                                  output_path: str,
                                  polarization: str = "VV",
                                  include_tc: bool = False) -> str:
        """Create SNAP GPT XML graph for S1 preprocessing.

        Args:
            input_path: Path to input Sentinel-1 product.
            output_path: Path for BEAM-DIMAP output.
            polarization: Polarization band to calibrate (VV, VH, HH, HV).
            include_tc: If True, append Terrain-Correction node.
        """
        # H-15: Added Lee Sigma speckle filter between Calibration and TC/Write
        # M-12: Note: Sigma0 normalizes incidence angle, which may suppress wave
        # modulation. Beta0 preserves intensity modulation but has range-dependent bias.

        # Build the core chain (always present)
        xml = f"""<graph id="S1-Preprocessing">
  <version>1.0</version>
  <node id="Read">
    <operator>Read</operator>
    <sources/>
    <parameters>
      <file>{xml_escape(str(input_path))}</file>
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
      <selectedPolarisations>{xml_escape(str(polarization))}</selectedPolarisations>
    </parameters>
  </node>
  <node id="Speckle-Filter">
    <operator>Speckle-Filter</operator>
    <sources>
      <sourceProduct refid="Calibration"/>
    </sources>
    <parameters>
      <filter>Lee Sigma</filter>
      <filterSizeX>7</filterSizeX>
      <filterSizeY>7</filterSizeY>
      <windowSize>7x7</windowSize>
      <sigmaStr>0.9</sigmaStr>
    </parameters>
  </node>"""

        if include_tc:
            # M-26: Configurable pixel spacing (default from self.pixel_spacing_m)
            xml += f"""
  <node id="Terrain-Correction">
    <operator>Terrain-Correction</operator>
    <sources>
      <sourceProduct refid="Speckle-Filter"/>
    </sources>
    <parameters>
      <demName>SRTM 1Sec HGT</demName>
      <pixelSpacingInMeter>{self.pixel_spacing_m}</pixelSpacingInMeter>
      <mapProjection>AUTO:42001</mapProjection>
    </parameters>
  </node>
  <node id="Write">
    <operator>Write</operator>
    <sources>
      <sourceProduct refid="Terrain-Correction"/>
    </sources>
    <parameters>
      <file>{xml_escape(str(output_path))}</file>
      <formatName>BEAM-DIMAP</formatName>
    </parameters>
  </node>
</graph>"""
        else:
            xml += f"""
  <node id="Write">
    <operator>Write</operator>
    <sources>
      <sourceProduct refid="Speckle-Filter"/>
    </sources>
    <parameters>
      <file>{xml_escape(str(output_path))}</file>
      <formatName>BEAM-DIMAP</formatName>
    </parameters>
  </node>
</graph>"""

        return xml

    def _load_snap_output(self, dim_path: Path,
                           polarization: str = "VV") -> OceanImage:
        """Load SNAP BEAM-DIMAP output as OceanImage using GDAL.

        Args:
            dim_path: Path to .dim file.
            polarization: Polarization used for calibration (for band lookup).
        """
        from ocean_rs.shared.raster_io import RasterIO

        data_dir = dim_path.with_suffix('.data')
        sigma0_files = list(data_dir.glob(f"Sigma0_{polarization}*.img"))
        if not sigma0_files:
            sigma0_files = sorted(data_dir.glob("Sigma0*.img"))
            if sigma0_files:
                logger.warning(f"Exact polarization '{polarization}' not found, using: {sigma0_files[0].name}")
        if not sigma0_files:
            raise FileNotFoundError(f"No Sigma0 band found in: {data_dir}")

        band_file = sigma0_files[0]
        data, metadata = RasterIO.read_raster(str(band_file))

        # GDAL geotransform: (origin_x, pixel_size_x, rot_x, origin_y, rot_y, pixel_size_y)
        gt = metadata['geotransform']
        geo = GeoTransform(
            origin_x=gt[0],
            origin_y=gt[3],
            pixel_size_x=gt[1],
            pixel_size_y=gt[5],
            crs_wkt=metadata.get('projection', ''),
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

    @staticmethod
    def _parse_datetime(scene_name: str) -> str:
        """Parse datetime from Sentinel-1 filename.

        M-14: S1 filenames follow: S1A_IW_GRDH_1SDV_YYYYMMDDTHHMMSS_...
        The acquisition start datetime is the 5th field.

        Args:
            scene_name: Sentinel-1 scene name (stem, no extension).

        Returns:
            ISO 8601 datetime string (YYYY-MM-DDTHH:MM:SSZ), or empty string
            if parsing fails.
        """
        match = re.search(r'(\d{8}T\d{6})', scene_name)
        if match:
            raw = match.group(1)
            # Convert YYYYMMDDTHHMMSS -> YYYY-MM-DDTHH:MM:SSZ
            dt_str = (
                f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]}T"
                f"{raw[9:11]}:{raw[11:13]}:{raw[13:15]}Z"
            )
            logger.info(f"Parsed acquisition datetime: {dt_str}")
            return dt_str

        logger.warning(f"Could not parse datetime from filename: {scene_name}")
        return ""

    # ------------------------------------------------------------------
    # InSAR SLC support
    # ------------------------------------------------------------------

    # Sentinel-1 C-band radar wavelength (m)
    S1_WAVELENGTH_M = 0.05546576

    def read_slc(self, input_path: Path, output_dir: Path,
                 polarization: str = "VV") -> SLCImage:
        """Read Sentinel-1 SLC for InSAR processing.

        For IW (TOPS) mode, automatically debursts before reading.
        For SM (Stripmap) mode, reads SLC directly.

        The complex SLC data is read from the measurement TIFF files
        inside the .SAFE directory. Orbit state vectors are parsed
        from the annotation XML.

        Args:
            input_path: Path to S1 SLC product (.SAFE or .zip)
            output_dir: Working directory for intermediate files
            polarization: Polarization channel (default: VV)

        Returns:
            SLCImage with complex data and orbit metadata.
        """
        polarization = self._validate_polarization(polarization)
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        scene_name = input_path.stem.replace('.SAFE', '')
        beam_mode = self._detect_beam_mode(scene_name)

        if beam_mode == 'IW':
            # TOPS mode: deburst first via SNAP GPT
            debursted_path = self.deburst_slc(
                input_path, output_dir, polarization=polarization
            )
            return self._load_slc_from_snap(
                debursted_path, polarization, scene_name, beam_mode
            )
        else:
            # Stripmap or other: read SLC directly from .SAFE
            return self._load_slc_from_safe(
                input_path, output_dir, polarization, scene_name, beam_mode
            )

    def deburst_slc(self, input_path: Path, output_dir: Path,
                    swaths: Optional[list] = None,
                    polarization: str = "VV") -> Path:
        """Deburst Sentinel-1 IW TOPS SLC using SNAP GPT.

        Processing chain:
            Read → TOPSAR-Split (per swath) → Apply-Orbit-File →
            TOPSAR-Deburst → TOPSAR-Merge → Write (BEAM-DIMAP)

        Args:
            input_path: Path to S1 IW SLC product (.SAFE or .zip)
            output_dir: Working directory for debursted output
            swaths: List of swaths to process (e.g. ['IW1','IW2','IW3']).
                    None = all swaths.
            polarization: Polarization channel (VV, VH, HH, HV). Default: VV.

        Returns:
            Path to debursted .dim file.
        """
        gpt = self._find_gpt()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        scene_name = Path(input_path).stem.replace('.SAFE', '')
        output_file = output_dir / f"{scene_name}_debursted.dim"

        if output_file.exists():
            logger.info(f"Debursted file exists, reusing: {output_file.name}")
            return output_file

        graph_xml = self._create_deburst_graph(
            str(input_path), str(output_file), swaths=swaths,
            polarization=polarization
        )

        graph_path = output_dir / f"{scene_name}_deburst_graph.xml"
        with open(graph_path, 'w') as f:
            f.write(graph_xml)

        logger.info(f"Running SNAP GPT TOPS deburst: {scene_name}")
        process = subprocess.Popen(
            [gpt, str(graph_path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        try:
            stdout, stderr = process.communicate(timeout=7200)
            if process.returncode != 0:
                raise RuntimeError(
                    f"SNAP GPT deburst failed (exit {process.returncode}):\n"
                    f"{stderr[-500:]}"
                )
            logger.info(f"SNAP GPT deburst completed: {scene_name}")
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise RuntimeError(
                f"SNAP GPT deburst timed out after 2 hours: {scene_name}"
            )

        graph_path.unlink(missing_ok=True)
        return output_file

    def _create_deburst_graph(self, input_path: str, output_path: str,
                               swaths: Optional[list] = None,
                               polarization: str = "VV") -> str:
        """Create SNAP GPT XML graph for TOPS deburst + merge.

        Chain: Read → Apply-Orbit-File → TOPSAR-Deburst → Write

        SNAP's TOPSAR-Deburst operator handles all sub-swaths.
        Explicit TOPSAR-Merge is not needed when debursting all swaths.
        """
        xml = f"""<graph id="S1-TOPS-Deburst">
  <version>1.0</version>
  <node id="Read">
    <operator>Read</operator>
    <sources/>
    <parameters>
      <file>{xml_escape(str(input_path))}</file>
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
  <node id="TOPSAR-Deburst">
    <operator>TOPSAR-Deburst</operator>
    <sources>
      <sourceProduct refid="Apply-Orbit-File"/>
    </sources>
    <parameters>
      <selectedPolarisations>{xml_escape(str(polarization))}</selectedPolarisations>
    </parameters>
  </node>
  <node id="Write">
    <operator>Write</operator>
    <sources>
      <sourceProduct refid="TOPSAR-Deburst"/>
    </sources>
    <parameters>
      <file>{xml_escape(str(output_path))}</file>
      <formatName>BEAM-DIMAP</formatName>
    </parameters>
  </node>
</graph>"""
        return xml

    def _load_slc_from_snap(self, dim_path: Path, polarization: str,
                             scene_name: str, beam_mode: str) -> SLCImage:
        """Load debursted SLC from SNAP BEAM-DIMAP output.

        Reads the complex (i+q) bands from the .data directory.
        """
        from ocean_rs.shared.raster_io import RasterIO

        data_dir = dim_path.with_suffix('.data')

        # SNAP stores complex as separate i_* and q_* bands
        i_files = sorted(data_dir.glob(f"i_{polarization}*.img"))
        q_files = sorted(data_dir.glob(f"q_{polarization}*.img"))

        if i_files and q_files:
            i_data, metadata = RasterIO.read_raster(str(i_files[0]))
            q_data, _ = RasterIO.read_raster(str(q_files[0]))
            complex_data = i_data.astype(np.float32) + 1j * q_data.astype(np.float32)
        else:
            # Fallback: try amplitude band
            amp_files = sorted(data_dir.glob(f"*{polarization}*.img"))
            if not amp_files:
                raise FileNotFoundError(
                    f"No SLC bands found for {polarization} in: {data_dir}"
                )
            logger.warning(
                f"Complex bands not found, loading amplitude only: {amp_files[0].name}"
            )
            real_data, metadata = RasterIO.read_raster(str(amp_files[0]))
            complex_data = real_data.astype(np.complex64)

        gt = metadata['geotransform']
        geo = GeoTransform(
            origin_x=gt[0], origin_y=gt[3],
            pixel_size_x=gt[1], pixel_size_y=gt[5],
            crs_wkt=metadata.get('projection', ''),
            rows=complex_data.shape[0], cols=complex_data.shape[1],
        )

        # Parse orbit state vectors from annotation XML
        orbit_vectors = self._parse_orbit_from_safe(dim_path)

        return SLCImage(
            data=complex_data,
            geo=geo,
            metadata={
                'sensor': 'Sentinel-1',
                'beam_mode': beam_mode,
                'acquisition_time': self._parse_datetime(scene_name),
                'source_file': str(dim_path),
                'orbit_state_vectors': orbit_vectors,
            },
            wavelength_m=self.S1_WAVELENGTH_M,
            pixel_spacing_range=abs(geo.pixel_size_x),
            pixel_spacing_azimuth=abs(geo.pixel_size_y),
            is_debursted=(beam_mode == 'IW'),
        )

    def _load_slc_from_safe(self, safe_path: Path, output_dir: Path,
                             polarization: str, scene_name: str,
                             beam_mode: str) -> SLCImage:
        """Load SLC directly from .SAFE directory (Stripmap mode).

        Reads the measurement TIFF files containing complex data.
        """
        safe_path = Path(safe_path)

        # Handle .zip files
        if safe_path.suffix.upper() == '.ZIP':
            raise NotImplementedError(
                "Direct SLC reading from .zip not supported. "
                "Extract the .SAFE directory first, or use deburst_slc() for IW mode."
            )

        measurement_dir = safe_path / 'measurement'
        if not measurement_dir.exists():
            raise FileNotFoundError(
                f"No measurement directory in: {safe_path}"
            )

        # Find matching TIFF file
        pol_lower = polarization.lower()
        tiff_files = sorted(measurement_dir.glob(f"*{pol_lower}*.tiff"))
        if not tiff_files:
            tiff_files = sorted(measurement_dir.glob("*.tiff"))
            if tiff_files:
                logger.warning(
                    f"Exact pol '{polarization}' not found, using: {tiff_files[0].name}"
                )
        if not tiff_files:
            raise FileNotFoundError(
                f"No measurement TIFF found for {polarization} in: {measurement_dir}"
            )

        # Read complex data via GDAL
        try:
            from osgeo import gdal
        except ImportError:
            raise ImportError("GDAL is required for SLC reading")

        ds = gdal.Open(str(tiff_files[0]))
        if ds is None:
            raise RuntimeError(f"GDAL failed to open: {tiff_files[0]}")

        try:
            # M15: Check memory before reading large SLC
            check_memory_for_array(
                ds.RasterYSize, ds.RasterXSize,
                bytes_per_pixel=8, description="Sentinel-1 SLC"
            )

            # S1 measurement TIFFs are stored as complex int16
            band = ds.GetRasterBand(1)
            complex_data = band.ReadAsArray().astype(np.complex64)

            gt = ds.GetGeoTransform()
            crs_wkt = ds.GetProjection() or ''
            rows, cols = ds.RasterYSize, ds.RasterXSize
        finally:
            ds = None

        geo = GeoTransform(
            origin_x=gt[0], origin_y=gt[3],
            pixel_size_x=gt[1], pixel_size_y=gt[5],
            crs_wkt=crs_wkt, rows=rows, cols=cols,
        )

        # m11: Geotransform in radar coordinates has pixel_size = 1.0 (pixel indices)
        # Use sensor defaults in that case
        pixel_spacing_range = abs(geo.pixel_size_x) if geo.pixel_size_x != 0 else 2.329
        pixel_spacing_azimuth = abs(geo.pixel_size_y) if geo.pixel_size_y != 0 else 13.97
        if abs(pixel_spacing_range - 1.0) < 0.01:
            pixel_spacing_range = 2.329  # S1 IW range pixel spacing (m)
            logger.debug("Using default S1 IW range pixel spacing (2.329m)")
        if abs(pixel_spacing_azimuth - 1.0) < 0.01:
            pixel_spacing_azimuth = 13.97  # S1 IW azimuth pixel spacing (m)
            logger.debug("Using default S1 IW azimuth pixel spacing (13.97m)")

        orbit_vectors = self._parse_orbit_from_safe(safe_path)

        return SLCImage(
            data=complex_data,
            geo=geo,
            metadata={
                'sensor': 'Sentinel-1',
                'beam_mode': beam_mode,
                'acquisition_time': self._parse_datetime(scene_name),
                'source_file': str(safe_path),
                'orbit_state_vectors': orbit_vectors,
            },
            wavelength_m=self.S1_WAVELENGTH_M,
            pixel_spacing_range=pixel_spacing_range,
            pixel_spacing_azimuth=pixel_spacing_azimuth,
            is_debursted=False,
        )

    def _parse_orbit_from_safe(self, path: Path) -> list:
        """Parse orbit state vectors from S1 annotation XML.

        Looks for orbit state vectors in the annotation directory
        of the .SAFE product, or from the SNAP .dim metadata.

        Returns:
            List of OrbitStateVector objects, or empty list if parsing fails.
        """
        import xml.etree.ElementTree as ET

        path = Path(path)
        orbit_vectors = []

        # Try annotation XML in .SAFE directory
        annotation_dir = None
        if path.suffix.upper() == '.SAFE':
            annotation_dir = path / 'annotation'
        elif path.suffix == '.dim':
            # For SNAP output, orbit info may be in the .dim XML
            safe_dir = self._find_original_safe(path)
            if safe_dir:
                annotation_dir = safe_dir / 'annotation'

        if annotation_dir and annotation_dir.exists():
            xml_files = sorted(annotation_dir.glob("*.xml"))
            if xml_files:
                try:
                    # m24: Prefer defusedxml for safer XML parsing
                    try:
                        from defusedxml import ElementTree as SafeET
                        tree = SafeET.parse(str(xml_files[0]))
                    except ImportError:
                        # defusedxml not available, use standard (safe for trusted local files)
                        tree = ET.parse(str(xml_files[0]))
                    root = tree.getroot()

                    for orbit_elem in root.iter('orbit'):
                        time_elem = orbit_elem.find('time')
                        pos = orbit_elem.find('position')
                        vel = orbit_elem.find('velocity')

                        if time_elem is not None and pos is not None and vel is not None:
                            orbit_vectors.append(OrbitStateVector(
                                time_utc=time_elem.text.strip(),
                                x=float(pos.find('x').text),
                                y=float(pos.find('y').text),
                                z=float(pos.find('z').text),
                                vx=float(vel.find('x').text),
                                vy=float(vel.find('y').text),
                                vz=float(vel.find('z').text),
                            ))

                    if orbit_vectors:
                        logger.info(
                            f"Parsed {len(orbit_vectors)} orbit state vectors"
                        )
                except (ET.ParseError, KeyError, ValueError, AttributeError, IndexError) as e:
                    logger.warning(f"Failed to parse orbit XML: {e}")

        if not orbit_vectors:
            logger.warning(
                "No orbit state vectors found. Baseline computation will "
                "require external orbit files (EOF)."
            )

        return orbit_vectors

    def _find_original_safe(self, dim_path: Path) -> Optional[Path]:
        """Try to find the original .SAFE directory from a SNAP .dim path.

        Looks for .SAFE directories in the parent directory matching
        the scene name extracted from the .dim filename.
        """
        scene_name = dim_path.stem.replace('_debursted', '').replace('_sigma0_fft', '')
        parent = dim_path.parent.parent
        for safe_dir in parent.glob(f"{scene_name}*.SAFE"):
            if safe_dir.is_dir():
                return safe_dir
        return None

    @staticmethod
    def _detect_beam_mode(scene_name: str) -> str:
        """Detect beam mode from Sentinel-1 filename.

        S1 filenames: S1A_IW_SLC__1SDV_... or S1B_SM_SLC__1SSH_...
        The beam mode is the 2nd field.
        """
        parts = scene_name.split('_')
        if len(parts) >= 2:
            mode = parts[1].upper()
            if mode in ('IW', 'EW', 'SM', 'WV'):
                return mode
        logger.warning(f"Could not detect beam mode from filename, defaulting to IW")
        return 'IW'  # Default to IW
