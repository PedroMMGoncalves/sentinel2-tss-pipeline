"""
Sentinel-1 sensor adapter.

Preprocesses Sentinel-1 SLC products using SNAP GPT:
    Apply-Orbit-File -> Thermal-Noise-Removal -> Calibration -> Terrain-Correction
"""

import os
import sys
import shutil
import logging
import subprocess
from pathlib import Path
from typing import Optional
from xml.sax.saxutils import escape as xml_escape

import numpy as np

from ..core.data_models import OceanImage, ImageType, GeoTransform
from .base import SensorAdapter

logger = logging.getLogger('ocean_rs')


class Sentinel1Adapter(SensorAdapter):
    """Preprocess Sentinel-1 SLC to calibrated sigma0 via SNAP GPT."""

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
        """Run SNAP GPT preprocessing chain on Sentinel-1 SLC.

        Processing chain:
            1. Apply-Orbit-File
            2. Thermal-Noise-Removal
            3. Calibration (to Sigma0)
            4. Terrain-Correction (Range-Doppler, to UTM)

        Args:
            input_path: Path to Sentinel-1 product (.SAFE or .zip).
            output_dir: Directory for intermediate outputs.
            snap_gpt_path: Optional explicit path to SNAP GPT executable.
            polarization: Polarization to calibrate (VV, VH, HH, HV).
        """
        gpt = self._find_gpt(snap_gpt_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        scene_name = input_path.stem.replace('.SAFE', '')
        output_file = output_dir / f"{scene_name}_sigma0.dim"

        if output_file.exists():
            logger.info(f"Preprocessed file exists, loading: {output_file.name}")
            return self._load_snap_output(output_file, polarization)

        graph_xml = self._create_processing_graph(
            str(input_path), str(output_file), polarization=polarization
        )

        graph_path = output_dir / f"{scene_name}_graph.xml"
        try:
            with open(graph_path, 'w') as f:
                f.write(graph_xml)
        except OSError as e:
            raise RuntimeError(f"Failed to write SNAP graph to '{graph_path}': {e}") from e

        logger.info(f"Running SNAP GPT preprocessing: {scene_name}")
        cmd = [gpt, str(graph_path)]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=7200
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"SNAP GPT failed (exit {result.returncode}):\n"
                    f"{result.stderr[-500:]}"
                )
            logger.info(f"SNAP GPT completed: {scene_name}")
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"SNAP GPT timed out after 2 hours: {scene_name}")

        graph_path.unlink(missing_ok=True)
        return self._load_snap_output(output_file, polarization)

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
                                  polarization: str = "VV") -> str:
        """Create SNAP GPT XML graph for S1 preprocessing.

        Args:
            input_path: Path to input Sentinel-1 product.
            output_path: Path for BEAM-DIMAP output.
            polarization: Polarization band to calibrate (VV, VH, HH, HV).
        """
        return f"""<graph id="S1-Preprocessing">
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
  <node id="Terrain-Correction">
    <operator>Terrain-Correction</operator>
    <sources>
      <sourceProduct refid="Calibration"/>
    </sources>
    <parameters>
      <demName>SRTM 1Sec HGT</demName>
      <pixelSpacingInMeter>10.0</pixelSpacingInMeter>
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
            sigma0_files = list(data_dir.glob("Sigma0*.img"))
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
