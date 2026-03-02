"""
Batch downloader for SAR scenes from ASF DAAC.

Handles authenticated downloads with resume support and progress reporting.
"""

import logging
from pathlib import Path
from typing import List, Callable, Optional

from .scene_discovery import SceneMetadata
from .credentials import CredentialManager

logger = logging.getLogger('ocean_rs')


class BatchDownloader:
    """Download SAR scenes from ASF with authentication and retry."""

    def __init__(self, credential_manager: CredentialManager):
        self.creds = credential_manager
        self._cancel_requested = False

    def cancel(self):
        """Request download cancellation."""
        self._cancel_requested = True

    def download_scenes(self,
                       scenes: List[SceneMetadata],
                       output_dir: str,
                       progress_callback: Optional[Callable] = None
                       ) -> List[Path]:
        """Download selected scenes to output directory.

        Args:
            scenes: List of scenes to download
            output_dir: Directory to save downloaded files
            progress_callback: Optional callback(scene_index, total, status_msg)

        Returns:
            List of paths to downloaded files
        """
        try:
            import asf_search as asf
        except ImportError:
            raise ImportError("asf_search required: pip install asf_search")

        self._cancel_requested = False
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        username, password = self.creds.get_earthdata_credentials()
        session = asf.ASFSession()
        session.auth_with_creds(username, password)

        downloaded = []
        total = len(scenes)

        for i, scene in enumerate(scenes):
            if self._cancel_requested:
                logger.info("Download cancelled by user")
                break

            logger.info(f"Downloading [{i+1}/{total}]: {scene.granule_id}")
            if progress_callback:
                progress_callback(i, total, f"Downloading: {scene.granule_id}")

            expected_file = output_path / f"{scene.granule_id}.zip"
            if expected_file.exists() and expected_file.stat().st_size > 0:
                logger.info(f"Already downloaded: {scene.granule_id}")
                downloaded.append(expected_file)
                continue

            try:
                asf_result = scene._asf_result
                if asf_result is None:
                    logger.warning(f"No ASF result for {scene.granule_id}, skipping")
                    continue

                asf_result.download(path=str(output_path), session=session)

                for ext in ['.zip', '.SAFE']:
                    candidate = output_path / f"{scene.granule_id}{ext}"
                    if candidate.exists():
                        downloaded.append(candidate)
                        logger.info(f"Downloaded: {candidate.name} "
                                   f"({candidate.stat().st_size / 1e6:.1f} MB)")
                        break
                else:
                    matches = list(output_path.glob(f"*{scene.granule_id}*"))
                    if matches:
                        downloaded.append(matches[0])
                        logger.info(f"Downloaded: {matches[0].name}")
                    else:
                        logger.warning(f"Download completed but file not found: "
                                      f"{scene.granule_id}")

            except Exception as e:
                logger.error(f"Download failed for {scene.granule_id}: {e}")
                if progress_callback:
                    progress_callback(i, total, f"FAILED: {scene.granule_id}")

        if progress_callback:
            progress_callback(total, total, "Download complete")

        logger.info(f"Downloaded {len(downloaded)}/{total} scenes")
        return downloaded
