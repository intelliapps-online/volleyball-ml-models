"""
Google Drive downloader utility for ML model weights.

This module provides functionality to download model weights from Google Drive
using a single ZIP file containing all models.
"""

import hashlib
import os
import tempfile
import zipfile
from pathlib import Path
from typing import BinaryIO, Dict, Optional, Union
from urllib.parse import urlparse, parse_qs

try:
    import gdown
except ImportError:
    gdown = None

from .logger import logger


# Google Drive file ID for the complete weights ZIP file
WEIGHTS_ZIP_DRIVE_ID = "1__zkTmGwZo2z0EgbJvC14I_3kOpgQx3o"

# Hex SHA-256 of the reviewed weights ZIP. Anyone with edit access to the Drive
# file can replace its contents, so the archive is only extracted when its bytes
# match this value. Update it in a reviewed commit whenever the archive changes;
# while it is unset, automatic download is refused.
WEIGHTS_ZIP_SHA256: Optional[str] = None

# Expected file paths for each model
MODEL_PATHS = {
    "ball_detection": "ball/weights/best.pt",
    "action_detection": "action/weights/best.pt",
    "court_detection": "court/weights/best.pt",
    "game_state": "game_state/"
}


def extract_drive_id(url: str) -> Optional[str]:
    """
    Extract Google Drive file ID from various Google Drive URL formats.
    
    Args:
        url: Google Drive URL in various formats
        
    Returns:
        Google Drive file ID or None if not found
    """
    # Direct file ID
    if len(url) == 33 and not url.startswith('http'):
        return url
    
    # Parse different Google Drive URL formats
    parsed = urlparse(url)
    
    # Format: https://drive.google.com/file/d/FILE_ID/view
    if 'drive.google.com' in parsed.netloc and '/file/d/' in parsed.path:
        parts = parsed.path.split('/')
        try:
            file_id_index = parts.index('d') + 1
            return parts[file_id_index]
        except (ValueError, IndexError):
            pass
    
    # Format: https://drive.google.com/open?id=FILE_ID
    if 'drive.google.com' in parsed.netloc and parsed.path == '/open':
        query_params = parse_qs(parsed.query)
        if 'id' in query_params:
            return query_params['id'][0]
    
    return None


def download_from_google_drive(file_id: str, output_path: Union[str, Path, BinaryIO], quiet: bool = False) -> bool:
    """
    Download a file from Google Drive using its file ID.
    
    Args:
        file_id: Google Drive file ID
        output_path: Local path where the file should be saved, or a binary
            file object opened for writing that receives the bytes
        quiet: Whether to suppress download progress output
        
    Returns:
        True if download was successful, False otherwise
    """
    if gdown is None:
        raise ImportError(
            "gdown is required for downloading from Google Drive. "
            "Install it with: pip install gdown"
        )

    if hasattr(output_path, "write"):
        try:
            logger.info(f"Downloading from Google Drive (ID: {file_id})")
            gdown.download(id=file_id, output=output_path, quiet=quiet)
            return True
        except Exception as e:
            logger.error(f"Failed to download from Google Drive: {e}")
            return False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info(f"Downloading from Google Drive (ID: {file_id}) to {output_path}")
        
        # Download file
        gdown.download(id=file_id, output=str(output_path), quiet=quiet)
        
        if not output_path.exists():
            logger.error(f"Download failed: {output_path} does not exist")
            return False
            
        logger.info(f"Successfully downloaded to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download from Google Drive: {e}")
        return False


def _sha256_of_stream(stream: BinaryIO) -> str:
    stream.seek(0)
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    stream.seek(0)
    return digest.hexdigest()


def download_all_models(weights_dir: Optional[Union[str, Path]] = None, force_download: bool = False,
                        quiet: bool = False) -> bool:
    """
    Download all model weights from the complete ZIP file.
    
    Args:
        weights_dir: Base directory for weights (defaults to 'weights' in current directory)
        force_download: Whether to download even if files already exist
        quiet: Whether to suppress download progress output
        
    Returns:
        True if download and extraction was successful, False otherwise
    """
    if weights_dir is None:
        weights_dir = Path.cwd() / "weights"
    else:
        weights_dir = Path(weights_dir)
    
    # Check if all models already exist
    if not force_download:
        existing_models = check_model_weights(weights_dir)
        if all(existing_models.values()):
            logger.info("All model weights already exist")
            return True
    
    expected_sha256 = WEIGHTS_ZIP_SHA256
    if not expected_sha256:
        logger.error(
            "Refusing to download weights: WEIGHTS_ZIP_SHA256 is not set in "
            "utils/downloader.py, so the archive cannot be verified"
        )
        return False
    expected_sha256 = expected_sha256.strip().lower()

    try:
        # Hashed and extracted through this one unnamed handle, so there is no
        # path that could be swapped between the digest check and extraction.
        with tempfile.TemporaryFile() as archive:
            logger.info("Downloading complete weights ZIP file from Google Drive...")
            success = download_from_google_drive(
                file_id=WEIGHTS_ZIP_DRIVE_ID,
                output_path=archive,
                quiet=quiet
            )

            if not success:
                logger.error("Failed to download weights ZIP file")
                return False

            actual_sha256 = _sha256_of_stream(archive)
            if actual_sha256 != expected_sha256:
                logger.error(
                    "Refusing to extract weights ZIP: SHA-256 "
                    f"{actual_sha256} does not match the pinned {expected_sha256}"
                )
                return False

            weights_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Extracting weights to {weights_dir}")
            with zipfile.ZipFile(archive, 'r') as zip_ref:
                zip_ref.extractall(weights_dir)

        logger.info("ZIP file verified and extracted")
        
        # Verify extraction
        extracted_models = check_model_weights(weights_dir)
        successful_extractions = sum(extracted_models.values())
        total_models = len(extracted_models)
        
        if successful_extractions == total_models:
            logger.success(f"Successfully extracted all {total_models} models")
            return True
        else:
            missing = [name for name, exists in extracted_models.items() if not exists]
            logger.warning(f"Some models missing after extraction: {missing}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to download/extract weights ZIP: {e}")
        return False


def check_model_weights(weights_dir: Optional[Union[str, Path]] = None) -> Dict[str, bool]:
    """
    Check which model weights are available locally.
    
    Args:
        weights_dir: Base directory for weights (defaults to 'weights' in current directory)
        
    Returns:
        Dictionary mapping model names to availability status
    """
    if weights_dir is None:
        weights_dir = Path.cwd()
    else:
        weights_dir = Path(weights_dir)
    
    results = {}
    
    for model_name, relative_path in MODEL_PATHS.items():
        target_path = weights_dir / relative_path
        
        if model_name == "game_state":
            # For game_state, check if directory exists and has files
            exists = target_path.exists() and any(target_path.iterdir())
        else:
            # For other models, check if the .pt file exists
            exists = target_path.exists()
        
        results[model_name] = exists
    
    return results