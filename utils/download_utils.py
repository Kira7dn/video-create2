"""
Download utility functions.
"""

import logging
import os
import uuid
from pathlib import Path
from typing import Union
from urllib.parse import urlparse

import aiofiles
import aiohttp

from app.core.config import settings

logger = logging.getLogger(__name__)


async def download_file(url: str, destination: Union[str, Path], **kwargs) -> str:
    """
    Download a file from URL to destination.

    Args:
        url: Source URL to download from
        destination: Local path or directory to save the downloaded file
        **kwargs: Additional download options
            - overwrite: bool - Whether to overwrite existing file (default: False)

    Returns:
        Path to the downloaded file or None if download fails

    """
    # If destination is a directory, generate a filename
    if os.path.isdir(str(destination)):
        filename = (
            os.path.basename(urlparse(url).path) or f"download_{uuid.uuid4().hex}"
        )
        dest_path = os.path.join(str(destination), filename)
    else:
        dest_path = str(destination)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    # Check if file exists and overwrite is False
    if not kwargs.get("overwrite", False) and os.path.exists(dest_path):
        logger.debug("File already exists, skipping download: %s", dest_path)
        return dest_path

    # Download the file
    result = await _download_file_internal(url, dest_path)
    if not result["success"]:
        logger.debug(f"Failed to download {url}: {result['error']}")
        return None

    return dest_path


async def _download_file_internal(url: str, dest_path: str) -> dict:
    """Internal function to download a single file"""
    try:
        timeout = aiohttp.ClientTimeout(total=settings.download_timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                response.raise_for_status()

                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)

                # Stream large files to avoid memory issues
                async with aiofiles.open(dest_path, "wb") as f:
                    async for chunk in response.content.iter_chunked(8192):
                        await f.write(chunk)

                logger.debug("✅ Downloaded %s to %s", url, dest_path)
                return {"success": True, "local_path": dest_path}

    except aiohttp.ClientError as e:
        logger.error("Failed to download %s: %s", url, str(e))
        return {"success": False, "error": f"Failed to download {url}: {e}"}
    except (OSError, IOError) as e:
        logger.error("File operation error downloading %s: %s", url, str(e))
        return {
            "success": False,
            "error": f"File operation error downloading {url}: {e}",
        }
    except Exception as e:
        logger.error("Unexpected error downloading %s: %s", url, str(e))
        return {"success": False, "error": f"Unexpected error downloading {url}: {e}"}
