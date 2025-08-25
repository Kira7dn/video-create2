"""
Resource management utilities for video processing
"""

import os
import gc
import time
import logging
import shutil
import uuid
import asyncio
import threading
from contextlib import contextmanager, asynccontextmanager
from typing import List, Optional, AsyncIterator
from app.core.config import settings

logger = logging.getLogger(__name__)


class ResourceManager:
    """Manages file resources and cleanup operations"""

    def __init__(self):
        self.tracked_files: List[str] = []

    def cleanup_files(self):
        """Clean up all tracked files"""
        for file_path in self.tracked_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.debug("✅ Removed file: %s", file_path)
            except (OSError, PermissionError) as e:
                logger.warning(
                    "Failed to remove file %s: %s. Scheduling delayed cleanup.",
                    file_path,
                    str(e),
                )
                # Schedule delayed cleanup
                self._schedule_delayed_cleanup(file_path)
        self.tracked_files.clear()

    def cleanup_all(self):
        """Clean up all tracked resources"""
        self.cleanup_files()

        if settings.performance_gc_enabled:
            gc.collect()

    def _schedule_delayed_cleanup(
        self, path: str, delay_seconds: Optional[float] = None
    ):
        """Schedule delayed cleanup for a file or directory"""
        if delay_seconds is None:
            delay_seconds = settings.temp_delayed_cleanup_delay

        def delayed_cleanup():
            try:
                time.sleep(delay_seconds)
                if os.path.exists(path):
                    if os.path.isfile(path):
                        os.remove(path)
                        logger.info("🕒 Delayed cleanup: Removed file %s", path)
                    elif os.path.isdir(path):
                        shutil.rmtree(path, ignore_errors=True)
                        logger.info("🕒 Delayed cleanup: Removed directory %s", path)
            except (OSError, PermissionError, shutil.Error) as e:
                logger.warning("🕒 Delayed cleanup failed for %s: %s", path, str(e))

        cleanup_thread = threading.Thread(target=delayed_cleanup, daemon=True)
        cleanup_thread.start()
        logger.info("🕒 Scheduled delayed cleanup for %s in %ss", path, delay_seconds)


@contextmanager
def managed_resources():
    """Context manager for automatic resource cleanup"""
    manager = ResourceManager()
    try:
        yield manager
    finally:
        manager.cleanup_all()


@asynccontextmanager
async def managed_temp_directory(prefix: Optional[str] = None) -> AsyncIterator[str]:
    """Async context manager for temporary directory with automatic cleanup

    Temp directories are created under a base directory which can be overridden
    via the TEMP_BASE_DIR environment variable (used by tests). Defaults to
    'data'.
    """

    base_dir = os.getenv("TEMP_BASE_DIR", "data/tmp")
    os.makedirs(base_dir, exist_ok=True)

    if prefix is None:
        dir_prefix = os.path.join(base_dir, settings.temp_dir_prefix)
    else:
        # Ensure trailing underscore once, and join with base_dir
        safe_prefix = prefix.rstrip("_") + "_"
        dir_prefix = os.path.join(base_dir, safe_prefix)

    temp_dir = f"{dir_prefix}{uuid.uuid4().hex}"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        yield temp_dir
    finally:
        # await _cleanup_temp_directory_async(temp_dir)
        pass


async def _cleanup_temp_directory_async(temp_dir: str):
    """Async cleanup for temporary directories with retries"""

    try:
        if not os.path.exists(temp_dir):
            return

        # Force garbage collection
        if settings.performance_gc_enabled:
            gc.collect()
            await asyncio.sleep(settings.performance_file_handle_delay)

        # Non-Windows systems
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.info("✅ Cleaned up temporary directory: %s", temp_dir)

    except (OSError, PermissionError, shutil.Error) as e:
        logger.warning("❌ Failed to clean up temp directory %s: %s", temp_dir, str(e))


def cleanup_old_temp_directories(
    base_pattern: Optional[str] = None, max_age_hours: Optional[float] = None
):
    """Clean up old temporary directories under the configured base dir"""
    if base_pattern is None:
        base_pattern = settings.temp_dir_prefix
    if max_age_hours is None:
        max_age_hours = settings.temp_cleanup_age_hours

    base_dir = os.getenv("TEMP_BASE_DIR", "data/tmp")

    try:
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600

        if not os.path.isdir(base_dir):
            return

        for item in os.listdir(base_dir):
            if item.startswith(base_pattern):
                path = os.path.join(base_dir, item)
                if os.path.isdir(path):
                    try:
                        dir_mtime = os.path.getmtime(path)
                        age_seconds = current_time - dir_mtime

                        if age_seconds > max_age_seconds:
                            logger.info(
                                "🧹 Cleaning up old temp directory: %s (age: %.1fh)",
                                path,
                                age_seconds / 3600,
                            )
                            shutil.rmtree(path, ignore_errors=True)
                            if not os.path.exists(path):
                                logger.info(
                                    "✅ Successfully removed old temp directory: %s",
                                    path,
                                )
                            else:
                                ResourceManager()._schedule_delayed_cleanup(
                                    path, delay_seconds=60.0
                                )
                    except (OSError, PermissionError, shutil.Error) as e:
                        logger.warning(
                            "Failed to process temp directory %s: %s", path, str(e)
                        )
    except (OSError, PermissionError) as e:
        logger.warning("Failed to cleanup old temp directories: %s", str(e))
