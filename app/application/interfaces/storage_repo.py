from __future__ import annotations
from typing import Protocol, Optional


class IStorageRepo(Protocol):
    """Abstract storage for temporary and persistent files.

    Implementations may back onto local FS, S3, GCS, etc.
    """

    def save_temp(self, data: bytes, suffix: str = "") -> str:
        """Persist bytes to a temp location and return absolute path."""

    def read_bytes(self, path: str) -> bytes:
        """Read bytes from a path (temp or persistent)."""

    def ensure_dir(self, path: str) -> None:
        """Ensure directory exists."""

    def remove(self, path: str) -> None:
        """Remove a file or directory recursively if exists."""

    def exists(self, path: str) -> bool:
        """Check if a path exists."""

    def get_public_url(self, path: str) -> Optional[str]:
        """Return a public URL if the path is uploaded to a public bucket, else None."""
