from __future__ import annotations
from typing import Protocol, Optional


class IUploader(Protocol):
    """Uploads local files to remote storage (e.g., S3) and returns public URLs."""

    async def upload_file(
        self,
        local_path: str,
        *,
        dest_path: Optional[str] = None,
        content_type: Optional[str] = None,
        public: bool = True,
    ) -> str:
        """Upload a file and return the public URL (or signed URL) string."""
        ...
