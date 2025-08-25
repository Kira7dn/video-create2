from __future__ import annotations
import os
import asyncio
from typing import Optional

import boto3

from app.application.interfaces.uploader import IUploader
from app.core.config import settings


class S3Uploader(IUploader):
    """Uploader that mirrors service-layer S3 behavior with local fallback."""

    def __init__(self) -> None:
        pass

    async def upload_file(
        self,
        local_path: str,
        *,
        dest_path: Optional[str] = None,
        content_type: Optional[str] = None,
        public: bool = True,
    ) -> str:
        bucket = settings.aws_s3_bucket
        region = settings.aws_s3_region
        aws_key = settings.aws_access_key_id
        aws_secret = settings.aws_secret_access_key
        key = dest_path or f"{settings.aws_s3_prefix}{os.path.basename(local_path)}"

        if not bucket or not region or not aws_key or not aws_secret:
            # Fallback: no S3 configured; return local path URL-like
            return f"local://{local_path}"

        def _upload_sync() -> str:
            s3_client = boto3.client(
                "s3",
                region_name=region,
                aws_access_key_id=aws_key,
                aws_secret_access_key=aws_secret,
            )
            extra_args = {}
            if content_type:
                extra_args["ContentType"] = content_type
            elif local_path.lower().endswith(".mp4"):
                extra_args["ContentType"] = "video/mp4"
            with open(local_path, "rb") as f:
                s3_client.upload_fileobj(f, bucket, key, ExtraArgs=extra_args)
            return f"https://{bucket}.s3.{region}.amazonaws.com/{key}"

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _upload_sync)
