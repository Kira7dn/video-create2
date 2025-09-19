import asyncio
import os

import pytest

from app.infrastructure.adapters.uploader_s3 import S3Uploader
from app.core.config import settings


@pytest.mark.adapters
@pytest.mark.asyncio
async def test_uploader_local_fallback_when_no_aws(monkeypatch, tmp_path):
    # Remove AWS config
    monkeypatch.setattr(settings, "aws_s3_bucket", "", raising=False)
    monkeypatch.setattr(settings, "aws_s3_region", "", raising=False)
    monkeypatch.setattr(settings, "aws_access_key_id", "", raising=False)
    monkeypatch.setattr(settings, "aws_secret_access_key", "", raising=False)

    # Create a dummy file
    f = tmp_path / "video.mp4"
    f.write_bytes(b"data")

    uploader = S3Uploader()
    url = await uploader.upload_file(str(f))

    assert url.startswith("local://")
    assert url.endswith("video.mp4")


@pytest.mark.adapters
@pytest.mark.asyncio
async def test_uploader_s3_upload(monkeypatch, tmp_path):
    # Provide fake AWS config
    monkeypatch.setattr(settings, "aws_s3_bucket", "my-bucket", raising=False)
    monkeypatch.setattr(settings, "aws_s3_region", "us-east-1", raising=False)
    monkeypatch.setattr(settings, "aws_access_key_id", "KEY", raising=False)
    monkeypatch.setattr(settings, "aws_secret_access_key", "SECRET", raising=False)
    monkeypatch.setattr(settings, "aws_s3_prefix", "videos/", raising=False)

    # Create a dummy file
    f = tmp_path / "video.mp4"
    f.write_bytes(b"data")

    # Stub boto3 client
    class FakeClient:
        def upload_fileobj(self, fobj, bucket, key, ExtraArgs=None):
            # simulate success
            return None

    class FakeBoto3:
        def client(self, *_args, **_kwargs):
            return FakeClient()

    import app.infrastructure.adapters.uploader_s3 as u

    monkeypatch.setattr(u, "boto3", FakeBoto3())

    uploader = S3Uploader()
    url = await uploader.upload_file(str(f))

    assert url == "https://my-bucket.s3.us-east-1.amazonaws.com/videos/video.mp4"
