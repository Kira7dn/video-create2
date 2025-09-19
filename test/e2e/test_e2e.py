#!/usr/bin/env python3
"""
E2E Test for video creation API with pytest integration and auto server startup
"""

import json
import os
import subprocess
import time
import sys
import requests
from typing import Generator
from contextlib import contextmanager
import pytest


@contextmanager
def uvicorn_server(
    host: str = "localhost", port: int = 8001, timeout: int = 10
) -> Generator[str, None, None]:
    """
    Pytest fixture context manager to start/stop uvicorn server automatically.

    Args:
        host: Server host (default: localhost)
        port: Server port (default: 8001 to avoid conflict)
        timeout: Server startup timeout in seconds (default: 10)

    Yields:
        str: Base URL of the running server
    """
    server_process = None

    def is_server_running() -> bool:
        """Check if server is responding"""
        try:
            # Try API health endpoint first (most reliable)
            response = requests.get(f"http://{host}:{port}/api/v1/health", timeout=5)
            if response.status_code == 200:
                return True

            # Fallback to other endpoints
            for endpoint in ["/", "/docs"]:
                try:
                    response = requests.get(
                        f"http://{host}:{port}{endpoint}", timeout=5
                    )
                    if response.status_code == 200:
                        return True
                except requests.exceptions.RequestException:
                    continue
            return False
        except Exception:
            return False

    def wait_for_server() -> bool:
        """Wait for server to start responding"""
        print(f"⏳ Waiting for server to start on {host}:{port}...")
        for i in range(timeout):
            if is_server_running():
                print(f"✅ Server is ready! (took {i+1} seconds)")
                return True
            if i % 5 == 0 and i > 0:
                print(f"⏳ Still waiting... ({i}/{timeout} seconds)")
            time.sleep(1)
        return False

    try:
        # Check if server is already running
        if is_server_running():
            print(f"✅ Server already running on {host}:{port}")
            yield f"http://{host}:{port}"
            return

        # Start server
        print(f"🚀 Starting uvicorn server on {host}:{port}...")
        server_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.presentation.main:app",
                "--host",
                host,
                "--port",
                str(port),
                "--log-level",
                "info",
                # Removed --workers to avoid complexity and startup issues
            ],
            # Don't capture output - let server run freely for faster startup
            cwd=os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            env={**os.environ, "PYTHONPATH": os.path.abspath(".")},
        )

        # Wait for server to be ready
        if wait_for_server():
            yield f"http://{host}:{port}"
        else:
            # Server failed to start within timeout
            print(f"❌ Server failed to start within {timeout} seconds")
            print(
                "💡 Try running manually: python -m uvicorn app.main:app --host localhost --port 8001"
            )
            raise RuntimeError(f"Server failed to start within {timeout} seconds")

    except Exception as e:
        print(f"❌ Error with server: {e}")
        raise
    finally:
        # Clean up
        if server_process:
            print("🛑 Stopping server...")
            try:
                server_process.terminate()
                server_process.wait(timeout=10)
                print("✅ Server stopped gracefully")
            except subprocess.TimeoutExpired:
                print("⚠️  Forcing server termination...")
                server_process.kill()
                server_process.wait()
                print("✅ Server terminated")


@pytest.fixture(scope="session")
def api_server():
    """Pytest fixture to provide a running API server for the entire test session"""
    with uvicorn_server() as server_url:
        yield server_url


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_e2e_video_creation_with_auto_server(api_server, valid_video_data):
    """
    E2E test for video creation API with automatic server management.

    This test will:
    1. Start a server automatically (via fixture)
    2. Send a video creation request
    3. Verify the response and download the video
    4. Clean up the server automatically
    """
    # API endpoint: router prefix is /video and POST "/create"
    url = f"{api_server}/api/v1/video/create"

    # Use valid_video_data from conftest
    # Server expects top-level {"segments": [...]}
    base_data = valid_video_data
    json_data = base_data.get("json_data", base_data)

    print(f"🧪 Testing video creation API at {url}")
    print(f"📝 Input data: {json.dumps(json_data, indent=2)[:200]}...")

    # Prepare request (multipart with file field)
    files = {"file": ("input_sample.json", json.dumps(json_data), "application/json")}

    # Send request
    response = requests.post(url, files=files, timeout=300)

    # Assert response
    assert (
        response.status_code == 200
    ), f"API call failed with status {response.status_code}: {response.text}"

    print("✅ API call successful!")
    print(f"📄 Content-Type: {response.headers.get('content-type')}")
    print(f"📏 Content-Length: {response.headers.get('content-length')}")

    # Handle response
    content_type = response.headers.get("content-type", "")

    if content_type.startswith("video/"):
        # Direct video response
        output_dir = "test/result/video"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "test_e2e_auto_server_output.mp4")

        with open(output_file, "wb") as f:
            f.write(response.content)

        # Verify file was created and has content
        assert os.path.exists(output_file), "Output video file was not created"
        file_size = os.path.getsize(output_file)
        assert file_size > 0, "Output video file is empty"

        print(f"✅ Video saved as: {output_file}")
        print(f"📁 File size: {file_size} bytes")

    elif content_type.startswith("application/json"):
        # JSON response with job_id or download_url
        result = response.json()
        print(f"📋 API returned JSON: {result}")

        if "job_id" in result:
            # Async job-based workflow
            job_id = result["job_id"]
            print(f"📝 Job ID: {job_id}")

            # Check job status endpoint
            status_url = f"{api_server}/api/v1/video/status/{job_id}"
            print(f"🔍 Checking job status at: {status_url}")

            # Poll for job completion
            max_attempts = (
                90  # 15 minutes with 10-second intervals (matching memory test)
            )
            for attempt in range(max_attempts):
                time.sleep(20)  # Wait 20 seconds like memory management test
                try:
                    status_resp = requests.get(status_url, timeout=10)

                    if status_resp.status_code == 200:
                        status_data = status_resp.json()
                        print(f"📊 Job status: {status_data}")

                        if status_data.get("status") in ["completed", "done"]:
                            # Handle both completed and done statuses
                            download_url = status_data.get(
                                "download_url"
                            ) or status_data.get("result")
                            if download_url:
                                # Check if it's already a complete S3 URL
                                if download_url.startswith("http"):
                                    print(f"✅ S3 URL received: {download_url}")
                                    print(f"📁 Video is available at: {download_url}")
                                    print(
                                        "🎬 E2E Test completed successfully - Video uploaded to S3!"
                                    )
                                    break
                                else:
                                    # Local download URL
                                    video_url = download_url
                                    if video_url.startswith("/"):
                                        video_url = f"{api_server}{video_url}"

                                    print(
                                        f"⬇️  Downloading completed video from: {video_url}"
                                    )
                                    video_resp = requests.get(video_url, timeout=60)

                                    assert (
                                        video_resp.status_code == 200
                                    ), f"Video download failed: {video_resp.status_code}"
                                    assert video_resp.headers.get(
                                        "content-type", ""
                                    ).startswith(
                                        "video/"
                                    ), "Downloaded content is not a video"

                                    # Save video
                                    output_dir = "test/result/video"
                                    os.makedirs(output_dir, exist_ok=True)
                                    output_file = os.path.join(
                                        output_dir, "test_e2e_auto_server_output.mp4"
                                    )

                                    with open(output_file, "wb") as f:
                                        f.write(video_resp.content)

                                    # Verify file
                                    assert os.path.exists(
                                        output_file
                                    ), "Output video file was not created"
                                    file_size = os.path.getsize(output_file)
                                    assert file_size > 0, "Output video file is empty"

                                    print(f"✅ Video saved as: {output_file}")
                                    print(f"📁 File size: {file_size} bytes")
                                    break
                            else:
                                pytest.fail(
                                    "Job completed but no download_url or result provided"
                                )

                        elif status_data.get("status") == "failed":
                            error_msg = status_data.get("error", "Unknown error")
                            pytest.fail(f"Job failed: {error_msg}")

                        elif status_data.get("status") in ["pending", "processing"]:
                            progress = status_data.get("progress", 0)
                            print(
                                f"⏳ Job still processing... (attempt {attempt + 1}/{max_attempts}) - Progress: {progress}%"
                            )
                            continue

                        else:
                            print(f"⚠️  Unknown job status: {status_data.get('status')}")
                            continue

                    else:
                        print(
                            f"⚠️  Failed to get job status: {status_resp.status_code} - {status_resp.text[:100]}"
                        )
                        continue

                except requests.exceptions.RequestException as e:
                    print(f"⚠️  Request error checking job status: {e}")
                    continue
                except Exception as e:
                    print(f"⚠️  Unexpected error checking job status: {e}")
                    continue

            else:
                # Check final status before failing
                try:
                    final_status_resp = requests.get(status_url, timeout=5)
                    if final_status_resp.status_code == 200:
                        final_status = final_status_resp.json()
                        print(f"📊 Final job status: {final_status}")
                        pytest.fail(
                            f"Job did not complete within {max_attempts * 10} seconds. Final status: {final_status.get('status')}"
                        )
                    else:
                        pytest.fail(
                            f"Job did not complete within {max_attempts * 10} seconds and status check failed"
                        )
                except Exception as e:
                    pytest.fail(
                        f"Job did not complete within {max_attempts * 10} seconds"
                    )

        elif "download_url" in result:
            # Direct download URL (legacy workflow)
            download_url = result["download_url"]

            # Download video
            video_url = download_url
            if video_url.startswith("/"):
                video_url = f"{api_server}{video_url}"

            print(f"⬇️  Downloading video from: {video_url}")
            video_resp = requests.get(video_url, timeout=300)

            assert (
                video_resp.status_code == 200
            ), f"Video download failed: {video_resp.status_code}"
            assert video_resp.headers.get("content-type", "").startswith(
                "video/"
            ), "Downloaded content is not a video"

            # Save video
            output_dir = "test/result/video"
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, "test_e2e_auto_server_output.mp4")

            with open(output_file, "wb") as f:
                f.write(video_resp.content)

            # Verify file
            assert os.path.exists(output_file), "Output video file was not created"
            file_size = os.path.getsize(output_file)
            assert file_size > 0, "Output video file is empty"

            print(f"✅ Video saved as: {output_file}")
            print(f"📁 File size: {file_size} bytes")

        else:
            pytest.fail(f"Unexpected API response format: {result}")

    else:
        pytest.fail(f"Unexpected content-type: {content_type}")


if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "-s"])
