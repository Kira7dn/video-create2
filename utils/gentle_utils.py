"""
Utils for working with Gentle forced alignment service.
Provides functionality to call Gentle API and verify alignment quality.
"""

import time
import json
import os
import sys
from typing import Dict, List, Tuple, Any

import logging
import requests

logger = logging.getLogger(__name__)


class GentleAlignmentError(Exception):
    """Base exception for Gentle alignment errors."""


class GentleAlignmentVerificationError(GentleAlignmentError):
    """Raised when alignment verification fails."""


def verify_alignment_quality(
    word_items: List[Dict[str, Any]], min_success_ratio: float = 0.5
) -> Dict[str, Any]:
    """
    Verify the quality of Gentle alignment results.

    Args:
        word_items: List of word items from Gentle API response
        min_success_ratio: Minimum ratio of successfully aligned words (0-1)

    Returns:
        Dict containing verification results and statistics

    Raises:
        GentleAlignmentVerificationError: If alignment quality is below thresholds
    """
    if not word_items:
        raise GentleAlignmentVerificationError(
            "No word items provided for verification"
        )

    total_words = len(word_items)
    success_words = filter_successful_words(word_items)
    success_count = len(success_words)
    success_ratio = success_count / total_words if total_words > 0 else 0

    # Check for alignment issues
    alignment_issues = []
    for word in word_items:
        if word.get("case") != "success":
            alignment_issues.append(
                {
                    "word": word.get("word"),
                    "case": word.get("case"),
                    "start": word.get("start"),
                    "end": word.get("end"),
                }
            )
    # Verify quality thresholds
    #   • Always check success_ratio
    # Confidence is not used for verification in default Gentle builds
    is_verified = success_ratio >= min_success_ratio

    result = {
        "is_verified": is_verified,
        "total_words": total_words,
        "success_count": success_count,
        "success_ratio": success_ratio,
        "alignment_issues": alignment_issues,
        "issues_count": len(alignment_issues),
        "verification_passed": is_verified,
    }
    return result


def align_audio_with_transcript(
    audio_path: str,
    transcript_path: str,
    gentle_url: str = "http://localhost:8765/transcriptions",
    timeout: int = 600,  # Tổng thời gian timeout cho toàn bộ quá trình
    min_success_ratio: float = 0.8,
    max_retries: int = 3,
    retry_delay: int = 10,
    request_timeout: int = 60,  # Timeout cho mỗi request riêng lẻ (giây)
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Align audio with transcript using Gentle API.

    Args:
        audio_path: Path to audio file
        transcript_path: Path to transcript text file
        gentle_url: Gentle API endpoint URL
        timeout: Total timeout for the entire operation in seconds
        min_success_ratio: Minimum ratio of successfully aligned words (0-1)
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        request_timeout: Timeout for each individual request in seconds

    Returns:
        Tuple of (Gentle API response, verification result)

    Raises:
        GentleAlignmentError: If alignment fails
    """
    start_time = time.time()
    last_error = None

    # Prepare files for upload
    for attempt in range(1, max_retries + 1):
        audio_file = None
        transcript_file = None
        session = None

        try:
            # Kiểm tra timeout tổng
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise GentleAlignmentError(
                    f"Total operation timeout of {timeout} seconds exceeded"
                )

            # Mở file trong mỗi lần thử để tránh file handle bị đóng
            audio_file = open(audio_path, "rb")
            transcript_file = open(transcript_path, "r", encoding="utf-8")

            files = {
                "audio": (os.path.basename(audio_path), audio_file, "audio/mp3"),
                "transcript": (os.path.basename(transcript_path), transcript_file),
            }

            # Tạo session mới cho mỗi lần thử
            session = requests.Session()
            session.headers.update(
                {"User-Agent": "Video-Create/1.0", "Accept": "application/json"}
            )

            # Gửi request với timeout riêng
            logger.info(
                "Sending request to Gentle API (attempt %s/%s)", attempt, max_retries
            )
            logger.info("Gentle URL: %s", gentle_url)
            logger.info(
                "Audio file: %s (exists: %s)", audio_path, os.path.exists(audio_path)
            )
            logger.info(
                "Transcript file: %s (exists: %s)",
                transcript_path,
                os.path.exists(transcript_path),
            )

            try:
                response = session.post(
                    f"{gentle_url}?async=false", files=files, timeout=request_timeout
                )
                logger.info("Gentle API response status: %s", response.status_code)
                logger.debug("Response headers: %s", response.headers)

                response.raise_for_status()
                result = response.json()
                logger.info("Successfully received response from Gentle API")
                logger.debug(
                    "Response sample: %s...",
                    json.dumps(result)[:200] if result else "Empty response",
                )

                # Xác minh kết quả
                word_items = result.get("words", [])
                verification_result = verify_alignment_quality(
                    word_items, min_success_ratio=min_success_ratio
                )

                return result, verification_result

            except Exception as e:
                logger.error(
                    "Error processing Gentle API response: %s", str(e), exc_info=True
                )
                raise

        except requests.exceptions.Timeout as e:
            last_error = (
                f"Request timed out after {request_timeout}s "
                f"(attempt {attempt}/{max_retries})"
            )
            logger.error("Gentle API timeout: %s", str(e), exc_info=True)
            print("Warning: %s", last_error, file=sys.stderr)

        except requests.exceptions.RequestException as e:
            last_error = f"Request failed (attempt {attempt}/{max_retries})"
            logger.error("Gentle API request failed: %s", str(e), exc_info=True)
            print("Warning: %s", last_error, file=sys.stderr)
            logger.error(
                "Response content: %s",
                (
                    e.response.text
                    if hasattr(e, "response") and e.response
                    else "No response"
                ),
            )

        except json.JSONDecodeError as e:
            last_error = f"Invalid JSON response (attempt {attempt}/{max_retries})"
            logger.error("JSON decode error: %s", str(e))
            if hasattr(response, "text"):
                logger.error("Response text: %s", response.text[:500])
            print("Warning: %s", last_error, file=sys.stderr)

        except (IOError, OSError) as e:
            last_error = f"I/O error during alignment (attempt {attempt}/{max_retries}): {str(e)}"
            logger.error("I/O error: %s", str(e), exc_info=True)
            print("Error: %s", last_error, file=sys.stderr)
            logger.error("Current working directory: %s", os.getcwd())
            logger.error(
                "Audio file exists: %s",
                os.path.exists(audio_path) if "audio_path" in locals() else "N/A",
            )
            logger.error(
                "Transcript file exists: %s",
                (
                    os.path.exists(transcript_path)
                    if "transcript_path" in locals()
                    else "N/A"
                ),
            )

        finally:
            # Đóng session và file handles
            if session:
                session.close()
            if audio_file:
                audio_file.close()
            if transcript_file:
                transcript_file.close()

            # Nếu còn lần thử, chờ một chút trước khi thử lại
            if attempt < max_retries:
                time.sleep(retry_delay)

    # Nếu đến đây có nghĩa là đã hết số lần thử
    raise GentleAlignmentError(
        f"Failed to align audio after {max_retries} attempts. Last error: {last_error}"
    )


def filter_successful_words(word_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter and return only successfully aligned words.

    Args:
        word_items: List of word items from Gentle API response

    Returns:
        List of successfully aligned word items
    """
    return [w for w in word_items if w.get("case") == "success"]
