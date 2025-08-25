"""
Custom exception handlers and error types
"""

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import logging
import traceback
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class VideoProcessingError(Exception):
    """Custom exception for video processing errors"""

    def __init__(self, message: str, error_code: "Optional[str]" = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


class VideoCreationError(VideoProcessingError):
    """Custom exception for video creation errors"""

    def __init__(self, message: str, error_code: Optional[str] = None):
        super().__init__(message, error_code)


class DownloadError(Exception):
    """Exception raised when asset download fails"""


class ProcessingError(Exception):
    """Exception raised when video processing fails"""


class UploadError(ProcessingError):
    """Exception raised when S3 upload fails
    Args:
        message (str): Error message
        video_id (Optional[str]): Video identifier (if available)
    Example:
        raise UploadError("Failed to upload", video_id="abc123")
    """

    def __init__(self, message: str, video_id: Optional[str] = None):
        super().__init__(message)
        self.video_id = video_id


class FileValidationError(Exception):
    """Custom exception for file validation errors"""

    def __init__(self, message: str, file_name: Optional[str] = None):
        self.message = message
        self.file_name = file_name
        super().__init__(self.message)


class ValidationError(VideoProcessingError):
    """Exception raised when input validation fails"""

    def __init__(self, message: str, validation_errors: Optional[list] = None):
        super().__init__(message, "VALIDATION_ERROR")
        self.validation_errors = validation_errors or []


class PipelineError(VideoProcessingError):
    """Exception raised when pipeline execution fails"""

    def __init__(
        self,
        message: str,
        stage_name: Optional[str] = None,
        stage_errors: Optional[list] = None,
    ):
        super().__init__(message, "PIPELINE_ERROR")
        self.stage_name = stage_name
        self.stage_errors = stage_errors or []


class ConcatenationError(ProcessingError):
    """Exception raised when video concatenation fails"""

    def __init__(self, message: str, video_segments: Optional[list] = None):
        super().__init__(message)
        self.video_segments = video_segments or []


class BatchProcessingError(ProcessingError):
    """Exception raised when batch processing fails"""

    def __init__(
        self,
        message: str,
        failed_items: Optional[list] = None,
        successful_items: Optional[list] = None,
    ):
        super().__init__(message)
        self.failed_items = failed_items or []
        self.successful_items = successful_items or []


class ResourceError(VideoProcessingError):
    """Exception raised when resource management fails"""

    def __init__(self, message: str, resource_type: Optional[str] = None):
        super().__init__(message, "RESOURCE_ERROR")
        self.resource_type = resource_type


class ConfigurationError(VideoProcessingError):
    """Exception raised when configuration is invalid"""

    def __init__(self, message: str, config_key: Optional[str] = None):
        super().__init__(message, "CONFIGURATION_ERROR")
        self.config_key = config_key


class AssetError(DownloadError):
    """Exception raised when asset handling fails"""

    def __init__(
        self,
        message: str,
        asset_type: Optional[str] = None,
        asset_url: Optional[str] = None,
    ):
        super().__init__(message)
        self.asset_type = asset_type
        self.asset_url = asset_url


class TranscriptError(VideoProcessingError):
    """Base exception for transcript processing errors"""

    def __init__(self, message: str, error_code: Optional[str] = None):
        super().__init__(message, error_code or "TRANSCRIPT_ERROR")


class AudioProcessingError(TranscriptError):
    """Exception raised when audio file processing fails"""

    def __init__(self, message: Optional[str] = None, file_path: Optional[str] = None):
        self.file_path = file_path
        msg = message or f"Error processing audio file: {file_path}"
        super().__init__(msg, "AUDIO_PROCESSING_ERROR")


class AlignmentError(TranscriptError):
    """Exception raised when audio-text alignment fails"""

    def __init__(
        self, message: Optional[str] = None, alignment_data: Optional[Dict] = None
    ):
        self.alignment_data = alignment_data or {}
        msg = message or "Error during alignment process"
        super().__init__(msg, "ALIGNMENT_ERROR")


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors"""
    logger.warning(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "detail": {
                "error": "Validation error",
                "details": "Invalid request data",
                "errors": exc.errors(),
            }
        },
    )


async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with consistent format"""
    logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")

    # Ensure detail is in our standard format
    if isinstance(exc.detail, dict):
        detail = exc.detail
    else:
        detail = {"error": "HTTP Error", "details": str(exc.detail)}

    return JSONResponse(status_code=exc.status_code, content={"detail": detail})


async def video_processing_exception_handler(
    request: Request, exc: VideoProcessingError
):
    """Handle video processing errors"""
    logger.error(f"Video processing error: {exc.message}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": {
                "error": "Video processing failed",
                "details": exc.message,
                "error_code": exc.error_code,
            }
        },
    )


async def file_validation_exception_handler(request: Request, exc: FileValidationError):
    """Handle file validation errors"""
    logger.warning(f"File validation error: {exc.message} (file: {exc.file_name})")
    return JSONResponse(
        status_code=400,
        content={
            "detail": {
                "error": "File validation failed",
                "details": exc.message,
                "file_name": exc.file_name,
            }
        },
    )


async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors"""
    logger.error(f"Unexpected error: {type(exc).__name__}: {str(exc)}")
    logger.error(f"Traceback: {traceback.format_exc()}")

    return JSONResponse(
        status_code=500,
        content={
            "detail": {
                "error": "Internal server error",
                "details": "An unexpected error occurred",
            }
        },
    )
