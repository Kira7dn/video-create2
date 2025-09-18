"""
Application configuration using Pydantic Settings
"""

import os
from typing import List, Union
from pydantic_settings import BaseSettings
from pydantic import field_validator


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    # API Settings
    api_title: str = "Video Creation API"
    api_description: str = "Professional video creation service with batch processing"
    api_version: str = "1.0.0"

    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # File Upload Settings
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_extensions: list = [".json"]
    upload_timeout: int = 300  # 5 minutes

    # Temporary Directory Settings
    temp_dir_prefix: str = "tmp_create_"
    cleanup_temp_files: bool = True

    # CORS Settings
    cors_origins: Union[List[str], str] = [
        "http://localhost:3000",
        "http://localhost:8080",
    ]
    cors_allow_credentials: bool = True
    cors_allow_methods: list = ["GET", "POST"]
    cors_allow_headers: list = ["*"]

    # Schema Validation Settings
    schema_path: str = "app/core/schema.json"
    input_sample_path: str = "app/core/input_sample.json"

    @field_validator("cors_origins")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string to list.

        Args:
            v: Can be either a list of origins or a comma-separated string.
               If "*" is provided, allows all origins.

        Returns:
            List[str]: List of allowed origins

        Example:
            >>> parse_cors_origins("http://localhost:3000,http://localhost:8080")
            ['http://localhost:3000', 'http://localhost:8080']
            >>> parse_cors_origins("*")
            ['*']
        """
        if isinstance(v, str):
            if v == "*":
                return ["*"]
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    # Logging Settings
    log_level: str = "INFO"
    log_format: str = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
    log_date_format: str = "%Y-%m-%d %H:%M:%S"

    # Video Processing Settings
    video_default_fps: int = 24
    video_default_codec: str = "libx264"
    video_default_audio_codec: str = "aac"
    # Video type specific presets
    video_resolution_long: str = "1280,720"
    video_resolution_short: str = "1080,1920"

    # Asset Processing Settings
    segment_asset_types: dict = {
        "image": "img",
        "video": "video",
        "voice_over": "voice_over",
    }
    # AWS S3 Settings
    aws_s3_bucket: str = ""
    aws_s3_region: str = "us-east-1"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_s3_prefix: str = "videos/"  # S3 object key prefix for uploads
    """
    AWS S3 configuration for video upload.
    aws_s3_bucket: S3 bucket name
    aws_s3_region: S3 region
    aws_s3_key: AWS access key
    aws_s3_secret: AWS secret key
    """
    video_default_segment_duration: float = 5.0
    video_default_start_delay: float = 0.5
    video_default_end_delay: float = 0.5

    # Audio Processing Settings
    audio_background_music_volume: float = 0.1
    audio_default_sample_rate: int = 44100
    audio_default_bitrate: str = "192k"
    # Enable loudness normalization and ducking for BGM during concat
    audio_auto_volume_enabled: bool = False

    # Per-Segment Audio Loudness Normalization (Renderer)
    segment_audio_normalize_enabled: bool = True
    segment_audio_target_i: float = -16.0
    segment_audio_target_tp: float = -1.5
    segment_audio_target_lra: float = 11.0
    segment_audio_limiter_enabled: bool = True
    # When true, renderer will log ffmpeg loudnorm summary output on success
    segment_audio_log_loudnorm: bool = False

    # Text Overlay Settings
    text_default_font_size: int = 48
    text_default_font_color: str = "white"
    text_default_font_file: str = "fonts/Roboto-Black.ttf"
    text_default_fade_in: float = 0.5
    text_default_fade_out: float = 0.5
    text_default_position_x: str = "(w-text_w)/2"
    text_default_position_y: str = "h-100"

    # Text Over Timing Heuristics (used when alignment missing/partial)
    text_over_word_seconds: float = 0.4  # seconds per word
    text_over_min_duration: float = 1.0  # min duration per line
    text_over_max_duration: float = 6.0  # max duration per line

    # FFmpeg Settings
    ffmpeg_binary_path: str = "ffmpeg"
    ffmpeg_timeout: int = 300
    ffmpeg_preset: str = "medium"
    ffmpeg_threads: int = 0  # 0 = auto

    # Download Settings
    download_timeout: int = 300  # 5 minutes for large video files
    download_max_concurrent: int = 10
    download_retry_attempts: int = 3

    # Pipeline Step Defaults - DownloadAssetsStep
    download_step_retries: int = 2
    download_step_retry_backoff: float = 0.5
    download_step_max_backoff: float = 3.0
    download_step_jitter: float = 0.2
    download_step_use_exp_backoff: bool = True
    download_step_timeout: float | None = None

    # Temp Directory Settings
    temp_dir_prefix: str = "tmp_create_"
    temp_batch_dir: str = "tmp_pipeline"
    temp_cleanup_age_hours: float = 1.0
    temp_cleanup_retry_attempts: int = 3
    temp_cleanup_retry_delay: float = 2.0
    temp_delayed_cleanup_delay: float = 30.0

    # Video Output Settings
    output_directory: str = "data/output"

    # Performance Settings
    performance_gc_enabled: bool = True
    performance_file_handle_delay: float = 1.0
    performance_max_memory_mb: int = 2048
    performance_max_concurrent_segments: int = 1

    # Security Settings
    request_timeout: int = 300  # 5 minutes
    max_concurrent_requests: int = 10

    # Ngrok Settings
    ngrok_authtoken: str = ""
    ngrok_url: str = ""

    # AI Pydantic Settings
    ai_pydantic_enabled: bool = True
    ai_pydantic_model: str = "gpt-4o-mini"

    # OpenAI API Key
    openai_api_key: str = ""

    # AI Keyword Extraction Settings
    ai_keyword_extraction_enabled: bool = True
    ai_keyword_extraction_timeout: int = 10
    ai_max_keywords_per_prompt: int = 3

    # Image Auto Processor Settings
    image_auto_enabled: bool = True
    video_min_image_width: int = 1024  # Minimum image width for video segments
    video_min_image_height: int = 576  # Minimum image height for video segments
    pixabay_api_key: str = ""  # API key for Pixabay image search

    # Pipeline Step Defaults - ImageAutoStep
    image_auto_retries: int = 1
    image_auto_retry_backoff: float = 0.5
    image_auto_max_backoff: float = 3.0
    image_auto_jitter: float = 0.2
    image_auto_use_exp_backoff: bool = True
    image_auto_timeout: float | None = None

    # Gentle Settings
    gentle_timeout: int = 120
    alignment_min_success_ratio: float = 0.8

    @property
    def gentle_url(self) -> str:
        """Get Gentle URL"""

        # Ưu tiên biến môi trường
        if os.getenv("GENTLE_URL"):
            return os.getenv("GENTLE_URL")
        # Nếu chạy Docker
        if os.getenv("DOCKER") == "1" or os.getenv("GENTLE_DOCKER") == "true":
            return "http://gentle:8765/transcriptions"
        # Mặc định local
        return "http://localhost:8765/transcriptions"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        # Allow unknown/legacy env vars without failing validation (e.g., VIDEO_DEFAULT_RESOLUTION)
        "extra": "ignore",
    }

    @property
    def video_long_resolution_tuple(self) -> tuple:
        """Resolution tuple for long video (landscape)."""
        try:
            parts = str(self.video_resolution_long).split(",")
            if len(parts) == 2:
                return (int(parts[0]), int(parts[1]))
        except Exception:
            pass
        return (1280, 720)

    @property
    def video_short_resolution_tuple(self) -> tuple:
        """Resolution tuple for short video (vertical)."""
        try:
            parts = str(self.video_resolution_short).split(",")
            if len(parts) == 2:
                return (int(parts[0]), int(parts[1]))
        except Exception:
            pass
        return (1080, 1920)

    def video_resolution_tuple_for(self, video_type: str | None) -> tuple:
        """Return resolution tuple based on video_type ('short' or 'long').

        Defaults to long if video_type is missing or unrecognized.
        """
        vt = (video_type or "long").strip().lower()
        if vt == "short":
            return self.video_short_resolution_tuple
        return self.video_long_resolution_tuple


# Global settings instance
settings = Settings()
