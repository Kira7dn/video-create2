"""
Các hàm tiện ích xử lý file audio.

Module này chứa các hàm hỗ trợ kiểm tra và xác thực file audio,
định dạng được hỗ trợ, và các thao tác liên quan đến audio.
"""

import logging
from pathlib import Path
from typing import Tuple

# Khởi tạo logger
audio_logger = logging.getLogger("audio_utils")

# Định dạng file audio được hỗ trợ
SUPPORTED_AUDIO_FORMATS = {".wav", ".mp3", ".m4a"}

# Kích thước file tối đa (100MB)
MAX_AUDIO_SIZE_MB = 100


def validate_audio_file(audio_path: str) -> Tuple[bool, str]:
    """
    Kiểm tra và xác thực file audio.

    Args:
        audio_path: Đường dẫn đến file audio cần kiểm tra

    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    try:
        if not audio_path:
            return False, "Đường dẫn audio không được để trống"

        path = Path(audio_path)

        # Kiểm tra sự tồn tại
        if not path.exists():
            return False, f"Không tìm thấy file audio: {audio_path}"

        if not path.is_file():
            return False, f"Đường dẫn không phải là file: {audio_path}"

        # Kiểm tra kích thước file
        file_size_mb = path.stat().st_size / (1024 * 1024)  # Chuyển sang MB
        if file_size_mb > MAX_AUDIO_SIZE_MB:
            return False, (
                f"Kích thước file quá lớn: {file_size_mb:.2f}MB "
                f"(tối đa {MAX_AUDIO_SIZE_MB}MB)"
            )

        # Kiểm tra định dạng file
        if path.suffix.lower() not in SUPPORTED_AUDIO_FORMATS:
            return False, (
                f"Định dạng file không được hỗ trợ: {path.suffix}. "
                f"Định dạng được hỗ trợ: {', '.join(SUPPORTED_AUDIO_FORMATS)}"
            )

        return True, ""

    except Exception as e:
        error_msg = f"Lỗi khi xác thực file audio: {str(e)}"
        audio_logger.error(error_msg, exc_info=True)
        return False, error_msg


def get_audio_duration(audio_path: str) -> float:
    """
    Lấy thời lượng của file audio (đơn vị: giây).

    Note: Hiện tại chỉ trả về 0, cần tích hợp với thư viện đọc audio thực tế
    """
    # TODO: Tích hợp với thư viện đọc audio (như librosa, pydub, ...)
    return 0.0


def is_audio_file(filename: str) -> bool:
    """
    Kiểm tra xem file có phải là file audio được hỗ trợ không.

    Args:
        filename: Tên file hoặc đường dẫn

    Returns:
        bool: True nếu là file audio được hỗ trợ
    """
    if not filename:
        return False
    return Path(filename).suffix.lower() in SUPPORTED_AUDIO_FORMATS
