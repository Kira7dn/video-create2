"""
Shared subprocess utilities for video processing
"""

import subprocess
import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)


class SubprocessError(Exception):
    """
    Custom exception for subprocess errors.

    This exception is raised when a subprocess command fails to execute properly.
    It provides a descriptive error message about the failure.
    """

    def __init__(self, message, command=None, returncode=None, stderr=None):
        """
        Initialize the exception with error details.

        Args:
            message: Primary error message
            command: Optional command that was executed (list or str)
            returncode: Optional exit code from the process
            stderr: Optional error output from the process
        """
        self.message = message
        self.command = command
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(self.message)

    def __str__(self):
        """Format the error message with available details."""
        parts = [self.message]
        if self.command:
            cmd_str = (
                " ".join(self.command)
                if isinstance(self.command, list)
                else self.command
            )
            parts.append(f"Command: {cmd_str}")
        if self.returncode is not None:
            parts.append(f"Exit code: {self.returncode}")
        if self.stderr:
            stderr = str(self.stderr)
            if len(stderr) > 500:  # Limit stderr length
                stderr = stderr[:500] + "... [truncated]"
            parts.append(f"Error output: {stderr}")

        return "\n".join(parts)


def safe_subprocess_run(
    cmd, operation_name="FFmpeg operation", custom_logger: Optional[Any] = None
):
    """
    Safely run subprocess with proper error handling

    Args:
        cmd: Command to run as list of strings
        operation_name: Descriptive name for the operation (for logging)
        custom_logger: Optional logger to use instead of default

    Returns:
        subprocess.CompletedProcess result

    Raises:
        SubprocessError: If subprocess fails or FFmpeg not found
    """
    active_logger = custom_logger or logger

    try:
        if active_logger:
            active_logger.debug(
                "Running %s: %s", operation_name, " ".join(str(x) for x in cmd)
            )
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
        return result
    except subprocess.CalledProcessError as e:
        error_msg = f"{operation_name} failed with return code {e.returncode}"

        # Windows-specific error code handling
        if e.returncode == -2147024896:  # 0x80004005 as signed int
            error_msg += " (Windows Error 0x80004005 - Access Denied or File in Use)"
        elif e.returncode == 3131621040:  # Another form of 0x80004005
            error_msg += " (Windows Error - Possible file access or permission issue)"

        if e.stderr:
            error_msg += f"\nFFmpeg stderr: {e.stderr}"
        if e.stdout:
            error_msg += f"\nFFmpeg stdout: {e.stdout}"
        if active_logger:
            active_logger.error(error_msg)
        raise SubprocessError(error_msg, cmd, e.returncode, e.stderr) from e
    except (OSError, PermissionError) as e:
        if isinstance(e, FileNotFoundError):
            error_msg = (
                f"{operation_name} failed: FFmpeg not found. "
                "Please ensure FFmpeg is installed and in PATH."
            )
        else:
            error_msg = f"{operation_name} failed with OS/Permission error: {e}"
        if active_logger:
            active_logger.error(error_msg)
        raise SubprocessError(error_msg, cmd) from e
