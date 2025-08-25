"""
Image processing utilities for video creation.

This module provides functions for image manipulation, enhancement, and validation
for use in video creation pipelines.
"""

import os
from typing import List, Union, Optional, Tuple
import cv2
import numpy as np


import numpy.typing as npt
import requests
from PIL import Image

# Type aliases
CV2Image = npt.NDArray[np.uint8]

# Add constants
cv2.INTER_AREA = cv2.INTER_AREA if hasattr(cv2, "INTER_AREA") else 3
cv2.BORDER_CONSTANT = cv2.BORDER_CONSTANT if hasattr(cv2, "BORDER_CONSTANT") else 0
cv2.COLOR_BGR2LAB = cv2.COLOR_BGR2LAB if hasattr(cv2, "COLOR_BGR2LAB") else 44
cv2.COLOR_LAB2BGR = cv2.COLOR_LAB2BGR if hasattr(cv2, "COLOR_LAB2BGR") else 56
cv2.COLOR_BGR2HSV = cv2.COLOR_BGR2HSV if hasattr(cv2, "COLOR_BGR2HSV") else 40
cv2.COLOR_HSV2BGR = cv2.COLOR_HSV2BGR if hasattr(cv2, "COLOR_HSV2BGR") else 54


def get_smart_pad_color(
    img: np.ndarray, method: str = "average_edge"
) -> Tuple[int, int, int]:
    """
    Automatically detect optimal padding color from image edges.

    Args:
        img: Input image (BGR format)
        method: Detection method - "average_edge", "corner_average", "median_edge"

    Returns:
        Tuple of (B, G, R) values for padding color
    """
    h, w = img.shape[:2]

    if method == "average_edge":
        # Simple average of edge pixels
        top_edge = img[0, :]
        bottom_edge = img[h - 1, :]
        left_edge = img[:, 0]
        right_edge = img[:, w - 1]

        edge_pixels = np.vstack(
            [
                top_edge.reshape(-1, 3),
                bottom_edge.reshape(-1, 3),
                left_edge.reshape(-1, 3),
                right_edge.reshape(-1, 3),
            ]
        )

        avg_color = np.mean(edge_pixels, axis=0)
        return (int(avg_color[0]), int(avg_color[1]), int(avg_color[2]))

    elif method == "median_edge":
        # Median of edge pixels (more robust to outliers)
        top_edge = img[0, :]
        bottom_edge = img[h - 1, :]
        left_edge = img[:, 0]
        right_edge = img[:, w - 1]

        edge_pixels = np.vstack(
            [
                top_edge.reshape(-1, 3),
                bottom_edge.reshape(-1, 3),
                left_edge.reshape(-1, 3),
                right_edge.reshape(-1, 3),
            ]
        )

        median_color = np.median(edge_pixels, axis=0)
        return (int(median_color[0]), int(median_color[1]), int(median_color[2]))

    elif method == "corner_average":
        # Average of 4 corner regions (10x10 pixels each)
        corner_size = min(10, h // 4, w // 4)

        corners = [
            img[0:corner_size, 0:corner_size],  # Top-left
            img[0:corner_size, w - corner_size : w],  # Top-right
            img[h - corner_size : h, 0:corner_size],  # Bottom-left
            img[h - corner_size : h, w - corner_size : w],  # Bottom-right
        ]

        corner_colors = []
        for corner in corners:
            avg_color = np.mean(corner.reshape(-1, 3), axis=0)
            corner_colors.append(avg_color)

        final_color = np.mean(corner_colors, axis=0)
        return (int(final_color[0]), int(final_color[1]), int(final_color[2]))

    # Default fallback
    return (0, 0, 0)


def process_image(
    image_paths: Union[str, List[str]],
    target_size=(1280, 720),
    pad_color=(0, 0, 0),
    smart_pad_color: bool = False,
    pad_color_method: str = "average_edge",
    auto_enhance: bool = False,
    enhance_brightness: bool = True,
    enhance_contrast: bool = True,
    enhance_saturation: bool = True,
    enhance_sharpness: bool = False,
    output_dir: Optional[str] = None,
    return_arrays: bool = False,
) -> Union[List, List[str]]:
    """
    Load, resize và padding ảnh về đúng target_size (w, h), giữ nguyên tỉ lệ.

    Args:
        image_paths: Đường dẫn ảnh (string) hoặc list đường dẫn
        target_size: Kích thước đích (width, height)
        pad_color: Màu padding (B, G, R)
        smart_pad_color: Sử dụng smart padding color detection
        pad_color_method: Phương pháp phát hiện màu padding
        auto_enhance: Tự động cải thiện chất lượng ảnh
        enhance_brightness: Tự động điều chỉnh độ sáng
        enhance_contrast: Tự động cải thiện độ tương phản
        enhance_saturation: Tự động tối ưu độ bão hòa màu
        enhance_sharpness: Tự động làm sắc nét ảnh
        output_dir: Thư mục lưu ảnh đã xử lý (nếu None thì trả về numpy arrays)
        return_arrays: Nếu True, trả về numpy arrays thay vì đường dẫn file

    Returns:
        List numpy arrays (nếu output_dir=None hoặc return_arrays=True)
        hoặc List đường dẫn file đã xử lý
    """
    # Ensure image_paths is a list
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    processed = []
    processed_paths = []
    target_w, target_h = target_size

    # Ensure target size is divisible by 2 for H.264 compatibility
    target_w = target_w - (target_w % 2)
    target_h = target_h - (target_h % 2)

    for _, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Image not found or unreadable: {path}")

        # Apply auto enhancement if enabled
        if auto_enhance:
            img = auto_enhance_image(
                img,
                enhance_brightness=enhance_brightness,
                enhance_contrast=enhance_contrast,
                enhance_saturation=enhance_saturation,
                enhance_sharpness=enhance_sharpness,
            )

        h, w = img.shape[:2]

        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        # Ensure resized dimensions are also even
        new_w = new_w - (new_w % 2)
        new_h = new_h - (new_h % 2)

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        pad_left = (target_w - new_w) // 2
        pad_right = target_w - new_w - pad_left
        pad_top = (target_h - new_h) // 2
        pad_bottom = target_h - new_h - pad_top

        # Determine padding color (use enhanced image for smart padding)
        if smart_pad_color:
            actual_pad_color = get_smart_pad_color(img, pad_color_method)
        else:
            actual_pad_color = pad_color

        padded = cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=actual_pad_color,
        )

        processed.append(padded)

        # Save processed image if output_dir is provided
        if output_dir and not return_arrays:
            os.makedirs(output_dir, exist_ok=True)

            # Generate output filename
            original_name = os.path.basename(path)
            name_without_ext = os.path.splitext(original_name)[0]
            processed_filename = f"processed_{name_without_ext}.jpg"
            processed_path = os.path.join(output_dir, processed_filename)

            # Save the processed image
            cv2.imwrite(processed_path, padded)
            processed_paths.append(processed_path)

    # Return based on what was requested
    if output_dir and not return_arrays:
        return processed_paths
    else:
        return processed


def auto_enhance_image(
    img: np.ndarray,
    enhance_brightness: bool = True,
    enhance_contrast: bool = True,
    enhance_saturation: bool = True,
    enhance_sharpness: bool = False,
) -> np.ndarray:
    """
    Tự động cải thiện chất lượng ảnh.

    Args:
        img: Input image (BGR format)
        enhance_brightness: Tự động điều chỉnh độ sáng
        enhance_contrast: Tự động cải thiện độ tương phản
        enhance_saturation: Tự động tối ưu độ bão hòa màu
        enhance_sharpness: Tự động làm sắc nét ảnh

    Returns:
        Enhanced image (BGR format)
    """
    enhanced = img.copy()

    if enhance_brightness or enhance_contrast:
        # Convert to LAB color space for better brightness/contrast control
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]

        if enhance_brightness:
            # Auto brightness adjustment using histogram analysis
            mean_brightness = np.mean(l_channel)
            target_brightness = 128  # Target middle brightness
            brightness_adjustment = target_brightness - mean_brightness

            # Apply brightness adjustment with clipping
            l_channel = np.clip(l_channel + brightness_adjustment * 0.3, 0, 255).astype(
                np.uint8
            )

        if enhance_contrast:
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_channel = clahe.apply(l_channel)

        # Merge back to LAB and convert to BGR
        lab[:, :, 0] = l_channel
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    if enhance_saturation:
        # Convert to HSV for saturation adjustment
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)

        # Calculate current saturation level
        saturation = hsv[:, :, 1]
        mean_saturation = np.mean(saturation)

        # Auto adjust saturation if image is too dull
        if mean_saturation < 100:  # Low saturation threshold
            saturation_factor = 1.2  # Increase saturation by 20%
            hsv[:, :, 1] = np.clip(saturation * saturation_factor, 0, 255).astype(
                np.uint8
            )

        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if enhance_sharpness:
        # Apply unsharp masking for sharpness enhancement
        gaussian = cv2.GaussianBlur(enhanced, (5, 5), 0)
        enhanced = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

    return enhanced


def is_image_size_valid(image_path: str, min_width: int, min_height: int) -> bool:
    """
    Kiểm tra kích thước ảnh có đạt chuẩn không.

    Args:
        image_path: Đường dẫn đến file ảnh
        min_width: Chiều rộng tối thiểu (pixels)
        min_height: Chiều cao tối thiểu (pixels)

    Returns:
        bool: True nếu ảnh đạt chuẩn kích thước, False nếu không

    Examples:
        >>> is_image_size_valid("image.jpg", 1280, 720)
        True
        >>> is_image_size_valid("small_image.jpg", 1920, 1080)
        False
    """
    # Validate input parameters
    if not image_path or not os.path.exists(image_path):
        return False

    if min_width <= 0 or min_height <= 0:
        return False

    try:
        with Image.open(image_path) as img:
            width, height = img.size
            is_valid = width >= min_width and height >= min_height

            return is_valid

    except (IOError, OSError, ValueError):
        return False


def search_pixabay_image(
    prompt: str, api_key: str, min_width: int, min_height: int
) -> Optional[str]:
    """
    Search for an image on Pixabay that matches the given criteria.

    Args:
        prompt: Search query string
        api_key: Pixabay API key
        min_width: Minimum required width in pixels
        min_height: Minimum required height in pixels

    Returns:
        URL of the first matching image, or None if no match found
    """

    url = "https://pixabay.com/api/"

    # Set reasonable API filter since we prioritize large/fullHD images
    # These are typically much larger than min requirements
    api_min_width = max(800, min_width)  # Don't go below 800px
    api_min_height = max(600, min_height)  # Don't go below 600px

    params = {
        "key": api_key,
        "q": prompt,
        "lang": "en",
        "image_type": "photo",
        "orientation": "horizontal",
        "order": "popular",
        "editors_choice": "true",
        # Removed hardcoded category to allow more diverse results
        "safesearch": "true",
        "min_width": api_min_width,
        "min_height": api_min_height,
        "per_page": 20,  # Increased for better chances
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        hits = data.get("hits", [])

        # Priority 1: Tìm largeImageURL từ tất cả hits
        for hit in hits:
            large_url = hit.get("largeImageURL")
            if large_url:
                # Assume large images are usually big enough (typically 1920x1080+)
                return large_url

        # Priority 2: Tìm fullHDURL từ tất cả hits
        for hit in hits:
            fullhd_url = hit.get("fullHDURL")
            if fullhd_url:
                # FullHD usually 1920x1080, good for most cases
                return fullhd_url

        # Priority 3: Tìm imageURL (preview) từ tất cả hits
        for hit in hits:
            image_url = hit.get("imageURL")
            if image_url:
                # Last resort, might be smaller but better than nothing
                return image_url

        # Fallback: remove size constraints entirely and retry
        if not hits:
            params.pop("min_width", None)
            params.pop("min_height", None)
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            hits = data.get("hits", [])

            if hits:
                first_hit = hits[0]
                return (
                    first_hit.get("largeImageURL")
                    or first_hit.get("fullHDURL")
                    or first_hit.get("imageURL")
                )

    except Exception as exc:  # Broad catch to ensure graceful fallback in tests
        # Log the error for debugging if needed
        print(f"Error searching Pixabay: {exc}")
    return None
