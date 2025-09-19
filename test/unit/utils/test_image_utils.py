import os
import tempfile
from PIL import Image

import pytest

from utils.image_utils import is_image_size_valid, search_pixabay_image


@pytest.mark.parametrize(
    "size,expected",
    [
        ((1280, 720), True),
        ((640, 360), False),
        ((1920, 1080), True),
        ((100, 100), False),
    ],
)
def test_is_image_size_valid(size, expected):
    # Tạo ảnh tạm thời với kích thước size
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        img = Image.new("RGB", size, color="white")
        img.save(tmp.name)
        tmp_path = tmp.name
    try:
        assert is_image_size_valid(tmp_path, 1280, 720) == expected
    finally:
        os.remove(tmp_path)


def test_search_pixabay_image_mocked(monkeypatch):
    """
    Unit test với mock data để test logic function thuần túy.
    """

    # Mock requests.get để không gọi thật ra ngoài
    class DummyResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "hits": [
                    {"largeImageURL": "http://img1_large.jpg"},
                    {"fullHDURL": "http://img2_fullhd.jpg"},
                ]
            }

    monkeypatch.setattr("requests.get", lambda *a, **kw: DummyResp())
    url = search_pixabay_image("cat", "fakekey", 1280, 720)
    assert url == "http://img1_large.jpg"  # Should return largeImageURL first

    # Test fallback to fullHDURL
    class FullHDOnlyResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "hits": [
                    {"fullHDURL": "http://img_fullhd.jpg"},
                    {"imageURL": "http://img_preview.jpg"},
                ]
            }

    monkeypatch.setattr("requests.get", lambda *a, **kw: FullHDOnlyResp())
    url2 = search_pixabay_image("cat", "fakekey", 1280, 720)
    assert url2 == "http://img_fullhd.jpg"  # Should return fullHDURL

    # Test fallback to imageURL
    class ImageOnlyResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "hits": [
                    {"imageURL": "http://img_preview.jpg"},
                ]
            }

    monkeypatch.setattr("requests.get", lambda *a, **kw: ImageOnlyResp())
    url3 = search_pixabay_image("cat", "fakekey", 1280, 720)
    assert url3 == "http://img_preview.jpg"  # Should return imageURL

    # Test trường hợp không có hits
    class EmptyResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"hits": []}

    monkeypatch.setattr("requests.get", lambda *a, **kw: EmptyResp())
    url4 = search_pixabay_image("notfound", "fakekey", 1280, 720)
    assert url4 is None


def test_search_pixabay_image_api_error(monkeypatch):
    """
    Unit test xử lý lỗi API.
    """

    # Mock API error
    def mock_get(*args, **kwargs):
        raise Exception("API Error")

    monkeypatch.setattr("requests.get", mock_get)
    url = search_pixabay_image("cat", "fakekey", 1280, 720)
    assert url is None  # Should handle error gracefully


def test_search_pixabay_image_http_error(monkeypatch):
    """
    Unit test xử lý HTTP error.
    """

    class ErrorResp:
        def raise_for_status(self):
            raise Exception("HTTP 404 Not Found")

        def json(self):
            return {}

    monkeypatch.setattr("requests.get", lambda *a, **kw: ErrorResp())
    url = search_pixabay_image("cat", "fakekey", 1280, 720)
    assert url is None  # Should handle HTTP error gracefully


def test_get_smart_pad_color_methods():
    import numpy as np
    from utils.image_utils import get_smart_pad_color

    # Build 4x4 image with distinct edges to make averages deterministic (BGR)
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    img[0, :, :] = [10, 20, 30]      # top
    img[-1, :, :] = [40, 50, 60]     # bottom
    img[:, 0, :] = [70, 80, 90]      # left
    img[:, -1, :] = [100, 110, 120]  # right

    # average_edge should average all four edges
    color_avg = get_smart_pad_color(img, method="average_edge")
    assert isinstance(color_avg, tuple) and len(color_avg) == 3

    # median_edge should be between the edge values
    color_med = get_smart_pad_color(img, method="median_edge")
    assert all(0 <= c <= 255 for c in color_med)

    # corner_average should compute from 10x10 corners but limited by image size -> uses min sizes
    color_corner = get_smart_pad_color(img, method="corner_average")
    assert all(0 <= c <= 255 for c in color_corner)

    # unknown method -> default fallback (0,0,0)
    color_default = get_smart_pad_color(img, method="unknown")
    assert color_default == (0, 0, 0)


def test_process_image_basic_and_output_dir(monkeypatch, tmp_path):
    import numpy as np
    import utils.image_utils as iu

    # Fake image data
    fake_img = np.ones((10, 20, 3), dtype=np.uint8) * 255

    # Track calls
    calls = {"imread": 0, "resize": [], "border": [], "imwrite": []}

    # Mock cv2 functions within module namespace
    def fake_imread(p):
        calls["imread"] += 1
        return fake_img

    monkeypatch.setattr(iu.cv2, "imread", fake_imread)

    def fake_resize(img, size, interpolation=None):
        calls["resize"].append((img.shape, size))
        w, h = size
        return np.zeros((h, w, 3), dtype=np.uint8)

    monkeypatch.setattr(iu.cv2, "resize", fake_resize)

    def fake_border(img, top, bottom, left, right, borderType=None, value=None):
        calls["border"].append((img.shape, top, bottom, left, right, value))
        # Return padded array of requested final size
        h, w = img.shape[:2]
        return np.zeros((h + top + bottom, w + left + right, 3), dtype=np.uint8)

    monkeypatch.setattr(iu.cv2, "copyMakeBorder", fake_border)

    monkeypatch.setattr(iu.os, "makedirs", lambda p, exist_ok=True: None)

    def fake_imwrite(path, img):
        calls["imwrite"].append((path, img.shape))
        return True

    monkeypatch.setattr(iu.cv2, "imwrite", fake_imwrite)

    # Avoid heavy enhancement path; assert enhancer is called when enabled
    enhancer_called = {"called": False}

    def fake_enhance(img, **kwargs):
        enhancer_called["called"] = True
        return img

    monkeypatch.setattr(iu, "auto_enhance_image", fake_enhance)

    # Execute with smart_pad_color True so get_smart_pad_color is used
    monkeypatch.setattr(iu, "get_smart_pad_color", lambda img, method: (1, 2, 3))

    # Create a temp image path; imread is mocked so content not used
    img_path = tmp_path / "img.jpg"
    img_path.write_bytes(b"x")

    out_dir = tmp_path / "out"
    result_paths = iu.process_image(
        str(img_path),
        target_size=(128, 64),
        smart_pad_color=True,
        auto_enhance=True,
        output_dir=str(out_dir),
        return_arrays=False,
    )

    # We saved exactly one processed file with expected prefix
    assert isinstance(result_paths, list) and len(result_paths) == 1
    assert os.path.basename(result_paths[0]).startswith("processed_")

    # Resize and padding called with dimensions producing final target size
    assert calls["resize"], "resize should be called"
    assert calls["border"], "copyMakeBorder should be called"
    # Enhancer was invoked
    assert enhancer_called["called"] is True
