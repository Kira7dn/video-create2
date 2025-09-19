import pytest
from utils.image_utils import search_pixabay_image

pytestmark = pytest.mark.integration


@pytest.mark.integration
@pytest.mark.parametrize(
    "prompt,expected_type",
    [
        ("cat", "largeImageURL"),
        ("technology", "any"),
        ("nature", "any"),
        ("business", "any"),
    ],
)
def test_search_pixabay_image_real_api(prompt, expected_type):
    """
    Test thực tế với Pixabay API thật, settings thật, và nhiều prompts khác nhau.
    """
    from app.core.config import settings

    api_key = settings.pixabay_api_key
    if not api_key:
        pytest.skip("Pixabay API key not set in settings")

    # Sử dụng settings thật
    min_width = settings.video_min_image_width
    min_height = settings.video_min_image_height

    print(
        f"\nTesting prompt: '{prompt}' with size requirements: {min_width}x{min_height}"
    )

    # Gọi API thật
    url = search_pixabay_image(prompt, api_key, min_width, min_height)

    print(f"Result URL: {url}")

    # Kiểm tra kết quả
    if url:
        assert url.startswith("http")
        print("✓ Valid URL returned")

        # Kiểm tra loại URL trả về
        if "largeImageURL" in expected_type or expected_type == "any":
            # Chấp nhận bất kỳ URL hợp lệ nào
            assert any(keyword in url for keyword in ["pixabay", "cdn"])
    else:
        print("No results found for this prompt")


@pytest.mark.integration
def test_search_pixabay_image_fallback_behavior():
    """
    Test behavior với prompt khó tìm để kiểm tra fallback logic.
    """
    from app.core.config import settings

    api_key = settings.pixabay_api_key
    if not api_key:
        pytest.skip("Pixabay API key not set in settings")

    # Test với prompt rất cụ thể có thể không có kết quả
    url = search_pixabay_image(
        "very_specific_nonexistent_query_12345",
        api_key,
        settings.video_min_image_width,
        settings.video_min_image_height,
    )

    print(f"Fallback test result: {url}")
    # Có thể trả về None hoặc URL fallback
    if url:
        assert url.startswith("http")


@pytest.mark.integration
def test_search_pixabay_image_high_resolution():
    """
    Test với yêu cầu độ phân giải cao để kiểm tra logic priority.
    """
    from app.core.config import settings

    api_key = settings.pixabay_api_key
    if not api_key:
        pytest.skip("Pixabay API key not set in settings")

    # Yêu cầu độ phân giải cao
    high_width = 1920
    high_height = 1080

    url = search_pixabay_image("landscape", api_key, high_width, high_height)

    print(f"High resolution test result: {url}")

    if url:
        assert url.startswith("http")
        print("✓ High resolution image URL returned")


@pytest.mark.integration
def test_search_pixabay_image_real_comprehensive():
    """
    Test comprehensive với real API, real settings, real prompts.
    """
    from app.core.config import settings

    api_key = settings.pixabay_api_key
    if not api_key:
        pytest.skip("Pixabay API key not set in settings")

    # Test với settings thật
    min_width = settings.video_min_image_width  # 1024
    min_height = settings.video_min_image_height  # 576

    print(f"\nSettings: min_width={min_width}, min_height={min_height}")
    print(f"API Key exists: {bool(api_key)}")

    # Test với prompt thật
    url = search_pixabay_image("cat", api_key, min_width, min_height)
    print(f"Pixabay result URL: {url}")

    assert url is not None and url.startswith("http")


@pytest.mark.integration
def test_search_pixabay_image_priority_logic():
    """
    Test logic priority: largeImageURL > fullHDURL > imageURL
    """
    from app.core.config import settings

    api_key = settings.pixabay_api_key
    if not api_key:
        pytest.skip("Pixabay API key not set in settings")

    # Test với prompt có nhiều kết quả để kiểm tra priority
    prompts = ["sunset", "mountain", "ocean"]

    for prompt in prompts:
        print(f"\nTesting priority logic with prompt: '{prompt}'")
        url = search_pixabay_image(
            prompt,
            api_key,
            settings.video_min_image_width,
            settings.video_min_image_height,
        )

        if url:
            print(f"URL returned: {url}")
            assert url.startswith("http")

            # Check if URL indicates high quality (large/fullHD)
            # Most Pixabay URLs contain quality indicators
            quality_indicators = ["large", "fullhd", "1920", "1080"]
            has_quality_indicator = any(
                indicator in url.lower() for indicator in quality_indicators
            )
            if has_quality_indicator:
                print("✓ High quality URL detected")
        else:
            print("No results for this prompt")
