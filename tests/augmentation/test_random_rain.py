import tempfile
from pathlib import Path

from PIL import Image

from SynthImage.random_rain_augmentation import RandomRainAugmentation

original_img_path = Path("./tests/page_image/data/expected_page_image.png")

original_img_obj = Image.open(original_img_path)

rainObject = RandomRainAugmentation(original_img_obj)


def test_random_rain():
    """Test the random rain augmentation function."""
    # Apply random rain augmentation
    rain_aug_img = rainObject.apply_random_rain()
    # Define the expected save directory
    expected_rain_save_path = Path(
        "./tests/augmentation/data/expected_rain_image/expected_rain_image.png"
    )
    with tempfile.TemporaryDirectory() as tempdirname:
        actual_rain_save_path = Path(tempdirname) / "actual_rain_image.png"
        rain_aug_img.save(actual_rain_save_path)
        expected_rain_image = Image.open(expected_rain_save_path)
        actual_rain_image = Image.open(actual_rain_save_path)
        assert expected_rain_image.size == actual_rain_image.size
