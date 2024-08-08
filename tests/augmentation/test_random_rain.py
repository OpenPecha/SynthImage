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
    expected_rain_save_dir = Path("./tests/augmentation/data/expected_rain_image")
    expected_rain_save_dir.mkdir(parents=True, exist_ok=True)

    expected_rain_save_path = Path(expected_rain_save_dir) / "expected_rain_image.png"
    rain_aug_img.save(expected_rain_save_path)
