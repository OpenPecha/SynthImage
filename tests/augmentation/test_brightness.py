import tempfile
from pathlib import Path

from PIL import Image

from SynthImage.Augmentation.brightness_augmentation import BrightnessAugmentation

original_img_path = Path("./tests/page_image/data/expected_page_image.png")

original_img_obj = Image.open(original_img_path)

brObject = BrightnessAugmentation(original_img_obj, 1.2)


def test_brightness_augmentation(utils):
    """Test the brightness augmentation function.

    Args:
        utils: A utility object that contains helper functions for testing,
               specifically an `is_same_img` function that compares two images.
    """
    # Apply brightness augmentation
    brightness_aug_img = brObject.apply()
    # Define the expected save directory
    expected_brightness_save_path = Path(
        "./tests/augmentation/data/expected_brightness_image/expected_brightness_image.png"
    )

    with tempfile.TemporaryDirectory() as tempdirname:
        actual_brightness_save_path = tempdirname + "/actual_brightness_image.png"
        brightness_aug_img.save(actual_brightness_save_path)
        expected_brightness_image = Image.open(expected_brightness_save_path)
        actual_brightness_image = Image.open(actual_brightness_save_path)
        assert utils.is_same_img(actual_brightness_image, expected_brightness_image)
