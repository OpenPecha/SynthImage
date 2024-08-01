import tempfile
from pathlib import Path

from PIL import Image

from SynthImage.brightness_augmentation import BrightnessAugmentations

original_img_path = Path("./tests/page_image/data/expected_page_image.png")

original_img_obj = Image.open(original_img_path)

brObject = BrightnessAugmentations(original_img_obj, 1.2)


def test_brightness_augmentation(utils):
    # Apply brightness augmentation
    brightness_aug_img = brObject.apply_brightness()
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
