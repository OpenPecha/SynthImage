import tempfile
from pathlib import Path

from PIL import Image

from SynthImage.contrast_augmentation import ContrastAugmentation

original_img_path = Path("./tests/page_image/data/expected_page_image.png")

original_img_obj = Image.open(original_img_path)

cObject = ContrastAugmentation(original_img_obj, 0.7)


def test_brightness_augmentation(utils):
    """Test the Contrast augmentation function.

    Args:
        utils: A utility object that contains helper functions for testing,
               specifically an `is_same_img` function that compares two images.
    """
    # Apply brightness augmentation
    contrast_aug_img = cObject.apply_contrast()
    # Define the expected save directory
    expected_contrast_save_path = Path(
        "./tests/augmentation/data/expected_contrast_image/expected_contrast_image.png"
    )
    with tempfile.TemporaryDirectory() as tempdirname:
        actual_contrast_save_path = tempdirname + "/actual_contrast_image.png"
        contrast_aug_img.save(actual_contrast_save_path)
        expected_contrast_img = Image.open(expected_contrast_save_path)
        actual_contrast_img = Image.open(actual_contrast_save_path)
        assert utils.is_same_img(expected_contrast_img, actual_contrast_img)
