import tempfile
from pathlib import Path

from PIL import Image

from SynthImage.Augmentation.sharpness_augmentation import SharpnessAugmentation

original_img_path = Path("./tests/page_image/data/expected_page_image.png")

original_img_obj = Image.open(original_img_path)

sObject = SharpnessAugmentation(original_img_obj, 0.7)


def test_sharpness_augmentation(utils):
    """Test the sharpness augmentation function.

    Args:
        utils: A utility object that contains helper functions for testing,
               specifically an `is_same_img` function that compares two images.
    """
    # Apply rotate augmentation
    sharpness_aug_img = sObject.apply_sharpness()
    expected_sharpness_save_path = Path(
        "./tests/augmentation/data/expected_sharpness_image/expected_sharpness_image.png"
    )
    with tempfile.TemporaryDirectory() as tempdirname:
        actual_sharpness_save_path = tempdirname + "/actual_sharpness_image.png"
        sharpness_aug_img.save(actual_sharpness_save_path)
        expected_sharpness_img = Image.open(expected_sharpness_save_path)
        actual_sharpness_img = Image.open(actual_sharpness_save_path)
        assert utils.is_same_img(expected_sharpness_img, actual_sharpness_img)
