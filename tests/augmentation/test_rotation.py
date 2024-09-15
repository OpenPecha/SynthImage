import tempfile
from pathlib import Path

from PIL import Image

from SynthImage.Augmentation.rotate_augmentation import RotateAugmentation

original_img_path = Path("./tests/page_image/data/expected_page_image.png")

original_img_obj = Image.open(original_img_path)

rObject = RotateAugmentation(original_img_obj, 3)


def test_rotate_augmentation(utils):
    """Test the Rotate augmentation function.

    Args:
        utils: A utility object that contains helper functions for testing,
               specifically an `is_same_img` function that compares two images.
    """

    # Apply rotate augmentation
    rotate_aug_img, angle = rObject.apply_rotate()
    expected_rotate_save_path = Path(
        "./tests/augmentation/data/expected_rotate_image/expected_rotate_image_3.png"
    )
    with tempfile.TemporaryDirectory() as tempdirname:
        actual_rotate_save_path = tempdirname + f"/actual_rotate_image_{angle}.png"
        rotate_aug_img.save(actual_rotate_save_path)
        expected_rotate_image = Image.open(expected_rotate_save_path)
        actual_rotate_image = Image.open(actual_rotate_save_path)
        assert utils.is_same_img(actual_rotate_image, expected_rotate_image)
