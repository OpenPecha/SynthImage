import tempfile
from pathlib import Path

from PIL import Image

from SynthImage.distort_augmentation import DistortAugmentation, DistortionMode

original_img_path = Path("./tests/page_image/data/expected_page_image.png")

original_img_obj = Image.open(original_img_path)

distortObject = DistortAugmentation(
    original_img_obj, DistortionMode.subtractive, 200, 100, 1, 2, 4
)


def test_distort_augmentation(utils):
    """Test the distortion augmentation function.

    Args:
         utils: A utility object that contains helper functions for testing,
               specifically an `is_same_img` function that compares two images.
    """
    # Apply distortion augmentation
    distort_aug_img = distortObject.apply_distort()

    # Create a temporary directory for storing the actual output
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Define paths for expected and actual output images
        expected_distort_save_path = "./tests/augmentation/data/expected_distort_image/expected_distort_image.png"  # noqa
        # noqa

        # Save the actual distorted image to the temporary directory
        actual_distort_save_path = tmpdirname + "/actual_distort_image.png"

        distort_aug_img.save(actual_distort_save_path)

        # Open the expected and actual images
        expected_distort_aug_img = Image.open(expected_distort_save_path)
        actual_distort_aug_img = Image.open(actual_distort_save_path)

        # Assert that the actual image matches the expected image
        assert utils.is_same_img(actual_distort_aug_img, expected_distort_aug_img)
