import tempfile
from pathlib import Path

from PIL import Image

from SynthImage.Augmentation.deform_augmentation import DeformAugmentation

original_img_path = Path("./tests/page_image/data/expected_page_image.png")

original_img_obj = Image.open(original_img_path)

deformObject = DeformAugmentation(original_img_obj, 30, 3, 90)


def test_deform_augmentation(utils):
    """Test the deformation augmentation function.

    Args:
        utils: A utility object that contains helper functions for testing,
               specifically an `is_same_img` function that compares two images.
    """
    # Create a temporary directory for storing the actual output
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Apply deformation augmentation
        deform_aug_img = deformObject.apply()

        expected_deform_save_path = Path(
            "./tests/augmentation/data/expected_deform_image/expected_deform_image.png"
        )

        # Save the actual deformed image to the temporary directory
        actual_deform_save_path = tmpdirname + "/actual_deform_image.png"
        deform_aug_img.save(actual_deform_save_path)

        # Open the expected and actual images
        expected_deform_aug_img = Image.open(expected_deform_save_path)
        actual_deform_aug_img = Image.open(actual_deform_save_path)

        # Assert that the actual image matches the expected image
        assert utils.is_same_img(actual_deform_aug_img, expected_deform_aug_img)
