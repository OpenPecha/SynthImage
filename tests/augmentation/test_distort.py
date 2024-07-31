import tempfile
from pathlib import Path

from PIL import Image
from test_aug_utils import is_same_img

from SynthImage.distort_augmentation import DistortAugmentation, DistortionMode

original_img_obj_path = Path(
    "../page_image/data/expected_page_output/expected_page_image.png"
)
original_img_obj = Image.open(original_img_obj_path)

distortObject = DistortAugmentation(
    original_img_obj, DistortionMode.subtractive, 200, 100, 1, 2, 4
)


def test_distort_augmentation():
    # Apply distortion augmentation
    distort_aug_img = distortObject.apply_distort()

    # Create a temporary directory for storing the actual output
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Define paths for expected and actual output images
        expected_distort_save_path = Path(
            "../../tests/augmentation/data/aug_output/expected_distort_aug_image_output/expected_distort_augmented_image.png"  # noqa
        )

        # Save the actual distorted image to the temporary directory
        actual_distort_save_path = (
            Path(tmpdirname) / "actual_distort_augmented_image.png"
        )
        distort_aug_img.save(actual_distort_save_path)

        # Open the expected and actual images
        expected_distort_aug_img = Image.open(expected_distort_save_path)
        actual_distort_aug_img = Image.open(actual_distort_save_path)

        # Assert that the actual image matches the expected image
        assert is_same_img(actual_distort_aug_img, expected_distort_aug_img)
