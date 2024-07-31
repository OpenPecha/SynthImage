import tempfile
from pathlib import Path

from PIL import Image
from test_aug_utils import is_same_img

from SynthImage.deform_augmentation import DeformAugmentation

original_img_obj_path = Path(
    "../page_image/data/expected_page_output/expected_page_image.png"
)
original_img_obj = Image.open(original_img_obj_path)

deformObject = DeformAugmentation(original_img_obj, 30, 3, 90)


def test_deform_augmentation():
    # Create a temporary directory for storing the actual output
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Apply deformation augmentation
        deform_aug_img = deformObject.apply_deform()

        expected_deform_save_path = Path(
            "../../tests/augmentation/data/aug_output/expected_deform_aug_image_output/expected_deform_augmented_image.png"  # noqa
        )

        # Save the actual deformed image to the temporary directory
        actual_deform_save_path = Path(tmpdirname) / "actual_deform_augmented_image.png"
        deform_aug_img.save(actual_deform_save_path)

        # Open the expected and actual images
        expected_deform_aug_img = Image.open(expected_deform_save_path)
        actual_deform_aug_img = Image.open(actual_deform_save_path)

        # Assert that the actual image matches the expected image
        assert is_same_img(actual_deform_aug_img, expected_deform_aug_img)
