import tempfile
from pathlib import Path

from PIL import Image

from SynthImage.Augmentation.flip_vertical_augmentation import FlipVerticalAugmentation

original_img_path = Path("./tests/page_image/data/expected_page_image.png")

original_img_obj = Image.open(original_img_path)

flipVerticalObject = FlipVerticalAugmentation(original_img_obj)


def test_vertical_flip():
    """Test the vertical flip augmentation function."""
    # Apply vertical flip augmentation
    vertical_flip_aug_img = flipVerticalObject.apply()
    # Define the expected save directory
    expected_vertical_flip_save_path = Path(
        "./tests/augmentation/data/expected_vertical_flip_image/expected_vertical_flip_image.png"
    )
    with tempfile.TemporaryDirectory() as tempdirname:
        actual_vertical_flip_save_path = (
            Path(tempdirname) / "actual_vertical_flip_image.png"
        )
        vertical_flip_aug_img.save(actual_vertical_flip_save_path)
        expected_vertical_flip_image = Image.open(expected_vertical_flip_save_path)
        actual_vertical_flip_image = Image.open(actual_vertical_flip_save_path)
        assert expected_vertical_flip_image.size == actual_vertical_flip_image.size
