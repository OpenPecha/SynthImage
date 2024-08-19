import tempfile
from pathlib import Path

from PIL import Image

from SynthImage.Augmentation.flip_horizontal_augmentation import (
    FlipHorizontalAugmentation,
)

original_img_path = Path("./tests/page_image/data/expected_page_image.png")

original_img_obj = Image.open(original_img_path)

flipHorizontalObject = FlipHorizontalAugmentation(original_img_obj)


def test_horizontal_flip():
    """Test the horizontal flip augmentation function."""
    # Apply horizontal flip augmentation
    horizontal_flip_aug_img = flipHorizontalObject.apply()
    # Define the expected save directory
    expected_horizontal_flip_save_path = Path(
        "./tests/augmentation/data/expected_horizontal_flip_image/expected_horizontal_flip_image.png"
    )
    with tempfile.TemporaryDirectory() as tempdirname:
        actual_horizontal_flip_save_path = (
            Path(tempdirname) / "actual_horizontal_flip_image.png"
        )
        horizontal_flip_aug_img.save(actual_horizontal_flip_save_path)
        expected_horizontal_flip_image = Image.open(expected_horizontal_flip_save_path)
        actual_horizontal_flip_image = Image.open(actual_horizontal_flip_save_path)
        assert expected_horizontal_flip_image.size == actual_horizontal_flip_image.size
