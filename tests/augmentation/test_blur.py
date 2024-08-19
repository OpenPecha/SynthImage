import tempfile
from pathlib import Path

from PIL import Image

from SynthImage.Augmentation.blur_augmentation import BlurAugmentation

original_img_path = Path("./tests/page_image/data/expected_page_image.png")

original_img_obj = Image.open(original_img_path)

rustingObject = BlurAugmentation(original_img_obj)


def test_blur():
    """Test the blur augmentation function."""
    # Apply blur augmentation
    blur_aug_img = rustingObject.apply()
    # Define the expected save directory
    expected_blur_save_path = Path(
        "./tests/augmentation/data/expected_blur_image/expected_blur_image.png"
    )
    with tempfile.TemporaryDirectory() as tempdirname:
        actual_blur_save_path = Path(tempdirname) / "actual_blur_image.png"
        blur_aug_img.save(actual_blur_save_path)
        expected_blur_image = Image.open(expected_blur_save_path)
        actual_blur_image = Image.open(actual_blur_save_path)
        assert expected_blur_image.size == actual_blur_image.size
