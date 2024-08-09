import tempfile
from pathlib import Path

from PIL import Image

from SynthImage.blur_augmentation import BlurAugmentation

original_img_path = Path("./tests/page_image/data/expected_page_image.png")

original_img_obj = Image.open(original_img_path)

rustingObject = BlurAugmentation(original_img_obj)


def test_blur():
    """Test the blur augmentation function."""
    # Apply blur augmentation
    blur_aug_img = rustingObject.apply_blur()
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


def test_median_blur():
    """Test the median blur augmentation function."""
    # Apply median blur augmentation
    median_blur_aug_img = rustingObject.apply_median_blur()
    # Define the expected save directory
    expected_median_blur_save_path = Path(
        "./tests/augmentation/data/expected_median_blur_image/expected_median_blur_image.png"
    )
    with tempfile.TemporaryDirectory() as tempdirname:
        actual_median_blur_save_path = (
            Path(tempdirname) / "actual_median_blur_image.png"
        )
        median_blur_aug_img.save(actual_median_blur_save_path)
        expected_median_blur_image = Image.open(expected_median_blur_save_path)
        actual_median_blur_image = Image.open(actual_median_blur_save_path)
        assert expected_median_blur_image.size == actual_median_blur_image.size


def test_motion_blur():
    """Test the motion blur augmentation function."""
    # Apply motion blur augmentation
    motion_blur_aug_img = rustingObject.apply_motion_blur()
    # Define the expected save directory
    expected_motion_blur_save_path = Path(
        "./tests/augmentation/data/expected_motion_blur_image/expected_motion_blur_image.png"
    )
    with tempfile.TemporaryDirectory() as tempdirname:
        actual_motion_blur_save_path = (
            Path(tempdirname) / "actual_motion_blur_image.png"
        )
        motion_blur_aug_img.save(actual_motion_blur_save_path)
        expected_motion_blur_image = Image.open(expected_motion_blur_save_path)
        actual_motion_blur_image = Image.open(actual_motion_blur_save_path)
        assert expected_motion_blur_image.size == actual_motion_blur_image.size
