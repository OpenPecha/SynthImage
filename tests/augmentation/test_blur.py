from pathlib import Path

from PIL import Image

from SynthImage.blur_augmentation import BlurAugmentation

original_img_path = Path("./tests/page_image/data/expected_page_image.png")

original_img_obj = Image.open(original_img_path)

rustingObject = BlurAugmentation(original_img_obj)


def test_blur():
    """Test the blur augmentation function.

    Args:
        utils: A utility object that contains helper functions for testing,
               specifically an `is_same_img` function that compares two images.
    """
    # Apply blur augmentation
    blur_aug_img = rustingObject.apply_blur()
    # Define the expected save directory
    expected_blur_save_dir = Path("./tests/augmentation/data/expected_blur_image")
    expected_blur_save_dir.mkdir(parents=True, exist_ok=True)

    expected_blur_save_path = Path(expected_blur_save_dir) / "expected_blur_image.png"
    blur_aug_img.save(expected_blur_save_path)


def test_median_blur():
    """Test the median blur augmentation function.

    Args:
        utils: A utility object that contains helper functions for testing,
               specifically an `is_same_img` function that compares two images.
    """
    # Apply median blur augmentation
    median_blur_aug_img = rustingObject.apply_median_blur()
    # Define the expected save directory
    expected_median_blur_save_dir = Path(
        "./tests/augmentation/data/expected_median_blur_image"
    )
    expected_median_blur_save_dir.mkdir(parents=True, exist_ok=True)

    expected_median_blur_save_path = (
        Path(expected_median_blur_save_dir) / "expected_median_blur_image.png"
    )
    median_blur_aug_img.save(expected_median_blur_save_path)


def test_motion_blur():
    """Test the motion blur augmentation function."""
    # Apply motion blur augmentation
    motion_blur_aug_img = rustingObject.apply_motion_blur()
    # Define the expected save directory
    expected_motion_blur_save_dir = Path(
        "./tests/augmentation/data/expected_motion_blur_image"
    )
    expected_motion_blur_save_dir.mkdir(parents=True, exist_ok=True)

    expected_motion_blur_save_path = (
        Path(expected_motion_blur_save_dir) / "expected_motion_blur_image.png"
    )
    motion_blur_aug_img.save(expected_motion_blur_save_path)
