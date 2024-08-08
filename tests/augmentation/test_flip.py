from pathlib import Path

from PIL import Image

from SynthImage.flip_augmentation import FlipAugmentation

original_img_path = Path("./tests/page_image/data/expected_page_image.png")

original_img_obj = Image.open(original_img_path)

flipObject = FlipAugmentation(original_img_obj)


def test_vertical_flip():
    """Test the vertical flip augmentation function."""
    # Apply vertical flip augmentation
    vertical_flip_aug_img = flipObject.apply_vertical_flip()
    # Define the expected save directory
    expected_vertical_flip_save_dir = Path(
        "./tests/augmentation/data/expected_vertical_flip_image"
    )
    expected_vertical_flip_save_dir.mkdir(parents=True, exist_ok=True)

    expected_vertical_flip_save_path = (
        Path(expected_vertical_flip_save_dir) / "expected_vertical_flip_image.png"
    )
    vertical_flip_aug_img.save(expected_vertical_flip_save_path)


def test_horizontal_flip():
    """Test the horizontal flip augmentation function."""
    # Apply horizontal flip augmentation
    horizontal_flip_aug_img = flipObject.apply_horizontal_flip()
    # Define the expected save directory
    expected_horizontal_flip_save_dir = Path(
        "./tests/augmentation/data/expected_horizontal_flip_image"
    )
    expected_horizontal_flip_save_dir.mkdir(parents=True, exist_ok=True)

    expected_horizontal_flip_save_path = (
        Path(expected_horizontal_flip_save_dir) / "expected_horizontal_flip_image.png"
    )
    horizontal_flip_aug_img.save(expected_horizontal_flip_save_path)
