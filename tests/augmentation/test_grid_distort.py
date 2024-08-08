from pathlib import Path

from PIL import Image

from SynthImage.grid_distort_augmentation import GridDistortAugmentation

original_img_path = Path("./tests/page_image/data/expected_page_image.png")

original_img_obj = Image.open(original_img_path)

distortObject = GridDistortAugmentation(original_img_obj)


def test_grid_distort():
    """Test the grid distort augmentation function."""
    # Apply grid distort augmentation
    distort_aug_img = distortObject.apply_grid_distort()
    # Define the expected save directory
    expected_distort_save_dir = Path("./tests/augmentation/data/expected_distort_image")
    expected_distort_save_dir.mkdir(parents=True, exist_ok=True)

    expected_distort_save_path = (
        Path(expected_distort_save_dir) / "expected_distort_image.png"
    )
    distort_aug_img.save(expected_distort_save_path)
