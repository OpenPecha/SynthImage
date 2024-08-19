import tempfile
from pathlib import Path

from PIL import Image

from SynthImage.Augmentation.grid_distort_augmentation import GridDistortAugmentation

original_img_path = Path("./tests/page_image/data/expected_page_image.png")

original_img_obj = Image.open(original_img_path)

distortObject = GridDistortAugmentation(original_img_obj)


def test_grid_distort():
    """Test the grid distort augmentation function."""
    # Apply grid distort augmentation
    grid_distort_aug_img = distortObject.apply()
    # Define the expected save directory
    expected_grid_distort_save_path = Path(
        "./tests/augmentation/data/expected_grid_distort_image/expected_grid_distort_image.png"
    )
    with tempfile.TemporaryDirectory() as tempdirname:
        actual_grid_distort_save_path = (
            Path(tempdirname) / "actual_grid_distort_image.png"
        )
        grid_distort_aug_img.save(actual_grid_distort_save_path)
        expected_grid_distort_image = Image.open(expected_grid_distort_save_path)
        actual_grid_distort_image = Image.open(actual_grid_distort_save_path)
        assert expected_grid_distort_image.size == actual_grid_distort_image.size
