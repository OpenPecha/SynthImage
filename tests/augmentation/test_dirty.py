import tempfile
from pathlib import Path

from PIL import Image

from SynthImage.dirty_augmentation import DirtySpotAugmentation

original_img_path = Path("./tests/page_image/data/expected_page_image.png")

original_img_obj = Image.open(original_img_path)
dirty_spots = [(50, 50, 20), (150, 100, 30), (200, 200, 25)]


dirtyObject = DirtySpotAugmentation(original_img_obj, dirty_spots)


def test_dirty_augmentation():
    """Test the dirty augmentation function."""
    # Apply dirty augmentation
    dirty_aug_img = dirtyObject.apply_dirty()

    expected_dirty_save_path = Path(
        "./tests/augmentation/data/expected_dirty_image/expected_dirty_image.png"
    )

    # Create a temporary directory for storing the actual output

    with tempfile.TemporaryDirectory() as tmpdirname:
        # Save the actual dirtyed image to the temporary directory
        actual_dirty_save_path = tmpdirname + "/actual_dirty_augmented_image.png"
        dirty_aug_img.save(actual_dirty_save_path)

        # Open the expected and actual images
        expected_dirty_aug_img = Image.open(expected_dirty_save_path)
        actual_dirty_aug_img = Image.open(actual_dirty_save_path)

        assert expected_dirty_aug_img.size == actual_dirty_aug_img.size
        assert expected_dirty_aug_img.mode == actual_dirty_aug_img.mode
