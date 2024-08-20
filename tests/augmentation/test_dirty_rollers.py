import tempfile
from pathlib import Path

from PIL import Image

from SynthImage.Augmentation.dirty_rollers_augmentation import DirtyRollersAugmentation

original_img_path = Path("./tests/page_image/data/expected_page_image.png")

original_img_obj = Image.open(original_img_path)

dirtyRollersObject = DirtyRollersAugmentation(original_img_obj)


def test_dirty_rollers():
    """Test the dirty rollers augmentation function."""

    # Apply the dirty rollers augmentation
    dirty_rollers_aug_img = dirtyRollersObject.apply_dirty_rollers()
    expected_dirty_rollers_path = Path(
        "./tests/augmentation/data/expected_dirty_rollers_image/expected_dirty_rollers_image.png"
    )
    with tempfile.TemporaryDirectory() as tempdirname:
        actual_dirty_rollers_path = Path(tempdirname) / "actual_dirty_rollers_image.png"
        dirty_rollers_aug_img.save(actual_dirty_rollers_path)
        expected_dirty_rollers_image = Image.open(expected_dirty_rollers_path)
        actual_dirty_rollers_image = Image.open(actual_dirty_rollers_path)
        assert expected_dirty_rollers_image.size == actual_dirty_rollers_image.size
