import tempfile
from pathlib import Path

from PIL import Image

from SynthImage.Augmentation.rusting_augmentation import RustingAugmentation

original_img_path = Path("./tests/page_image/data/expected_page_image.png")

original_img_obj = Image.open(original_img_path)

rustingObject = RustingAugmentation(original_img_obj)


def test_rusting():
    """Test the rusting augmentation function."""
    # Apply rusting augmentation
    rusting_aug_img = rustingObject.apply_rusting()
    # Define the expected save directory
    expected_rusting_save_path = Path(
        "./tests/augmentation/data/expected_rusting_image/expected_rusting_image.png"
    )
    with tempfile.TemporaryDirectory() as tempdirname:
        actual_rusting_save_path = Path(tempdirname) / "actual_rusting_image.png"
        rusting_aug_img.save(actual_rusting_save_path)
        expected_rusting_image = Image.open(expected_rusting_save_path)
        actual_rusting_image = Image.open(actual_rusting_save_path)
        assert expected_rusting_image.size == actual_rusting_image.size
