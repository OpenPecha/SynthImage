import tempfile
from pathlib import Path

from PIL import Image

from SynthImage.Augmentation.ink_bleed_augmentation import InkBleedAugmentation

original_img_path = Path("./tests/page_image/data/expected_page_image.png")

original_img_obj = Image.open(original_img_path)

inkBleedObject = InkBleedAugmentation(original_img_obj)


def test_ink_bleed():
    """Test the ink bleed augmentation function."""

    # Apply the ink bleed augmentation
    ink_bleed_aug_img = inkBleedObject.apply_ink_bleed()
    expected_ink_bleed_path = Path(
        "./tests/augmentation/data/expected_ink_bleed_image/expected_ink_bleed_image.png"
    )
    with tempfile.TemporaryDirectory() as tempdirname:
        actual_ink_bleed_path = Path(tempdirname) / "actual_ink_bleed_image.png"
        ink_bleed_aug_img.save(actual_ink_bleed_path)
        expected_ink_bleed_image = Image.open(expected_ink_bleed_path)
        actual_ink_bleed_image = Image.open(actual_ink_bleed_path)
        assert expected_ink_bleed_image.size == actual_ink_bleed_image.size
