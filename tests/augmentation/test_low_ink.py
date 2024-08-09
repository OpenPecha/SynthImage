import tempfile
from pathlib import Path

from PIL import Image

from SynthImage.low_ink_augmentation import LowInkPeriodicLinesAugmentation

original_img_path = Path("./tests/page_image/data/expected_page_image.png")

original_img_obj = Image.open(original_img_path)

paperTextureObject = LowInkPeriodicLinesAugmentation(original_img_obj)


def test_low_ink_periodic_lines():
    """Test the low ink periodic lines augmentation function."""

    # Apply the low ink periodic lines augmentation
    low_ink_periodic_lines_aug_img = paperTextureObject.apply_low_ink_periodic_lines()
    expected_low_ink_periodic_lines_path = Path(
        "./tests/augmentation/data/expected_low_ink_periodic_lines_image/expected_low_ink_periodic_lines_image.png"
    )
    with tempfile.TemporaryDirectory() as tempdirname:
        actual_low_ink_periodic_lines_path = (
            Path(tempdirname) / "actual_low_ink_periodic_lines_image.png"
        )
        low_ink_periodic_lines_aug_img.save(actual_low_ink_periodic_lines_path)
        expected_low_ink_periodic_lines_image = Image.open(
            expected_low_ink_periodic_lines_path
        )
        actual_low_ink_periodic_lines_image = Image.open(
            actual_low_ink_periodic_lines_path
        )
        assert (
            expected_low_ink_periodic_lines_image.size
            == actual_low_ink_periodic_lines_image.size
        )
