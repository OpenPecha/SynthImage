import tempfile
from pathlib import Path

from PIL import Image

from SynthImage.bad_photocopy_augmentation import BadPhotoCopyAugmentation

original_img_path = Path("./tests/page_image/data/expected_page_image.png")

original_img_obj = Image.open(original_img_path)

badPhotoCopyObject = BadPhotoCopyAugmentation(original_img_obj)


def test_bad_photocopy():
    """Test the bad photocopy augmentation function."""

    # Apply the bad photocopy augmentation
    bad_photocopy_aug_img = badPhotoCopyObject.apply_bad_photocopy()
    expected_bad_photocopy_path = Path(
        "./tests/augmentation/data/expected_bad_photocopy_image/expected_bad_photocopy_image.png"
    )
    with tempfile.TemporaryDirectory() as tempdirname:
        actual_bad_photocopy_path = Path(tempdirname) / "actual_bad_photocopy_image.png"
        bad_photocopy_aug_img.save(actual_bad_photocopy_path)
        expected_bad_photocopy_image = Image.open(expected_bad_photocopy_path)
        actual_bad_photocopy_image = Image.open(actual_bad_photocopy_path)
        assert expected_bad_photocopy_image.size == actual_bad_photocopy_image.size
