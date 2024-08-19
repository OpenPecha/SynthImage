import tempfile
from pathlib import Path

from PIL import Image

from SynthImage.Augmentation.faxify_augmentation import FaxifyAugmentation

original_img_path = Path("./tests/page_image/data/expected_page_image.png")

original_img_obj = Image.open(original_img_path)
faxifyObject = FaxifyAugmentation(original_img_obj)


def test_faxify():
    """Test the faxify augmentation function."""

    # Apply the faxify augmentation
    faxify_aug_img = faxifyObject.apply()
    expected_faxify_path = Path(
        "./tests/augmentation/data/expected_faxify_image/expected_faxify_image.png"
    )
    with tempfile.TemporaryDirectory() as tempdirname:
        actual_faxify_path = Path(tempdirname) / "actual_faxify_image.png"
        faxify_aug_img.save(actual_faxify_path)
        expected_faxify_image = Image.open(expected_faxify_path)
        actual_faxify_image = Image.open(actual_faxify_path)
        assert expected_faxify_image.size == actual_faxify_image.size
