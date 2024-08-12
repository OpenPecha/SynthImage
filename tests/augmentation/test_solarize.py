import tempfile
from pathlib import Path

from PIL import Image

from SynthImage.Augmentation.solarize_augmentation import SolarizeAugmentation

original_img_path = Path("./tests/page_image/data/expected_page_image.png")

original_img_obj = Image.open(original_img_path)

solarizeObject = SolarizeAugmentation(original_img_obj)


def test_solarize():
    """Test the solarize augmentation function."""
    # Apply solarize augmentation
    solarize_aug_img = solarizeObject.apply_solarize()
    # Define the expected save directory
    expected_solarize_save_path = Path(
        "./tests/augmentation/data/expected_solarize_image/expected_solarize_image.png"
    )
    with tempfile.TemporaryDirectory() as tempdirname:
        actual_solarize_save_path = Path(tempdirname) / "actual_solarize_image.png"
        solarize_aug_img.save(actual_solarize_save_path)
        expected_solarize_image = Image.open(expected_solarize_save_path)
        actual_solarize_image = Image.open(actual_solarize_save_path)
        assert expected_solarize_image.size == actual_solarize_image.size
