import tempfile
from pathlib import Path

from PIL import Image

from SynthImage.Augmentation.sun_flare_augmentation import SunFlareAugmentation

original_img_path = Path("./tests/page_image/data/expected_page_image.png")

original_img_obj = Image.open(original_img_path)

sun_flare_Object = SunFlareAugmentation(original_img_obj)


def test_sun_flare():
    """Test the sun flare augmentation function."""
    # Apply sun flare augmentation
    sun_flare_aug_img = sun_flare_Object.apply_sun_flare()
    # Define the expected save directory
    expected_sun_flare_save_path = Path(
        "./tests/augmentation/data/expected_sun_flare_image/expected_sun_flare_image.png"
    )
    with tempfile.TemporaryDirectory() as tempdirname:
        actual_sun_flare_save_path = Path(tempdirname) / "actual_sun_flare_image.png"
        sun_flare_aug_img.save(actual_sun_flare_save_path)
        expected_sun_flare_image = Image.open(expected_sun_flare_save_path)
        actual_sun_flare_image = Image.open(actual_sun_flare_save_path)
        assert expected_sun_flare_image.size == actual_sun_flare_image.size
