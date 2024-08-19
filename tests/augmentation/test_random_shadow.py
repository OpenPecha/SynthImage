import tempfile
from pathlib import Path

from PIL import Image

from SynthImage.Augmentation.random_shadow_augmentation import RandomShadowAugmentation

original_img_path = Path("./tests/page_image/data/expected_page_image.png")

original_img_obj = Image.open(original_img_path)

shadowObject = RandomShadowAugmentation(original_img_obj)


def test_random_shadow():
    """Test the random shadow augmentation function."""
    # Apply random shadow augmentation
    shadow_aug_img = shadowObject.apply()
    # Define the expected save directory
    expected_shadow_save_path = Path(
        "./tests/augmentation/data/expected_shadow_image/expected_shadow_image.png"
    )
    with tempfile.TemporaryDirectory() as tempdirname:
        actual_shadow_save_path = Path(tempdirname) / "actual_shadow_image.png"
        shadow_aug_img.save(actual_shadow_save_path)
        expected_shadow_image = Image.open(expected_shadow_save_path)
        actual_shadow_image = Image.open(actual_shadow_save_path)
        assert expected_shadow_image.size == actual_shadow_image.size
