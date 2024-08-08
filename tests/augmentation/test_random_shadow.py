from pathlib import Path

from PIL import Image

from SynthImage.random_shadow_augmentation import RandomShadowAugmentation

original_img_path = Path("./tests/page_image/data/expected_page_image.png")

original_img_obj = Image.open(original_img_path)

shadowObject = RandomShadowAugmentation(original_img_obj)


def test_random_shadow():
    """Test the random shadow augmentation function."""
    # Apply random shadow augmentation
    shadow_aug_img = shadowObject.apply_random_shadow()
    # Define the expected save directory
    expected_shadow_save_dir = Path("./tests/augmentation/data/expected_shadow_image")
    expected_shadow_save_dir.mkdir(parents=True, exist_ok=True)

    expected_shadow_save_path = (
        Path(expected_shadow_save_dir) / "expected_shadow_image.png"
    )
    shadow_aug_img.save(expected_shadow_save_path)
