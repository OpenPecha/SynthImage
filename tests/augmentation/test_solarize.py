from pathlib import Path

from PIL import Image

from SynthImage.solarize_augmentation import SolarizeAugmentation

original_img_path = Path("./tests/page_image/data/expected_page_image.png")

original_img_obj = Image.open(original_img_path)

solarizeObject = SolarizeAugmentation(original_img_obj)


def test_solarize():
    """Test the solarize augmentation function."""
    # Apply solarize augmentation
    solarize_aug_img = solarizeObject.apply_solarize()
    # Define the expected save directory
    expected_solarize_save_dir = Path(
        "./tests/augmentation/data/expected_solarize_image"
    )
    expected_solarize_save_dir.mkdir(parents=True, exist_ok=True)

    expected_solarize_save_path = (
        Path(expected_solarize_save_dir) / "expected_solarize_image.png"
    )
    solarize_aug_img.save(expected_solarize_save_path)
