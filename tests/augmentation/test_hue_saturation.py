from pathlib import Path

from PIL import Image

from SynthImage.hue_saturation_augmentation import HueSaturationAugmentation

original_img_path = Path("./tests/page_image/data/expected_page_image.png")

original_img_obj = Image.open(original_img_path)

hueSatObject = HueSaturationAugmentation(original_img_obj)


def test_hue_saturation():
    """Test the hue saturation augmentation function."""
    # Apply hue saturation augmentation
    hue_sat_aug_img = hueSatObject.apply_hue_saturation()
    # Define the expected save directory
    expected_hue_sat_save_dir = Path(
        "./tests/augmentation/data/expected_hue_saturation_image"
    )
    expected_hue_sat_save_dir.mkdir(parents=True, exist_ok=True)

    expected_hue_sat_save_path = (
        Path(expected_hue_sat_save_dir) / "expected_hue_saturation_image.png"
    )
    hue_sat_aug_img.save(expected_hue_sat_save_path)
