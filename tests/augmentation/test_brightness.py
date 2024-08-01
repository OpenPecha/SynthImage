from pathlib import Path

from PIL import Image

from SynthImage.brightness_augmentation import BrightnessAugmentations

original_img_obj_path = (
    Path(__file__).parent
    / "../../tests/page_image/data/expected_page_output/expected_page_image.png"
)

original_img_obj = Image.open(original_img_obj_path)

brObject = BrightnessAugmentations(original_img_obj, 1.2)


def test_brightness_augmentation():
    # Apply brightness augmentation
    brightness_aug_img = brObject.apply_brightness()

    # Define the expected save directory
    expected_brightness_save_dir = (
        Path(__file__).parent
        / "../../tests/augmentation/data/aug_output/expected_brightness_aug_image_output"
    )

    # Create the directory if it doesn't exist
    expected_brightness_save_dir.mkdir(parents=True, exist_ok=True)

    # Define the expected save path
    expected_brightness_save_path = (
        expected_brightness_save_dir / "expected_brightness_augmented_image.png"
    )

    brightness_aug_img.save(expected_brightness_save_path)
