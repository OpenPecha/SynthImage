from pathlib import Path

from PIL import Image

from SynthImage.rusting_augmentation import RustingAugmentation

original_img_path = Path("./tests/page_image/data/expected_page_image.png")

original_img_obj = Image.open(original_img_path)

rustingObject = RustingAugmentation(original_img_obj)


def test_rusting():
    """Test the rusting augmentation function."""
    # Apply rusting augmentation
    rusting_aug_img = rustingObject.apply_rusting()
    # Define the expected save directory
    expected_rusting_save_dir = Path("./tests/augmentation/data/expected_rusting_image")
    expected_rusting_save_dir.mkdir(parents=True, exist_ok=True)

    expected_rusting_save_path = (
        Path(expected_rusting_save_dir) / "expected_rusting_image.png"
    )
    rusting_aug_img.save(expected_rusting_save_path)
