from pathlib import Path

from PIL import Image

from SynthImage.perspective_augmentation import PerspectiveAugmentation

original_img_path = Path("./tests/page_image/data/expected_page_image.png")

original_img_obj = Image.open(original_img_path)

perspectiveObject = PerspectiveAugmentation(original_img_obj)


def test_perspective():
    """Test the perspective augmentation function."""
    # Apply perspective augmentation
    perspective_aug_img = perspectiveObject.apply_perspective()
    # Define the expected save directory
    expected_perspective_save_dir = Path(
        "./tests/augmentation/data/expected_perspective_image"
    )
    expected_perspective_save_dir.mkdir(parents=True, exist_ok=True)

    expected_perspective_save_path = (
        Path(expected_perspective_save_dir) / "expected_perspective_image.png"
    )
    perspective_aug_img.save(expected_perspective_save_path)
