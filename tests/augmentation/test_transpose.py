import tempfile
from pathlib import Path

from PIL import Image

from SynthImage.transpose_augmentation import TransposeAugmentation

original_img_path = Path("./tests/page_image/data/expected_page_image.png")

original_img_obj = Image.open(original_img_path)

transposeObject = TransposeAugmentation(original_img_obj)


def test_transpose():
    """Test the transpose augmentation function."""
    # Apply transpose augmentation
    transpose_aug_img = transposeObject.apply_transpose()
    # Define the expected save directory
    expected_transpose_save_path = Path(
        "./tests/augmentation/data/expected_transpose_image/expected_transpose_image.png"
    )
    with tempfile.TemporaryDirectory() as tempdirname:
        actual_transpose_save_path = Path(tempdirname) / "actual_transpose_image.png"
        transpose_aug_img.save(actual_transpose_save_path)
        expected_transpose_image = Image.open(expected_transpose_save_path)
        actual_transpose_image = Image.open(actual_transpose_save_path)
        assert expected_transpose_image.size == actual_transpose_image.size
