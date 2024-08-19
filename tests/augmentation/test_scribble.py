import tempfile
from pathlib import Path

from PIL import Image

from SynthImage.Augmentation.scribble_augmentation import ScribbleAugmentation

original_img_path = Path("./tests/page_image/data/expected_page_image.png")

original_img_obj = Image.open(original_img_path)

scribbleObject = ScribbleAugmentation(original_img_obj)


def test_scribble():
    """Test the scribble augmentation function."""

    # Apply the scribble augmentation
    scribble_aug_img = scribbleObject.apply()
    expected_scribble_path = Path(
        "./tests/augmentation/data/expected_scribble_image/expected_scribble_image.png"
    )
    with tempfile.TemporaryDirectory() as tempdirname:
        actual_scribble_path = Path(tempdirname) / "actual_scribble_image.png"
        scribble_aug_img.save(actual_scribble_path)
        expected_scribble_image = Image.open(expected_scribble_path)
        actual_scribble_image = Image.open(actual_scribble_path)
        assert expected_scribble_image.size == actual_scribble_image.size
