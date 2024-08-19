import tempfile
from pathlib import Path

from PIL import Image

from SynthImage.Augmentation.water_mark_augmentation import WaterMarkAugmentation

original_img_path = Path("./tests/page_image/data/expected_page_image.png")

original_img_obj = Image.open(original_img_path)

waterMarkObject = WaterMarkAugmentation(original_img_obj)


def test_water_mark():
    """Test the water mark augmentation function."""

    # Apply the water mark augmentation
    water_mark_aug_img = waterMarkObject.apply()
    expected_water_mark_path = Path(
        "./tests/augmentation/data/expected_water_mark_image/expected_water_mark_image.png"
    )
    with tempfile.TemporaryDirectory() as tempdirname:
        actual_water_mark_path = Path(tempdirname) / "actual_water_mark_image.png"
        water_mark_aug_img.save(actual_water_mark_path)
        expected_water_mark_image = Image.open(expected_water_mark_path)
        actual_water_mark_image = Image.open(actual_water_mark_path)
        assert expected_water_mark_image.size == actual_water_mark_image.size
