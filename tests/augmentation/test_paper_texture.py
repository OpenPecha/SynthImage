import tempfile
from pathlib import Path

from PIL import Image

from SynthImage.Augmentation.paper_texture_augmentation import PaperTextureAugmentation

original_img_path = Path("./tests/page_image/data/expected_page_image.png")

original_img_obj = Image.open(original_img_path)

paperTextureObject = PaperTextureAugmentation(original_img_obj)


def test_paper_texture():
    """Test the paper texture augmentation function."""

    # Apply the paper texture augmentation
    paper_texture_aug_img = paperTextureObject.apply_paper_texture()
    expected_paper_texture_path = Path(
        "./tests/augmentation/data/expected_paper_texture_image/expected_paper_texture_image.png"
    )
    with tempfile.TemporaryDirectory() as tempdirname:
        actual_paper_texture_path = Path(tempdirname) / "actual_paper_texture_image.png"
        paper_texture_aug_img.save(actual_paper_texture_path)
        expected_paper_texture_image = Image.open(expected_paper_texture_path)
        actual_paper_texture_image = Image.open(actual_paper_texture_path)
        assert expected_paper_texture_image.size == actual_paper_texture_image.size
