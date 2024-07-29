from pathlib import Path

from PIL import Image, ImageChops

from SynthImage.augmentation import Augmentation

original_img_obj_dir = (
    Path(__file__).parent / ".." / "page_image" / "data" / "expected_page_output"
)
original_img_obj_path = str(original_img_obj_dir / "expected_page_image.png")
original_img_obj = Image.open(original_img_obj_path)

agObject = Augmentation(original_img_obj, False)


def test_deform_augmentation():
    aug_img = agObject.apply_deform(30, 3, 90)
    expected_aug_img_obj_dir = (
        Path(__file__).parent / "data" / "aug_output" / "expected_aug_image_output"
    )
    expected_save_path = expected_aug_img_obj_dir / "expected_augmented_image.png"
    actual_aug_img_obj_dir = (
        Path(__file__).parent / "data" / "aug_output" / "actual_aug_image_output"
    )
    actual_aug_img_obj_dir.mkdir(parents=True, exist_ok=True)
    actual_save_path = actual_aug_img_obj_dir / "actual_augmented_image.png"
    aug_img.save(actual_save_path)
    expected_aug_img = Image.open(expected_save_path)
    actual_aug_img = Image.open(actual_save_path)
    assert images_are_equal(actual_aug_img, expected_aug_img)
    Path(actual_save_path).unlink()


def images_are_equal(img1, img2):
    if img1.size != img2.size:
        return False
    diff = ImageChops.difference(img1, img2)
    if diff.getbbox():
        return False
    return True
