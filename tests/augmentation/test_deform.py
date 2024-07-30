from pathlib import Path

from PIL import Image, ImageChops

from SynthImage.deform_augmentation import DeformAugmentation

original_img_obj_dir = (
    Path(__file__).parent / ".." / "page_image" / "data" / "expected_page_output"
)
original_img_obj_path = str(original_img_obj_dir / "expected_page_image.png")
original_img_obj = Image.open(original_img_obj_path)

deformObject = DeformAugmentation(original_img_obj, 30, 3, 90)


def test_deform_augmentation():
    deform_aug_img = deformObject.apply_deform()
    expected_deform_aug_img_obj_dir = (
        Path(__file__).parent
        / "data"
        / "aug_output"
        / "expected_deform_aug_image_output"
    )
    expected_deform_save_path = (
        expected_deform_aug_img_obj_dir / "expected_deform_augmented_image.png"
    )
    actual_deform_aug_img_obj_dir = (
        Path(__file__).parent / "data" / "aug_output" / "actual_deform_aug_image_output"
    )
    actual_deform_aug_img_obj_dir.mkdir(parents=True, exist_ok=True)
    actual_deform_save_path = (
        actual_deform_aug_img_obj_dir / "actual_deform_augmented_image.png"
    )
    deform_aug_img.save(actual_deform_save_path)
    expected_deform_aug_img = Image.open(expected_deform_save_path)
    actual_deform_aug_img = Image.open(actual_deform_save_path)
    assert is_same_img(actual_deform_aug_img, expected_deform_aug_img)
    Path(actual_deform_save_path).unlink()


def is_same_img(img1, img2):
    if img1.size != img2.size:
        return False
    diff = ImageChops.difference(img1, img2)
    if diff.getbbox():
        return False
    return True
