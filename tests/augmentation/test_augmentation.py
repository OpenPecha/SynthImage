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
    deform_aug_img = agObject.apply_deform(original_img_obj, 30, 3, 90)
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


def test_distort_augmentation():
    distort_aug_img = agObject.apply_distort(original_img_obj, 1, 200, 100, 1, 2, 4)
    expected_distort_aug_img_obj_dir = (
        Path(__file__).parent
        / "data"
        / "aug_output"
        / "expected_distort_aug_image_output"
    )
    expected_distort_save_path = (
        expected_distort_aug_img_obj_dir / "expected_distort_augmented_image.png"
    )
    actual_distort_aug_img_obj_dir = (
        Path(__file__).parent
        / "data"
        / "aug_output"
        / "actual_distort_aug_image_output"
    )
    actual_distort_aug_img_obj_dir.mkdir(parents=True, exist_ok=True)
    actual_distort_save_path = (
        actual_distort_aug_img_obj_dir / "actual_distort_augmented_image.png"
    )
    distort_aug_img.save(actual_distort_save_path)
    expected_distort_aug_img = Image.open(expected_distort_save_path)
    actual_distort_aug_img = Image.open(actual_distort_save_path)
    assert is_same_img(actual_distort_aug_img, expected_distort_aug_img)


def is_same_img(img1, img2):
    if img1.size != img2.size:
        return False
    diff = ImageChops.difference(img1, img2)
    if diff.getbbox():
        return False
    return True


if __name__ == "__main__":
    test_distort_augmentation()
