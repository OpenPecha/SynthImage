from pathlib import Path

from PIL import Image

from SynthImage.augmentation import PageAugmentationGenerator

page_image_dir = (
    Path(__file__).parent / ".." / "page_image" / "data" / "expected_page_output"
)
generated_augmented_output_dir = (
    Path(__file__).parent / "generated_augmented_page_output"
)

generated_augmented_output_dir.mkdir(parents=True, exist_ok=True)
generated_augmented_path = str(generated_augmented_output_dir)
text_folder = Path(__file__).parent / "data" / "text_folder"
page_image_path = str(page_image_dir / "expected_page_image.png")
page_image = Image.open(page_image_path)
augObj = PageAugmentationGenerator(text_folder, generated_augmented_path)


def test_apply_brightness():
    aug_img, factor = augObj.augment_brightness(page_image)

    # Define the save directory and ensure it exists
    save_dir = Path(__file__).parent / "output_brightness"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = str(save_dir / "augmented_brightness_image.png")
    # Save the augmented image
    aug_img.save(save_path)
    print(factor)
    # Path(save_path).unlink()


def test_apply_augmentation():
    aug_img, rotation_angle = augObj.apply_augmentation(page_image)
    # Define the save directory and ensure it exists
    aug_save_dir = Path(__file__).parent / "output_apply_augmentation"
    aug_save_dir.mkdir(parents=True, exist_ok=True)
    aug_save_path = str(aug_save_dir / "apply_augmented_image.png")
    aug_img.save(aug_save_path)
    print(rotation_angle)


def test_generate_augmented_page_and_line():

    augObj.generate_augmented_page_and_line()


if __name__ == "__main__":
    test_apply_brightness()
    test_apply_augmentation()
    test_generate_augmented_page_and_line()
