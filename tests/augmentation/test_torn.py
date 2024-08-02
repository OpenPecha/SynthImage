import tempfile
from pathlib import Path

from PIL import Image

from SynthImage.torn_augmentation import TornAugmentation

# Define the path to the original image
original_img_path = Path("./tests/page_image/data/expected_page_image.png")

# Open the original image
original_img_obj = Image.open(original_img_path)

# Create the TornAugmentation object
tornObject = TornAugmentation(original_img_obj, 7, 40)


def test_torn_augmentation():
    """Tests the TornAugmentation class by applying the torn effect to an image and comparing it to an expected output image."""  # noqa
    # Apply torn augmentation
    torn_aug_img = tornObject.apply_torn()

    expected_torn_save_path = Path(
        "./tests/augmentation/data/expected_torn_image/expected_torn_image.png"  # noqa
    )
    # Create a temporary directory for storing the actual output
    with tempfile.TemporaryDirectory() as tmpdirname:

        # Save the actual torned image to the temporary directory
        actual_torn_save_path = tmpdirname + "/actual_torn_augmented_image.png"
        torn_aug_img.save(actual_torn_save_path)

        # Open the expected and actual images
        expected_torn_aug_img = Image.open(expected_torn_save_path)
        actual_torn_aug_img = Image.open(actual_torn_save_path)

        assert expected_torn_aug_img.size == actual_torn_aug_img.size
        assert expected_torn_aug_img.mode == actual_torn_aug_img.mode
