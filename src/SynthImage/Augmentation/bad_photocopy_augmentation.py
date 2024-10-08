import numpy as np
from augraphy.augmentations import BadPhotoCopy
from PIL import Image


class BadPhotoCopyAugmentation:
    def __init__(self, original_img_obj):
        """Initialize the BadPhtoCopyAugmentation class with an image.

        Args:
            original_img_obj (PIL.Image.Image): The input image to be augmented.
        """
        self.original_img_obj = original_img_obj

    def apply_bad_photocopy(self):
        """Apply a bad photocopy effect to the image.

        Returns:
            PIL.Image.Image: The bad photocopy image.
        """
        aug = BadPhotoCopy()
        image_array = np.array(self.original_img_obj)

        # Apply augmentation
        aug_img = aug(image=image_array)

        return Image.fromarray(aug_img)
