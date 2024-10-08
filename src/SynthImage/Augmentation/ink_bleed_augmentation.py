import numpy as np
from augraphy.augmentations import InkBleed
from PIL import Image


class InkBleedAugmentation:
    def __init__(self, original_img_obj):
        """Initialize the InkBleedAugmentation class with an image.

        Args:
            original_img_obj (PIL.Image.Image): The input image to be augmented.
        """
        self.original_img_obj = original_img_obj

    def apply_ink_bleed(self):
        """Apply an ink bleed effect to the image.

        Returns:
            PIL.Image.Image: The ink bled image.
        """
        aug = InkBleed()
        image_array = np.array(self.original_img_obj)

        # Apply augmentation
        aug_img = aug(image=image_array)

        return Image.fromarray(aug_img)
