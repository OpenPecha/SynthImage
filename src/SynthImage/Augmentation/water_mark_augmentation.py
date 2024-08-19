import numpy as np
from augraphy.augmentations import WaterMark
from PIL import Image


class WaterMarkAugmentation:
    def __init__(self, original_img_obj):
        """Initialize the WaterMarkAugmentation class with an image.

        Args:
            original_img_obj (PIL.Image.Image): The input image to be augmented.
        """
        self.original_img_obj = original_img_obj

    def apply(self):
        """Apply a water mark effect to the image.

        Returns:
            PIL.Image.Image: The water marked image.
        """
        aug = WaterMark()
        image_array = np.array(self.original_img_obj)

        # Apply augmentation
        aug_img = aug(image=image_array)

        return Image.fromarray(aug_img)
