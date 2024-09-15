import albumentations as A
import numpy as np
from PIL import Image


class FlipAugmentation:
    def __init__(self, original_img_obj):
        """Initialize the FlipAugmentation class with an image.

        Args:
            original_img_obj (PIL.Image.Image): The input image to be augmented.
        """
        self.original_img_obj = original_img_obj

    def apply_vertical_flip(self):
        """Apply vertical flip to the image.

        Returns:
            PIL.Image.Image: The vertical flip applied image.
        """
        aug = A.VerticalFlip()
        aug_img = aug(image=np.array(self.original_img_obj))["image"]
        return Image.fromarray(aug_img)

    def apply_horizontal_flip(self):
        """Apply horizontal flip to the image.

        Returns:
            PIL.Image.Image: The horizontal flip applied image.
        """
        aug = A.HorizontalFlip()
        aug_img = aug(image=np.array(self.original_img_obj))["image"]
        return Image.fromarray(aug_img)
