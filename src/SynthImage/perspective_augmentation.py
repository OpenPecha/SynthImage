import albumentations as A
import numpy as np
from PIL import Image


class PerspectiveAugmentation:
    def __init__(self, original_img_obj):
        """Initialize the PerspectiveAugmentation class with an image.

        Args:
            original_img_obj (PIL.Image.Image): The input image to be augmented.
        """
        self.original_img_obj = original_img_obj

    def apply_perspective(self):
        """Apply perspective to the image.

        Returns:
            PIL.Image.Image: The perspective applied to the image
        """
        aug = A.Perspective(p=1)
        aug_img = aug(image=np.array(self.original_img_obj))["image"]
        return Image.fromarray(aug_img)
