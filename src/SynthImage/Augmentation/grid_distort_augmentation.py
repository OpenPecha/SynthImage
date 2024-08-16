import albumentations as A
import numpy as np
from PIL import Image


class GridDistortAugmentation:
    def __init__(self, original_img_obj):
        """Initialize the GridDistortAugmentation class with an image.

        Args:
            original_img_obj (PIL.Image.Image): The input image to be augmented.
        """
        self.original_img_obj = original_img_obj

    def apply_grid_distort(self):
        """Apply grid distortion to the image.

        Returns:
            PIL.Image.Image: The grid distorted image.
        """
        aug = A.GridDistortion()
        aug_img = aug(image=np.array(self.original_img_obj))["image"]
        return Image.fromarray(aug_img)
