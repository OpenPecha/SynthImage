import albumentations as A
import numpy as np
from PIL import Image


class RandomRainAugmentation:
    def __init__(self, original_img_obj):
        """Initialize the RandomRainAugmentation class with an image.

        Args:
            original_img_obj (PIL.Image.Image): The input image to be augmented.
        """
        self.original_img_obj = original_img_obj

    def apply_random_rain(self):
        """Apply random rain to the image.

        Returns:
            PIL.Image.Image: The image with random rain applied.
        """
        aug = A.RandomRain()
        aug_img = aug(image=np.array(self.original_img_obj))["image"]
        return Image.fromarray(aug_img)
