import albumentations as A
import numpy as np
from PIL import Image


class SolarizeAugmentation:
    def __init__(self, original_img_obj):
        """Initialize the SolarizeAugmentation class with an image.

        Args:
            original_img_obj (PIL.Image.Image): The input image to be augmented.
        """
        self.original_img_obj = original_img_obj

    def apply(self):
        """Apply solarize to the image.

        Returns:
            PIL.Image.Image: The image with solarize applied.
        """
        aug = A.Solarize()
        aug_img = aug(image=np.array(self.original_img_obj))["image"]
        return Image.fromarray(aug_img)
