import albumentations as A
import numpy as np
from PIL import Image


class HueSaturationAugmentation:
    def __init__(self, original_img_obj):
        """Initialize the HueSaturatiomAugmentation class with an image.

        Args:
            original_img_obj (PIL.Image.Image): The input image to be augmented.
        """
        self.original_img_obj = original_img_obj

    def apply_hue_saturation(self):
        """Apply hue saturation to the image.

        Returns:
            PIL.Image.Image: The hue saturation applied image.
        """
        aug = A.HueSaturationValue()
        aug_img = aug(image=np.array(self.original_img_obj))["image"]
        return Image.fromarray(aug_img)
