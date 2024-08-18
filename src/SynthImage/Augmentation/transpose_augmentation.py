import albumentations as A
import numpy as np
from PIL import Image


class TransposeAugmentation:
    def __init__(self, original_img_obj):
        """Initialize the TransposeAugmentation class with an image.

        Args:
            original_img_obj (PIL.Image.Image): The input image to be augmented.
        """
        self.original_img_obj = original_img_obj

    def apply(self):
        """Apply transpose to the image.

        Returns:
            PIL.Image.Image: The transpose applied to the image
        """
        aug = A.Transpose(p=1)
        aug_img = aug(image=np.array(self.original_img_obj))["image"]
        return Image.fromarray(aug_img)
