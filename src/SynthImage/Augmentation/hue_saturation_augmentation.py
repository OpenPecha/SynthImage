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

    def apply(self):
        """Apply random shadow to the image.

        Returns:
            PIL.Image.Image: The image with random shadow applied.
        """
        # Convert the image to RGB if it's not already
        if self.original_img_obj.mode != "RGB":
            img_rgb = self.original_img_obj.convert("RGB")
        else:
            img_rgb = self.original_img_obj

        # Convert the image to a NumPy array
        img_np = np.array(img_rgb)

        # Apply the RandomShadow augmentation
        aug = A.RandomShadow()
        aug_img = aug(image=img_np)["image"]

        # Convert the augmented NumPy array back to a PIL Image
        return Image.fromarray(aug_img)
