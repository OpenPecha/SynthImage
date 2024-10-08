import albumentations as A
import numpy as np
from PIL import Image


class BlurAugmentation:
    def __init__(self, original_img_obj):
        """Initialize the BlurAugmentation class with an image.

        Args:
            original_img_obj (PIL.Image.Image): The input image to be augmented.
        """
        self.original_img_obj = original_img_obj

    def apply_blur(self):
        """Apply a blur effect to the image.

        Returns:
            PIL.Image.Image: The blurred image.
        """
        aug = A.Blur()
        aug_img = aug(image=np.array(self.original_img_obj))["image"]
        return Image.fromarray(aug_img)

    def apply_median_blur(self):
        """Apply a median blur effect to the image.

        Returns:
            PIL.Image.Image: The median blurred image.
        """
        aug = A.MedianBlur()
        aug_img = aug(image=np.array(self.original_img_obj))["image"]
        return Image.fromarray(aug_img)

    def apply_motion_blur(self):
        """Apply a motion blur effect to the image.

        Returns:
            PIL.Image.Image: The motion blurred image.
        """
        aug = A.MotionBlur()
        aug_img = aug(image=np.array(self.original_img_obj))["image"]
        return Image.fromarray(aug_img)
