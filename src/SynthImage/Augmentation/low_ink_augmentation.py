import numpy as np
from augraphy.augmentations import LowInkPeriodicLines
from PIL import Image


class LowInkPeriodicLinesAugmentation:
    def __init__(self, original_img_obj):
        """Initialize the LowInkPeriodicLinesAugmentation class with an image.

        Args:
            original_img_obj (PIL.Image.Image): The input image to be augmented.
        """
        self.original_img_obj = original_img_obj

    def apply(self):
        """Apply a low ink periodic lines effect to the image.

        Returns:
            PIL.Image.Image: The low ink periodic lines image.
        """
        aug = LowInkPeriodicLines(p=1)
        image_array = np.array(self.original_img_obj)

        # Apply augmentation
        aug_img = aug(image=image_array)

        return Image.fromarray(aug_img)
