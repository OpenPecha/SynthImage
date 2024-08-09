import numpy as np
from augraphy.augmentations import Scribbles
from PIL import Image


class ScribbleAugmentation:
    def __init__(self, original_img_obj):
        """Initialize the ScribblesAugmentation class with an image.

        Args:
            original_img_obj (PIL.Image.Image): The input image to be augmented.
        """
        self.original_img_obj = original_img_obj

    def apply_scribble(self):
        """Apply a scribbles effect to the image.

        Returns:
            PIL.Image.Image: The scribbled image.
        """
        aug = Scribbles(p=1)
        image_array = np.array(self.original_img_obj)

        # Apply augmentation
        aug_img = aug(image=image_array)
        aug_img = aug_img.astype(np.uint8)

        return Image.fromarray(aug_img)
