import numpy as np
from augraphy.augmentations import BadPhotoCopy
from PIL import Image


class BadPhotoCopyAugmentation:
    def __init__(self, original_img_obj):
        """Initialize the BadPhotoCopyAugmentation class with an image.

        Args:
            original_img_obj (PIL.Image.Image): The input image to be augmented.
        """
        self.original_img_obj = original_img_obj

    def apply(self):
        """Apply a bad photocopy effect to the image.

        Returns:
            PIL.Image.Image: The bad photocopy image.
        """
        # Convert the image to RGB if it's not already
        if self.original_img_obj.mode != "RGB":
            img_rgb = self.original_img_obj.convert("RGB")
        else:
            img_rgb = self.original_img_obj

        # Convert the image to a NumPy array with uint8 type
        image_array = np.array(img_rgb).astype(np.uint8)

        # Apply the BadPhotoCopy augmentation
        aug = BadPhotoCopy()
        augmented_image = aug(image=image_array)

        # Convert the augmented NumPy array back to a PIL Image
        return Image.fromarray(augmented_image)
