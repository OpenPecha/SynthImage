import numpy as np
from augraphy.augmentations import Scribbles
from PIL import Image


class ScribbleAugmentation:
    def __init__(self, original_img_obj):
        """Initialize the ScribbleAugmentation class with an image.

        Args:
            original_img_obj (PIL.Image.Image): The input image to be augmented.
        """
        self.original_img_obj = original_img_obj

    def apply_scribble(self):
        """Apply a scribbles effect to the image.

        Returns:
            PIL.Image.Image: The scribbled image.
        """
        # Convert the image to numpy array
        image_array = np.array(self.original_img_obj)

        # Create the Scribbles augmentation object
        aug = Scribbles(p=1)

        # Apply the scribbles augmentation
        aug_img = aug(image=image_array)

        # Convert the augmented image to uint8
        aug_img = np.clip(aug_img, 0, 255).astype(np.uint8)
        # Convert numpy array back to PIL Image
        return Image.fromarray(aug_img)
