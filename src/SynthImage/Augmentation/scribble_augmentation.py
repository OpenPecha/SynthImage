import numpy as np
from augraphy.augmentations import Scribbles
from PIL import Image


class ScribbleAugmentation:
    """
    Applies scribble-style augmentation to an input image using the Scribbles class from the Augraphy library.

    Attributes:
        original_img_obj (PIL.Image.Image): The input image to be augmented.

    Methods:
        apply():
            Applies the scribble-style augmentation to the input image and returns the augmented image.
    """

    def __init__(self, original_img_obj):
        """
        Initializes the ScribbleAugmentation class with an image object.

        Args:
            original_img_obj (PIL.Image.Image): The input image to be augmented.
        """
        self.original_img_obj = original_img_obj

    def apply(self):
        """
        Applies the scribble-style augmentation to the input image.

        The method performs the following steps:
            1. Converts the input PIL image to a numpy array.
            2. Ensures the numpy array is in uint8 format.
            3. Configures and applies scribble-style augmentation using the Scribbles class from Augraphy.
            4. Handles potential errors during augmentation and returns the original image if an error occurs.
            5. Ensures the augmented image is in uint8 format and converts it back to a PIL Image.

        Returns:
            PIL.Image.Image: The augmented image after applying the scribble-style effect.
        """
        # Convert the image to a numpy array
        image_array = np.array(self.original_img_obj)

        # Ensure the image array is in uint8 format
        if image_array.dtype != np.uint8:
            image_array = image_array.astype(np.uint8)

        # Configure Scribbles parameters if necessary
        aug = Scribbles()
        try:
            # Apply the scribbles augmentation
            aug_img = aug(image=image_array)
        except Exception as e:
            print(f"Error applying ScribbleAugmentation: {e}")
            return self.original_img_obj  # Return original image on error

        # Ensure the augmented image is a numpy array and in uint8 format
        if isinstance(aug_img, np.ndarray):
            # Clip values to the range [0, 255] and ensure type is uint8
            aug_img = np.clip(aug_img, 0, 255).astype(np.uint8)
        else:
            print("Error: Augmentation output is not a numpy array.")
            return self.original_img_obj  # Return original image on error

        # Convert numpy array back to PIL Image
        return Image.fromarray(aug_img)
