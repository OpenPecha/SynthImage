import random

from PIL import ImageEnhance


class SharpnessAugmentation:
    def __init__(self, original_img_obj, factor: float = None):
        """Initialize the SharpnessAugmentation object.

        Args:
            original_img_obj (PIL.Image.Image): The input image to be augmented.
            factor (float, optional): The factor by which to adjust the sharpness. Defaults to None.
        """
        self.original_img_obj = original_img_obj
        self.factor = factor

    def apply_sharpness(self):
        """Apply sharpness augmentation to the input image.
            If the factor was not provided during initialization, a random factor
            between 0.7 and 1.3 will be generated.

        Returns:
            PIL.Image.Image: The image with adjusted sharpness.
        """
        aug_img = self.original_img_obj
        if self.factor is None:
            self.factor = random.uniform(0.7, 1.3)
        enhancer = ImageEnhance.Sharpness(aug_img)
        aug_img = enhancer.enhance(self.factor)
        return aug_img
