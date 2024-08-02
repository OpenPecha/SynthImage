import random

from PIL import ImageEnhance


class ContrastAugmentation:
    def __init__(self, original_img_obj, factor: float = None):
        """Initialize the ContrastAugmentation object.

        Args:
            original_img_obj (PIL.Image.Image): The input image to be augmented.
            factor (float, optional): The factor by which to adjust the contrast. Defaults to None.
        """
        self.original_img_obj = original_img_obj
        self.factor = factor

    def apply_contrast(self):
        """Apply contrast augmentation to the input image.

        Returns:
            PIL.Image.Image: The image with adjusted contrast.
        """
        aug_img = self.original_img_obj
        if self.factor is None:
            self.factor = random.uniform(0.7, 1.3)
        enhancer = ImageEnhance.Contrast(aug_img)
        aug_img = enhancer.enhance(self.factor)
        return aug_img
