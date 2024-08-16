from PIL import ImageEnhance


class BrightnessAugmentation:
    def __init__(self, original_img_obj, factor: float = 1.1):
        """Initialize the BrightnessAugmentations object.

        Args:
            original_img_obj (PIL.Image.Image): The input image to be augmented.
            factor (float, optional):The factor by which to adjust the brightness.
                                      A factor of 1.0 means no change, greater than 1.0 increases brightness,
                                      and less than 1.0 decreases brightness. Defaults to 1.1.
        """
        self.original_img_obj = original_img_obj
        self.factor = factor

    def apply_brightness(self):
        """Apply brightness augmentation to the input image.

        Returns:
           PIL.Image.Image: The augmented image with adjusted brightness.
        """
        aug_img = self.original_img_obj
        aug_factor = self.factor
        enhancer = ImageEnhance.Brightness(aug_img)
        aug_img = enhancer.enhance(aug_factor)
        return aug_img
