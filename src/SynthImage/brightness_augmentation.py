import random

from PIL import ImageEnhance


class BrightnessAugmentations:
    def __init__(self, original_img_obj, factor: float = 1.1, random: bool = False):
        self.original_img_obj = original_img_obj
        self.factor = factor
        self.random = random

    def apply_brightness(self):
        aug_img = self.original_img_obj
        if self.random is True:
            aug_factor = random.uniform(0.7, 1.3)
        else:
            aug_factor = self.factor
        enhancer = ImageEnhance.Brightness(aug_img)
        aug_img = enhancer.enhance(aug_factor)
        return aug_img
