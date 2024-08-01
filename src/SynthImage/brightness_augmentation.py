from PIL import ImageEnhance


class BrightnessAugmentation:
    def __init__(self, original_img_obj, factor: float = 1.1):
        self.original_img_obj = original_img_obj
        self.factor = factor

    def apply_brightness(self):
        aug_img = self.original_img_obj
        aug_factor = self.factor
        enhancer = ImageEnhance.Brightness(aug_img)
        aug_img = enhancer.enhance(aug_factor)
        return aug_img
