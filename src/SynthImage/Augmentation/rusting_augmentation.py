import albumentations as A
import numpy as np
from PIL import Image


class RustingAugmentation:
    def __init__(
        self,
        original_img_obj,
        gauss_noise_var_limit=(10.0, 50.0),
        ison_noise_color_shift=(0.01, 0.05),
        ison_noise_intensity=(0.1, 0.5),
        multiplicative_noise_multiplier=(0.9, 1.1),
        random_fog_coef_range=(0.1, 0.3),
        random_fog_alpha_coef=0.1,
        hue_shift_limit=20,
        sat_shift_limit=30,
        val_shift_limit=20,
        brightness_limit=0.2,
        contrast_limit=0.2,
    ):
        """Initialize the RustingAugmentation class with an image and parameters.

        Args:
            original_img_obj (PIL.Image.Image): The input image to be augmented.
            gauss_noise_var_limit (tuple): Variance limits for Gaussian noise.
            ison_noise_color_shift (tuple): Color shift limits for ISO noise.
            ison_noise_intensity (tuple): Intensity limits for ISO noise.
            multiplicative_noise_multiplier (tuple): Multiplier limits for multiplicative noise.
            random_fog_coef_range (tuple): Fog coefficient range for random fog.
            random_fog_alpha_coef (float): Alpha coefficient for random fog.
            hue_shift_limit (int): Hue shift limit for hue adjustment.
            sat_shift_limit (int): Saturation shift limit for color adjustment.
            val_shift_limit (int): Value shift limit for color adjustment.
            brightness_limit (float): Brightness limit for random brightness contrast.
            contrast_limit (float): Contrast limit for random brightness contrast.
        """
        self.original_img_obj = original_img_obj
        self.gauss_noise_var_limit = gauss_noise_var_limit
        self.ison_noise_color_shift = ison_noise_color_shift
        self.ison_noise_intensity = ison_noise_intensity
        self.multiplicative_noise_multiplier = multiplicative_noise_multiplier
        self.random_fog_coef_range = random_fog_coef_range
        self.random_fog_alpha_coef = random_fog_alpha_coef
        self.hue_shift_limit = hue_shift_limit
        self.sat_shift_limit = sat_shift_limit
        self.val_shift_limit = val_shift_limit
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit

    def apply(self):
        """
        Apply a rusting effect to the image.

        Returns:
            PIL.Image.Image: Augmented image with rusting effect.
        """
        # Combine various transformations to simulate rust
        aug = A.Compose(
            [
                A.OneOf(
                    [
                        A.GaussNoise(
                            var_limit=self.gauss_noise_var_limit, p=0.5
                        ),  # Add noise to simulate rust particles
                        A.ISONoise(
                            color_shift=self.ison_noise_color_shift,
                            intensity=self.ison_noise_intensity,
                            p=0.5,
                        ),
                    ],
                    p=0.5,
                ),
                A.OneOf(
                    [
                        A.MultiplicativeNoise(
                            multiplier=self.multiplicative_noise_multiplier,
                            per_channel=True,
                            elementwise=True,
                            p=0.5,
                        ),
                        A.RandomFog(
                            fog_coef_range=self.random_fog_coef_range,
                            alpha_coef=self.random_fog_alpha_coef,
                            p=0.5,
                        ),
                    ],
                    p=0.5,
                ),
                A.HueSaturationValue(
                    hue_shift_limit=self.hue_shift_limit,
                    sat_shift_limit=self.sat_shift_limit,
                    val_shift_limit=self.val_shift_limit,
                    p=0.5,
                ),  # Adjust colors to create rust-like hues
                A.RandomBrightnessContrast(
                    brightness_limit=self.brightness_limit,
                    contrast_limit=self.contrast_limit,
                    p=0.5,
                ),  # Adjust brightness and contrast to simulate rust
            ],
            p=1,
        )

        aug_img = aug(image=np.array(self.original_img_obj))["image"]
        return Image.fromarray(aug_img)
