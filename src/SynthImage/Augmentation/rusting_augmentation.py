import albumentations as A
import numpy as np
from PIL import Image


class RustingAugmentation:
    def __init__(self, original_img_obj):
        """Initialize the RustingAugmentation class with an image.

        Args:
            original_img_obj (PIL.Image.Image): The input image to be augmented.
        """
        self.original_img_obj = original_img_obj

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
                            var_limit=(10.0, 50.0), p=0.5
                        ),  # Add noise to simulate rust particles
                        A.ISONoise(
                            color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5
                        ),
                    ],
                    p=0.5,
                ),
                A.OneOf(
                    [
                        A.MultiplicativeNoise(
                            multiplier=(0.9, 1.1),
                            per_channel=True,
                            elementwise=True,
                            p=0.5,
                        ),
                        A.RandomFog(
                            fog_coef_range=(0.1, 0.3),
                            alpha_coef=0.1,
                            p=0.5,
                        ),
                    ],
                    p=0.5,
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5
                ),  # Adjust colors to create rust-like hues
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),  # Adjust brightness and contrast to simulate rust
            ],
            p=1,
        )

        aug_img = aug(image=np.array(self.original_img_obj))["image"]
        return Image.fromarray(aug_img)
