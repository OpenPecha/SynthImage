import random

import cv2
import numpy as np
from PIL import Image

from SynthImage.Augmentation.background import BackgroundAugmentation
# Import augmentations with correct initialization
from SynthImage.Augmentation.bad_photocopy_augmentation import BadPhotoCopyAugmentation
from SynthImage.Augmentation.blur_augmentation import BlurAugmentation
from SynthImage.Augmentation.brightness_augmentation import BrightnessAugmentation
from SynthImage.Augmentation.contrast_augmentation import ContrastAugmentation
from SynthImage.Augmentation.distort_augmentation import (
    DistortAugmentation,
    DistortionMode,
)
from SynthImage.Augmentation.grid_distort_augmentation import GridDistortAugmentation
from SynthImage.Augmentation.hue_saturation_augmentation import (
    HueSaturationAugmentation,
)
from SynthImage.Augmentation.ink_bleed_augmentation import InkBleedAugmentation
from SynthImage.Augmentation.random_shadow_augmentation import RandomShadowAugmentation


def apply_augmentations(image, augmentations):
    """
    Apply a random selection of image augmentations to the input image.

    Args:
        image (numpy.ndarray): The input image in BGR format.
        augmentations (list): A list of tuples, where each tuple contains an augmentation class and its probability.

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: The augmented image in BGR format.
            - list: A list of strings describing the applied augmentations.

    This function randomly selects 1 to 3 augmentations from the provided list and applies them to the input image.
    Each augmentation is applied with its associated probability. The function handles the initialization of each
    augmentation with specific parameters where necessary. It also keeps track of the applied augmentations and
    their parameters for logging purposes.

    The function supports various types of augmentations, including:
    - Bad photocopy effect
    - Brightness adjustment
    - Contrast adjustment
    - Distortion (additive or subtractive)
    - Blur (Gaussian, median, or box)
    - Grid distortion
    - Hue and saturation adjustment
    - Ink bleed effect
    - Random shadow addition
    - Background replacement

    For each augmentation, the function randomly selects appropriate parameters within predefined ranges.
    """
    applied_augmentations = []
    augmented_image = image.copy()

    # Randomly select 1 to 3 augmentations
    num_augmentations = random.randint(1, 3)
    selected_augmentations = random.sample(augmentations, num_augmentations)

    for aug, prob in selected_augmentations:
        if random.random() < prob:
            # Initialize augmentation with specific arguments if necessary
            if aug == BadPhotoCopyAugmentation:
                augmentation = aug(
                    original_img_obj=Image.fromarray(
                        cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB)
                    )
                )
            elif aug == BrightnessAugmentation:
                random_factor = random.uniform(
                    0.8, 1.2
                )  # Random brightness factor between 0.8 and 1.2
                augmentation = aug(
                    original_img_obj=Image.fromarray(
                        cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB)
                    ),
                    factor=random_factor,
                )
            elif aug == ContrastAugmentation:
                random_factor = random.uniform(
                    0.8, 1.5
                )  # Random contrast factor between 0.8 and 1.5
                augmentation = aug(
                    original_img_obj=Image.fromarray(
                        cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB)
                    ),
                    factor=random_factor,
                )
            elif aug == DistortAugmentation:
                mode = random.choice(
                    [DistortionMode.additive, DistortionMode.subtractive]
                )  # Randomly choose mode
                edge_tresh1 = random.randint(50, 150)  # Random edge threshold 1
                edge_tresh2 = random.randint(150, 300)  # Random edge threshold 2
                kernel_width = random.randint(1, 5)  # Random kernel width
                kernel_height = random.randint(1, 5)  # Random kernel height
                kernel_iterations = random.randint(
                    1, 5
                )  # Random number of kernel iterations
                augmentation = aug(
                    original_img_obj=Image.fromarray(
                        cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB)
                    ),
                    mode=mode,
                    edge_tresh1=edge_tresh1,
                    edge_tresh2=edge_tresh2,
                    kernel_width=kernel_width,
                    kernel_height=kernel_height,
                    kernel_iterations=kernel_iterations,
                )
            elif aug == BlurAugmentation:
                blur_type = random.choice(["gaussian", "median", "box"])
                kernel_size = random.choice([3, 5, 7])
                augmentation = aug(
                    original_img_obj=Image.fromarray(
                        cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB)
                    ),
                    blur_type=blur_type,
                    kernel_size=kernel_size,
                )
            elif aug == GridDistortAugmentation:
                num_steps = random.randint(5, 15)
                distort_limit = random.uniform(0.1, 0.5)
                augmentation = aug(
                    original_img_obj=Image.fromarray(
                        cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB)
                    ),
                    num_steps=num_steps,
                    distort_limit=distort_limit,
                )
            elif aug == HueSaturationAugmentation:
                hue_shift = random.uniform(-20, 20)
                saturation_shift = random.uniform(0.8, 1.2)
                augmentation = aug(
                    original_img_obj=Image.fromarray(
                        cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB)
                    ),
                    hue_shift=hue_shift,
                    saturation_shift=saturation_shift,
                )
            elif aug == InkBleedAugmentation:
                intensity = random.uniform(0.1, 0.5)
                kernel_size = random.choice([3, 5, 7])
                augmentation = aug(
                    original_img_obj=Image.fromarray(
                        cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB)
                    ),
                    intensity=intensity,
                    kernel_size=kernel_size,
                )
            elif aug == RandomShadowAugmentation:
                num_shadows = random.randint(1, 3)
                shadow_dimension = random.randint(50, 200)
                augmentation = aug(
                    original_img_obj=Image.fromarray(
                        cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB)
                    ),
                    num_shadows=num_shadows,
                    shadow_dimension=shadow_dimension,
                )
            elif aug == BackgroundAugmentation:
                background_folder = "./data/backgrounds"
                augmentation = aug(
                    original_img_obj=Image.fromarray(
                        cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB)
                    ),
                    background_folder=background_folder,
                )
                augmented_image = np.array(augmentation.apply())
                augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
                applied_augmentations.append("BKG")
                continue
            else:
                augmentation = aug(
                    original_img_obj=Image.fromarray(
                        cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB)
                    )
                )

            augmented_image = np.array(augmentation.apply())
            if augmented_image.ndim == 3 and augmented_image.shape[2] == 3:
                augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)

            # For augmentations with factors or parameters, include them in the applied augmentations list
            if aug in [BrightnessAugmentation, ContrastAugmentation]:
                applied_augmentations.append(
                    f"{aug.__name__[:3]}(factor={random_factor: .2f})"
                )
            elif aug == DistortAugmentation:
                applied_augmentations.append(
                    f"{aug.__name__[:3]}(mode={mode.name[:3]}, et1={edge_tresh1}, et2={edge_tresh2}, kw={kernel_width}, kh={kernel_height}, ki={kernel_iterations})"  # noqa
                )
            elif aug == BlurAugmentation:
                applied_augmentations.append(
                    f"{aug.__name__[:3]}(type={blur_type}, size={kernel_size})"
                )
            elif aug == GridDistortAugmentation:
                applied_augmentations.append(
                    f"{aug.__name__[:3]}(steps={num_steps}, limit={distort_limit: .2f})"
                )
            elif aug == HueSaturationAugmentation:
                applied_augmentations.append(
                    f"{aug.__name__[:3]}(hue={hue_shift: .2f}, sat={saturation_shift: .2f})"
                )
            elif aug == InkBleedAugmentation:
                applied_augmentations.append(
                    f"{aug.__name__[:3]}(int={intensity: .2f}, size={kernel_size})"
                )
            elif aug == RandomShadowAugmentation:
                applied_augmentations.append(
                    f"{aug.__name__[:3]}(num={num_shadows}, dim={shadow_dimension})"
                )
            else:
                applied_augmentations.append(f"{aug.__name__[:3]}")

    return augmented_image, applied_augmentations
