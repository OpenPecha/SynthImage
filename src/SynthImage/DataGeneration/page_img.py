import os
import random

from SynthImage.Augmentation.bad_photocopy_augmentation import BadPhotoCopyAugmentation
from SynthImage.Augmentation.blur_augmentation import BlurAugmentation
from SynthImage.Augmentation.brightness_augmentation import BrightnessAugmentation
from SynthImage.Augmentation.contrast_augmentation import ContrastAugmentation
from SynthImage.Augmentation.deform_augmentation import DeformAugmentation
from SynthImage.Augmentation.dirty_augmentation import DirtySpotAugmentation
from SynthImage.Augmentation.dirty_rollers_augmentation import DirtyRollersAugmentation
from SynthImage.Augmentation.distort_augmentation import DistortAugmentation
from SynthImage.Augmentation.faxify_augmentation import FaxifyAugmentation
from SynthImage.Augmentation.flip_horizontal_augmentation import (
    FlipHorizontalAugmentation,
)
from SynthImage.Augmentation.flip_vertical_augmentation import FlipVerticalAugmentation
from SynthImage.Augmentation.grid_distort_augmentation import GridDistortAugmentation
from SynthImage.Augmentation.hue_saturation_augmentation import (
    HueSaturationAugmentation,
)
from SynthImage.Augmentation.ink_bleed_augmentation import InkBleedAugmentation
from SynthImage.Augmentation.low_ink_augmentation import LowInkPeriodicLinesAugmentation
from SynthImage.Augmentation.paper_texture_augmentation import PaperTextureAugmentation
from SynthImage.Augmentation.random_rain_augmentation import RandomRainAugmentation
from SynthImage.Augmentation.random_shadow_augmentation import RandomShadowAugmentation
from SynthImage.Augmentation.rotate_augmentation import RotationAugmentation
from SynthImage.Augmentation.rusting_augmentation import RustingAugmentation
from SynthImage.Augmentation.scribble_augmentation import ScribbleAugmentation
from SynthImage.Augmentation.sharpness_augmentation import SharpnessAugmentation
from SynthImage.Augmentation.solarize_augmentation import SolarizeAugmentation
from SynthImage.Augmentation.sun_flare_augmentation import SunFlareAugmentation
from SynthImage.Augmentation.torn_augmentation import TornAugmentation
from SynthImage.Augmentation.transpose_augmentation import TransposeAugmentation
from SynthImage.Augmentation.water_mark_augmentation import WaterMarkAugmentation
from SynthImage.SynthPageImage.page_image import PageGenerator


def apply_augmentations(image):
    """Apply a random number of augmentations (1 to 3) to the image.

    Args:
        image (PIL.Image.Image): The image to augment.

    Returns:
        PIL.Image.Image: The augmented image.
    """
    # Define possible augmentations with short names
    augmentation_classes = {
        "BPC": BadPhotoCopyAugmentation,
        "BLR": BlurAugmentation,
        "BRI": BrightnessAugmentation,
        "CON": ContrastAugmentation,
        "DEF": DeformAugmentation,
        "DSTP": DirtySpotAugmentation,
        "DRL": DirtyRollersAugmentation,
        "DST": DistortAugmentation,
        "FAX": FaxifyAugmentation,
        "FLH": FlipHorizontalAugmentation,
        "FLV": FlipVerticalAugmentation,
        "GDS": GridDistortAugmentation,
        "HUE": HueSaturationAugmentation,
        "INK": InkBleedAugmentation,
        "LNK": LowInkPeriodicLinesAugmentation,
        "PPR": PaperTextureAugmentation,
        "RRN": RandomRainAugmentation,
        "RSH": RandomShadowAugmentation,
        "ROT": RotationAugmentation,
        "RST": RustingAugmentation,
        "SCR": ScribbleAugmentation,
        "SHP": SharpnessAugmentation,
        "SOL": SolarizeAugmentation,
        "SUN": SunFlareAugmentation,
        "TRN": TornAugmentation,
        "TRP": TransposeAugmentation,
        "WMK": WaterMarkAugmentation,
    }
    # Randomly select the number of augmentations to apply (1 to 3)
    num_augmentations = random.randint(1, 3)
    # Convert the keys to a list to ensure it is a sequence
    augmentation_keys = list(augmentation_classes.keys())

    # Randomly select the augmentations to apply
    selected_augmentations = random.sample(augmentation_keys, num_augmentations)
    applied_augmentations = []

    for aug_short_name in selected_augmentations:
        # Apply the augmentation
        aug_class = augmentation_classes[aug_short_name]
        if aug_class == BadPhotoCopyAugmentation:
            aug = aug_class(image)
        elif aug_class == BlurAugmentation:
            aug = aug_class(image)
        elif aug_class == BrightnessAugmentation:
            factor = random.uniform(
                0.8, 1.2
            )  # Random brightness factor between 0.8 and 1.2
            aug = aug_class(image, factor)
        elif aug_class == ContrastAugmentation:
            factor = random.uniform(
                0.8, 1.2
            )  # Random contrast factor between 0.8 and 1.2
            aug = aug_class(image, factor)
        elif aug_class == DeformAugmentation:
            grid = random.randint(10, 30)  # Random grid size between 10 and 30
            multiplier = random.uniform(
                5.0, 15.0
            )  # Random multiplier between 5.0 and 15.0
            offset = random.uniform(50.0, 100.0)  # Random offset between 50.0 and 100.0
            aug = aug_class(image, grid, multiplier, offset)
        elif aug_class == DirtySpotAugmentation:
            num_spots = random.randint(
                1, 5
            )  # Random number of dirty spots between 1 and 5
            dirty_spots = [
                (
                    random.randint(0, image.width - 1),
                    random.randint(0, image.height - 1),
                    random.randint(10, 30),
                )
                for _ in range(num_spots)
            ]
            aug = aug_class(image, dirty_spots)
        elif aug_class == DirtyRollersAugmentation:
            aug = aug_class(image)
        elif aug_class == DistortAugmentation:
            severity = random.uniform(
                0.5, 1.5
            )  # Random severity factor between 0.5 and 1.5
            aug = aug_class(image, severity)
        elif aug_class == FaxifyAugmentation:
            aug = aug_class(image)
        elif aug_class == FlipHorizontalAugmentation:
            aug = aug_class(image)
        elif aug_class == FlipVerticalAugmentation:
            aug = aug_class(image)
        elif aug_class == GridDistortAugmentation:
            aug = aug_class(image)
        elif aug_class == HueSaturationAugmentation:
            aug = aug_class(image)
        elif aug_class == InkBleedAugmentation:
            aug = aug_class(image)
        elif aug_class == LowInkPeriodicLinesAugmentation:
            aug = aug_class(image)
        elif aug_class == PaperTextureAugmentation:
            texture_types = ["rough_stains", "fine_grain", "linen", "grain"]
            texture_type = random.choice(texture_types)
            texture_width = random.randint(100, 1000)
            texture_height = random.randint(100, 1000)
            quilt_texture = random.randint(0, 10)
            alpha = random.uniform(0.5, 0.7)
            beta = random.uniform(0.3, 0.5)
            aug = aug_class(
                image,
                texture_type=texture_type,
                texture_width=texture_width,
                texture_height=texture_height,
                quilt_texture=quilt_texture,
                alpha=alpha,
                beta=beta,
            )
        elif aug_class == RandomRainAugmentation:
            aug = aug_class(image)
        elif aug_class == RandomShadowAugmentation:
            aug = aug_class(image)
        elif aug_class == RotationAugmentation:
            angle = random.randint(-5, 5)  # Random angle between -30 and 30 degrees
            aug = aug_class(image, angle)
        elif aug_class == RustingAugmentation:
            rust_intensity = random.uniform(
                0.1, 1.0
            )  # Random rust intensity between 0.1 and 1.0
            aug = aug_class(image, rust_intensity)
        elif aug_class == ScribbleAugmentation:
            aug = aug_class(image)
        elif aug_class == SharpnessAugmentation:
            factor = random.uniform(
                0.5, 2.0
            )  # Random sharpness factor between 0.5 and 2.0
            aug = aug_class(image, factor)
        elif aug_class == SolarizeAugmentation:
            aug = aug_class(image)
        elif aug_class == SunFlareAugmentation:
            aug = aug_class(image)
        elif aug_class == TornAugmentation:
            num_tears = random.randint(3, 7)  # Random number of tears between 3 and 7
            tear_size = random.randint(20, 50)  # Random tear size between 20 and 50
            jagged_step = random.randint(3, 7)  # Random jagged step between 3 and 7
            jagged_variability = random.randint(
                1, 10
            )  # Random jagged variability between 1 and 10
            aug = aug_class(
                image, num_tears, tear_size, jagged_step, jagged_variability
            )
        elif aug_class == TransposeAugmentation:
            aug = aug_class(image)
        elif aug_class == WaterMarkAugmentation:
            aug = aug_class(image)
        else:
            raise ValueError(f"Unsupported augmentation class: {aug_class}")

        # Apply the augmentation
        image = aug.apply()
        # Add the name of the applied augmentation to the list
        applied_augmentations.append(aug_short_name)

    return image, applied_augmentations


def main():
    text_file_path = "./data/texts/kangyur/v001_plain.txt"
    fonts_folder = "/Users/ogyenthoga/Desktop/Work/SynthImage/data/fonts"
    output_dir = "/Users/ogyenthoga/Desktop/Work/SynthImage/data/SynthPageImages"

    # Extract the base name of the text file (without extension) to use as a prefix
    text_file_name = os.path.basename(text_file_path).split(".")[0]

    # Read the content of the text file
    with open(text_file_path, encoding="utf-8") as file:
        vol_text = file.read()

    # Font sizes to randomly choose from
    font_sizes = [20, 25, 30, 35]

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the PageGenerator (shared across different font sizes)
    pgobject = PageGenerator(
        left_padding=80,
        right_padding=80,
        top_padding=40,
        bottom_padding=40,
    )

    # Iterate through each font file in the fonts folder
    for subfolder in os.listdir(fonts_folder):
        subfolder_path = os.path.join(fonts_folder, subfolder)

        if os.path.isdir(subfolder_path):
            for font_file in os.listdir(subfolder_path):
                if font_file.endswith(".ttf") or font_file.endswith(".otf"):
                    font_path = os.path.join(subfolder_path, font_file)

                    # Generate images with a different random font size for each page
                    pages = pgobject.get_pages(vol_text)
                    for i, page_text in enumerate(pages):
                        # Randomize font size for each page
                        font_size = random.choice(font_sizes)

                        # Generate the page image
                        page_image = pgobject.generate_page_image(
                            page_text, font_size, font_path
                        )

                        # Apply random number of augmentations (1 to 3)
                        augmented_image, applied_aug_names = apply_augmentations(
                            page_image
                        )

                        # Create a string of augmentation names
                        aug_name = "_".join(applied_aug_names)

                        # Save the generated image with the font size and page sequence in the file name
                        font_name = os.path.basename(font_path).split(".")[0]
                        output_filename = f"{aug_name}_{text_file_name}_size{font_size}_{subfolder}_{font_name}_page_{i+1:04d}.png"  # noqa
                        output_path = os.path.join(output_dir, output_filename)

                        augmented_image.save(output_path)  # Save the image
                        print(
                            f"Saved image {output_filename} to {output_path}."
                        )  # Debug: Print save info


if __name__ == "__main__":
    main()
