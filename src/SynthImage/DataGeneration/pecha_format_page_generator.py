import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from SynthImage.Augmentation.background import BackgroundAugmentation
# Import augmentations with correct initialization
from SynthImage.Augmentation.bad_photocopy_augmentation import BadPhotoCopyAugmentation
from SynthImage.Augmentation.blur_augmentation import BlurAugmentation
from SynthImage.Augmentation.brightness_augmentation import BrightnessAugmentation
from SynthImage.Augmentation.contrast_augmentation import ContrastAugmentation
from SynthImage.Augmentation.distort_augmentation import DistortAugmentation
from SynthImage.Augmentation.grid_distort_augmentation import GridDistortAugmentation
from SynthImage.Augmentation.hue_saturation_augmentation import (
    HueSaturationAugmentation,
)
from SynthImage.Augmentation.ink_bleed_augmentation import InkBleedAugmentation
from SynthImage.Augmentation.random_shadow_augmentation import RandomShadowAugmentation
from SynthImage.DataGeneration.augmentation import apply_augmentations
from SynthImage.LineExtraction.line_extraction import extract_lines_with_preprocessing
from SynthImage.SynthPageImage.pecha_format_page_image import PechaPageGenerator


def extract_green_contours(image):
    """
    Extract green pixel coordinates of rectangle and polygon bounding boxes surrounding text lines.

    Args:
    image (numpy.ndarray): The input augmented image in BGR format.

    Returns:
    tuple: Two lists of dictionaries, one for rectangle bounding boxes and one for polygon bounding boxes.
    """
    # Convert BGR to HSV color space for better green detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range for green color in HSV
    lower_green = np.array([35, 50, 50])  # Lower bound for green detection
    upper_green = np.array([85, 255, 255])  # Upper bound for green detection

    # Create a mask for green pixels
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Perform morphological operations to remove noise (optional but useful in some cases)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours of green pixels
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rectangle_contours = []
    polygon_contours = []
    for i, contour in enumerate(contours):
        # Get the bounding rectangle of the contour
        x, y, w, h = cv2.boundingRect(contour)  # Rectangle bounding box

        # Get all points of the contour (polygon)
        points = contour.reshape(-1, 2).tolist()

        # Save rectangle bounding box and polygon points
        rectangle_contours.append(
            {
                "id": f"rect_contour_{i}",
                "rect": (x, y, w, h),  # Rectangle format: (x, y, width, height)
            }
        )

        polygon_contours.append(
            {
                "id": f"poly_contour_{i}",
                "points": points,  # List of points in the polygon
            }
        )

    print(f"Number of text line contours found: {len(contours)}")

    return rectangle_contours, polygon_contours


def apply_text_color(image):
    """
    Apply red, faded red, or black color to text with given probabilities.

    Args:
    image (PIL.Image): The input image.

    Returns:
    PIL.Image: The image with potentially colored text.
    """
    # Convert to RGBA to preserve transparency
    image = image.convert("RGBA")
    data = np.array(image)

    # Define colors (RGBA)
    red = (255, 0, 0, 255)
    faded_red = (255, 100, 100, 255)
    black = (0, 0, 0, 255)

    # Find all non-white (text) pixels
    r, g, b, a = data.T
    text_areas = (r < 245) & (g < 245) & (b < 245)

    # Apply color based on probability
    rand_val = random.random()
    if rand_val < 0.1:  # 10% chance for red
        chosen_color = red
    elif rand_val < 0.2:  # 10% chance for faded red
        chosen_color = faded_red
    else:  # 80% chance for black
        chosen_color = black

    # Apply the chosen color to text areas
    data[..., :-1][text_areas.T] = chosen_color[:-1]

    return Image.fromarray(data)


if __name__ == "__main__":
    # Path to the text file
    text_file_path = "./data/texts/kangyur/v001_plain.txt"

    # Path to the font folder containing multiple font files
    font_folder_path = Path("./data/fonts/Drutsa_short/")
    font_sizes = [10, 11, 30]  # Adjust the font size as needed
    dimensions = [
        (1123, 265),
        (794, 265),
        (1680, 402),
        (1800, 630),
        (2864, 680),
    ]
    background_folder = "./data/backgrounds"

    # Define probabilities for each dimension
    dimension_probs = [0.4, 0.1, 0.1, 0.2, 0.2]

    # Define probabilities for each font size
    font_size_probs = {10: 0, 11: 0, 30: 1}

    # Read the volume text from the file
    with open(text_file_path, encoding="utf-8") as file:
        vol_text = file.read()

    # Initialize the PageGenerator
    page_generator = PechaPageGenerator(
        left_padding=50,
        right_padding=50,
        top_padding=45,
        bottom_padding=45,
        background_folder=background_folder,
        dimensions=dimensions,
    )

    augmentations = [
        (BadPhotoCopyAugmentation, 0.15),
        (BlurAugmentation, 0.1),
        (DistortAugmentation, 0.1),
        (GridDistortAugmentation, 0.1),
        (RandomShadowAugmentation, 0.05),
        (InkBleedAugmentation, 0.2),
        (BrightnessAugmentation, 0.15),
        (ContrastAugmentation, 0.1),
        (HueSaturationAugmentation, 0.05),
        (BackgroundAugmentation, 0.3),
    ]

    # Iterate over all font files in the font folder
    for font_path in font_folder_path.glob("*.ttf"):
        font_name = font_path.stem  # Extract the font name from the font path

        # Generate the page images and get bbox details
        (
            pages_with_bbox,
            pages_without_bbox,
            dimension_counter,
            bbox_details,
            polygon_bbox_details,
        ) = page_generator.generate_pecha_page_images(
            vol_text, font_sizes, font_path, dimension_probs, font_size_probs
        )

        # Store bbox details in JSONL file for plain synthetic pages
        plain_bbox_jsonl_file = "plain_bbox_details.jsonl"
        with open(plain_bbox_jsonl_file, "w") as f:
            for bbox in bbox_details:
                f.write(json.dumps(bbox) + "\n")

        # Store polygon bbox details in a separate JSON file for plain synthetic pages
        plain_polygon_jsonl_file = "plain_polygon_bbox_details.jsonl"
        with open(plain_polygon_jsonl_file, "w") as f:
            for polygon_bbox in polygon_bbox_details:
                f.write(json.dumps(polygon_bbox) + "\n")

        # Define the output paths for the current font
        original_output_path = Path(f"./data/pecha_output/original/{font_name}")
        augmented_output_path_with_bbox = Path(
            f"./data/pecha_output/augmented/{font_name}/with_bbox"
        )
        augmented_output_path_without_bbox = Path(
            f"./data/pecha_output/augmented/{font_name}/without_bbox"
        )
        original_lines_output_path = Path(
            f"./data/pecha_output/original/{font_name}/lines"
        )
        augmented_lines_output_path = Path(
            f"./data/pecha_output/augmented/{font_name}/lines"
        )

        # Create directories
        original_output_path.mkdir(parents=True, exist_ok=True)
        augmented_output_path_with_bbox.mkdir(parents=True, exist_ok=True)
        augmented_output_path_without_bbox.mkdir(parents=True, exist_ok=True)
        original_lines_output_path.mkdir(parents=True, exist_ok=True)
        augmented_lines_output_path.mkdir(parents=True, exist_ok=True)

        # Track the count of images for each dimension
        for i, (
            (page_img_with_bbox, font_size),
            (page_img_without_bbox, _),
        ) in enumerate(zip(pages_with_bbox, pages_without_bbox)):
            # Apply text color with probabilities
            page_img_with_bbox = apply_text_color(page_img_with_bbox)
            page_img_without_bbox = apply_text_color(page_img_without_bbox)

            # Get the dimension of the current page image
            page_width, page_height = page_img_with_bbox.size

            # Create the filename with dimension, font size, and font name
            dimension_prefix = f"{page_width}x{page_height}"
            # Increment the dimension counter
            dimension_counter[dimension_prefix] += 1
            count = dimension_counter[dimension_prefix]

            # Filename for the original page image
            filename = f"page_{i + 1}_{dimension_prefix}_count_{count}_font{font_size}_{font_name}.png"
            page_output_path_with_bbox = original_output_path / f"with_bbox_{filename}"
            page_output_path_without_bbox = (
                original_output_path / f"without_bbox_{filename}"
            )

            # Save the original page images
            page_img_with_bbox.save(page_output_path_with_bbox)
            page_img_without_bbox.save(page_output_path_without_bbox)

            # Convert the PIL images to a format suitable for OpenCV
            page_img_cv_with_bbox = cv2.cvtColor(
                np.array(page_img_with_bbox), cv2.COLOR_RGB2BGR
            )
            page_img_cv_without_bbox = cv2.cvtColor(
                np.array(page_img_without_bbox), cv2.COLOR_RGB2BGR
            )

            # Apply random augmentations to both types of images
            (
                augmented_img_with_bbox,
                applied_augmentations_with_bbox,
            ) = apply_augmentations(page_img_cv_with_bbox, augmentations)
            (
                augmented_img_without_bbox,
                applied_augmentations_without_bbox,
            ) = apply_augmentations(page_img_cv_without_bbox, augmentations)

            # Create the filename for the augmented page images with augmentation details
            aug_filename_prefix_with_bbox = (
                "_".join([f"aug_{aug[:3]}" for aug in applied_augmentations_with_bbox])
                if applied_augmentations_with_bbox
                else "none"
            )
            aug_filename_prefix_without_bbox = (
                "_".join(
                    [f"aug_{aug[:3]}" for aug in applied_augmentations_without_bbox]
                )
                if applied_augmentations_without_bbox
                else "none"
            )

            aug_filename_with_bbox = f"with_bounding_box_aug_page_{i+1}_{dimension_prefix}_count_{count}_font{font_size}_{font_name}_{aug_filename_prefix_with_bbox}.png"  # noqa
            aug_filename_without_bbox = f"without_bounding_box_aug_page_{i+1}_{dimension_prefix}_count_{count}_font{font_size}_{font_name}_{aug_filename_prefix_without_bbox}.png"  # noqa

            aug_page_output_path_with_bbox = (
                augmented_output_path_with_bbox / aug_filename_with_bbox
            )
            aug_page_output_path_without_bbox = (
                augmented_output_path_without_bbox / aug_filename_without_bbox
            )

            # Save the augmented page images
            cv2.imwrite(str(aug_page_output_path_with_bbox), augmented_img_with_bbox)
            cv2.imwrite(
                str(aug_page_output_path_without_bbox), augmented_img_without_bbox
            )

            # Extract green contours from the augmented image with bbox
            rectangle_contours, polygon_contours = extract_green_contours(
                augmented_img_with_bbox
            )

            # Create bbox details for the augmented image and crop text lines
            for j, (rect_contour, poly_contour) in enumerate(
                zip(rectangle_contours, polygon_contours)
            ):
                augmented_rect_bbox_details = {
                    "id": f"{aug_filename_with_bbox}_rect_bbox_{j}",
                    "image": f"https://s3.amazonaws.com/monlam.ai.ocr/line_segmentations/Images/{aug_filename_with_bbox}_rect_bbox_{j}",  # noqa
                    "rect": rect_contour["rect"],
                    "page_number": i + 1,
                }

                augmented_poly_bbox_details = {
                    "id": f"{aug_filename_with_bbox}_poly_bbox_{j}",
                    "image": f"https://s3.amazonaws.com/monlam.ai.ocr/line_segmentations/Images/{aug_filename_with_bbox}_poly_bbox_{j}",  # noqa
                    "points": poly_contour["points"],
                    "page_number": i + 1,
                }

                # Store augmented bbox details in separate JSON files
                augmented_rect_bbox_jsonl_file = (
                    f"augmented_rect_bbox_details_{font_name}.jsonl"
                )
                augmented_poly_bbox_jsonl_file = (
                    f"augmented_poly_bbox_details_{font_name}.jsonl"
                )

                with open(augmented_rect_bbox_jsonl_file, "a") as f:
                    f.write(json.dumps(augmented_rect_bbox_details) + "\n")

                with open(augmented_poly_bbox_jsonl_file, "a") as f:
                    f.write(json.dumps(augmented_poly_bbox_details) + "\n")

            # Extract lines from the original page image without bbox
            original_lines = extract_lines_with_preprocessing(
                cv2.cvtColor(np.array(page_img_without_bbox), cv2.COLOR_RGB2BGR),
                filename,
                i + 1,
            )

            # Save original extracted lines
            for j, line_img in enumerate(original_lines):
                line_filename = f"original_{filename}_line_{j}.png"
                line_output_path = original_lines_output_path / line_filename
                cv2.imwrite(str(line_output_path), line_img)

            # Extract lines from the augmented image without bbox
            augmented_lines = extract_lines_with_preprocessing(
                augmented_img_without_bbox, aug_filename_without_bbox, i + 1
            )

            # Save augmented extracted lines
            for j, line_img in enumerate(augmented_lines):
                line_filename = f"{aug_filename_without_bbox}_line_{j}.png"
                line_output_path = augmented_lines_output_path / line_filename
                cv2.imwrite(str(line_output_path), line_img)
