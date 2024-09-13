import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from SynthImage.Augmentation.background import BackgroundAugmentation
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
from SynthImage.SynthPageImage.modern_book_format_page_image import (
    ModernBookPageGenerator,
)


def extract_green_contours(image):
    """
    Extracts green contours from an input image.

    Args:
        image (numpy.ndarray): Input image in BGR format.

    Returns:
        tuple: A tuple containing two lists:
            - rectangle_contours: List of dictionaries with 'id' and 'rect' (x, y, w, h) for each contour.
            - polygon_contours: List of dictionaries with 'id' and 'points' for each contour.

    This function converts the image to HSV color space, creates a mask for green colors,
    applies morphological operations, and finds contours. It then processes these contours
    to create both rectangular and polygon representations.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rectangle_contours = []
    polygon_contours = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        points = contour.reshape(-1, 2).tolist()
        rectangle_contours.append({"id": f"rect_contour_{i}", "rect": (x, y, w, h)})
        polygon_contours.append({"id": f"poly_contour_{i}", "points": points})

    print(f"Number of text line contours found: {len(contours)}")
    return rectangle_contours, polygon_contours


if __name__ == "__main__":
    text_file_path = "./data/texts/kangyur/v001_plain.txt"
    font_folder_path = Path("./data/fonts/Drutsa_short/")
    font_sizes = [12, 14, 16, 30]
    dimensions = [
        (626, 771),
        (1063, 1536),
        (259, 194),
        (349, 522),
        (974, 1500),
        (968, 1440),
    ]
    background_folder = "./data/backgrounds"
    dimension_probs = [0.3, 0.3, 0.1, 0.1, 0.1, 0.1]
    font_size_probs = {12: 0.5, 14: 0.2, 16: 0.15, 30: 0.15}

    with open(text_file_path, encoding="utf-8") as file:
        vol_text = file.read()

    page_generator = ModernBookPageGenerator(
        left_padding=45,
        right_padding=45,
        top_padding=50,
        bottom_padding=50,
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

    for font_path in font_folder_path.glob("*.ttf"):
        font_name = font_path.stem

        (
            pages_with_bbox,
            pages_without_bbox,
            dimension_counter,
            bbox_details,
            polygon_bbox_details,
        ) = page_generator.generate_modern_page_images(
            vol_text, font_sizes, font_path, dimension_probs, font_size_probs
        )

        plain_bbox_jsonl_file = "plain_bbox_details.jsonl"
        with open(plain_bbox_jsonl_file, "w") as f:
            for bbox in bbox_details:
                f.write(json.dumps(bbox) + "\n")

        plain_polygon_jsonl_file = "plain_polygon_bbox_details.jsonl"
        with open(plain_polygon_jsonl_file, "w") as f:
            for polygon_bbox in polygon_bbox_details:
                f.write(json.dumps(polygon_bbox) + "\n")

        original_output_path = Path(f"./data/modern_book_output/original/{font_name}")
        augmented_output_path_with_bbox = Path(
            f"./data/modern_book_output/augmented/{font_name}/with_bbox"
        )
        augmented_output_path_without_bbox = Path(
            f"./data/modern_book_output/augmented/{font_name}/without_bbox"
        )
        original_lines_output_path = Path(
            f"./data/modern_book_output/original/{font_name}/lines"
        )
        augmented_lines_output_path = Path(
            f"./data/modern_book_output/augmented/{font_name}/lines"
        )

        original_output_path.mkdir(parents=True, exist_ok=True)
        augmented_output_path_with_bbox.mkdir(parents=True, exist_ok=True)
        augmented_output_path_without_bbox.mkdir(parents=True, exist_ok=True)
        original_lines_output_path.mkdir(parents=True, exist_ok=True)
        augmented_lines_output_path.mkdir(parents=True, exist_ok=True)

        dimension_counter = defaultdict(int)
        for i, (
            (page_img_with_bbox, font_size),
            (page_img_without_bbox, _),
        ) in enumerate(zip(pages_with_bbox, pages_without_bbox)):
            page_width, page_height = page_img_with_bbox.size
            dimension_prefix = f"{page_width}x{page_height}"
            dimension_counter[dimension_prefix] += 1
            count = dimension_counter[dimension_prefix]

            filename = f"page_{i + 1}_{dimension_prefix}_count_{count}_font{font_size}_{font_name}.png"
            page_output_path_with_bbox = original_output_path / f"with_bbox_{filename}"
            page_output_path_without_bbox = (
                original_output_path / f"without_bbox_{filename}"
            )

            page_img_with_bbox.save(page_output_path_with_bbox)
            page_img_without_bbox.save(page_output_path_without_bbox)

            page_img_cv_with_bbox = cv2.cvtColor(
                np.array(page_img_with_bbox), cv2.COLOR_RGB2BGR
            )
            page_img_cv_without_bbox = cv2.cvtColor(
                np.array(page_img_without_bbox), cv2.COLOR_RGB2BGR
            )

            (
                augmented_img_with_bbox,
                applied_augmentations_with_bbox,
            ) = apply_augmentations(page_img_cv_with_bbox, augmentations)
            (
                augmented_img_without_bbox,
                applied_augmentations_without_bbox,
            ) = apply_augmentations(page_img_cv_without_bbox, augmentations)

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

            cv2.imwrite(str(aug_page_output_path_with_bbox), augmented_img_with_bbox)
            cv2.imwrite(
                str(aug_page_output_path_without_bbox), augmented_img_without_bbox
            )

            rectangle_contours, polygon_contours = extract_green_contours(
                augmented_img_with_bbox
            )

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
