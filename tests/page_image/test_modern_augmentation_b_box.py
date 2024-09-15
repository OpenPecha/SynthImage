import json
import os

import cv2
import numpy as np

from SynthImage.DataGeneration.modern_book_format_page_generator import (
    extract_green_contours,
)


def test_extract_green_contours():
    # Load the image from the specified path
    image_path = (
        "tests/data/modern_format_page_img/1063x1536_modern_format_page_image.png"
    )
    image = cv2.imread(image_path)

    # Extract green contours
    rectangle_contours, polygon_contours = extract_green_contours(image)

    # Create a blank white image
    height, width = image.shape[:2]
    blank_image = np.ones((height, width, 3), np.uint8) * 255

    # Draw contours on the blank image
    for contour in polygon_contours:
        cv2.drawContours(blank_image, [np.array(contour["points"])], -1, (0, 255, 0), 2)

    # Save the visualized bounding box
    cv2.imwrite("visualized_bounding_box_contours.png", blank_image)

    # Create a copy of the original image to draw contours on
    image_with_contours = image.copy()

    # Draw contours on the original image
    for contour in polygon_contours:
        cv2.drawContours(
            image_with_contours, [np.array(contour["points"])], -1, (0, 255, 0), 2
        )

    # Save the image with contours
    cv2.imwrite("original_image_with_contours.png", image_with_contours)

    # Create bbox details for the augmented image
    for j, (rect_contour, poly_contour) in enumerate(
        zip(rectangle_contours, polygon_contours)
    ):
        augmented_bbox_details = {
            "id": f"{os.path.basename(image_path)}_bbox_{j}",
            "image": f"https://s3.amazonaws.com/monlam.ai.ocr/line_segmentations/Images/{os.path.basename(image_path)}_bbox_{j}",  # noqa
            "rect": rect_contour["rect"],
            "points": poly_contour["points"],
            "page_number": 1,
        }

        # Store augmented bbox details in a JSON file
        augmented_bbox_jsonl_file = "augmented_bbox_details.jsonl"

        # Append the new details to the file
        with open(augmented_bbox_jsonl_file, "a") as f:
            f.write(json.dumps(augmented_bbox_details) + "\n")

    # Add assertions to make this a proper test
    assert len(rectangle_contours) > 0, "No rectangle contours found"
    assert len(polygon_contours) > 0, "No polygon contours found"
    assert len(rectangle_contours) == len(
        polygon_contours
    ), "Number of rectangle and polygon contours do not match"
