from typing import List

import cv2
import numpy as np

from SynthImage.SynthPageImage.pecha_format_page_image import (
    draw_tight_line_bounding_boxes,
)


def extract_lines_from_image(
    image: np.ndarray, filename: str, page_number: int
) -> List[np.ndarray]:
    """
    Extract lines from a page image using rectangle bounding box details.
    Adds fixed padding to top and bottom to make the extraction more consistent.

    Args:
        image (np.ndarray): Input page image.
        filename (str): Name of the image file.
        page_number (int): Page number of the image.

    Returns:
        List[np.ndarray]: A list of extracted line images.

    This function uses the draw_tight_line_bounding_boxes function to get bounding box details,
    sorts them by vertical position, and extracts each line with added padding.
    """
    # Get bounding box details from pecha_format_page_image.py
    _, _, bbox_details, _ = draw_tight_line_bounding_boxes(image, filename, page_number)

    line_images = []
    page_height, page_width = image.shape[:2]

    # Sort bounding boxes by their vertical position (y-coordinate)
    sorted_bboxes = sorted(bbox_details, key=lambda x: x["center"][1])

    for bbox in sorted_bboxes:
        x, y = bbox["points"][0]
        w = bbox["width"]
        h = bbox["height"]

        # Add fixed padding of 6 pixels to both top and bottom
        padding = 15
        top = max(0, int(y) - padding)
        bottom = min(page_height, int(y + h) + padding)
        left = int(x)
        right = int(x + w)

        # Extract the line image
        line_img = image[top:bottom, left:right]

        line_images.append(line_img)

    return line_images


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """
    Enhance the contrast of the image to improve text extraction.

    Args:
        image (np.ndarray): Input image in BGR format.

    Returns:
        np.ndarray: Contrast-enhanced image.

    This function converts the image to LAB color space, applies CLAHE (Contrast Limited
    Adaptive Histogram Equalization) to the L channel, and then converts it back to BGR.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_image = cv2.merge((cl, a, b))
    return cv2.cvtColor(enhanced_image, cv2.COLOR_LAB2BGR)


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess the image to improve text extraction.

    Args:
        image (np.ndarray): Input image in BGR format.

    Returns:
        np.ndarray: Preprocessed binary image.

    This function enhances contrast, converts to grayscale, applies adaptive thresholding,
    and performs morphological operations to prepare the image for line extraction.
    """
    enhanced_image = enhance_contrast(image)
    gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.dilate(binary, kernel, iterations=1)
    return binary


def extract_lines_with_preprocessing(
    image: np.ndarray, filename: str, page_number: int
) -> List[np.ndarray]:
    """
    Extract lines from a page image with preprocessing, using rectangle bounding box details.

    Args:
        image (np.ndarray): Input page image from which to extract lines.
        filename (str): Name of the image file.
        page_number (int): Page number of the image.

    Returns:
        List[np.ndarray]: A list of extracted and preprocessed line images.

    This function preprocesses the input image and then extracts lines using the
    extract_lines_from_image function.
    """
    preprocessed_image = preprocess_image(image)
    line_images = extract_lines_from_image(preprocessed_image, filename, page_number)
    return line_images
