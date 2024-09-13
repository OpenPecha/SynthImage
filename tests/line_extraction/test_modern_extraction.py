from pathlib import Path

import cv2
import numpy as np

from SynthImage.LineExtraction.line_extraction import extract_lines_from_image


def test_extract_lines_from_image():
    # Load the test image
    image_path = Path(
        "tests/data/modern_format_page_img/1063x1536_modern_format_page_image_without_bbox.png"
    )
    image = cv2.imread(str(image_path))

    # Extract lines from the image
    filename = image_path.name
    page_number = 1
    extracted_lines = extract_lines_from_image(image, filename, page_number)

    # Assertions to verify the extracted lines
    assert len(extracted_lines) > 0, "No lines were extracted from the image"

    for i, line_img in enumerate(extracted_lines):
        # Check if each extracted line is a valid image
        assert isinstance(
            line_img, np.ndarray
        ), f"Extracted line {i} is not a numpy array"
        assert (
            line_img.shape[0] > 0 and line_img.shape[1] > 0
        ), f"Extracted line {i} has invalid dimensions"

        # Save each extracted line for visual inspection
        output_dir = Path("./tests/output/extracted_lines_modern")
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_dir / f"line_{i}.png"), line_img)
